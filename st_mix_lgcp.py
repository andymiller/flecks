import numpy as np
from pproc import DiscretizedPointProcess
from mcmc.ess import elliptical_slice
from mcmc.whitened_mh import whitened_mh
from util import normalize_rows, resample_assignments
from kernel import Kernel, MultiKronKernel
from kron_util import kron_mat_vec_prod
from numpy.random import rand, randn
import pylab as plt 

# Spatio-temporal LGCP's
#
#   A static spatial component (or a few of them) are modulated by weights that
#   evolve over time (according to an exponentiated Gaussian Process).  
#   The idea is to de-couple spatial bases from temporal phenomena that 
#   modulate them, such as periodicities, bursts in intensities, etc.  
#
#   The model defines the following pieces: 
#     - B_1, ..., B_K    : non-negative basis surfaces over the 
#                          space (normalized to unit vol)
#     - w_t1, dots, w_tK : non-negative weights that modulate B_k at 
#                          each time (normalized to sum to 1)
#     - s_t              : non-negative global scaling process
#
#
#   For each space-time-bin (x_i,t_i) (with width dx, dt) we define the 
#   likelihood over observations 
#     N_{x_i, t_i} ~ poisson( s_t * sum( w(t,:) * B(i) ) dVol ) 
#

class SpatioTemporalMixLGCP(DiscretizedPointProcess): 
  def __init__(self, xdim=2, K=5, \
                     xgrid_dims=(50,50), xbbox=[(0,1), (0,1)], \
                     tgrid_dims=[20], tbbox=[0,1], \
                     xkernel = "sqeu", \
                     tkernel = "per" ):
    ## initalize discretized grid (space,time)
    all_grid_dim = np.concatenate( (xgrid_dims, tgrid_dims) )
    all_bbox     = np.concatenate( (xbbox, [tbbox]) )
    all_dim      = xdim + 1
    super(SpatioTemporalMixLGCP, self).__init__(all_dim, all_grid_dim, all_bbox)
    
    #set up kernel for each dimension 
    if isinstance(xkernel, basestring): 
        self._xkern = MultiKronKernel([xkernel for d in range(xdim)])
    else: 
        self._xkern = xkernel
    if isinstance(tkernel, basestring):
        self._tkern = Kernel.factory(tkernel)
    else: 
        self._tkern = tkernel

    #figure out parameter sizes, etc
    self._xdim    = xdim
    self._K       = K
    self._V       = np.prod(xgrid_dims)
    self._T       = tgrid_dims[0]
    self._Nbeta   = (self._V+1)*self._K        #spatial GP vals
    self._Nomega  = (self._T+1)*self._K        #temporal GP vals
    self._Nth     = self._Nbeta + self._Nomega #all GP prior vals
    self._Nxhypers = len(self._xkern.hypers())  #number of X space cov hypers
    self._Nthypers = len(self._tkern.hypers())  #number of T space cov hypers
    self._Nhypers = self._Nxhypers + self._Nthypers 

    #size of a time tile and area/volume of a spatial tile
    self._dt = float(tbbox[1] - tbbox[0]) / tgrid_dims[0]
    self._dx = self._cell_vol / self._dt # np.prod([float(xbbox[i][1]-xbbox[i][0])/xgrid_dims[i] for i in range(xdim)])
 
  def describe(self): 
    """ short desc of model and params """
    print """SpatioTemporalMixLGCP: 
             xdim      = %d
             K (bases) = %d
             V (space) = %d
             T (time)  = %d
             Nbeta     = %d
             Nomega    = %d
             Nth       = %d
             Nhypers   = %d
           """%(self._xdim, self._K, self._V, self._T, self._Nbeta, \
               self._Nomega, self._Nth, self._Nhypers)

  def fit(self, data, Nsamps=100, prop_scale=1, \
                verbose=True, burnin=500, num_ess=10, init_th=None): 
    """ fit the spatiotemporal mixture of LGCPs: 
    data should come in as a dim by N dimensional numpy array 
    """
    #initialize a 'point process' for each example, bin data
    #TODO make sure this VECTORIZES the X space, leaving the t space
    self._count_data(data)
    print self._grid_obs.shape

    # initialize current weights and bases
    w_curr = np.zeros( self._K*(self._T+1) )       # current log weights
    b_curr = np.zeros( self._K*(self._V+1) )       # current log bases
    z_curr = np.zeros( (self._T, self._V, self._K), dtype=np.int ) # current assignments

    # initialize hyperparameters for spatial and temporal kernels
    xh_curr = np.ones(self._K * self._Nxhypers)
    th_curr = np.ones( (self._K,self._Nthypers) )
    th_curr = np.array( [[5., 5., 10.], [5., 10., 10.]] )
    
    # keep track of all samples
    w_samps = np.zeros( (Nsamps, len(w_curr)) )
    b_samps = np.zeros( (Nsamps, len(b_curr)) )
    xh_samps = np.zeros( (Nsamps, len(xh_curr)) )
    th_samps = np.zeros( (Nsamps, th_curr.shape[0], th_curr.shape[1]) )
    lls     = np.zeros( Nsamps )
    accept_rate = 0
    for i in range(Nsamps):
      if i%10==0 and verbose: 
          print "  samp %d of %d (num_accepted = %d)"%(i,Nsamps,accept_rate)

      ##
      ## Sample Z values (which basis each obs comes from) given B, W
      ##
      W = self._omega_to_weight(w_curr)
      B = self._beta_to_basis(b_curr)
      z_curr = resample_assignments(W, B, self._grid_obs.T, z_curr)

      ##
      ## sample W values (conjugate)
      ##
      #th_mat = th_curr.reshape((self._K, self._Nthypers)) #unpack time hypers
      w_mat  = w_curr.reshape((self._K, self._T+1))       #unpack time samp
      Ztime  = z_curr.sum(axis=1).T   #summed over space (spatial basis is fixed)
      for k in range(self._K):
        def weight_log_like(omega): 
            logOmega = omega[0] + omega[1:]
            W        = self._dt * np.exp(logOmega)
            ll = np.sum( logOmega*Ztime[k] - W )  #independent poisson 
            return ll
        Kmat = self._tkern.K(self._grids[-1], self._grids[-1], hypers=th_curr[k])
        L = np.linalg.cholesky(Kmat + 1e-8*np.eye(len(self._grids[-1])))
        for ii in range(10):
          #omega_samp = self._tkern.gen_prior(th_curr[k], self._grids[-1])
          omega_samp = L.dot(np.random.randn(len(self._grids[-1])))
          prior_w = np.concatenate(([3.16*randn()], omega_samp))
          w_mat[k], log_lik = elliptical_slice( initial_theta = w_mat[k], \
                                                prior         = prior_w, \
                                                lnpdf         = weight_log_like )
      w_curr = w_mat.ravel()
      w_samps[i] = w_curr

      ##
      ## Sample ESS surface given weights and Z values
      ##
      ## sample latent surface (multiple runs of ESS)
      xh_mat = xh_curr.reshape((self._K, self._Nxhypers)) #unpack space hypers
      b_mat  = b_curr.reshape((self._K, self._V+1))        #unpack basis samp
      Zbasis = z_curr.sum(axis=0).T  #all shots from each basis
      for k in range(self._K):
        beta_samp = self._xkern.gen_prior(5*xh_mat[k], self._grids[0:-1])
        prior_b = np.concatenate(([np.sqrt(10)*randn()], beta_samp))
        def basis_log_like(beta):
            Bk   = self._single_beta_to_basis(beta)
            lams = W[k,:].sum()*Bk
            loglam = np.log(lams)
            ll = np.sum(loglam*Zbasis[k] - lams)
            return ll
        b_mat[k], log_lik = elliptical_slice( initial_theta = b_mat[k], \
                                              prior         = prior_b, \
                                              lnpdf         = basis_log_like )
      b_curr = b_mat.ravel()
      b_samps[i] = b_curr

      ##
      ## Sample cov function hyper parameters
      ## 
      # whitens/unwhitens 
      def whiten(th, f): 
        th = np.concatenate([[5],th])
        return self._tkern.whiten_process(f, th, self._grids[-1])
      def unwhiten(thp, nu): 
        thp = np.concatenate([[5],thp])
        return  self._tkern.gen_prior(thp, self._grids[-1], nu=nu)
      for k in range(self._K):
        def w_log_like(omega): 
            logOmega = omega[0] + omega[1:]
            W        = self._dt * np.exp(logOmega)
            ll = np.sum( logOmega*Ztime[k] - W )  #independent poisson 
            return ll
        #sample hyper params (ignore scale)
        th_curr[k,1:], w_mat[k,1:], accepted, ll = \
          whitened_mh( th            = th_curr[k,1:], 
                       f             = w_mat[k,1:], 
                       whiten_func   = whiten, 
                       unwhiten_func = unwhiten, 
                       like_func = lambda(w): w_log_like(np.append(w_mat[k,0], w)),
                       ln_prior  = lambda(h): self._tkern.hyper_prior_lnpdf(h),
                       prop_scale = .25 )
        #if k==1:
        #  th_mat[k] = np.array([1, 20, 5])
        #else: 
        #  th_mat[k] = np.array([1, 5, 5])
      th_samps[i] = th_curr

      ##
      ## compute log like
      ## 
      W = self._omega_to_weight(w_curr)
      B = self._beta_to_basis(b_curr)
      lams = B.T.dot(W)
      loglam_all = np.log(lams)
      lls[i] = np.sum( loglam_all*self._grid_obs - lams )
      ll_curr = lls[i]

      #chain stats
      accept_rate += accepted

    #print "final acceptance rate: ", float(accept_rate)/Nsamps
    self.w_samps = w_samps
    self.b_samps = b_samps
    self.th_samps = th_samps
    self.xh_samps = xh_samps
    self._lls      = lls
    self._burnin   = burnin
    return w_samps, b_samps, th_samps, lls

  def posterior_mean_lambda(self, thin=1): 
    """ return the posterior mean lambda surface """
    zvecs  = (self._z_samps[:,0] + self._z_samps[:,1:].T).T
    zvecs  = zvecs[ np.arange(0,zvecs.shape[0],thin), :]  #thin
    lamvec = self._cell_vol * np.exp(zvecs).mean(axis=0)
    return np.reshape(lamvec, self._grid_dim, order='C')

  def posterior_max_ll_lambda(self):
    """ returns the max samp lambda """
    max_i = self._lls[self._burnin:].argmax()
    zvec = self._z_samps[max_i,0] + self._z_samps[max_i,1:]
    lamvec = self._cell_vol * np.exp(zvec)
    return np.reshape(lamvec, self._grid_dim, order='C')

  def posterior_var_lambda(self):
    """ return the posterior variance of the lambda surface """
    zvecs = (self._z_samps[:,0] + self._z_samps[:,1:].T).T
    lamvec = self._cell_vol * np.exp(zvecs).var(axis=0)
    return np.reshape(lamvec, self._grid_dim, order='C')

  def plot_basis_from_samp(self, b_samp, ax=None):
    """  approximate log like is just independent poissons """
    B = self._beta_to_basis(b_samp) 
    B /= self._dx
    #B, W = self._vec_to_basis_weights(th)
    if self._xdim==1: 
      if ax is None: 
        f = plt.figure()
        plt.plot(self._grids[0], B.T)
        plt.title('Basis')
        plt.xlabel('space')
      else: 
        ax.plot(self._grids[0], B.T)
        ax.set_title("Basis")
        ax.set_xlabel("X space")

  def plot_weights_from_samp(self, w_samp, ax=None):
    """ """
    W = self._omega_to_weight(w_samp) 
    W /= self._dt
    #B, W = self._vec_to_basis_weights(th)
    if self._xdim==1:
      if ax is None: 
        f = plt.figure()
        plt.plot(self._grids[-1], W.T)
        plt.title('Weights')
        plt.xlabel('time')
      else: 
        ax.plot(self._grids[-1], W.T)
        plt.title("Weights")
        plt.xlabel("time")
    #make weights positive 
    #th_ws = th[self._K*self._Nz:]
    #ws = np.exp(th_ws.reshape((self._Nw, self._K)))

  def plot_time_hypers(self):
    fig, axarr = plt.subplots(self._Nthypers,1)
    param_names = self._tkern.hyper_names()
    for n in range(self._Nthypers): 
      hypers = self.th_samps[:,:,n]
      axarr[n].plot(hypers)
      axarr[n].set_title('Time cov, param %s'%param_names[n])
    fig.tight_layout()
    #for k in range(self._K):
    #  startI = k*self._Nthypers
    #  endI   = startI + self._Nthypers
    #  thk    = self.th_samps[:, startI:endI]
    #  l      = axarr[k].plot(thk) 
    #  axarr[k].set_title('Time cov hypers, k = %d'%k)
    #  axarr[k].legend(l, ['scale', 'per', 'length_scale'])

  #########################################################
  # "private" helper methods for fitting the model
  #########################################################
  #def _unravel_params(self, th):
  #  """ take flat vector and produce basis and weight matrices """
  #  #compute positive normalized basis
  #  Bs = self._beta_to_basis(th[0:self._K*self._Nz])
  #  #make weights positive
  #  th_ws = th[self._K*self._Nz:]
  #  ws = np.exp(th_ws.reshape((self._Nw, self._K)))
  #  return Bs, ws

  def _beta_to_basis(self, bs):
    bs    = bs.reshape((self._K, self._V+1))
    logBs = (bs[:,0] + bs[:,1:].T).T
    Bs    = normalize_rows(self._dx * np.exp(logBs))
    return Bs

  def _omega_to_weight(self, ws):
    ws = ws.reshape((self._K, self._T+1))
    logWs = (ws[:,0] + ws[:,1:].T).T
    Ws    = self._dt * np.exp(logWs)              #weights are not normalized!!
    return Ws

  def _single_beta_to_basis(self, beta):
    logB = beta[0] + beta[1:]
    B_unnorm = self._dx * np.exp(logB)
    return B_unnorm/B_unnorm.sum()

  #def _log_like(self, th):
  #  """ approximate log like is just independent poissons """
  #  #compute the intensity for each point process
  #  Bs, ws = self._unravel_params(th)
  #  lams = ws.dot(Bs)

  #  #compute log probs: pr(x | lam) = lam^x * exp(-lam) / x!
  #  loglam_all = np.log(lams)
  #  ll = np.sum( loglam_all*self._grid_counts - lams )
  #  return ll


  #def _unravel_gp_params(self, th):
  #  """ take flat vector and produce basis and weight matrices: 
  #  Note: storage of these params are th = (beta,omega), i.e. (space,time)
  #  """
  #  #log spatial basis
  #  Beta  = th[0:self._Nbeta].reshape((self._K, self._V+1))
  #  Omega = th[self._Nbeta:].reshape((self._K, self._T+1))
  #  return Beta, Omega 

  #def _vec_to_basis_weights(self, th):
  #  Beta, Omega = self._unravel_gp_params(th)
  #  W      = np.exp((Omega[:,0].T + Omega[:,1:].T).T) #positve temporal weights
  #  B      = np.exp((Beta[:,0].T + Beta[:,1:].T).T)   
  #  B_sums = B.sum(axis=1)
  #  B      = self._cell_vol*B/B_sums[:,np.newaxis]                  #normalized spatial basis
  #  return B, W
  #
  #def _log_like(self, th): 
  #  """ computes log likelihood log(Pr(X | \Lambda=W.T.dot(B))) 
  #  th is the vector of parameters th = [vec(Omega), vec(Beta)]
  #        - Omega is a (K x T+1) matrix (num time tiles + bias term)
  #        - Beta is a (K x V+1) matrix (num space tiles + bias term)
  #  """
  #  B, W = self._vec_to_basis_weights(th)
  #  Lambda = B.T.dot(W)
  #  return np.sum( self._grid_obs*np.log(Lambda) - Lambda )

  #def _gen_prior(self, cov_hypers):
  #  """ generate from GP prior over spatial basis Beta 
  #  and temporal weights Omega """
  #  xhypers = cov_hypers[0:self._Nxhypers]
  #  thypers = cov_hypers[self._Nxhypers:]
  #  Beta  = np.zeros((self._K, self._V+1))   #sample basis
  #  for k in range(self._K):
  #      Beta[k,0]  = np.random.randn()*np.sqrt(10)
  #      Beta[k,1:] = self._xkern.gen_prior(xhypers, self._grids[0:-1])
  #  Omega = np.zeros((self._K, self._T+1))   #sample temp weights
  #  for k in range(self._K):
  #      Omega[k,0]  = np.random.randn()*np.sqrt(10)
  #      Omega[k,1:] = self._tkern.gen_prior(thypers, self._grids[-1])
  #  return np.concatenate( (Beta.ravel(), Omega.ravel()) )


