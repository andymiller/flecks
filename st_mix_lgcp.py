import numpy as np
from pproc import DiscretizedPointProcess
from ess import elliptical_slice
from util import  whitened_mh, spherical_proposal
from kernel import Kernel, MultiKronKernel
from kron_util import kron_mat_vec_prod
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
                     tkernel = "sqeu" ):
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
                verbose=True, burnin=500, num_ess=10): 
    """ fit the spatiotemporal mixture of LGCPs: 
    data should come in as a dim by N dimensional numpy array 
    """
    #initialize a 'point process' for each example, bin data
    #TODO make sure this VECTORIZES the X space, leaving the t space
    self._count_data(data)
    print self._grid_obs.shape

    ## set up samples
    th_curr  = np.zeros(self._Nth)  #samples of gaussian prior parameters
    th_samps = np.zeros( (Nsamps, len(th_curr)) )
    h_curr   = np.ones(self._Nhypers) #samples of hyper params
    h_samps  = np.zeros( (Nsamps, len(h_curr)) )
    ll_curr  = None
    lls      = np.zeros( Nsamps ) 
    accept_rate = 0
    for i in range(Nsamps):
      if i%10==0 and verbose: 
          print "  samp %d of %d (num_accepted = %d)"%(i,Nsamps,accept_rate)

      ##
      ## Sample cov function hyper parametersHYPER PARAMETERS
      ## 
      # whitens/unwhitens 
      def whiten(th, f): 
        return self._kern.whiten_process(f, th, self._grids)
      def unwhiten(thp, nu): 
        return  self._kern.gen_prior(thp, self._grids, nu=nu)

      ##sample hyper params
      #h_curr, z_hyper, accepted, ll = whitened_mh( h_curr, \
      #         z_curr[1:], \
      #         whiten   = whiten, #lambda(th, f): self._kern.whiten_process(f, th, self._grids), \
      #         unwhiten = unwhiten, #lambda(thp, nu): self._kern.gen_prior(thp, self._grids, nu=nu), \
      #         Lfn      = lambda(z): self._log_like(np.append(z_curr[0], z)), \
      #         ln_prior = lambda(h): self._kern.hyper_prior_lnpdf(h), \
      #         prop_dist = lambda(th): spherical_proposal(th, prop_scale))
      #z_curr = np.append(z_curr[0], z_hyper)
      
      ##
      ## sample latent surface (multiple runs of ESS)
      ##
      for resamp_i in range(num_ess):
        th_curr, log_lik = \
          elliptical_slice(initial_theta = th_curr, \
                           prior         = self._gen_prior(h_curr),\
                           lnpdf         = self._log_like )

      # store samples
      th_samps[i,] = th_curr
      h_samps[i,] = h_curr
      lls[i] = log_lik

      #chain stats
      #accept_rate += accepted

    #print "final acceptance rate: ", float(accept_rate)/Nsamps
    self._th_samps = th_samps[burnin:,]
    self._h_samps  = h_samps[burnin:,]
    self._lls      = lls
    self._burnin   = burnin
    return th_samps, h_samps, lls

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

  def plot_basis_from_samp(self, th):
    """  approximate log like is just independent poissons """
    B, W = self._vec_to_basis_weights(th)
    if self._xdim==1: 
      f = plt.figure()
      plt.plot(self._grids[0], B.T)
      plt.title('Basis')
      plt.xlabel('space')

  def plot_weights_from_samp(self, th):
    """ """
    B, W = self._vec_to_basis_weights(th)
    if self._xdim==1:
      f = plt.figure()
      plt.plot(self._grids[-1], W.T)
      plt.title('Weights')
      plt.xlabel('time')
    #make weights positive 
    #th_ws = th[self._K*self._Nz:]
    #ws = np.exp(th_ws.reshape((self._Nw, self._K)))

  #########################################################
  # "private" helper methods for fitting the model
  #########################################################
  def _unravel_gp_params(self, th):
    """ take flat vector and produce basis and weight matrices: 
    Note: storage of these params are th = (beta,omega), i.e. (space,time)
    """
    #log spatial basis
    Beta  = th[0:self._Nbeta].reshape((self._K, self._V+1))
    Omega = th[self._Nbeta:].reshape((self._K, self._T+1))
    return Beta, Omega 

  def _vec_to_basis_weights(self, th):
    Beta, Omega = self._unravel_gp_params(th)
    W      = np.exp((Omega[:,0].T + Omega[:,1:].T).T) #positve temporal weights
    B      = np.exp((Beta[:,0].T + Beta[:,1:].T).T)   
    B_sums = B.sum(axis=1)
    B      = self._cell_vol*B/B_sums[:,np.newaxis]                  #normalized spatial basis
    return B, W
  
  def _log_like(self, th): 
    """ computes log likelihood log(Pr(X | \Lambda=W.T.dot(B))) 
    th is the vector of parameters th = [vec(Omega), vec(Beta)]
          - Omega is a (K x T+1) matrix (num time tiles + bias term)
          - Beta is a (K x V+1) matrix (num space tiles + bias term)
    """
    B, W = self._vec_to_basis_weights(th)
    Lambda = B.T.dot(W)
    return np.sum( self._grid_obs*np.log(Lambda) - Lambda )

  def _gen_prior(self, cov_hypers):
    """ generate from GP prior over spatial basis Beta 
    and temporal weights Omega """
    xhypers = cov_hypers[0:self._Nxhypers]
    thypers = cov_hypers[self._Nxhypers:]
    Beta  = np.zeros((self._K, self._V+1))   #sample basis
    for k in range(self._K):
        Beta[k,0]  = np.random.randn()*np.sqrt(10)
        Beta[k,1:] = self._xkern.gen_prior(xhypers, self._grids[0:-1])
    Omega = np.zeros((self._K, self._T+1))   #sample temp weights
    for k in range(self._K):
        Omega[k,0]  = np.random.randn()*np.sqrt(10)
        Omega[k,1:] = self._tkern.gen_prior(thypers, self._grids[-1])
    return np.concatenate( (Beta.ravel(), Omega.ravel()) )


