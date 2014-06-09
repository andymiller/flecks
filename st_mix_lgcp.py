import numpy as np
import numpy.random as npr
from pproc import DiscretizedPointProcess
from mcmc.ess import elliptical_slice
from mcmc.whitened_mh import whitened_mh
from mcmc.slicesample import slice_sample
from mcmc.pseudo_marginal import approx_log_marg_like, poisson_log_like
from mcmc.sample_assignments import sample_assignments
from util.misc import logmulexp
from kernel import Kernel, MultiKronKernel, SQEKernelUnscaled
from scipy.misc import logsumexp
import scipy.weave
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
                     tkernel = "per_unscaled" ):
    ## initalize discretized grid (space,time)
    all_grid_dim = np.concatenate( (xgrid_dims, tgrid_dims) )
    all_bbox     = np.concatenate( (xbbox, [tbbox]) )
    all_dim      = xdim + 1
    super(SpatioTemporalMixLGCP, self).__init__(all_dim, all_grid_dim, all_bbox)
    
    #set up kernel for each dimension 
    if isinstance(xkernel, basestring): 
        #self._xkern = MultiKronKernel([xkernel for d in range(xdim)])
        self._xkern = MultiKronKernel([SQEKernelUnscaled(length_scale=1, 
                                                         alpha0 = 2, 
                                                         beta0  = 2) for d in range(xdim)])
    else: 
        self._xkern = xkernel
    if isinstance(tkernel, basestring):
        self._tkern = Kernel.factory(tkernel)
    else: 
        self._tkern = tkernel

    #figure out parameter sizes, etc
    self._xdim    = xdim
    self._xbbox   = xbbox
    self._tbbox   = tbbox
    self._xgrid_dims = xgrid_dims
    self._tgrid_dims = tgrid_dims
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

    #some diagnostic random vars
    self._num_accepts = 0
 
  def describe(self): 
    """ short desc of model and params """
    print """SpatioTemporalMixLGCP: 
             xdim      = %d
             tbbox     = (%d, %d)
             K (bases) = %d
             V (space) = %d  (dx = %2.4f)
             T (time)  = %d  (dt = %2.4f)
             Nbeta     = %d
             Nomega    = %d
             Nth       = %d
             Nhypers   = %d
           """%(self._xdim, self._tbbox[0], self._tbbox[1], self._K, self._V, 
                self._dx, self._T, self._dt,
                self._Nbeta, self._Nomega, self._Nth, self._Nhypers)
  
  def fit(self, data,             # numpy array with time last [x1,x2,...,t]
                Nsamps=100,       # number of samples to draw
                prop_scale=1,     # proposal scale for MH 
                verbose=True,     # print lls and acceptance rate
                burnin=500,       # num burnin
                num_ess=10,       # number of ESS runs for each GP sample
                init_xh=None,     # initial setting of spatial hyper params
                init_th = None,   # initial setting of temporal hyper params
                pseudo_marg=True  # do pseudomarginal inference for temp hypers
                ): 
    """ fit the spatiotemporal mixture of LGCPs: 
    data should come in as a dim by N dimensional numpy array 
    """
    #initialize a 'point process' for each example, bin data
    self._count_data(data)
    if len(self._grid_obs.shape) > 2:
      self._grid_obs = np.array([ self._grid_obs[:,:,i].ravel(order='C') 
                                  for i in range(self._grid_obs.shape[-1]) ]).T
    print "Grid obs shape: ", self._grid_obs.shape

    # initialize current weights and bases
    w_curr = np.zeros( self._K*(self._T+1) )       # current log weights
    b_curr = np.zeros( self._K*(self._V+1) )       # current log bases
    z_curr = np.zeros( (self._T, self._V, self._K), dtype=np.int ) # current assignments

    # initialize hyperparameters for spatial and temporal kernels
    xh_curr = 5*np.ones( (self._K,self._Nxhypers) )
    th_curr = 10*np.ones( (self._K,self._Nthypers) )
    ll_curr = -np.inf
    if init_xh is not None:
      xh_curr = init_xh
    if init_th is not None:
      th_curr = init_th

    #keep track of all samples
    w_samps  = np.zeros( (Nsamps, len(w_curr)) )
    b_samps  = np.zeros( (Nsamps, len(b_curr)) )
    xh_samps = np.zeros( (Nsamps, xh_curr.shape[0], xh_curr.shape[1]) )
    th_samps = np.zeros( (Nsamps, th_curr.shape[0], th_curr.shape[1]) )

    #set up approx marg likes for temporal cov params if running pseudo_marg
    if pseudo_marg: 
      log_py_samps = np.zeros( (Nsamps, self._K) )
      log_py_curr  = -np.inf * np.ones(self._K)
    
    #keep track of all sample log likelihoods
    lls = np.zeros( Nsamps )
    num_per_jump_accepts = 0
    for i in range(Nsamps):
      if i%1==0 and verbose: 
          print "  samp %d of %d (curr_ll = %2.4f) (num th accepts: %d)"%\
            (i, Nsamps, ll_curr, self._num_accepts) 

      ##
      ## Sample Z values (which basis each obs comes from) given B, W
      ##
      z_curr = sample_assignments( W = self._omega_to_log_weight(w_curr),
                                   B = self._beta_to_log_basis(b_curr), 
                                   grid_counts = self._grid_obs.T, 
                                   z_curr = z_curr, is_log=True )
      #xs = self._grid_obs[v,n]
      #plt.imshow( z_curr[:,:,0].sum(axis=0).reshape(self._xgrid_dims) )
      #plt.show()

      #marginalize zs over space and time
      Ztime  = z_curr.sum(axis=1).T #summed over space (spatial basis is fixed)
      Zbasis = z_curr.sum(axis=0).T  #all shots from each basis
      print "    => finished sampling assignments "

      ##
      ## Sample Basis surfaces given weights and Z values
      ##
      ## sample latent surface (multiple runs of ESS)
      logW = self._omega_to_log_weight(w_curr)  
      W    = self._omega_to_weight(w_curr)
      #xh_mat = xh_curr.reshape((self._K, self._Nxhypers)) #unpack space hypers
      b_mat  = b_curr.reshape((self._K, self._V+1))        #unpack basis samp
      for k in range(self._K):
        def basis_log_like(beta):
          beta = beta[0] + beta[1:]
          logBk = beta - logsumexp(beta) - np.log(self._dx)
          logLams = logsumexp(logW[k,:]) + logBk + np.log(self._dt) + np.log(self._dx)
          stable_ll = np.sum(logLams*Zbasis[k] - np.exp(logLams))
          #print "!!!!!!!!!! basis_log_like %2.5f, %2.5f"%(ll, stable_ll)
          return stable_ll

        nreps = 100 if i<10 else 20
        for ii in range(nreps):
          beta_samp = self._xkern.gen_prior(xh_curr[k], self._grids[0:-1])
          prior_b = np.concatenate(([np.sqrt(10)*npr.randn()], beta_samp))
          b_mat[k], log_lik = elliptical_slice( initial_theta = b_mat[k],
                                                prior         = prior_b,
                                                lnpdf         = basis_log_like )
      b_curr = b_mat.ravel()
      b_samps[i] = b_curr
      print "    => finished sampling basis surfaces"
      ll_curr = self._check_ll_diff(ll_curr, w_curr, b_curr, "B")

      ##
      ## Sample spatial cov function hyper parameters
      ## 
      # whitens/unwhitens 
      logW = self._omega_to_log_weight(w_curr)  
      W    = self._omega_to_weight(w_curr)
      b_mat  = b_curr.reshape((self._K, self._V+1))       #unpack time samp
      for k in range(self._K):
        #whiten spatial process Bk
        nu  = self._xkern.whiten_process(b_mat[k,1:], xh_curr[k,:], self._grids[0:-1])
        def whitened_spatial_log_like(th):
          ll_prior = self._xkern.hyper_prior_lnpdf(th)
          if ll_prior < -1e50:
            return -np.inf
          beta = b_mat[k,0] + self._xkern.gen_prior(th, self._grids[0:-1], nu=nu)
          logBk = beta - logsumexp(beta) - np.log(self._dx)
          logLams = logsumexp(logW[k,:]) + logBk + np.log(self._dt) + np.log(self._dx)
          stable_ll = np.sum(logLams*Zbasis[k] - np.exp(logLams))
          return stable_ll+ll_prior
        xh_curr[k,:] = slice_sample(xh_curr[k,:], whitened_spatial_log_like)
        b_mat[k,1:]  = self._xkern.gen_prior(xh_curr[k,:], self._grids[0:-1], nu=nu)
      xh_samps[i] = xh_curr
      b_curr = b_mat.ravel()
      print "    => finished sampling basis cov hypers"
      ll_curr = self._check_ll_diff(ll_curr, w_curr, b_curr, "xhyper")

      ##
      ## Sample TEMPORAL cov function hyper parameters
      ## 
      w_mat  = w_curr.reshape((self._K, self._T+1))       #unpack time samp
      for k in range(self._K):
        print "    => sampling temporal covariance parameters k=%d"%k
        if pseudo_marg: 
          th_curr[k,:], w_mat[k,:], log_py_curr[k] = \
              self._sample_temp_hypers( th_curr     = th_curr[k,:],
                                        w           = w_mat[k,:],
                                        Ztime       = Ztime[k], 
                                        log_py_curr = log_py_curr[k] )
        else: 
          th_curr[k,:], w_mat[k,:] = \
              self._sample_temp_hypers( th_curr     = th_curr[k,:],
                                        w           = w_mat[k,:],
                                        Ztime       = Ztime[k] ) 
      if pseudo_marg: 
        log_py_samps[i] = log_py_curr
      th_samps[i] = th_curr
      w_curr = w_mat.ravel()
      print "    => finished sampling temporal cov hypers"
      ll_curr = self._check_ll_diff(ll_curr, w_curr, b_curr, "temp hypers")

      ##
      ## sample W values (ESS, not conjugate anymore)
      ##
      #th_mat = th_curr.reshape((self._K, self._Nthypers)) #unpack time hypers
      w_mat  = w_curr.reshape((self._K, self._T+1))       #unpack time samp
      for k in range(self._K):
        def weight_log_like(omega): 
            logW = omega[0] + omega[1:] + np.log(self._dt) #scaled by dt
            return np.sum( logW*Ztime[k] - np.exp(logW) )  #independent poisson 
        Kmat = self._tkern.K(self._grids[-1], self._grids[-1], hypers=th_curr[k])
        L = np.linalg.cholesky(Kmat + 1e-8*np.eye(len(self._grids[-1])))
        for ii in range(50):
          #omega_samp = self._tkern.gen_prior(th_curr[k], self._grids[-1])
          omega_samp = L.dot(npr.randn(len(self._grids[-1])))
          prior_w = np.concatenate(([3.16*npr.randn()], omega_samp))
          w_mat[k], log_lik = elliptical_slice( initial_theta = w_mat[k],
                                                prior         = prior_w,
                                                lnpdf         = weight_log_like )
      w_curr = w_mat.ravel()
      w_samps[i] = w_curr
      print "    => finished sampling temporal weights"
      ll_curr = self._check_ll_diff(ll_curr, w_curr, b_curr, "W")

      ##
      ## compute log like
      ##
      lls[i]  = self._log_like(w_curr, b_curr)
      ll_curr = lls[i]

      ## compute volume of intensity function
      W = self._omega_to_weight(w_curr)
      B = self._beta_to_basis(b_curr)
      lams = B.T.dot(W) * self._dx*self._dt #V tiles by T time steps intensity matrix
      print "    intensity volume: ", np.sum(lams)
      print "    num observations: ", np.sum(self._grid_obs)

    #print "final acceptance rate: ", float(accept_rate)/Nsamps
    self.w_samps = w_samps
    self.b_samps = b_samps
    self.th_samps = th_samps
    self.xh_samps = xh_samps
    self._lls      = lls
    self._burnin   = burnin
    return w_samps, b_samps, th_samps, lls

  def _sample_temp_hypers(self, 
                          th_curr, #current temporal hypers (for some k)
                          w,       #current GP (for some k)
                          Ztime,   #counts for Z 
                          log_py_curr = None, #if not none, do pseudo_marg
                          prop_scale  = .025,  #proposal scale for MH steps
                          jitter      = 1e-6, #jitter for gram mat
                          sig2_bias   = 25    #variance of bias term for W
                          ): 
    """ note that conditioned on Z, fitting W and cov hypers of W is like
    a simple Poisson regression proglem """
    
    ##
    ## If no log_py_curr estimate, do Whitened slice sample 
    ##
    if log_py_curr is None: 
      nu  = self._tkern.whiten_process(w[1:], th_curr, self._grids[-1])
      def whitened_log_like(th, nu=nu):
        ll_prior = self._tkern.hyper_prior_lnpdf(th)
        if ll_prior < -1e50:
          return -np.inf
        logW = w[0] + np.log(self._dt) + \
               self._tkern.gen_prior(th, self._grids[-1], nu=nu)
        ll   = np.sum( logW*Ztime - np.exp(logW) )  
        return ll + ll_prior
      th_curr = slice_sample(th_curr, whitened_log_like)
      w[1:]   = self._tkern.gen_prior(th_curr, self._grids[-1], nu=nu)
      return th_curr, w

    ##
    ## otherwise Sample covariance hypers with PSEUDOMARGINAL method
    ##
    #gen proposal, stop early if it's out of bounds
    for nn in range(1):
      th_prop = th_curr.copy() 
      for i,hyper_name in enumerate(self._tkern.hyper_names()):
        if hyper_name=="Period":         # special proposal for periodic params
          jump_prop=False
          if npr.rand() < .5:
            jump_prop=True
            factor = npr.geometric(p=.5) + 1  
            if npr.rand() < .5: 
                factor = 1./factor
            th_prop[i] = factor * th_curr[i] + .05*npr.randn()
          else: 
            th_prop[i] = th_curr[i] + prop_scale*npr.randn()
        else: 
          th_prop[i] = th_curr[i] + prop_scale*npr.randn()

      #prior prob of hypers, stop early if it's no bueno
      log_prior_prop = self._tkern.hyper_prior_lnpdf(th_prop)
      if log_prior_prop < -1e20: 
        continue
        #return th_curr, w, log_py_curr

      #likelihood function 
      def bias_ll(y, f, cK_inv, grad=False, hess=False, hessp=False, p=None):
        return poisson_log_like(y, f, cK_inv, f0=np.log(self._dt), 
                                   grad=grad, hess=hess, hessp=hessp, p=p) 

      #compute current log marginal like (important, because Z changes)
      tgrid = self._grids[-1]
      N     = len(tgrid)
      K_curr = ( self._tkern.K(tgrid, tgrid, th_curr) + #temp covariance
                 jitter*np.eye(N) +                 #jitter for num stability
                 sig2_bias*np.ones((N,N)) )         #prior uncertainty from bias

      #compute log marginal like
      K_prop = ( self._tkern.K(tgrid, tgrid, th_prop) + #temp covariance
                 jitter*np.eye(N) +                 #jitter for num stability
                 sig2_bias*np.ones((N,N)) )         #prior uncertainty from bias

      log_py_curr = approx_log_marg_like( Ztime, K_curr, like_func=bias_ll )
      log_py_prop = approx_log_marg_like( Ztime, K_prop, like_func = bias_ll )
      #print "period: ", th_prop
      #print "length: ", ell_prop
      #print log_py_prop - log_py_curr
      rat = ( log_py_prop - log_py_curr +
              log_prior_prop - self._tkern.hyper_prior_lnpdf(th_curr) )
      if np.log(npr.rand()) < rat: 
          print "accepting temporal proposal"
          self._num_accepts += 1 
          th_curr = th_prop
          log_py_curr = log_py_prop
    return th_curr, w, log_py_curr

    ##update state
    #th_samps[ii] = th_curr
    #ell_samps[ii] = ell_curr
    #log_py_samps[ii] = log_py_curr
    ##propose an integer fraction jump for period
    #for ii in range(10): 
    #  factor = np.random.geometric(p=.22) + 1
    #  if rand() < .5: 
    #     factor = 1./factor
    #  th_prop = np.array([ factor*th_curr[k,0], 1./factor * th_curr[k,1] ])
    #  #nu = self._tkern.whiten_process(w_mat[k,1:], th_curr[k,:], self._grids[-1])
    #  jitter = 1e-6
    #  alpha  = .1
    #  K_prime = self._tkern.K(self._grids[-1], self._grids[-1], th_prop) + jitter*np.eye(N)
    #  K       = self._tkern.K(self._grids[-1], self._grids[-1], th_curr[k]) + jitter*np.eye(N)
    #  K_i     = np.linalg.inv(K)
    #  K_prime_i = np.linalg.inv(K_prime)
    #  Sig_prime = np.linalg.inv( (1/alpha)*K_i + K_prime_i)

    #  ll_th_prop = whitened_log_like(th_prop, nu=nu)
    #  ll_th_curr = whitened_log_like(th_curr[k,:], nu=nu)
    #  ratio = ll_th_prop-ll_th_curr
    #  #print "    per jump: prop, curr, (diff) %2.4f, %2.4f, (%2.4f)"%(ll_th_prop, ll_th_curr, ratio)
    #  if np.random.exponential() < ratio:
    #    print "  accepting per jump from %2.2f to %2.2f"%(th_curr[k,0],th_prop[0])
    #    num_per_jump_accepts += 1
    #    th_curr[k,:] = th_prop
    #    w_mat[k,1:]  = self._tkern.gen_prior(th_curr[k,:], self._grids[-1], nu=nu)


  def _check_ll_diff(self, ll_curr, w_curr, b_curr, varname, tol=-50):
    post_th_ll = self._log_like(w_curr, b_curr)
    if post_th_ll - ll_curr < tol: 
      err_message = " post %s sample log like diff: %2.4f"%(varname, 
                                                            post_th_ll-ll_curr)
      raise Exception(err_message)
    return post_th_ll

  ############################################################
  # Prediction methods
  ############################################################
  #TODO write prediction methods!!!! 
  def sample_conditional_intensity(self, w_samp, th_samp, tgrid_pred):
    w_mat = w_samp.reshape((self._K, self._T+1))
    Tpred = len(tgrid_pred)
    w_pred = np.zeros((self._K, Tpred+1))
    w_pred[:,0] = w_mat[:,0]
    for k in range(self._K): 
      #1. sample omega(tgrid_pred) given omega(tgrid) (w_samp) (standard GP)
      mu_pred, Sig_pred = self._tkern.conditional_params( f = w_mat[k,1:], 
                                                          pts = self._grids[-1],
                                                          pred_pts = tgrid_pred,
                                                          hypers = th_samp[k], 
                                                          jitter = 1e-8 )
      L_pred = np.linalg.cholesky(Sig_pred) 
      w_pred[k,1:] = mu_pred + L_pred.dot(np.random.randn(Tpred))
    return w_pred

  def test_likelihood(self, xtest, tgrid_dims, tbbox, num_samps=100): 
    """ return out of sample likelihood for test data, on a 
    test grid (that extends past the existing tgrid """

    ## grid test observations 
    all_grid_dim = np.concatenate( (self._xgrid_dims, tgrid_dims) )
    all_bbox     = np.concatenate( (self._xbbox, [tbbox]) )
    all_dim      = self._xdim + 1
    test_pproc = DiscretizedPointProcess(all_dim, all_grid_dim, all_bbox)
    test_pproc._count_data(xtest)  #populates test_pproc._grid_obs
  
    ## put together average lambda in test region 
    b_samps = self.b_samps[-num_samps:]
    w_samps = self.w_samps[-num_samps:]
    th_samps = self.th_samps[-num_samps:]
    xh_samps = self.xh_samps[-num_samps:]

    Lambda = np.zeros( (self._V, tgrid_dims[0]) )
    for n in range(len(b_samps)):
      w_pred = self.sample_conditional_intensity( w_samps[n], 
                                                  th_samps[n], 
                                                  test_pproc._grids[-1] )
      W = self._omega_to_weight(w_pred, T=len(test_pproc._grids[-1]))
      B = self._beta_to_basis(b_samps[n])
      Lambda += B.T.dot(W)
    Lambda /= len(b_samps)
    return Lambda, test_pproc._grids[-1]


  #############################################################
  # Model summary methods 
  #############################################################
  def posterior_mean_var_lambda(self, samp_start=500, thin=1): 
    """ return the posterior mean lambda surface """
    Lambda = np.zeros( (self._V, self._T) )
    w_thinned = self.w_samps[samp_start::thin]
    b_thinned = self.b_samps[samp_start::thin]
    Nsamps    = len(w_thinned)
    for n in range(Nsamps):
      W  = self._omega_to_weight(w_thinned[n])
      B  = self._beta_to_basis(b_thinned[n])
      Lambda += B.T.dot(W)
    Lambda /= Nsamps

    #variance sweep
    Lambda_var = np.zeros( (self._V, self._T) )
    for n in range(Nsamps):
      W  = self._omega_to_weight(w_thinned[n])
      B  = self._beta_to_basis(b_thinned[n])
      Lambda_var += (B.T.dot(W)-Lambda)*(B.T.dot(W)-Lambda)
    Lambda_var /= (Nsamps-1)
    return Lambda, Lambda_var

  def posterior_max_ll_lambda(self):
    """ returns the max samp lambda """
    max_i = self._lls[self._burnin:].argmax()
    zvec = self._z_samps[max_i,0] + self._z_samps[max_i,1:]
    lamvec = self._cell_vol * np.exp(zvec)
    return np.reshape(lamvec, self._grid_dim, order='C')

  def plot_basis_from_samp(self, b_samp, ax=None):
    """  approximate log like is just independent poissons """
    B = self._beta_to_basis(b_samp) 
    #B /= self._dx
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
    elif self._xdim==2:
      if ax is None:
        fig, ax = plt.subplots(self._K)
      for k in range(self._K): 
        ax[k].imshow( B[k].reshape(self._xgrid_dims).T,  
                      origin = 'lower', 
                      extent = self._xbbox[0]+self._xbbox[1] )
        ax[k].set_title("Spatial basis, $B_%d$"%k)
    else: 
      raise NotImplementedError

  def plot_weights_from_samp(self, w_samp, ax=None):
    W = self._omega_to_weight(w_samp) 
    if ax is None: 
      f = plt.figure()
      plt.plot(self._grids[-1], W.T)
      plt.title('Weights')
      plt.xlabel('time')
    else: 
      ax.plot(self._grids[-1], W.T)
      plt.title("Weights")
      plt.xlabel("time")

  def plot_time_hypers(self):
    fig, axarr = plt.subplots(self._Nthypers,1)
    param_names = self._tkern.hyper_names()
    for n in range(self._Nthypers): 
      hypers = self.th_samps[:,:,n]
      axarr[n].plot(hypers)
      axarr[n].set_title('Time cov, param %s'%param_names[n])
    fig.tight_layout()

  def plot_space_hypers(self):
    fig, axarr = plt.subplots(self._Nxhypers,1)
    param_names = self._xkern.hyper_names()
    for n in range(self._Nxhypers): 
      hypers = self.xh_samps[:,:,n]
      axarr[n].plot(hypers)
      axarr[n].set_title('Space cov, param %s'%param_names[n])
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
    """ returns the discretized function approximation!!!! - need to mulitply
    by self._dx to obtain correctly scaled volume under a tile!"""
    return np.exp( self._beta_to_log_basis(bs) )

  def _beta_to_log_basis(self, bs):
    bs    = bs.reshape((self._K, self._V+1))
    logBs = (bs[:,0] + bs[:,1:].T).T
    logBs = (logBs.T - logsumexp(logBs, axis=1) - np.log(self._dx)).T
    return logBs

  def _omega_to_weight(self, ws, T=None):
    """ returns the discretized function approx! same as above - need to 
    mulitply by self._dt to obtain scaled volume in one time bin """
    #if T is None: 
    #    T = self._T
    #ws = ws.reshape((self._K, T+1))
    #logWs = (ws[:,0] + ws[:,1:].T).T
    #Ws    = np.exp(logWs)              #weights are not normalized!!
    #return Ws
    return np.exp( self._omega_to_log_weight(ws, T) )

  def _omega_to_log_weight(self, ws, T=None):
    if T is None: 
        T = self._T
    ws = ws.reshape((self._K, T+1))
    return (ws[:,0] + ws[:,1:].T).T

  def _single_beta_to_basis(self, beta):
    maxBeta = np.max(beta)
    logB = beta[0] + beta[1:] - maxBeta
    B_unnorm = np.exp(logB)
    return B_unnorm/(self._dx*B_unnorm.sum()) #dx in norm term again

  def _log_like(self, w_curr, b_curr):
    """ approximate log like is just independent poissons """
    logWs = self._omega_to_log_weight(w_curr)
    logBs = self._beta_to_log_basis(b_curr)
    logLams = logmulexp(logBs.T, logWs) + np.log(self._dx) + np.log(self._dt)
    stable_ll = np.sum(logLams*self._grid_obs - np.exp(logLams))
    
    ## compute unstable LL for copmarison
    #W = self._omega_to_weight(w_curr)
    #B = self._beta_to_basis(b_curr)
    #lams = B.T.dot(W) * self._dx*self._dt #V tiles by T time steps intensity matrix
    #loglam_all = np.log(lams)
    #unstable_ll = np.sum( loglam_all*self._grid_obs - lams )
    #print "!!!!!!!!!!!!!! comparison: %2.6f, %2.6f"%(stable_ll, unstable_ll)
    return stable_ll #  #compute log probs: pr(x | lam) = lam^x * exp(-lam) / x!


