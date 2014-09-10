import numpy as np
from pproc import DiscretizedPointProcess
from ..inference.ess import elliptical_slice
from ..inference.whitened_mh import  whitened_mh, spherical_proposal
from ..kernel import MultiKronKernel
from ..util.kron_util import kron_mat_vec_prod

#
# models a discretized log gaussian cox process in N dimensions
#
#  initialize with: 
#    - dimension, D
#    - number of bins in each dimension 
#    - range of each dimension (bbox)
#    - covariance of each dimension 
#
#  For LGCP.fit
#  input: a set of N-D points, X
#  output: posterior samples for intensity function value 
#          at specified grid
#
class LGCP(DiscretizedPointProcess): 

  def __init__(self, dim=2, grid_dim=(50,50), bbox=[(0,1), (0,1)], \
                     kernel_types=["sque", "sque"], kern=None):
    ## initalize discretized grid
    super(LGCP, self).__init__(dim, grid_dim, bbox)

    #set up kernel for each dimension 
    if kern is None: 
      self._kern = MultiKronKernel(kernel_types)
    else: 
      self._kern = Kern

  def fit(self, data, Nsamps=2000, prop_scale=1, \
                do_pseudo_marg=False, verbose=True, burnin=500, num_ess=10):
    """ data should come in as a dim by N dimensional numpy array """

    # grid data into tile counts
    self._count_data(data)

    # number of latent z's to sample (including bias term)
    Nz = len(self._grid_counts) + 1

    # sample from the posterior w/ ESS, slice sample Hypers
    z_curr  = np.zeros( Nz )                    # current state of latent GP (Z)
    h_curr  = self._kern.hypers()               # current values of hyper params
    ll_curr = None                              # current log likelihood of obs

    # keep track of all samples
    z_samps = np.zeros( (Nsamps, Nz) )
    h_samps = np.zeros( (Nsamps, len(h_curr) ) )
    lls     = np.zeros( Nsamps )
    accept_rate = 0
    for i in range(Nsamps):
      if i%10==0 and verbose: print "  samp %d of %d"%(i,Nsamps)

      #sample hyper params
      if do_pseudo_marg: # approximate marg likelihood
        pass
        #h_curr, accepted, ll = pseudo_mh( h_curr

      else: # do whitened MH
        def whiten(th, f): 
          return self._kern.whiten_process(f, th, self._grids)
        def unwhiten(thp, nu): 
          return  self._kern.gen_prior(thp, self._grids, nu=nu)
        h_curr, z_hyper, accepted, ll = whitened_mh( h_curr,
           z_curr[1:],
           whiten_func   = whiten,
           unwhiten_func = unwhiten,
           like_func     = lambda(z): self._log_like(np.append(z_curr[0], z)),
           ln_prior      = lambda(h): self._kern.hyper_prior_lnpdf(h),
           prop_dist     = lambda(th): spherical_proposal(th, prop_scale))
        z_curr = np.append(z_curr[0], z_hyper)

      ## sample latent surface (multiple runs of ESS)
      for resamp_i in range(num_ess):
        #gen bias and spatial comp
        prior_samp = np.append( np.sqrt(10)*np.random.randn(),
                                self._kern.gen_prior(h_curr, self._grids) ) 
        z_curr, log_lik = elliptical_slice( z_curr, \
                                            prior_samp, \
                                            self._log_like )


      #store samples
      z_samps[i,] = z_curr
      h_samps[i,] = h_curr
      lls[i] = ll

      #chain stats
      accept_rate += accepted

    print "final acceptance rate: ", float(accept_rate)/Nsamps
    self._z_samps = z_samps[burnin:,]
    self._h_samps = h_samps[burnin:,]
    self._lls     = lls
    self._burnin  = burnin
    return z_samps, h_samps

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

  def _log_like(self, th): 
    """ approximate log like is just independent poissons """
    loglam = th[0] + th[1:]
    lam     = self._cell_vol * np.exp(loglam)
    #pr(x | lam) = lam^x * exp(-lam) / x!
    ll = loglam * self._grid_counts - lam  # log-like
    return np.sum(ll)


