import numpy as np
from ess import elliptical_slice
from sample_covar_hypers import whitened_mh
from util import nd_grid_centers
from kernel import SQEKernel

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
#

class LGCP: 

  def __init__(self, dim=2, grid_dim=(25,25), bbox=[(0,1), (0,1)], covar=["sqe", "sqe"]):
    self._dim      = dim              # dimensionality of the point process
    self._grid_dim = grid_dim         # number of tiles for each dimension
    self._bbox     = bbox             # range of each dimension
    self._data     = None
    self._sigma_chol = None
    assert self._dim == len(self._grid_dim), 'LGCP Space dimensions not coherent'

    # set up covariance parameters
    # different dimensions may have different covariance functions/params
    self._kern = SQEKernel(length_scale = 20*np.ones(self._dim), sigma2=1)

    #set up kernel for each dimension 
    self._kerns = []
    for d in range(self._dim): 
        self._kerns.append( Kernel.factory(covar[d]) )

  def fit(self, data, Nsamps=550, verbose=T): 
    """ data should come in as a dim by N dimensional numpy array """
    assert self._dim == data.shape[1]
    
    # grid data into tile counts
    grid_counts, edges = np.histogramdd( data, \
                                         bins=self._grid_dim, \
                                         range=self._bbox )
    self._grid_counts  = grid_counts.flatten()
    self._grid_centers = nd_grid_centers(edges)

    # number of latent z's to sample (including bias term)
    Nz = len(self._grid_counts) + 1

    # sample from the posterior w/ ESS, slice sample Hypers
    z_curr  = np.zeros( Nz )
    h_curr  = self._kern.hyper_params()
    ll_curr = None
    z_samps = np.zeros( (Nsamps, Nz) )
    h_samps = np.zeros( (Nsamps, len(h_curr) ) )
    lls     = np.zeros( Nsamps )
    accept_rate = 0
    for i in range(Nsamps):
      if i%10==0 and verbose: print "  samp %d of %d"%(i,Nsamps)

      #sample latent surface
      prior_samp = self._gen_prior()
      z_curr, log_lik = elliptical_slice( z_curr, \
                                          prior_samp, \
                                          self._log_like, \
                                          cur_lnpdf=ll_curr)
      #sample hyper params
      h_curr, z_hyper, accepted, ll = whitened_mh( h_curr, \
                                                  z_curr[1:], \
                                                  lambda(h): self._kern.K(self._grid_centers, self._grid_centers, h), \
                                                  lambda(z): self._log_like(np.append(z_curr[0], z)), \
                                                  lambda(h): self._kern.prior_lnpdf(h) )
      z_curr = np.append(z_curr[0], z_hyper)
    
      #store samples
      z_samps[i,] = z_curr
      h_samps[i,] = h_curr
      lls[i] = ll

      #chain stats
      accept_rate += accepted

    print "final acceptance rate: ", float(accept_rate)/Nsamps
    self._z_samps = z_samps[100:,]
    self._h_samps = h_samps[100:,]
    return z_samps, h_samps

  def posterior_mean_lambda(self): 
    """ return the posterior mean lambda surface """
    zvecs  = (self._z_samps[:,0] + self._z_samps[:,1:].T).T
    lamvec = np.exp(zvecs).mean(axis=0)
    return np.reshape(lamvec, self._grid_dim)

  def posterior_var_lambda(self):
    """ return the posterior variance of the lambda surface """
    zvecs = (self._z_samps[:,0] + self._z_samps[:,1:].T).T
    lamvec = np.exp(zvecs).var(axis=0)

  def plot_3d(self):
    X,T = np.meshgrid(xgrid, tgrid)
    Z   = lam(X, T, mus, omegas)
    fig = plt.figure(figsize=(14,6))
    # `ax` is a 3D-aware axis instance because of the projection='3d' keyword argument to add_subplot
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    # surface_plot with color grading and color bar
    p = ax.plot_surface(X, T, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    cb = fig.colorbar(p, shrink=0.5)




  def _log_like(self, th): 
    """ approximate log like is just independent poissons """
    loglam = th[0] + th[1:]
    lam     = np.exp(loglam)
    #pr(x | lam) = lam^x * exp(-lam) / x!
    ll = loglam * self._grid_counts - lam  # log-like
    return np.sum(ll)

  def _gen_prior(self):
    """ populate covariance matrix and generate a sample """
    if self._sigma_chol is None:
        Sigma = self._kern.K(self._grid_centers, self._grid_centers) + np.diag(1e-6*np.ones(len(self._grid_centers)))
        self._sigma_chol = np.linalg.cholesky(Sigma)  # TODO use goddamn toeplitz or something for this
    y = np.random.randn(len(self._grid_centers))  # standard normal
    z0 = np.sqrt(10)*np.random.randn()
    z = np.append(z0, self._sigma_chol.dot(y))
    return z

if __name__=="__main__":
  
  import pylab as plt
  #x = np.random.rand(500,2)
  x = np.random.beta(a=1, b=2, size=(1000,2))
  lgcp = LGCP()
  z_samps = lgcp.fit(x)

  lam_mean = lgcp.posterior_mean_lambda()
  #print lam_mean
  #print lam_mean.mean()
  fig = plt.figure()
  plt.imshow(lam_mean, interpolation='none', origin='lower', extent=[0,1,0,1])
  plt.colorbar()
  plt.hold(True)
  plt.scatter(x[:,0], x[:,1])
  plt.show()
  
  #plot hyperparam posterior dists
  hsamps = lgcp._h_samps
  fig = plt.figure()
  f, axarr = plt.subplots(2, sharex=True)
  axarr[0].hist(hsamps[:,0], 20, normed=True)
  axarr[1].hist(hsamps[:,1], 20, normed=True)
  plt.show()
  fig.savefig('lgcp_length_scale_dist_test.pdf')



