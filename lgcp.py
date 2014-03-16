import numpy as np
from ess import elliptical_slice
from util import nd_grid_centers, whitened_mh, x_grid_centers
from kernel import MultiKronKernel #Kernel, SQEKernel
from kron_util import kron_mat_vec_prod
from mpl_toolkits.mplot3d.axes3d import Axes3D

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

  def __init__(self, dim=2, grid_dim=(50,50), bbox=[(0,1), (0,1)], kern=None):
    self._dim      = dim              # dimensionality of the point process
    self._grid_dim = grid_dim         # number of tiles for each dimension
    self._bbox     = bbox             # range of each dimension
    self._data     = None
    self._sigma_chol = None
    assert self._dim == len(self._grid_dim), 'LGCP Space dimensions not coherent'

    #compute cell volume
    self._cell_vol = 1.0
    for d in range(self._dim):
      self._cell_vol *= float(bbox[d][1] - bbox[d][0]) / grid_dim[d] 

    #set up kernel for each dimension 
    if kern is None: 
      self._kern = MultiKronKernel( ["sqeu", "sqeu"] )
    else: 
      self._kern = Kern
  
  def fit(self, data, Nsamps=550, verbose=True): 
    """ data should come in as a dim by N dimensional numpy array """
    assert self._dim == data.shape[1]
    
    # grid data into tile counts
    grid_counts, edges = np.histogramdd( data, \
                                         bins=self._grid_dim, \
                                         range=self._bbox )
    self._grid_counts  = grid_counts.flatten('F')   # counts in each box
    self._grids        = x_grid_centers(edges)      # list of grid centers for each dimension 
    #self._grid_centers = nd_grid_centers(edges)     # center of each box (list of points)

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

      #sample latent surface
      prior_samp = self._gen_prior(h_curr)
      z_curr, log_lik = elliptical_slice( z_curr, \
                                          prior_samp, \
                                          self._log_like, \
                                          cur_lnpdf=ll_curr)
      
      ## whitens latent field (probably should find better place for this)
      def whiten(th, f): 
        Ks = self._kern.gram_list(th, self._grids)
        Ls = [np.linalg.cholesky(K) for K in Ks]
        Ls_inverse = [np.linalg.inv(L) for L in Ls]
        nu = kron_mat_vec_prod(Ls_inverse, f)
        return nu

      def unwhiten(thp, nu): 
        Ks = self._kern.gram_list(thp, self._grids)
        Ls = [np.linalg.cholesky(K) for K in Ks]
        fp = kron_mat_vec_prod(Ls, nu)
        return fp

      #sample hyper params
      h_curr, z_hyper, accepted, ll = whitened_mh( h_curr, \
               z_curr[1:], \
               whiten, \
               unwhiten, \
               lambda(z): self._log_like(np.append(z_curr[0], z)), \
               lambda(h): self._kern.hyper_prior_lnpdf(h) )
      z_curr = np.append(z_curr[0], z_hyper)
    
      #store samples
      z_samps[i,] = z_curr
      h_samps[i,] = h_curr
      lls[i] = ll

      #chain stats
      accept_rate += accepted

    print "final acceptance rate: ", float(accept_rate)/Nsamps
    self._z_samps = z_samps[300:,]
    self._h_samps = h_samps[300:,]
    self._lls     = lls
    return z_samps, h_samps

  def posterior_mean_lambda(self): 
    """ return the posterior mean lambda surface """
    zvecs  = (self._z_samps[:,0] + self._z_samps[:,1:].T).T
    lamvec = self._cell_vol * np.exp(zvecs).mean(axis=0)
    return np.reshape(lamvec, self._grid_dim, order='F')

  def posterior_var_lambda(self):
    """ return the posterior variance of the lambda surface """
    zvecs = (self._z_samps[:,0] + self._z_samps[:,1:].T).T
    lamvec = self._cell_vol * np.exp(zvecs).var(axis=0)
    return np.reshape(lamvec, self._grid_dim, order='F')

  #def plot_3d(self):
  #  X,T = np.meshgrid(self._grids[0], self._grids[1])
  #  Z   = self.posterior_mean_lambda()
  #  fig = plt.figure(figsize=(14,6))
  #  # `ax` is a 3D-aware axis instance because of the projection='3d' keyword argument to add_subplot
  #  ax = fig.add_subplot(1, 1, 1, projection='3d')
  #  # surface_plot with color grading and color bar
  #  p = ax.plot_surface(X, T, Z, rstride=1, cstride=1, linewidth=0, antialiased=False)
  #  cb = fig.colorbar(p, shrink=0.5)

  def _log_like(self, th): 
    """ approximate log like is just independent poissons """
    loglam = th[0] + th[1:]
    lam     = self._cell_vol * np.exp(loglam)
    #pr(x | lam) = lam^x * exp(-lam) / x!
    ll = loglam * self._grid_counts - lam  # log-like
    return np.sum(ll)

  def _gen_prior(self, hparams):
    """ generate from the GP prior (with the object's grid points, and 
    some flattened vector of hyperparameters for the list of covariance funcs """
    Ks = self._kern.gram_list(hparams, self._grids)  # covariance mats for each dimension
    Ls = [np.linalg.cholesky(K) for K in Ks]         # cholesky of each cov mat

    #generate spatial component and bias 
    z_spatial = kron_mat_vec_prod( Ls, np.random.randn(len(self._grid_counts)) )
    z_bias    = np.sqrt(10)*np.random.randn()
    return np.append(z_bias, z_spatial)


if __name__=="__main__":
  
  import pylab as plt
  #x = np.random.rand(500,2)
  x = np.row_stack( (np.random.beta(a=5, b=2, size=(500,2)), 
                     np.random.beta(a=2, b=5, size=(500,2)) ) )
  #x[500:,0] = 1.0 - x[500:,0]

  lgcp = LGCP()
  z_samps = lgcp.fit(x)

  plt.plot(lgcp._lls)
  plt.show()
  lam_mean = lgcp.posterior_mean_lambda()
  
  fig = plt.figure()
  plt.imshow(lam_mean.T, interpolation='none', origin='lower', extent=[0,1,0,1])
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

  #fig.savefig('lgcp_length_scale_dist_test.pdf')

  from mpl_toolkits.mplot3d import Axes3D
  from matplotlib import cm
  from matplotlib.ticker import LinearLocator, FormatStrFormatter
  import matplotlib.pyplot as plt
  import numpy as np

  fig = plt.figure()
  ax = fig.gca(projection='3d')
  X = lgcp._grids[0]
  Y = lgcp._grids[1]
  X, Y = np.meshgrid(X, Y)
  Z = lam_mean 
  surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, \
                         linewidth=0, antialiased=False)
  ax.set_zlim(np.min(lam_mean), np.max(lam_mean))
  ax.zaxis.set_major_locator(LinearLocator(10))
  ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
  fig.colorbar(surf, shrink=0.5, aspect=5)
  
  plt.show()
  
