import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist
from scipy.stats import gamma
from util.kron_util import kron_mat_vec_prod

"""
Kernel objects: these are very Gaussian Process oriented kernels.  A 
                Kernel object should have the following capabilities: 

      0. init(prior_type, hyper_params): 
        - initialize a certain type of kernel with a certain type of prior and 
          initial hyper_params
      1. K(pts, pts, hypers): 
        - Return covariance matrix given two lists of points (each pair)
      2. hypers(): 
        - Return hyper parameters (or a sample from the prior), 
          really anything that is the appropriate length 
      3. hyper_prior_lnpdf(hparms): 
        - log prior over hyper parameters passed 
      4. gen_prior(hyper_parameters, grids, nu=None): 
        - draws from prior at cartesian grid points defined
          by grids.  This is useful when certain types of 
          structures don't necessitate the instantiation of 
          an entire Gram matrix over a huge set of points (which can 
          be computationally prohibitive)
      5. whiten_process(f, hypers, pts): 
        - whitens the process passed in, using setting of hypers and 
          process spatial locations
        - unwhiten process is just self.gen_prior(hypers, pts, nu) 
          where nu is the whitened process (which is equal in distribution
          to a spherical normal draw).
"""

##TODO figure out how to make sure methods above are 
##     implemented in all subclasses!
class Kernel(object): 
  """ simple model for a kernel - PSD function 
  """
  def __init__(self): 
    pass

  @staticmethod
  def factory(kernel_name = "sqe"): 
    return {
      'sqe'  : SQEKernel(), 
      'sqeu' : SQEKernelUnscaled(),
      'sm'   : SpectralMixtureKernel(),
      'kron' : MultiKronKernel(), 
      'per'  : PerKernel(),
      'per_unscaled' : PerKernelUnscaled()
      }.get(kernel_name, SQEKernel())

  def K(self, Xi, Xj, hypers=None): 
    raise NotImplementedError

  def hypers(self): 
    raise NotImplementedError

  def hyper_prior_lnpdf(self, hypers):
    raise NotImplementedError

  def gen_prior(self, hypers, pts, nu=None): 
    """ return sample from prior - not optimized at all """
    L = np.linalg.cholesky(self.K(pts, pts, hypers) + np.eye(len(pts))*1e-8)
    if nu is None: 
      nu = np.random.randn(len(pts))
    return L.dot(nu)

  def whiten_process(self, f, hypers, pts): 
    #TODO whitening is messed up here
    L = np.linalg.cholesky( self.K(pts, pts, hypers) + 1e-8*np.eye(len(pts)) )
    nu = sp.linalg.solve_triangular(L, f, lower=True)
    if nu.var() > 10: 
      print nu.mean(), nu.var()
      raise "!!!! ===> nu moments not right!!!: "
    return nu

  def conditional_params(self, f, pts, pred_pts, hypers, jitter=0.):
    Kxx = self.K(pts, pts, hypers) + np.eye(len(pts))*jitter
    Kxp = self.K(pts, pred_pts, hypers) 
    #Kxx_inv = np.linalg.inv(Kxx)
    mu_pred = Kxp.T.dot( np.linalg.solve(Kxx,f) )
    Kpp = self.K(pred_pts, pred_pts, hypers) + jitter*np.eye(len(pred_pts))
    Sig_pred = Kpp - Kxp.T.dot( np.linalg.solve(Kxx, Kxp) )
    return mu_pred, Sig_pred

  def hyper_names(self): 
    return self._hyper_names

#
# kronecker kernel (composed of multiple kernels)
#
class MultiKronKernel(Kernel): 
  """ maintains a list of kernels corresponding to each dimension. 
  This class assumes that the Kernel over the multi-dimensional space
  is a tensor product kernel, so it's Gram matrix can only be instantiated 
  over *regular grids* in each dimension.  This is ideal for a discretized 
  approximation to a space. """
  def __init__(self, kernel_names=["sqeu", "sqeu"], kernels=None): 
    self._scale = 1.
    self._kerns = []
    if kernels is not None: 
      self._kerns = kernels
    else: 
      #TODO fix the factory?
      for kname in kernel_names:
        self._kerns.append( SQEKernelUnscaled() ) # Kernel.factory(kname) )
      #self._kerns.append( Kernel.factory(kname) )
  
  def _gram_list(self, hparams, grids): 
    """ generate a list of gram matrices, one for each dimension, 
    given the fixed grid """
    scale, kern_hypers = self._chunk_hypers(hparams)
    #print "multikron kernel hypers in: ", scale, kern_hypers
    scale = scale ** (1./len(grids))
    Ks = []
    for d in range(len(self._kerns)):
        Kd = self._kerns[d].K(grids[d], grids[d], kern_hypers[d]) + \
                               np.diag(1e-8*np.ones(len(grids[d])) )
        Ks.append(scale*Kd)
    return Ks
 
  def hyper_prior_lnpdf(self, h): 
    """ prior over hyper parameters for this kernel - 
    using scale invariant for now """
    scale, hparams = self._chunk_hypers(h)
    lls = 0
    for d in range(len(hparams)):
      lls += self._kerns[d].hyper_prior_lnpdf(hparams[d])
    lls += gamma(2, scale=2).logpdf(scale)
    return lls

  def hypers(self): 
    """ returns the hyperparameters associated with each kernel 
    as a vector """
    hypers = [self._scale]
    for k in self._kerns: 
        hypers.append( k.hypers() )
    return np.reshape(hypers, (-1,))

  def set_hypers(self, hypers):
    scale, kern_hypers = self._chunk_hypers(hypers)
    self._scale = scale
    for k in self._kerns:
      k.set_hyper_params(kern_hypers)

  def _chunk_hypers(self, hypers): 
    """ separate out a flattened vector of hyperparameters to a 
    list of arrays of them (using kernel information) """
    kern_hypers = []
    scale = hypers[0]
    startI = 1
    for d in range(len(self._kerns)):
      endI = startI + len(self._kerns[d].hypers())
      kern_hypers.append( hypers[startI:endI] )
      startI = endI
    return scale, kern_hypers

  def gen_prior(self, hparams, grids, nu=None):
    """ generate from the GP prior (with the object's grid points, and 
    some flattened vector of hyperparameters for the list of covariance funcs """
    Ks = self._gram_list(hparams, grids)  # covariance mats for each dimension
    Ls = [np.linalg.cholesky(K) for K in Ks]         # cholesky of each cov mat

    #generate spatial component and bias 
    Nz = np.prod( [len(g) for g in grids] )
    if nu is None: 
      nu = np.random.randn(Nz)
    return kron_mat_vec_prod(Ls, nu) 

  def whiten_process(self, f, hypers, grids): 
    Ks = self._gram_list(hypers, grids)
    Ls = [np.linalg.cholesky(K) for K in Ks]
    Ls_inverse = [np.linalg.inv(L) for L in Ls]
    nu = kron_mat_vec_prod(Ls_inverse, f)
    return nu

  def hyper_names(self): 
    hnames = ["scale"]
    for kern in self._kerns:
      hnames += kern.hyper_names()
    return hnames 

class SpectralMixtureKernel(Kernel):
  """ Simple, one-dimensional spectral mixture kernel.  The 
  spectral density of this stationary kernel is represented 
  by a mixture of Gaussians, which provides a closed form for 
  the resulting spatial kernel. 

  Refs: 
   - Wilson and Adams. Gaussian Process Kernels for Pattern 
     Discovery and Extrapolation. ICML, 2013.
  """
  def __init__(self, num_comp = 3):
    #default weights, default means, default vars for spectral density
    self._num_comp = num_comp
    ws  = 1./num_comp * np.ones(num_comp)  # weights
    mus = np.arange(num_comp)              # means
    vs  = np.ones(num_comp)                # variances
    self._set_hyper_params( np.append(ws, np.append(mus, vs)) )

  def K(self, Xi, Xj, hypers=None):
    assert len(Xi.shape)==len(Xj.shape) and len(Xi.shape)==1, "multi-dim not supported"
    if hypers is not None:
      self._set_hyper_params( hypers )

    #pairwise dists
    tau = np.subtract.outer(Xi, Xj)
    k   = np.zeros( tau.shape )
    for q in range(self._num_comp): 
      kq = np.exp(-2.*np.pi*np.pi * tau*tau * self._vars[q]) * \
           np.cos( 2.*np.pi*tau * self._means[q] )
      kq *= self._weights[q]
      k += kq
    return k

  def hypers(self):
    """ return flat vector of hyper parameters """
    return np.append( self._weights, np.append(self._means, self._vars) )
  
  def hyper_prior_lnpdf(self, hypers): 
    """ on mus, place fat tailed prior """
    self._weights, self._means, self._vars = np.split(hypers, 3)
    return -np.sum(np.log(self._vars))
  
  def _set_hyper_params(self, hypers): 
    """ pass in flat vector of hyper parameters """
    assert len(hypers) == self._num_comp*3, "SMKernel, num hypers not correct"
    self._weights, self._means, self._vars = np.split(hypers, 3)


class SQEKernelUnscaled(Kernel):
  """ SQE Kernel (only one dimension )with only one 
  hyper paramter, the length scale """
  def __init__(self, length_scale=1, alpha0=12, beta0=.5):
    self._length_scale = alpha0*beta0
    self._alpha0 = alpha0      #shape/convolution parameter
    self._beta0 = beta0        #scale, inverse rate parameter
    self._hyper_names = ["length_scale"]

  def K(self, Xi, Xj, length_scale=None): 
    """ gram mat between Xi and Xj """
    if length_scale is not None:
      self._length_scale = length_scale

    # reshape to make sure they are stacks of vecs
    if len(Xi.shape) == 1:
      Xi = np.reshape( Xi, (-1, 1) )
      Xj = np.reshape( Xj, (-1, 1) )
    dists = cdist(Xi, Xj, 'sqeuclidean')
    return np.exp(-.5*dists/(self._length_scale*self._length_scale))
 
  def hypers(self): 
    return np.array([self._length_scale])

  def hyper_prior_lnpdf(self, hypers): 
    """ jeffrey's/scale-free prior on length scales """
    self._length_scale = hypers[0]
    ll_prior = gamma(self._alpha0, scale=self._beta0).logpdf(self._length_scale)
    return ll_prior

#########################################################
# Periodic kernel and its unscaled (fixed scaled)version
#########################################################
class PerKernel(Kernel):
  """ Periodic kernel """
  def __init__(self, scale=5., per=5., length_scale=10.): 
    self._set_hypers([scale, per, length_scale])
    self._hyper_names = ["Scale", "Period", "Length Scale"]

  def _set_hypers(self, hypers=[5., 5., 10.]):
    self._scale, self._per, self._length_scale = hypers

  def K(self, Xi, Xj, hypers=None):
    if hypers is not None:
      self._set_hypers(hypers)

    if len(Xi.shape)==1:
      Xi = np.reshape(Xi, (-1,1))
      Xj = np.reshape(Xj, (-1,1))

    dists    = cdist(Xi, Xj, 'euclidean')
    sin_term = np.sin(np.pi*np.abs(dists)/self._per)  
    l2       = self._length_scale*self._length_scale
    s2       = self._scale * self._scale
    return s2 * np.exp( -2. * sin_term*sin_term / l2 )

  def hypers(self):
    return np.array([self._scale, self._per, self._length_scale])

  def hyper_prior_lnpdf(self, hypers):
    self._set_hypers(hypers)
    ll_scale  = gamma(2, scale=2).logpdf(self._scale)
    ll_per    = gamma(2, scale=2).logpdf(self._per)
    ll_lscale = gamma(2, scale=2).logpdf(self._length_scale)
    return ll_scale+ll_per+ll_lscale

class PerKernelUnscaled(PerKernel):
  """ Periodic kernel without the scale parameter """
  def __init__(self, fixed_scale=1., per=5., length_scale=10., min_per=2.):
    self._fixed_scale = fixed_scale
    self._min_per     = min_per             #fix a minimum period to fix aliasing effects 
                                            #(tiny, periods=high frequencies which can 
                                            # look like lower frequencies/bigger periods with weird jaggies
    self._set_hypers([per, length_scale])
    self._hyper_names = ["Period", "Length Scale"]

  def _set_hypers(self, hypers=[5.,10.]):
    self._scale = self._fixed_scale
    self._per, self._length_scale  = hypers

  def hypers(self):
    return np.array([self._per, self._length_scale])

  def hyper_prior_lnpdf(self, hypers):
    self._set_hypers(hypers)
    ll_per    = gamma(2, scale=2).logpdf(self._per - self._min_per)  #min shifted
    ll_lscale = gamma(2, scale=2).logpdf(self._length_scale)
    return ll_per+ll_lscale

class LinearKernel(Kernel): 
  """ Linear kernel, K_lin(x,x') = sig2_b + sig2_v(x-c)(x'-c) """
  def __init__(self, bias_var=.5, slope_var=.1, offset=0): 
    self._set_hypers([bias_var, slope_var, offset])
    self._hyper_names = ["Bias var", "Slope var", "Offset"] 

  def _set_hypers(self, hypers=[5., 5., 10.]):
    self._bias_var, self._slope_var, self._offset = hypers

  def K(self, Xi, Xj, hypers=None):
    if hypers is not None:
      self._set_hypers(hypers)
    if len(Xi.shape)==1:
      Xi = np.reshape(Xi, (-1,1))
      Xj = np.reshape(Xj, (-1,1))
    outers = np.outer(Xi-self._offset, Xj-self._offset) 
    return self._bias_var + self._slope_var*outers

  def hypers(self):
    return np.array([self._bias_var, self._slope_var, self._offset])

  def hyper_prior_lnpdf(self, hypers):
    self._set_hypers(hypers)
    ll_scale  = gamma(2, scale=2).logpdf(self._bias_var)
    ll_per    = gamma(2, scale=2).logpdf(self._slope_var)
    ll_lscale = normal(mean=0,scale=10).logpdf(self._offset)
    return ll_scale+ll_per+ll_lscale

##################################################
# TODO implement parent class functions
# for following kernels 
#
class SQEKernel(Kernel): 
  """ Simple squared exponential kernel (independent length scales)"""
  def __init__(self, length_scale=1, sigma2=1):
    hyps = np.append(length_scale, sigma2)
    self._set_hypers( np.append(length_scale, sigma2) )
    self._input_dim = len(self._lscales)

  def _set_hypers(self, hypers): 
    """ local func to set hyperparams from a vector """
    self._sigma2    = hypers[-1]                     # covariance marginal variance
    self._lscales   = hypers[:-1]                    # length scales
    self._inv_V     = np.diag( .5/(self._lscales*self._lscales)) # inverse diag length scale matrix (for mahal dist)

  def K(self, Xi, Xj, hypers=None): 
    if len(Xi.shape) == 1:
      assert self._input_dim==1, "inputs don't match kernel dim"
    elif len(Xi.shape) > 1: 
      assert Xi.shape[1]==self._input_dim, "inputs don't match kernel dim"

    # resets hypers if passed in (l_1, ..., l_d, sigma2)
    if hypers is not None: 
      assert len(hypers)==self._input_dim+1, "num hypers doesn't match"
      self._set_hypers(hypers)

    # reshape to make sure they are stacks of vecs
    Xi = np.reshape( Xi, (-1, self._input_dim) )
    Xj = np.reshape( Xj, (-1, self._input_dim) )
    
    # one dimension is a special case...
    if len(Xi.shape)==1: 
      dists = cdist(Xi, Xj)
      return self._sigma2*np.exp(-(.5/self._lscales)*dists*dists)

    # force Xi, Xj to be 2-d arrays
    dists = cdist(Xi, Xj, 'mahalanobis', VI=self._inv_V)
    return self._sigma2 * np.exp( - dists*dists )

  def hyper_params(self):
    return np.append( self._lscales, self._sigma2 )

  def prior_lnpdf(self, hypers): 
    """ jeffrey's/scale-free prior on length scales """
    return -np.sum(np.log(hypers))



