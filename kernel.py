import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import gamma
####################################################
# Kernel classes 
#   - MultiKronKernel: class that takes advantage of 
#     kronecker structure in multi dimensional covariance matrices
#     for optimized inverse, cholesky, multiplications, etc. 
#     This class strings together multiple 1-d kernels
#
#   - Kernel and subclasses: implements simple covariance 
#     functions for one-d kernels
####################################################


class MultiKronKernel: 
  """ maintains a list of kernels corresponding to each dimension. 
  This class assumes that the Kernel over the multi-dimensional space
  is a tensor product kernel, so it's Gram matrix can only be instantiated 
  over *regular grids* in each dimension.  This is ideal for a discretized 
  approximation to a space. """
  
  def __init__(self, kernel_names): 
    self._scale = 1.
    self._kerns = []
    for kname in kernel_names:
      self._kerns.append( Kernel.factory(kname) )
    #self.set_hypers(hypers)
  
  def gram_list(self, hparams, grids): 
    """ generate a list of gram matrices, one for each dimension, 
    given the fixed grid """
    kern_hypers = self._chunk_hypers(hparams)
    Ks = []
    scale = self._scale ** (1./len(grids))
    for d in range(len(self._kerns)):
        Kd = self._kerns[d].K( grids[d], grids[d], kern_hypers[d]) + \
                               np.diag(1e-8*np.ones(len(grids[d])) )
        Ks.append(scale*Kd)
    return Ks
 
  def hyper_prior_lnpdf(self, h): 
    """ prior over hyper parameters for this kernel - 
    using scale invariant for now """
    hparams = self._chunk_hypers(h)
    lls = 0
    for d in range(len(hparams)):
      lls += self._kerns[d].prior_lnpdf(hparams[d])
    return lls

  def hypers(self): 
    """ returns the hyperparameters associated with each kernel 
    as a vector """
    hypers = [self._scale]
    for k in self._kerns: 
        hypers.append( k.hyper_params() )
    return np.reshape(hypers, (-1,))

  def set_hypers(self, hypers):
    kern_hypers = self._chunk_hypers(hypers)
    for k in self._kerns:
      k.set_hyper_params(kern_hypers)

  def _chunk_hypers(self, hypers): 
    """ separate out a flattened vector of hyperparameters to a 
    list of arrays of them (using kernel information) """
    kern_hypers = []
    startI = 0
    for d in range(len(self._kerns)):
      endI = startI + len(self._kerns[d].hyper_params())
      kern_hypers.append( hypers[startI:endI] )
      startI = endI
    return kern_hypers


#
# collection of 1-d kernels to be used with LGCP 
#
class Kernel: 
  """ simple model for a kernel - PSD function 
  """
  def __init__(self): 
    pass

  @staticmethod
  def factory(kernel_name = "sqe"): 
    return {
      'sqe'  : SQEKernel(), 
      'sqeu' : SQEKernelUnscaled(),
      'per'  : PerKernel(),
      }.get(kernel_name, SQEKernel())

  def K(self, Xi, Xj, hypers=None): 
    raise NotImplementedError

  def gram_mat(self, Xs):
    raise NotImplementedError

  def prior_lnpdf(self, hypers):
    raise NotImplementedError

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
    self.set_hyper_params( np.append(ws, np.append(mus, vs)) )

  def K(self, Xi, Xj, hypers=None):
    assert len(Xi.shape)==len(Xj.shape) and len(Xi.shape)==1, "multi-dim not supported"
    if hypers is not None:
      self.set_hyper_params( hypers )

    #pairwise dists
    tau = np.subtract.outer(Xi, Xj)
    k = np.zeros( tau.shape )
    for q in range(self._num_comp): 
      kq = np.exp(-2.*np.pi*np.pi * tau*tau * self._vars[q]) * \
           np.cos( 2.*np.pi * tau * self._means[q] )
      kq *= self._weights[q]
      k += kq
    return k

  def hyper_params(self):
    """ return flat vector of hyper parameters """
    return np.append( self._weights, np.append(self._means, self._vars) )

  def set_hyper_params(self, hypers): 
    """ pass in flat vector of hyper parameters """
    assert len(hypers) == self._num_comp*3, "SMKernel, num hypers not correct"
    self._weights, self._means, self._vars = np.split(hypers, 3)

  def prior_lnpdf(self, hypers): 
    """ constant jeffrey's/scale-free prior on length scales """
    return -np.sum(np.log(hypers))



class SQEKernelUnscaled(Kernel):
  """ SQE Kernel (only one dimension )with only one 
  hyper paramter, the length scale """
  def __init__(self, length_scale=1):
    self._length_scale = length_scale
  
  def K(self, Xi, Xj, length_scale=None): 
    """ gram mat between Xi and Xj """
    if length_scale is not None:
      self._length_scale = length_scale
    
    # reshape to make sure they are stacks of vecs
    if len(Xi.shape) == 1:
      Xi = np.reshape( Xi, (-1, 1) )
      Xj = np.reshape( Xj, (-1, 1) )
    dists = cdist(Xi, Xj, 'sqeuclidean')
    return np.exp( -(1./(2.*self._length_scale*self._length_scale)) * dists)
 
  def hyper_params(self):
    return np.array([self._length_scale])

  def set_hyper_params(self, hypers):
    self._length_scale = hypers[0]

  def prior_lnpdf(self, hypers): 
    """ jeffrey's/scale-free prior on length scales """
    #return -np.sum(np.log(hypers))
    return np.sum(gamma(12, scale=.5).logpdf(hypers))


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

class PerKernel(Kernel):
  """ Periodic Kernel """
  pass








