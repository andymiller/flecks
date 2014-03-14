import numpy as np
from scipy.spatial.distance import cdist
#
# collection of 1-d kernels to be used with LGCP 
#

class Kernel: 
  """ simple model for a kernel - PSD function 
  """
  def __init__(self): 
    pass

  def factory(kernel_name = "sqe"): 
    if kernel_name == "sqe": 
      return SQEKernel()
    elif kernel_name == "per": 
      return PerKernel()
  
  def K(self, Xi, Xj, hypers=None): 
    raise NotImplementedError

  def gram_mat(self, Xs):
    raise NotImplementedError

  def prior_lnpdf(self, hypers):
    raise NotImplementedError

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
    self._inv_V     = np.diag( 1./(self._lscales*self._lscales)) # inverse diag length scale matrix (for mahal dist)

  def K(self, Xi, Xj, hypers=None): 
    if len(Xi.shape) == 1:
      assert self._input_dim==1, "inputs don't match kernel dim"
    elif len(Xi.shape) > 1: 
      assert Xi.shape[1]==self._input_dim, "inputs don't match kernel dim"

    # resets hypers if passed in (l_1, ..., l_d, sigma2)
    if hypers is not None: 
      assert len(hypers)==self._input_dim+1, "num hypers doesn't match"
      self._set_hypers(hypers)

    # force Xi, Xj to be 2-d arrays
    Xi = np.reshape( Xi, (-1, self._input_dim) )
    Xj = np.reshape( Xj, (-1, self._input_dim) )
    dists = cdist(Xi, Xj, 'mahalanobis', VI=self._inv_V)
    return self._sigma2 * np.exp( -.5 * dists )

  def hyper_params(self):
    return np.append( self._lscales, self._sigma2 )

  def prior_lnpdf(self, hypers): 
    """ jeffrey's/scale-free prior on length scales """
    return -np.sum(np.log(hypers))



class PerKernel(Kernel):
  """ Periodic Kernel """
  pass








