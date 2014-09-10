import numpy as np
#from scipy.misc import logsumexp
#from scipy.stats import poisson
from scipy.optimize import minimize
from numpy.linalg import inv, cholesky, solve

def laplace_approx(K_theta, log_like, y=None): 
    """ laplace_approx: finds a MAP solution given a Gaussian prior and a 
    differentiable likelihood. 

    Computes the solution:

      argmax_f p(y | f) p(f | K_theta)

    where the likelihood is determined by the function handle log_like.  Note
    that the log_like function must implement it's own Gradient and Hessian
    product functions, and take in the inverse cholesky of K_theta, 
    that is we need to be able to call:

      ll    = log_like(y, f, cK_inv)
      grad  = log_like(y, f, cK_inv, grad=True)
      hessp = log_like(y, f, cK_inv, hessp=True, p=p)

    Args:
      y:        Nx1 vector, observations that go into likelihood
      log_like: function handle, determines the term p(y | f)
      K_theta:  NxN matrix, determines the Gaussian prior covariance over f

    Returns: 
      f_mu: (MAP solution for f)
      f_Sig: (covariance - inverse Hessian at MAP)
      cK_theta: the cholesky decomposition of K_theta
      cK_inv:   the inverse of K_theta
      TODO: add option for inputting cK_theta and K_inv/cK_inv 
    """
    cK_theta = cholesky(K_theta)
    cK_inv   = inv(cK_theta)
    Nparam = K_theta.shape[0]
    x0 = cK_theta.dot(np.random.randn(Nparam))

    # if y is none, then it is assumed data is implicit in likelihood func call
    if y is None: 
        nlog_like      = lambda(f): -log_like(f, cK_inv)
        nlog_like_grad = lambda(f): -log_like(f, cK_inv, grad=True)
        def nlog_like_hessp(f, p): 
            return -log_like(f, cK_inv, hessp=True, p=p)
    else:
        nlog_like      = lambda(f): -log_like(y, f, cK_inv)
        nlog_like_grad = lambda(f): -log_like(y, f, cK_inv, grad=True)
        def nlog_like_hessp(f, p): 
            return -log_like(y, f, cK_inv, hessp=True, p=p)
    res0 = minimize(x0      = x0,
                    fun     = nlog_like,
                    jac     = nlog_like_grad,
                    hessp   = nlog_like_hessp,
                    method  = 'Newton-CG', 
                    options = {'maxiter':1000} )
    f_mu  = res0.x

    # TODO: there is probably some woodbury speedup one can do here - 
    # if i already have the inverse of K
    f_Sig = inv( cK_inv.T.dot(cK_inv) + np.diag(np.exp(f_mu)) )
    #cf_Sig = cholesky(f_Sig)
    return f_mu, f_Sig, cK_theta, cK_inv


