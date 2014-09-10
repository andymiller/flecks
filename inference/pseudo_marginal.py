#
# Implementation of Pseudo marginal MCMC for Poisson Regression
import numpy as np
from laplace_approx import laplace_approx
from scipy.misc import logsumexp
#from scipy.stats import poisson
from numpy.linalg import inv, cholesky, solve

def approx_log_marg_like(K_theta, like_func, y=None): 
    """ approx_log_marg_like: returns approximate log marginal likelihood 
    (based on importance sampling - unbiased)

    Computes the importance sampling based approximation 

        log p(y | K_theta) = log \int p(y | f) p(f | K_theta)
                           ~ log \sum_i p(y | f_i) p(f_i | K_theta) / q(f_i)

    where q(f_i) is an approximate distribution that can be sampled (e.g. a
    laplace approximation).  The log like function must support gradient
    and hessian product calculations: 

      ll    = log_like(y, f, cK_inv)
      grad  = log_like(y, f, cK_inv, grad=True)
      hessp = log_like(y, f, cK_inv, hessp=True, p=p)

    Args:
      y:        Nx1 vector, observations that go into likelihood
      log_like: function handle, determines the term p(y | f)
      K_theta:  NxN matrix, determines the Gaussian prior covariance over f

    Returns: 
      log_py: (unbiased IS-based approximation of log p(y | K_theta)
    """

    # cache like func such that you don't have to pass data around
    if y is not None:
      like_func = lambda(f): like_func(y, f) 

    # first find params of a laplace approx q(f | f_mu, f_Sig), 
    f_mu, f_Sig, cK_theta, cK_inv = laplace_approx(K_theta, like_func)

    # cache covariance manipulations
    cf_Sig      = cholesky(f_Sig)
    cf_Sig_inv  = inv(cf_Sig)
    (sign, ldetK )  = np.linalg.slogdet(K_theta) 
    (sign, ldetSig) = np.linalg.slogdet(f_Sig)
    
    #sample a bunch to find weights for unbiased est of marg like
    N_imp = 500
    log_rats = np.zeros(N_imp)
    for n in range(N_imp): 

        # generate from the MVN posterior approx, evaluate log like
        fi = cf_Sig.dot(np.random.randn(len(f_mu))) + f_mu 
        ll = like_func(fi, cK_inv)

        # compute log prior over sample
        p_fi = -.5 * ldetK   - .5 * fi.dot(cK_inv.T).dot( cK_inv.dot(fi) ) #- .5*N*np.log(2*np.pi)
        q_fi = -.5 * ldetSig - .5 * (fi-f_mu).dot(cf_Sig_inv.T).dot( cf_Sig_inv.dot(fi-f_mu) ) #- .5*N*np.log(2*np.pi) 
        log_rats[n] = ll + p_fi - q_fi 

    log_py = logsumexp(log_rats) - np.log(N_imp)
    return log_py


##########################################################################
# Proposal distributions for pseudo marginal GP MH
# TODO: Make them more generic
def spherical_proposal_scalar(theta, prop_scale=.05): 
  """ jitters theta with spherical gaussian noise """
  thp = theta + prop_scale*np.random.randn()
  return thp

def jump_proposal_scalar(th, prop_scale=.05):
  """ defines a noisy jump from a value to an integer multiplied/divided
  value (used for periods in periodic covariance functions) """
  factor = np.random.geometric(p=.5) + 1  
  if np.random.rand() < .5: 
    factor = 1./factor
  th_prop = factor * th + prop_scale*np.random.randn()
  return th_prop

def pseudo_gp_mh( th,                # current state of hyper parameters
                  kern_func,         # kernel function
                  like_func,         # likelihood func
                  ln_prior,          # log prior over hyper parameters 
                  curr_marg_ll  = None,    #current log marg liklihood value
                  proposal_func = None): 
  """ performs a psuedo-marginal MCMC step.  It uses a laplace approx + 
  importance sampling to integrate out the value f and computes an unbiased
  estimate of the (log) marginal likelihood.  Then it does a simple MH 
  accept/reject step

  INPUT: 
    - th       : current state of the cov func hyperparams
    - kern_func: function kern_func(th) = K_th yields a gram matrix, 
                 where locations K(x,x') are implicit
    - like_func: log pr(y | f) (e.g. poisson, bernoulli, gaussian likelihood)
    - ln_prior : prior over the hyper parameters
    - prop_dist: proposal distribution for th (function of curr th)

  OUTPUT: 
    - th      : new covariance kernel parameters
    - accept  : boolean (accepted proposal or rejected?)
    - marg_ll : new value of the marginal likelihood (dont have to recompute)
  """
  if curr_marg_ll is None: 
    curr_marg_ll = approx_log_marg_like(kern_func(th), like_func=like_func)

  #gen proposal, stop early if it's out of bounds
  th_prop = proposal_func(th)
  prop_log_prior = ln_prior(th_prop)
  if prop_log_prior < -1e30:   # HACK - the prior makes this impossible
    return th, False, curr_marg_ll

  # compute (approximate) log marginal like
  K_prop = kern_func(th_prop)
  prop_marg_ll = approx_log_marg_like(K_prop, like_func=like_func)

  # pr(accept) = (likelihood ratio)           (prior ratio)
  log_ratio = (prop_marg_ll - curr_marg_ll) + (prop_log_prior - ln_prior(th))
  if np.log(np.random.rand()) < log_ratio:
    return th_prop, True, prop_marg_ll
  else:
    return th, False, curr_marg_ll


