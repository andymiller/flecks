import numpy as np

def spherical_proposal(theta, prop_scale=.05): 
  """ jitters theta with spherical gaussian noise """
  thp = theta + prop_scale*np.random.randn(len(theta))
  if np.any(thp < 0): 
    print thp
    return None
  else: 
    return thp

def whitened_mh( th,                #current state of hyper parameters
                 f,                 #current state of GP governed by th
                 whiten_func,       #function handle to whiten a GP sample
                 unwhiten_func,     #function handle to unwhiten a GP sample
                 like_func,         #likelihood func
                 ln_prior,          #log prior over hyper parameters 
                 prop_scale = None, #proposal distribution scale
                 prop_dist = spherical_proposal): 
  """ returns a sample of theta (cov funciont hyper parameters
  given the state of the MVN f.  It first whitens f into nu, 
  leaving that fixed for a higher acceptance rate

  INPUT: 
    - th      : current state of the cov func hyperparams
    - f       : current state of the latent gaussian proc/vars
    - whiten  : function takes in (th, f) to find whitened version of f
                e.g. K_th = Cov_Func(th, x)
                     L_th = chol(K_th)
                     nu   = inv(L_th) * f
                user specified, so this can be optimized version can be 
                passed in
    - unwhiten: function takes in (th_p, nu) and computes unwhitened version 
                of nu
                e.g. K_thp = Cov_Func(th_p, nu)
                     L_thp = chol(K_thp)
                     fp    = L_thp * nu
                again, it's user specified so optimized versions can be passed in
    - Lfn     : likelihood function, func of f
    - prop_dist: proposal distribution for th (function of curr th)

  OUTPUT: 
    - th-new  : new covariance kernel parameters
    - f-new   : new version of the multivariate normal 
  """
  #set proposal function (scale takes over)
  if prop_scale is not None:
    prop_dist = lambda(th): spherical_proposal(th, prop_scale=prop_scale)

  # solve for nu ~ Normal(0, I)      (whiten)
  # whitening function incorporates Covariance 
  # function (but optimized to handle kronecker stuff)
  nu = whiten_func(th, f)   

  # propose th' ~ q(th' ; th)
  thp = prop_dist(th)
  if thp is None: 
    print "bad proposal, returning none"
    return th, f, False, like_func(f) + ln_prior(th)

  # compute implied values f' = L_thp*nu (unwhiten)
  fp = unwhiten_func(thp, nu)

  # mh accept/reject
  ll_p = like_func(fp) + ln_prior(thp)
  #print "=========="
  #print "  proposal likelihood: ", like_func(fp)
  #print "  proposal prior: ", ln_prior(thp)
  ll_o = like_func(f) + ln_prior(th)
  #print "  current likelihood: ", like_func(f)
  #print "  current prior: ", ln_prior(th)
  if -np.random.exponential() < ll_p - ll_o:
    return thp, fp, True, ll_p
  else:
    return th, f, False, ll_o



