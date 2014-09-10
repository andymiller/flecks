import numpy as np
import numpy.random as npr
# import numpy.random
 
def spherical_proposal(theta, prop_scale=.05): 
  """ jitters theta with spherical gaussian noise """
  thp = theta + prop_scale*np.random.randn(len(theta))
  if np.any(thp < 0): 
    print thp
    return None
  else: 
    return thp

def slicesample_cov_hyper( th,            #current state of hyper parameters
                           f,             #current state of GP governed by th
                           whiten_func,   #function handle to whiten a GP sample
                           unwhiten_func, #function handle to unwhiten a GP sample
                           like_func,     #likelihood func (of gaussian)
                           ln_prior ):    #log prior over hyper parameters 
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


def uni_slice_sample(init_x, logprob, lower, upper):
    llh_s = np.log(npr.rand()) + logprob(init_x)
    while True:
        new_x = npr.rand()*(upper-lower) + lower
        new_llh = logprob(new_x)
        if new_llh > llh_s:
            return new_x
        elif new_x < init_x:
            lower = new_x
        elif new_x > init_x:
            upper = new_x
        else:
            raise Exception("Slice sampler shrank to zero!")

def slice_sample(init_x, logprob, sigma=1.0, step_out=True, max_steps_out=1000, 
                 compwise=True, doubling_step=True, verbose=False):

    def direction_slice(direction, init_x):
        def dir_logprob(z):
            return logprob(direction*z + init_x)

        def acceptable(z, llh_s, L, U):
            while (U-L) > 1.1*sigma:
                middle = 0.5*(L+U)
                splits = (middle > 0 and z >= middle) or (middle <= 0 and z < middle)
                if z < middle:
                    U = middle
                else:
                    L = middle
                # Probably these could be cached from the stepping out.
                if splits and llh_s >= dir_logprob(U) and llh_s >= dir_logprob(L):
                    return False
            return True
    
        upper = sigma*npr.rand()
        lower = upper - sigma
        llh_s = np.log(npr.rand()) + dir_logprob(0.0)

        l_steps_out = 0
        u_steps_out = 0
        if step_out:
            if doubling_step:
                while (dir_logprob(lower) > llh_s or dir_logprob(upper) > llh_s) and (l_steps_out + u_steps_out) < max_steps_out:
                    if npr.rand() < 0.5:
                        l_steps_out += 1
                        lower       -= (upper-lower)                        
                    else:
                        u_steps_out += 1
                        upper       += (upper-lower)
            else:
                while dir_logprob(lower) > llh_s and l_steps_out < max_steps_out:
                    l_steps_out += 1
                    lower       -= sigma                
                while dir_logprob(upper) > llh_s and u_steps_out < max_steps_out:
                    u_steps_out += 1
                    upper       += sigma

        start_upper = upper
        start_lower = lower

        steps_in = 0
        while True:
            steps_in += 1
            new_z     = (upper - lower)*npr.rand() + lower
            new_llh   = dir_logprob(new_z)
            if np.isnan(new_llh):
                print new_z, direction*new_z + init_x, new_llh, llh_s, init_x, logprob(init_x)
                raise Exception("Slice sampler got a NaN")
            if new_llh > llh_s and acceptable(new_z, llh_s, start_lower, start_upper):
                break
            elif new_z < 0:
                lower = new_z
            elif new_z > 0:
                upper = new_z
            else:
                raise Exception("Slice sampler shrank to zero!")

        if verbose:
            print "Steps Out:", l_steps_out, u_steps_out, " Steps In:", steps_in

        return new_z*direction + init_x

    if type(init_x) == float or isinstance(init_x, np.number):
        init_x = np.array([init_x])
        scalar = True
    else:
        scalar = False

    dims = init_x.shape[0]
    if compwise:
        ordering = range(dims)
        npr.shuffle(ordering)
        new_x = init_x.copy()
        for d in ordering:
            direction    = np.zeros((dims))
            direction[d] = 1.0
            new_x = direction_slice(direction, new_x)

    else:
        direction = npr.randn(dims)
        direction = direction / np.sqrt(np.sum(direction**2))
        new_x = direction_slice(direction, init_x)

    if scalar:
        return float(new_x[0])
    else:
        return new_x
             
#def slicesample(xx, llh_func, last_llh=None, sigma=1, step_out=True):
#    """ simple slice sampling function """
#    dims = xx.shape[0]
#    perm = range(dims)
#    np.random.shuffle(perm)
#     
#    if (type(sigma).__name__ == 'int') or (type(sigma).__name__ == 'float'):
#        sigma = np.tile(sigma, dims)
#    elif (type(sigma).__name__ == 'tuple') or (type(sigma).__name__ == 'list'):
#        sigma = np.array(sigma)
# 
#    if last_llh is None:
#        last_llh = llh_func(xx)
# 
#    for d in perm:
#        llh0   = last_llh + np.log(numpy.random.rand())
#        rr     = np.random.rand(1)
#        x_l    = xx.copy()
#        x_l[d] = x_l[d] - rr*sigma[d]
#        x_r    = xx.copy()
#        x_r[d] = x_r[d] + (1-rr)*sigma[d]
#         
#        if step_out:
#            llh_l = llh_func(x_l)
#            while llh_l > llh0:
#                x_l[d] = x_l[d] - sigma[d]
#                llh_l  = llh_func(x_l)
#            llh_r = llh_func(x_r)
#            while llh_r > llh0:
#                x_r[d] = x_r[d] + sigma[d]
#                llh_r  = llh_func(x_r)
# 
#        x_cur = xx.copy()
#        while True:
#            xd       = np.random.rand(1)*(x_r[d] - x_l[d]) + x_l[d]
#            x_cur[d] = xd
#            last_llh = llh_func(x_cur)
#            if last_llh > llh0:
#                xx[d] = xd
#                break
#            elif xd > xx[d]:
#                x_r[d] = xd
#            elif xd < xx[d]:
#                x_l[d] = xd
#            else:
#                raise RuntimeException("Slice sampler shrank too far.")
#    return xx, last_llh
#
