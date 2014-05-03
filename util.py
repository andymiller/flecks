import re
import numpy        as np
import numpy.random as npr
import math
import sys

# This should eventually go somewhere else, but I don't know where yet
TERMINATION_SIGNAL = "terminate"

def x_grid_centers( edges ): 
    """ Given D lists of grid edges, return a list of 
    grid centers """
    x_centers = []
    for d in range(len(edges)):
      x_centers.append(edges[d][1:] - .5*(edges[d][1] - edges[d][0]))
    return x_centers
 
def nd_grid_centers( edges ): 
    """ Given D lists of grid edges (as returned by np.histogramdd), 
    returns a list of D dimensional points corresponding to each tile center """
    return nd_grid_points( x_grid_centers(edges) )

def nd_grid_points( dim_points ): 
    """ Given D grid centers along each axis, 
    enumerate all of the points in a long array """

    #hacky way to check this - any way to automate? 
    if len(dim_points)==1:
      return dim_points
    elif len(dim_points)==2:
      return np.dstack( np.meshgrid( dim_points[0], dim_points[1] ) ).reshape(-1,2)
    elif len(dim_points)==3:
      grids = np.meshgrid(dim_points[0], dim_points[1], dim_points[2])
      return np.squeeze( np.dstack( grids[0].flatten(), \
                                    grids[1].flatten(), \
                                    grids[2].flatten() ) )
    else: 
      raise NotImplementedError

def ndmesh(grids):
   # args = map(np.asarray,args)
   return np.broadcast_arrays(*[x[(slice(None),)+(None,)*i] for i, x in enumerate(grids)]) 

def spherical_proposal(theta, scale=.05): 
  """ jitters theta with spherical gaussian noise """
  thp = theta + scale*np.random.randn(len(theta))
  if np.any(thp < 0): 
    return None
  else: 
    return thp

def whitened_mh(th, f, whiten, unwhiten, Lfn, ln_prior, \
                prop_dist=spherical_proposal): 
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
                user specified, so this can be optimized version can be passed in
    - unwhiten: function takes in (th_p, nu) and computes unwhitened version of nu
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
  
  # solve for nu ~ Normal(0, I)      (whiten)
  nu = whiten(th, f)  #whitening function incorporates Covariance 
                      #function (but optimized to handle kronecker stuff)

  # propose th' ~ q(th' ; th)
  thp = prop_dist(th)
  if thp is None: 
    return th, f, False, Lfn(f) + ln_prior(th)
  
  # compute implied values f' = L_thp*nu (unwhiten)
  fp = unwhiten(thp, nu)
  
  # mh accept/reject
  ll_p = Lfn(fp) + ln_prior(thp)
  ll_o = Lfn(f) + ln_prior(th)
  if -np.random.exponential() < ll_p - ll_o:
    return thp, fp, True, ll_p
  else:
    return th, f, False, ll_o


############################################################
# TODO: incorporate slice sampling for covar hypers later
############################################################
def slice_sample(init_x, logprob, sigma=1.0, step_out=True, max_steps_out=1000, 
                 compwise=False, verbose=False, returnFunEvals=False):
    
    # Keep track of the number of evaluations of the logprob function
    funEvals = {'funevals': 0} # sorry, i don't know how to actually do this properly with all these nested function. pls forgive me -MG
    
    # this is a 1-d sampling subrountine that only samples along the direction "direction"
    def direction_slice(direction, init_x):
        
        def dir_logprob(z): # logprob of the proposed point (x + dir*z) where z must be the step size
            funEvals['funevals'] += 1
            try:
                return logprob(direction*z + init_x)
            except:
                print 'ERROR: Logprob failed at input %s' % str(direction*z + init_x)
                raise
                
    
        # upper and lower are step sizes -- everything is measured relative to init_x
        upper = sigma*npr.rand()  # random thing above 0
        lower = upper - sigma     # random thing below 0
        llh_s = np.log(npr.rand()) + dir_logprob(0.0)  # = log(prob_current * rand) 
        # (above) uniformly sample vertically at init_x
    
    
        l_steps_out = 0
        u_steps_out = 0
        if step_out:
            # increase upper and decrease lower until they overshoot the curve
            while dir_logprob(lower) > llh_s and l_steps_out < max_steps_out:
                l_steps_out += 1
                lower       -= sigma  # make lower smaller by sigma
            while dir_logprob(upper) > llh_s and u_steps_out < max_steps_out:
                u_steps_out += 1
                upper       += sigma
        
        
        # rejection sample along the horizontal line (because we don't know the bounds exactly)
        steps_in = 0
        while True:
            steps_in += 1
            new_z     = (upper - lower)*npr.rand() + lower  # uniformly sample between upper and lower
            new_llh   = dir_logprob(new_z)  # new current logprob
            if np.isnan(new_llh):
                print new_z, direction*new_z + init_x, new_llh, llh_s, init_x, logprob(init_x)
                raise Exception("Slice sampler got a NaN logprob")
            if new_llh > llh_s:  # this is the termination condition
                break       # it says, if you got to a better place than you started, you're done
                
            # the below is only if you've overshot, meaning your uniform sample from the horizontal
            # slice ended up outside the curve because the bounds lower and upper were not tight
            elif new_z < 0:  # new_z is to the left of init_x
                lower = new_z  # shrink lower to it
            elif new_z > 0:
                upper = new_z
            else:  # upper and lower are both 0...
                raise Exception("Slice sampler shrank to zero!")

        if verbose:
            print "Steps Out:", l_steps_out, u_steps_out, " Steps In:", steps_in, "Final logprob:", new_llh

        # return new the point
        return new_z*direction + init_x  

    
    # begin main
    
    # # This causes an extra "logprob" function call -- might want to turn off for speed
    initial_llh = logprob(init_x)
    if verbose:
        sys.stderr.write('Logprob before sampling: %f\n' % initial_llh)
    if np.isneginf(initial_llh):
        sys.stderr.write('Values passed into slice sampler: %s\n' % init_x)
        raise Exception("Initial value passed into slice sampler has logprob = -inf")
    
    if not init_x.shape:  # if there is just one dimension, stick it in a numpy array
        init_x = np.array([init_x])

    dims = init_x.shape[0]
    if compwise:   # if component-wise (independent) sampling
        ordering = range(dims)
        npr.shuffle(ordering)
        cur_x = init_x.copy()
        for d in ordering:
            direction    = np.zeros((dims))
            direction[d] = 1.0
            cur_x = direction_slice(direction, cur_x)
            
    else:   # if not component-wise sampling
        direction = npr.randn(dims)
        direction = direction / np.sqrt(np.sum(direction**2))  # pick a unit vector in a random direction
        cur_x = direction_slice(direction, init_x)  # attempt to sample in that direction
    
    return (cur_x, funEvals['funevals']) if returnFunEvals else cur_x
    

# Check the gradients of function "fun" at location(s) "test_x"
#   fun: a function that takes in test_x and returns a tuple of the form (function value, gradient)
#       # the function value should have shape (n_points) and  the gradient should have shape (n_points, D)
#   test_x: the points. should have shape (n_points by D)
#       special case: if test_x is a single *flat* array, then gradient should also be a flat array
#   delta: finite difference step length
#   error_tol: tolerance for error of numerical to symbolic gradient
# Returns a boolean of whether or not the gradients seem OK
def check_grad(fun, test_x, error_tol=1e-3, delta=1e-5, verbose=False):
    if verbose:
        sys.stderr.write('Checking gradients...\n')
        
    state_before_checking = npr.get_state()
    fixed_seed = 5      # arbitrary
    
    npr.seed(fixed_seed)
    analytical_grad = fun(test_x)[1]
    D = test_x.shape[1] if test_x.ndim > 1 else test_x.size
    grad_check = np.zeros(analytical_grad.shape) if analytical_grad.size > 1 else np.zeros(1)
    for i in range(D):
        unit_vector = np.zeros(D)
        unit_vector[i] = delta
        npr.seed(fixed_seed)
        forward_val = fun(test_x + unit_vector)[0]
        npr.seed(fixed_seed)
        backward_val = fun(test_x - unit_vector)[0]
        grad_check_i = (forward_val - backward_val)/(2*delta)
        if test_x.ndim > 1:
            grad_check[:,i] = grad_check_i
        else:
            grad_check[i] = grad_check_i
    grad_diff = grad_check - analytical_grad
    err = np.sqrt(np.sum(grad_diff**2))

    if verbose:        
        sys.stderr.write('Analytical grad: %s\n' % str(analytical_grad))
        sys.stderr.write('Estimated grad:  %s\n' % str(grad_check))
        sys.stderr.write('L2-norm of gradient error = %g\n' % err)

    npr.set_state(state_before_checking)

    return err < error_tol



# For converting a string of args into a dict of args
# (one could then call parse_args on the output)
def unpack_args(str):
    if len(str) > 1:
        eq_re = re.compile("\s*=\s*")
        return dict(map(lambda x: eq_re.split(x),
                        re.compile("\s*,\s*").split(str)))
    else:
        return {}
            
# For parsing the input arguments to a Chooser. 
# "argTypes" is a dict with keys of argument names and
# values of tuples with the (argType, argDefaultValue)
# args is the dict of arguments passd in by the used
def parse_args(argTypes, args):
    opt = dict() # "options"
    for arg in argTypes:
        if arg in args:
            try:
                opt[arg] = argTypes[arg][0](args[arg])
            except:
                # If the argument cannot be parsed into the data type specified by argTypes (e.g., float)
                sys.stderr.write("Cannot parse user-specified value %s of argument %s" % (args[arg], arg))
        else:
            opt[arg] = argTypes[arg][1]
 
    return opt
