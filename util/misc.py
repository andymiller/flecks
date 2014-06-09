import re
import numpy        as np
import numpy.random as npr
import math
import sys
from scipy.misc import logsumexp
import scipy.weave

# This should eventually go somewhere else, but I don't know where yet
TERMINATION_SIGNAL = "terminate"
def normalize_rows(A):
  """ makes rows of a matrix sum to one """
  row_sums = A.sum(axis=1)
  return A / row_sums[:,np.newaxis]

def mvnorm_lnpdf(x, mean=0, cov=None): 
  """ simple log multivariate normal pdf """
  #number of samples and dimensionality of samples
  if len(x.shape)==1:
    N = 1
    D = len(x)
  else: 
    N,D = x.shape
  invCov = np.linalg.inv(cov)
  (sign, ldCov) = np.linalg.slogdet(cov)
  lls = -.5*D*np.log(2*np.pi) -.5*ldCov -.5 * (x-mean).T.dot(invCov).dot(x-mean)
  return lls


############################################################
# TODO: incorporate slice sampling for covar hypers later
############################################################
#def slice_sample(init_x, logprob, sigma=1.0, step_out=True, max_steps_out=1000, 
#                 compwise=False, verbose=False, returnFunEvals=False):
#    
#    # Keep track of the number of evaluations of the logprob function
#    funEvals = {'funevals': 0} # sorry, i don't know how to actually do this properly with all these nested function. pls forgive me -MG
#    
#    # this is a 1-d sampling subrountine that only samples along the direction "direction"
#    def direction_slice(direction, init_x):
#        
#        def dir_logprob(z): # logprob of the proposed point (x + dir*z) where z must be the step size
#            funEvals['funevals'] += 1
#            try:
#                return logprob(direction*z + init_x)
#            except:
#                print 'ERROR: Logprob failed at input %s' % str(direction*z + init_x)
#                raise
#                
#    
#        # upper and lower are step sizes -- everything is measured relative to init_x
#        upper = sigma*npr.rand()  # random thing above 0
#        lower = upper - sigma     # random thing below 0
#        llh_s = np.log(npr.rand()) + dir_logprob(0.0)  # = log(prob_current * rand) 
#        # (above) uniformly sample vertically at init_x
#    
#    
#        l_steps_out = 0
#        u_steps_out = 0
#        if step_out:
#            # increase upper and decrease lower until they overshoot the curve
#            while dir_logprob(lower) > llh_s and l_steps_out < max_steps_out:
#                l_steps_out += 1
#                lower       -= sigma  # make lower smaller by sigma
#            while dir_logprob(upper) > llh_s and u_steps_out < max_steps_out:
#                u_steps_out += 1
#                upper       += sigma
#        
#        
#        # rejection sample along the horizontal line (because we don't know the bounds exactly)
#        steps_in = 0
#        while True:
#            steps_in += 1
#            new_z     = (upper - lower)*npr.rand() + lower  # uniformly sample between upper and lower
#            new_llh   = dir_logprob(new_z)  # new current logprob
#            if np.isnan(new_llh):
#                print new_z, direction*new_z + init_x, new_llh, llh_s, init_x, logprob(init_x)
#                raise Exception("Slice sampler got a NaN logprob")
#            if new_llh > llh_s:  # this is the termination condition
#                break       # it says, if you got to a better place than you started, you're done
#                
#            # the below is only if you've overshot, meaning your uniform sample from the horizontal
#            # slice ended up outside the curve because the bounds lower and upper were not tight
#            elif new_z < 0:  # new_z is to the left of init_x
#                lower = new_z  # shrink lower to it
#            elif new_z > 0:
#                upper = new_z
#            else:  # upper and lower are both 0...
#                raise Exception("Slice sampler shrank to zero!")
#
#        if verbose:
#            print "Steps Out:", l_steps_out, u_steps_out, " Steps In:", steps_in, "Final logprob:", new_llh
#
#        # return new the point
#        return new_z*direction + init_x  
#
#    
#    # begin main
#    
#    # # This causes an extra "logprob" function call -- might want to turn off for speed
#    initial_llh = logprob(init_x)
#    if verbose:
#        sys.stderr.write('Logprob before sampling: %f\n' % initial_llh)
#    if np.isneginf(initial_llh):
#        sys.stderr.write('Values passed into slice sampler: %s\n' % init_x)
#        raise Exception("Initial value passed into slice sampler has logprob = -inf")
#    
#    if not init_x.shape:  # if there is just one dimension, stick it in a numpy array
#        init_x = np.array([init_x])
#
#    dims = init_x.shape[0]
#    if compwise:   # if component-wise (independent) sampling
#        ordering = range(dims)
#        npr.shuffle(ordering)
#        cur_x = init_x.copy()
#        for d in ordering:
#            direction    = np.zeros((dims))
#            direction[d] = 1.0
#            cur_x = direction_slice(direction, cur_x)
#            
#    else:   # if not component-wise sampling
#        direction = npr.randn(dims)
#        direction = direction / np.sqrt(np.sum(direction**2))  # pick a unit vector in a random direction
#        cur_x = direction_slice(direction, init_x)  # attempt to sample in that direction
#    
#    return (cur_x, funEvals['funevals']) if returnFunEvals else cur_x
    

# Check the gradients of function "fun" at location(s) "test_x"
#   fun: a function that takes in test_x and returns a tuple of the form (function value, gradient)
#       # the function value should have shape (n_points) and  the gradient should have shape (n_points, D)
#   test_x: the points. should have shape (n_points by D)
#       special case: if test_x is a single *flat* array, then gradient should also be a flat array
#   delta: finite difference step length
#   error_tol: tolerance for error of numerical to symbolic gradient
# Returns a boolean of whether or not the gradients seem OK
#def check_grad(fun, test_x, error_tol=1e-3, delta=1e-5, verbose=False):
#    if verbose:
#        sys.stderr.write('Checking gradients...\n')
#        
#    state_before_checking = npr.get_state()
#    fixed_seed = 5      # arbitrary
#    
#    npr.seed(fixed_seed)
#    analytical_grad = fun(test_x)[1]
#    D = test_x.shape[1] if test_x.ndim > 1 else test_x.size
#    grad_check = np.zeros(analytical_grad.shape) if analytical_grad.size > 1 else np.zeros(1)
#    for i in range(D):
#        unit_vector = np.zeros(D)
#        unit_vector[i] = delta
#        npr.seed(fixed_seed)
#        forward_val = fun(test_x + unit_vector)[0]
#        npr.seed(fixed_seed)
#        backward_val = fun(test_x - unit_vector)[0]
#        grad_check_i = (forward_val - backward_val)/(2*delta)
#        if test_x.ndim > 1:
#            grad_check[:,i] = grad_check_i
#        else:
#            grad_check[i] = grad_check_i
#    grad_diff = grad_check - analytical_grad
#    err = np.sqrt(np.sum(grad_diff**2))
#
#    if verbose:        
#        sys.stderr.write('Analytical grad: %s\n' % str(analytical_grad))
#        sys.stderr.write('Estimated grad:  %s\n' % str(grad_check))
#        sys.stderr.write('L2-norm of gradient error = %g\n' % err)
#
#    npr.set_state(state_before_checking)
#
#    return err < error_tol


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

def logmulexp(a, b): 
  """ returns log( exp(a).dot(exp(b)) ) in a numerically stable way """
  K,M  = a.shape
  M2,N = b.shape
  s = np.tile(a, (N,1)) + np.kron(b.T, np.ones((K,1))) #np.ones( (K,1) ))
  s = np.reshape( logsumexp(s,axis=1), (K,N), order='F' )
  return s

def gen_outer(a, b): 
  """ returns genearlized outer product """
  K,M  = a.shape
  M2,N = b.shape
  s = np.tile(a, (N,1)) + np.kron(b.T, np.ones((K,1))) #np.ones( (K,1) ))
  return s

if __name__=="__main__": 

  #test logmulexp
  A = 10*np.random.rand(5*10).reshape(5,10)
  B = 10*np.random.rand(5*20).reshape(5,20)
  a = np.log(A)
  b = np.log(B)
  logAB = np.log( np.exp(a).T.dot(np.exp(b)) )
  logsmAB = logmulexp(a.T,b)
  if np.allclose(logAB, logsmAB): 
    print "logmulexp success!"
  else: 
    print "logmulexp failure :("
    print logAB - logsmAB

  import time


  ##
  ## test multinomial correctness
  ##
  M = 20
  K = 30
  Xs = 1000 + np.random.randint(100, size=(M,))
  Ps = np.random.rand(M,K)
  Ps /= Ps.sum(axis=1)[:,np.newaxis]
  samps = fast_multinomials(Ps, Xs)
  print samps
  print "convergence...", np.sum( np.abs((samps / samps.sum(axis=1)[:,np.newaxis]) - Ps) )
  if np.allclose( samps.sum(axis=1), Xs ): 
    print "  ... all samples accounted for"
  else: 
    print "  ... counts don't match! "
    print samps.sum(axis=1) - Xs

  ##
  ## test old vs new multinomial sampling 
  ##
  V = 100
  T = 100
  K = 100
  Xs = 1000 + np.random.randint(100, size=(V,T))
  logB = np.log(np.random.rand(K,V))
  logW = np.log(np.random.rand(K,T))
  t = time.time()
  logLamList = gen_outer(logB.T, logW)
  Lams = np.exp(logLamList)
  Lams  /= Lams.sum(axis=1)[:,np.newaxis]
  samps = fast_multinomials( Lams, Xs.ravel() )
  samps = samps.reshape( (T,V,K) )
  elapsed = time.time() - t
  print "  weave function time: ", elapsed
  #print Lams.shape, "lams shape"
  #print samps.sum(axis=1)

  #test old 
  t = time.time()
  for n in range(T): 
    for v in range(V): 
      N_nv = Xs[v,n]
      if N_nv > 0: 
        Lam = np.exp(logW[:,n] + logB[:,v])
        Lam = Lam / Lam.sum()
        samps[n,v,:] = np.random.multinomial(N_nv, Lam, size=1)[0]
      else:
        samps[n,v,:] = np.zeros(K)

  elapsed = time.time() - t
  print "  non eave function time: ", elapsed
 

