import pymc
import numpy        as np
import numpy.random as npr

def elliptical_slice(xx, chol_Sigma, log_like_fn):
    D  = xx.size

    # Select a random ellipse.
    nu = np.dot(chol_Sigma, npr.randn(D))

    # Select the slice threshold.
    hh = np.log(npr.rand()) + log_like_fn(xx)

    # Randomly center the bracket.
    phi     = npr.rand()*2*np.pi
    phi_max = phi
    phi_min = phi_max - 2*np.pi

    # Loop until acceptance.
    while True:

        # Compute the proposal.
        xx_prop = xx*np.cos(phi) + nu*np.sin(phi)

        # If on the slice, return the proposal.
        if log_like_fn(xx_prop) > hh:
            return xx_prop

        if phi > 0:
            phi_max = phi
        elif phi < 0:
            phi_min = phi
        else:
            raise Exception("Shrank to zero!")

        phi = npr.rand()*(phi_max - phi_min) + phi_min

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
                    
if __name__ == '__main__':
    npr.seed(1)

    import pylab as pl

    D  = 10
    fn = lambda x: -0.5*np.sum(x**2)

    iters = 1000
    samps = np.zeros((iters,D))
    for ii in xrange(1,iters):
        samps[ii,:] = slice_sample(samps[ii-1,:], fn, sigma=0.1, step_out=False, doubling_step=True, verbose=False)

    ll = -0.5*np.sum(samps**2, axis=1)

    scores = pymc.geweke(ll)
    pymc.Matplot.geweke_plot(scores, 'test')

    pymc.raftery_lewis(ll, q=0.025, r=0.01)

    pymc.Matplot.autocorrelation(ll, 'test')

