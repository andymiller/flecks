import numpy as np
from numpy.linalg import cholesky, inv

def poisson_log_posterior(y, f, cK_inv, f0=0.0, 
                          grad=False, hess=False, 
                          hessp=False, p=None): 
    """ Poisson regression log likelihood with correlated gaussian prior

    Computes the log likelihood (technically the full log unnormalized 
    posterior) of a Poisson regression with a Gaussian prior over the log
    intensity values.  The likelihood is

      log p(y | f) p(f | K_theta) = log dpoiss(y | lam=exp(f)) + log N(f|0,K)

    Args: 
      y (len N array of int): a list of poisson counts
      f (len N array of float): log intensity function for each count
      cK_inv (NxN lower tri mat): Inverse of the cholesky decomposition of
        the covariance matrix governing f.  Note that there is no Bias term - 
        this can be integrated out by adding the bias variance to every entry
        of K.
      grad (bool): set to True for the gradient 
      hess (bool): set to True for the hessian
      hessp (bool): set to True for the hessian times an arbitrary vector p
        (must set p to a value!)
      p (len N array): arbitrary vector multiplied by the Hessian when 
        hessp=True

    Returns: 
      log_like (float) by default
      grad vector (array float) if grad=True
      Hessian matrix (array float) if hess=True
      Hessian product (array float) if hessp=True and p is not None

    """
    #if bias term is included, shift fs
    f = f + f0  #
    assert len(f)==cK_inv.shape[0], "log_like f doesn't match covariance mat"
    if grad: 
        K_inv_f = cK_inv.T.dot(cK_inv.dot(f)) 
        return y - np.exp(f) - K_inv_f
    elif hess: 
        K_inv = cK_inv.T.dot(cK_inv) 
        return -np.diag(np.exp(f)) - K_inv
    elif hessp:
        K_inv_p = cK_inv.T.dot(cK_inv.dot(p)) 
        return -np.exp(f)*p - K_inv_p
    else: 
        K_inv_f = cK_inv.T.dot(cK_inv.dot(f))
        return np.sum(y*f - np.exp(f)) - .5*f.dot( K_inv_f )


if __name__=="__main__":
    import pylab as plt
    from flecks.kernel import PerKernelUnscaled, SQEKernelUnscaled, LinearKernel

    #generate poisson counts with a periodic kernel
    sig_noise = 1e-6
    np.random.seed(45)
    pkern = PerKernelUnscaled()
    skern = SQEKernelUnscaled()
    lkern = LinearKernel()
    N = 200
    T = 50
    ell = 1
    per = 10
    tgrid = np.linspace(0, T, N)
    dt    = tgrid[1]-tgrid[0]
    K     = pkern.K(tgrid, tgrid, hypers=[per, ell]) + sig_noise*np.eye(N) #+ .01*skern.K(tgrid,tgrid)
    f_gt  = cholesky(K).dot(np.random.randn(N))
    f0_gt = 3
    lam_gt = np.exp(f_gt + f0_gt)
    y      = np.random.poisson(lam = dt*lam_gt) 

    ## double check gradients for poisson log like
    K_theta  = pkern.K(tgrid, tgrid, hypers=[per, ell]) + 25*np.ones((N,N)) + sig_noise*np.eye(N)
    cK_theta = cholesky(K_theta)
    cK_inv   = inv(cK_theta)
    x0 = cholesky(K_theta).dot(np.random.randn(N))
    num_grad = np.zeros(N)
    for n in range(N): 
        de = np.zeros(N)
        de[n] = 1e-6
        num_grad[n] = (poisson_log_posterior(y, x0+de, cK_inv) - 
                       poisson_log_posterior(y, x0-de, cK_inv)) / (2*de[n])
    an_grad = poisson_log_posterior(y, x0, cK_inv, grad=True)
    if not np.allclose(an_grad, num_grad, atol=1e-3): 
        print "Gradients don't match"
        print an_grad - num_grad
    else: 
        print "Gradients match! Good job, Andy!"

