#
# Implementation of Pseudo marginal MCMC for Poisson Regression
#
import numpy as np
from scipy.misc import logsumexp
from scipy.stats import poisson
from scipy.optimize import minimize
from numpy.linalg import inv, cholesky, solve

def poisson_log_like(y, f, cK_inv, f0=0.0, 
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


def laplace_approx(y, K_theta, log_like=poisson_log_like): 
    cK_theta = cholesky(K_theta)
    cK_inv   = inv(cK_theta)
    Nparam = K_theta.shape[0]
    x0 = cK_theta.dot(np.random.randn(Nparam))
    def nlog_like_hessp(f, p): 
        return -log_like(y, f, cK_inv, hessp=True, p=p)
    res0 = minimize( x0 = x0,
                    fun = lambda(f): -log_like(y, f, cK_inv), 
                    jac = lambda(f): -log_like(y, f, cK_inv, grad=True), 
                    #hess = lambda(f): -log_like(y, f, cK_inv, hess=True, bias=bias), #nlog_like_hessp, 
                    hessp = nlog_like_hessp,
                    method = 'Newton-CG', 
                    options = {'maxiter':1000} )
    f_mu  = res0.x
    f_Sig = inv( cK_inv.T.dot(cK_inv) + np.diag(np.exp(f_mu)) )
    cf_Sig = cholesky(f_Sig)
    return f_mu, f_Sig, cK_theta, cK_inv


def approx_log_marg_like(y, K_theta, like_func = poisson_log_like): 
    """ returns approximate log marginal likelihood 
    (based on importance sampling - unbiased)
    """
    # first find params of a laplace approx q(f | f_mu, f_Sig), 
    f_mu, f_Sig, cK_theta, cK_inv = laplace_approx(y, K_theta, like_func)

    # cache covariance manipulations
    cf_Sig      = cholesky(f_Sig)
    cf_Sig_inv  = inv(cf_Sig)
    (sign, ldetK )  = np.linalg.slogdet(K_theta) 
    (sign, ldetSig) = np.linalg.slogdet(f_Sig)
    
    #sample a bunch to find weights for unbiased est of marg like
    N_imp = 500
    log_rats = np.zeros(N_imp)
    for n in range(N_imp): 
        fi = cf_Sig.dot(np.random.randn(len(f_mu))) + f_mu 
        #ll = np.sum(poisson.logpmf(y, mu=np.exp(fi)))
        ll = np.sum( -np.exp(fi) + y*fi )
        p_fi = -.5 * ldetK   - .5 * fi.dot(cK_inv.T).dot( cK_inv.dot(fi) ) #- .5*N*np.log(2*np.pi)
        q_fi = -.5 * ldetSig - .5 * (fi-f_mu).dot(cf_Sig_inv.T).dot( cf_Sig_inv.dot(fi-f_mu) ) #- .5*N*np.log(2*np.pi) 
        log_rats[n] = ll + p_fi - q_fi 
    log_py = logsumexp(log_rats) - np.log(N_imp)
    return log_py


if __name__=="__main__":
    import pylab as plt
    from lgcp.kernel import PerKernelUnscaled, SQEKernelUnscaled, LinearKernel

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
        num_grad[n] = (poisson_log_like(y, x0+de, cK_inv) - 
                       poisson_log_like(y, x0-de, cK_inv)) / (2*de[n])
    an_grad = poisson_log_like(y, x0, cK_inv, grad=True)
    if not np.allclose(an_grad, num_grad, atol=1e-3): 
        print "Gradients don't match"
        print an_grad - num_grad
    else: 
        print "Gradients match! Good job, Andy!"

    # fit laplace approximation, Mu, Sig (with bias term)
    K_theta = pkern.K(tgrid, tgrid, hypers=[per, ell]) + \
              25*np.ones((N,N)) + \
              sig_noise*np.eye(N)
    f_mu, f_Sig, _, _ = laplace_approx(y, K_theta)

    #plot laplace mean
    plt.plot(tgrid, np.exp(f_mu))
    plt.plot(tgrid, np.exp(f_gt+f0_gt))
    plt.legend(["$\hat \mathbf{f}$", "$\mathbf{f}_{gt}$"])
    cf_Sig = cholesky(f_Sig)
    for n in range(3): 
        fi = cf_Sig.dot(np.random.randn(N)) + f_mu
        plt.plot(tgrid, np.exp(fi), c='grey')
    plt.plot(tgrid, y, 'r.')
    plt.show()

