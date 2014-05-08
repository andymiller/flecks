import numpy as np
import scipy as sp
from scipy.integrate import quad

def gen_synthetic_data(xdim = 1, K=2, 
                       xgrid_dims=[100], xbbox=[[-10,10]], 
                       tgrid_dims=[100], tbbox=[0,20]): 
    """ Given X (space) grid size and bounding box, and T grid size and 
    bounding box, returns a sample point process and ground truth 
    functions to check inference - returns: 
      - data: unordered point process with time last
      - B_gt: ground truth bases (K functions on xdim space)
      - W_gt: ground truth temporal (K 1d functions over time)
      - xgrid: grid used for B_gt (for plotting)
      - tgrid: grid used for W_gt (for plotting)
      - X, T, Z: the meshgrid X, T and lambda function (if in 1-d)
    """
    if xdim==1:
        # Spatial bump and sinusoidal Wfunc 
        def bump(x, mu):
            return 1.0/np.sqrt(2*np.pi) * np.exp( -.5*(x - mu)*(x-mu) ) 
        def wfunc(t, omega): 
            return np.sin(t*omega) + 75
        
        #create spatial basis location of the bumps
        mus = np.array([ -5, 2])
        xgrid = np.linspace(xbbox[0][0], xbbox[0][1], xgrid_dims[0])
        B_gt  = np.array([bump(xgrid, mu) for mu in mus])

        #create temporal weights
        omegas = np.array([.6, 1.5])
        tgrid = np.linspace(tbbox[0], tbbox[1], tgrid_dims[0])
        W_gt = np.array([wfunc(tgrid, omega) for omega in omegas])
        
        ## sample point process
        # define lambda function to be integrated
        def lam(x, t, mus, omegas):         
            if isinstance(x, (np.ndarray, np.generic)):
                lam_xt = np.zeros(x.shape)
            else:
                lam_xt = 0
            for k in range(len(mus)):
                w_kt = wfunc(t, omegas[k])
                b_kx = bump(x, mus[k])
                lam_xt += w_kt*b_kx
            return lam_xt
        def lam2(t, x0, x1): 
            return quad(lam, x0, x1, args=(t,mus,omegas,))[0]
        vol = quad(lam2, -10, 10, args=(0, 20))[0] #total volume in bbox 
        print "total volume: ", vol
        N = np.random.poisson(lam=vol)
        X,T = np.meshgrid(xgrid, tgrid)
        Z   = lam(X, T, mus, omegas)
        print "num points sampled: ", N
        normZ = Z / Z.sum()
        samp_idx = np.random.choice(Z.size, size=N, p=normZ.flatten())
        x = (X.flat)[samp_idx] + .05*np.random.randn(len(samp_idx))
        t = (T.flat)[samp_idx] + .05*np.random.randn(len(samp_idx))   
        return np.column_stack((x,t)), B_gt, W_gt, xgrid, tgrid, X, T, Z
    else: 
        raise NotImplementedError, 'Synthetic x dim greater than 1 not implemented'



