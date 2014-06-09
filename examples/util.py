import numpy as np
import scipy as sp
import numpy.random as npr
from scipy.integrate import quad
from scipy.stats import multivariate_normal
import pylab as plt

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
            n1 = 1.0/np.sqrt(2*np.pi) * np.exp( -.5*(x - mu)*(x-mu) ) 
            n2 = 1.0/np.sqrt(2*np.pi*.5) * np.exp( -(x-mu+3)*(x-mu+3))
            return .6*n1 + .4*n2
        def wfunc(t, per): 
            return 30*np.sin(2*np.pi*t/per) + 50
        
        #create spatial basis location of the bumps
        mus = np.array([ -5, 2])
        xgrid = np.linspace(xbbox[0][0], xbbox[0][1], xgrid_dims[0])
        B_gt  = np.array([bump(xgrid, mu) for mu in mus])

        #create temporal weights
        omegas = np.array([10, 5])
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

        # compute volume of product function
        dx = float(xbbox[0][1] - xbbox[0][0])/xgrid_dims[0]
        dt = float(tbbox[1] - tbbox[0])/tgrid_dims[0]
        vol = np.sum(B_gt.T.dot(W_gt)*dx*dt)

        #vol = quad(lam2, xbbox[0][0], xbbox[0][1], args=(tbbox[0],tbbox[1]))[0] #total volume in bbox 
        N = np.random.poisson(lam=vol)
        print "total volume: ", vol
        print "num points sampled: ", N
        X,T = np.meshgrid(xgrid, tgrid, indexing='ij')
        Z   = lam(X, T, mus, omegas)
 
        #generate data from discretized intensity (directly)
        Lambda  = B_gt.T.dot(W_gt) * dx * dt #intensity is outer prod of B and W
        print "... gen data volume 2: ", np.sum(Lambda)
        normLam = Lambda / Lambda.sum()
        samp_idx = np.random.choice(normLam.size, size=N, p=normLam.ravel(order="C"))
        x = (X.ravel(order="C"))[samp_idx] + .25*dx*np.random.randn(len(samp_idx))
        t = (T.ravel(order="C"))[samp_idx] + .25*dt*np.random.randn(len(samp_idx))   
        return np.column_stack((x,t)), B_gt, W_gt, xgrid, tgrid, X, T, Z
    elif xdim==2: 

        # compute volume of product function
        V = np.prod(xgrid_dims)
        print "number of spatial tiles: ", V
        dx = np.prod([float(xbbox[i][1] - xbbox[i][0])/xgrid_dims[i] for i in range(len(xgrid_dims))])
        dt = float(tbbox[1] - tbbox[0])/tgrid_dims[0]
        print "  dx = ", dx
        print "  dt = ", dt

        # create spatial bump funcs
        xgs = [np.linspace(xbbox[i][0], xbbox[i][1], xgrid_dims[i]) for i in range(len(xgrid_dims))]
        xv, yv = np.meshgrid(xgs[0], xgs[1], indexing='ij')
        xgrid = np.column_stack( (xv.ravel(), yv.ravel()) )
        B_gt = np.zeros((K, len(xgrid)))
        for k in range(K): 
          num_means = npr.randint(3) + 2
          means = 2.5*npr.randn(num_means,2) + 3*npr.randn(2)
          Sigs  = np.zeros((num_means, 2, 2))
          ws    = npr.rand(num_means)
          ws   /= ws.sum()
          for ii in range(num_means): 
            sig = 2
            A = sig*sig*np.eye(2)
            rho = npr.rand()*sig*sig
            rho = -rho if npr.rand() > .5 else rho
            A[0,1] = A[1,0] = rho
            Sigs[ii,:,:] = A
            Zii = multivariate_normal.pdf(xgrid, mean=means[ii], cov=Sigs[ii])
            B_gt[k] += ws[ii]*Zii

        # normalize - truncated gaussians
        row_sums = B_gt.sum(axis=1)*dx
        B_gt /= row_sums[:,np.newaxis]
        #plt.imshow(B_gt[1].reshape(xgrid_dims))
        #plt.show()

        #create temporal weights
        def wfunc(t, per): 
           return 30*np.sin(2*np.pi*t/per) + 150*npr.rand()
        omegas = np.array([23.5, 10, 15, 34.5, 20])[0:K]
        tgrid = np.linspace(tbbox[0], tbbox[1], tgrid_dims[0])
        W_gt = np.array([wfunc(tgrid, omega) for omega in omegas])
    
        # sample points
        vol = np.sum(B_gt.T.dot(W_gt)*dx*dt)
        N   = np.random.poisson(lam=vol)
        print "total volume: ", vol
        print "num points sampled: ", N
        X, T = np.meshgrid(np.arange(V), np.arange(tgrid_dims[0]), indexing='ij')
 
        #generate data from discretized intensity (directly)
        Lambda  = B_gt.T.dot(W_gt) * dx * dt #intensity is outer prod of B and W
        print "... gen data volume 2: ", np.sum(Lambda)
        normLam = Lambda / Lambda.sum()
        samp_idx = npr.choice(normLam.size, size=N, p=normLam.ravel(order="C"))
        vs = (X.ravel(order="C"))[samp_idx]
        print vs.shape
        print min(vs)
        print max(vs)
        print xgrid.shape
        ts = (T.ravel(order="C"))[samp_idx]
        x  = xgrid[vs,:] + dx*npr.randn(len(vs),2)
        t  = tgrid[ts] + dt*npr.randn(len(ts))
        return np.column_stack((x,t)), B_gt, W_gt, xgrid, tgrid

    else:
        raise NotImplementedError, 'Synthetic x dim greater than 1 not implemented'



