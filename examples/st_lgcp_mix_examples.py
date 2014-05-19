import pylab as plt
import numpy as np
from util import gen_synthetic_data
from lgcp.st_mix_lgcp import SpatioTemporalMixLGCP

def st_lgcp_mix_examples_1d():
    """ Very simple 1d space example """

    # example parameters and sythetic data
    xdim = 1
    K    = 2
    xgrid_dims = [40]
    xbbox      = [[-10,10]]
    tgrid_dims = [60]
    tbbox      = [0,20]
    data, B_gt, W_gt, xgrid, tgrid, X, T, Z = \
        gen_synthetic_data( xdim = xdim, K=K,
                            xgrid_dims = xgrid_dims, 
                            xbbox = xbbox, 
                            tgrid_dims = tgrid_dims,
                            tbbox = tbbox )

    #visualize 
    fig, axarr = plt.subplots(3,2)
    axarr[0,0].plot(xgrid, B_gt.T) 
    axarr[0,0].set_title("spatial bumps")
    axarr[0,0].set_xlabel('X space')
    axarr[0,0].set_xlim((-10,10))

    axarr[1,0].plot(tgrid, W_gt.T) 
    axarr[1,0].set_title("temporal weights")
    axarr[1,0].set_ylabel('T space')
    axarr[1,0].set_xlim((0,20))

    # `ax` is a 3D-aware axis instance because of the projection='3d' keyword argument to add_subplot
    #fig = plt.figure(figsize=(14,6))
    #ax = fig.add_subplot(1, 1, 1, projection='3d')
    # surface_plot with color grading and color bar
    #p = ax.plot_surface(X, T, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    #plt.set_title("Spatiotemporal Intensity")
    #cb = fig.colorbar(p, shrink=0.5)
    #plt.show()

    axarr[2,0].scatter(data[:,0], data[:,1], marker='.')
    axarr[2,0].set_title("observed point process")
    axarr[2,0].set_xlabel('X space')
    axarr[2,0].set_ylabel("TIME")
    axarr[2,0].set_xlim((-10,10))
    axarr[2,0].set_ylim((0,20))
    fig.tight_layout()

    #
    # fit model
    #
    K = 2
    dim = 1
    xgrid_dims = [40]  # downsample space and time
    tgrid_dims = [60]  
    model = SpatioTemporalMixLGCP( xdim       = xdim, 
                                   K          = K,
                                   xgrid_dims = xgrid_dims, 
                                   xbbox      = xbbox, 
                                   tgrid_dims = tgrid_dims,
                                   tbbox      = tbbox )

    model.describe()
    w_samps, b_samps, th_samps, lls = model.fit(data, Nsamps=1000)
    max_idx = lls.argmax()
    model.plot_basis_from_samp(b_samps[max_idx], axarr[0,1])
    model.plot_weights_from_samp(w_samps[max_idx], axarr[1,1])
    f = plt.figure()
    plt.plot(lls)
    #plt.show()

    # plot Temporal Hyper parameter traces
    model.plot_time_hypers()
    plt.show()

    # examine fit
    #sanity check
    #create ground truth basis/weights 
    Lambda = B_gt.T.dot(W_gt) * model._cell_vol    #V by T matrix
    ll_gt = np.sum(model._grid_obs*np.log(Lambda) - Lambda )

    #Beta_gt_biased = np.log(B_gt)
    #Beta_gt_0 = Beta_gt_biased.mean(axis=1)
    #Beta_gt = np.column_stack((Beta_gt_0, (Beta_gt_biased.T-Beta_gt_0).T))
    #Omega_gt_biased = np.log(W_gt)
    #Omega_gt_0 = Omega_gt_biased.mean(axis=1)
    #Omega_gt = np.column_stack((Omega_gt_0, (Omega_gt_biased.T-Omega_gt_0).T))
    #th_gt = np.concatenate( (Beta_gt.ravel(), Omega_gt.ravel()) ) 
    print "Ground truth loglike: ", ll_gt #model._log_like(th_gt)
    print "sample max loglike: ", lls.max()

if __name__=="__main__":
    st_lgcp_mix_examples_1d()


