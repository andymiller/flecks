import pylab as plt
import numpy as np
from lgcp.examples.util import gen_synthetic_data
from lgcp.st_mix_lgcp import SpatioTemporalMixLGCP

#def st_lgcp_mix_examples_1d():
if __name__=="__main__":
    """ Very simple 1d space example """
    np.random.seed(100)

    # example parameters and sythetic data
    xdim = 1
    K    = 2
    xgrid_dims = [100]
    xbbox      = [[-10,10]]
    tgrid_dims = np.array([100])
    tbbox      = np.array([0,50])
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
    axarr[0,0].set_xlim(xbbox[0])

    axarr[1,0].plot(tgrid, W_gt.T) 
    axarr[1,0].set_title("temporal weights")
    axarr[1,0].set_ylabel('T space')
    axarr[1,0].set_xlim(tbbox/2)

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
    axarr[2,0].set_xlim(xbbox[0])
    axarr[2,0].set_ylim(tbbox/2)
    fig.tight_layout()

    #
    # fit model
    #
    train_data = data[ data[:,1] < tbbox[1]/2 ]  #grab 
    K = 2
    dim = 1
    #xgrid_dims = xgrid_dims # downsample space and time
    #tgrid_dims = tgrid_dims
    #tbbox     /= 2
    model = SpatioTemporalMixLGCP( xdim       = xdim, 
                                   K          = K,
                                   xgrid_dims = xgrid_dims, 
                                   xbbox      = xbbox, 
                                   tgrid_dims = tgrid_dims/2,
                                   tbbox      = tbbox/2 )
    model.describe()


    init_xh = np.array( [[1, .3], [1, .3]] )
    init_th = np.array( [[15., 20.], [15, 20.]] )
    w_samps, b_samps, th_samps, lls = model.fit( data, 
                                                 Nsamps  = 500, 
                                                 init_xh = init_xh,  
                                                 init_th = init_th )
    max_idx = lls.argmax()
    model.plot_basis_from_samp(b_samps[max_idx], axarr[0,1])
    model.plot_weights_from_samp(w_samps[max_idx], axarr[1,1])
    f = plt.figure()
    plt.plot(lls)

    # plot Temporal Hyper parameter traces
    model.plot_time_hypers()
    model.plot_space_hypers()

    ##
    # visualize resulting posterior intensity surfs
    ##
    #create ground truth basis/weights 
    #compute posterior surface
    Lambda_mean, Lambda_var = model.posterior_mean_var_lambda(samp_start=10, thin=5)
    Lambda_gt = B_gt.T.dot(W_gt)  #V by T matrix
    Lambda_gt_train = Lambda_gt[:,0:np.floor( Lambda_gt.shape[1]/2 )]

    #plot comparison
    fig, axarr = plt.subplots(3,1)
    vmin = np.min( np.column_stack([Lambda_gt, Lambda_mean]) )
    vmax = np.max( np.column_stack([Lambda_gt, Lambda_mean]) )
    axarr[0].imshow(Lambda_gt_train, origin='lower', vmin=vmin, vmax=vmax, 
                               extent=[0,tbbox[1]/2, xbbox[0][0], xbbox[0][1]])
    axarr[0].set_title('Ground truth intensity function')
    mean_im = axarr[1].imshow(Lambda_mean, origin='lower', vmin=vmin, vmax=vmax, 
                              extent=[0,tbbox[1]/2, xbbox[0][0], xbbox[0][1]])
    axarr[1].set_title('Posterior mean intensity function')
    fig.subplots_adjust(right=0.8)
    mean_cbar_ax = fig.add_axes([0.85, 0.65, 0.025, 0.25])
    fig.colorbar(mean_im, cax=mean_cbar_ax)
   
    #plot lambda variance
    var_im = axarr[2].imshow(Lambda_var, origin='lower', 
                             extent=[0,tbbox[1]/2, xbbox[0][0], xbbox[0][1]])
    axarr[2].set_title('Posterior uncertainty')
    var_cbar_ax = fig.add_axes([0.85, .05, 0.025, 0.25])
    fig.colorbar(var_im, cax=var_cbar_ax)
    #fig.tight_layout()

    #Beta_gt_biased = np.log(B_gt)
    #Beta_gt_0 = Beta_gt_biased.mean(axis=1)
    #Beta_gt = np.column_stack((Beta_gt_0, (Beta_gt_biased.T-Beta_gt_0).T))
    #Omega_gt_biased = np.log(W_gt)
    #Omega_gt_0 = Omega_gt_biased.mean(axis=1)
    #Omega_gt = np.column_stack((Omega_gt_0, (Omega_gt_biased.T-Omega_gt_0).T))
    #th_gt = np.concatenate( (Beta_gt.ravel(), Omega_gt.ravel()) ) 

    #################################################
    # make predictions
    #################################################
    #visualize test lambda
    test_data  = data[data[:,1] > tbbox[1]/2]
    test_tbbox = np.array([tbbox[1]/2, tbbox[1]])
    test_lam, test_grid  = model.test_likelihood( test_data, 
                                                  tgrid_dims/2, 
                                                  test_tbbox, 
                                                  num_samps = 100)
    fullLam    = np.column_stack((Lambda_mean, test_lam))
    fig, axarr = plt.subplots(2,1) 
    axarr[1].imshow(fullLam, origin='lower',
                             extent=[tbbox[0], tbbox[1], xbbox[0][0], xbbox[0][1]])
    axarr[1].set_title('Inferred $\lambda^{old}$ (left) and projected $\lambda^{new}$')
    axarr[0].imshow(Lambda_gt, origin='lower',
                               extent=[tbbox[0], tbbox[1], xbbox[0][0], xbbox[0][1]])
    axarr[0].set_title("Ground truth, both post and post predictive")

    #visualize forward sampled bases
    w_pred = model.sample_conditional_intensity( w_samps[max_idx], 
                                                 th_samps[max_idx], 
                                                 test_grid )
    w_mat = w_samps[max_idx].reshape((K,-1))
    full_w = np.column_stack((w_mat[:,1:], w_pred[:,1:]))
    full_t = np.concatenate( [model._grids[-1], test_grid] )
    plt.figure()
    plt.plot(full_t, full_w.T)
    plt.show()

    #ground truth loglike
    ll_gt     = np.sum(model._grid_obs*np.log(Lambda_gt_train) - Lambda_gt_train )
    print "Ground truth loglike: ", ll_gt #model._log_like(th_gt)
    print "sample max loglike: ", lls.max()


