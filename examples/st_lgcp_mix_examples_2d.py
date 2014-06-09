import pylab as plt
import numpy as np
from util import gen_synthetic_data
from lgcp.st_mix_lgcp import SpatioTemporalMixLGCP

if __name__ == "__main__":
    #""" Very simple 1d space example """
    np.random.seed(50)

    # example parameters and sythetic data
    xdim = 2
    K    = 2
    xgrid_dims = [90, 110]
    xbbox      = [[-10,10], [-8, 12]]
    tgrid_dims = np.array([100])
    tbbox      = np.array([0,50])
    data, B_gt, W_gt, xgrid, tgrid = \
        gen_synthetic_data( xdim = xdim, K=K,
                            xgrid_dims = xgrid_dims, 
                            xbbox = xbbox, 
                            tgrid_dims = tgrid_dims,
                            tbbox = tbbox )

    #plt.scatter(data[1:2000,0], data[1:2000,1], marker='.')
    #aplt.contour(

    #Plot synthetic basis and weights
    fig, axarr = plt.subplots(K, 2)
    for k in range(K): 
        axarr[k,0].imshow(B_gt[k].reshape(xgrid_dims).T, origin='lower', extent=xbbox[0]+xbbox[1])
        axarr[k,0].set_title("Spatial basis, $B_%d$"%k)
        axarr[k,1].plot(tgrid, W_gt[k])
        axarr[k,1].set_title("Temporal weights, $W_%d$"%k)
    fig.tight_layout()
    #plt.show()

    ##
    ## fit model
    ##
    train_data = data[ data[:,-1] < tbbox[1]/2 ]
    print "training on %d events"%len(train_data)
    model = SpatioTemporalMixLGCP( xdim       = xdim, 
                                   K          = K,
                                   xgrid_dims = xgrid_dims, 
                                   xbbox      = xbbox, 
                                   tgrid_dims = tgrid_dims/2,
                                   tbbox      = tbbox/2 )
    model.describe()
    w_samps, b_samps, th_samps, lls = model.fit(data, Nsamps=100)
    
    #plot highest like samp
    fig, axarr = plt.subplots(K,2)
    max_idx = lls.argmax()
    model.plot_basis_from_samp(b_samps[max_idx], axarr[:,0])
    Wmax = model._omega_to_weight(w_samps[max_idx])
    for k in range(K):
      axarr[k,1].plot(model._grids[-1], Wmax[k])
      axarr[k,1].set_title("Temporal weight fit: $W_%d$"%k)

    ## plot Temporal Hyper parameter traces
    model.plot_time_hypers()
    model.plot_space_hypers()

    ###
    ## visualize resulting posterior intensity surfs
    ###
    ##create ground truth basis/weights 
    Lambda_gt = B_gt.T.dot(W_gt[:,0:tgrid_dims[0]/2])  #V by T matrix
    V,T       = Lambda_gt.shape
    ll_gt     = np.sum(model._grid_obs*np.log(Lambda_gt) - Lambda_gt )
    print "ground truth loglike: ", ll_gt
    #compute posterior surface
    Lambda_mean, Lambda_var = model.posterior_mean_var_lambda(samp_start=500, thin=5)

    ##plot comparison
    #fig, axarr = plt.subplots(3,1)
    #vmin = np.min( np.concatenate([Lambda, Lambda_mean]) )
    #vmax = np.max( np.concatenate([Lambda, Lambda_mean]) )
    #axarr[0].imshow(Lambda_gt[:,0:np.floor(T/2)], origin='lower', vmin=vmin, vmax=vmax, 
    #                           extent=[0,tbbox[1]/2, xbbox[0][0], xbbox[0][1]])
    #axarr[0].set_title('Ground truth intensity function')
    #mean_im = axarr[1].imshow(Lambda_mean, origin='lower', vmin=vmin, vmax=vmax, 
    #                          extent=[0,tbbox[1]/2, xbbox[0][0], xbbox[0][1]])
    #axarr[1].set_title('Posterior mean intensity function')
    #fig.subplots_adjust(right=0.8)
    #mean_cbar_ax = fig.add_axes([0.85, 0.65, 0.025, 0.25])
    #fig.colorbar(mean_im, cax=mean_cbar_ax)
   
    ##plot lambda variance
    #var_im = axarr[2].imshow(Lambda_var, origin='lower', 
    #                         extent=[0,tbbox[1]/2, xbbox[0][0], xbbox[0][1]])
    #axarr[2].set_title('Posterior uncertainty')
    #var_cbar_ax = fig.add_axes([0.85, .05, 0.025, 0.25])
    #fig.colorbar(var_im, cax=var_cbar_ax)
    ##fig.tight_layout()

    ##Beta_gt_biased = np.log(B_gt)
    ##Beta_gt_0 = Beta_gt_biased.mean(axis=1)
    ##Beta_gt = np.column_stack((Beta_gt_0, (Beta_gt_biased.T-Beta_gt_0).T))
    ##Omega_gt_biased = np.log(W_gt)
    ##Omega_gt_0 = Omega_gt_biased.mean(axis=1)
    ##Omega_gt = np.column_stack((Omega_gt_0, (Omega_gt_biased.T-Omega_gt_0).T))
    ##th_gt = np.concatenate( (Beta_gt.ravel(), Omega_gt.ravel()) ) 
    #print "Ground truth loglike: ", ll_gt #model._log_like(th_gt)
    #print "sample max loglike: ", lls.max()


    ##################################################
    ## make predictions
    ##################################################
    ##visualize test lambda
    #test_data  = data[data[:,1] > tbbox[1]/2]
    #test_tbbox = np.array([tbbox[1]/2, tbbox[1]])
    #test_lam, test_grid  = model.test_likelihood( test_data, 
    #                                              tgrid_dims/2, 
    #                                              test_tbbox, 
    #                                              num_samps = 100)
    #fullLam    = np.column_stack((Lambda_mean, test_lam))
    #fig, axarr = plt.subplots(2,1) 
    #axarr[1].imshow(fullLam, origin='lower',
    #                         extent=[tbbox[0], tbbox[1], xbbox[0][0], xbbox[0][1]])
    #axarr[1].set_title('Inferred $\lambda^{old}$ (left) and projected $\lambda^{new}$')
    #axarr[0].imshow(Lambda_gt, origin='lower',
    #                           extent=[tbbox[0], tbbox[1], xbbox[0][0], xbbox[0][1]])
    #axarr[0].set_title("Ground truth, both post and post predictive")

    ##visualize forward sampled bases
    #w_pred = model.sample_conditional_intensity( w_samps[max_idx], 
    #                                             th_samps[max_idx], 
    #                                             test_grid )
    #w_mat = w_samps[max_idx].reshape((K,-1))
    #full_w = np.column_stack((w_mat[:,1:], w_pred[:,1:]))
    #full_t = np.concatenate( [model._grids[-1], test_grid] )
    #plt.figure()
    #plt.plot(full_t, full_w.T)
    plt.show()


