import numpy as np
from pproc import DiscretizedPointProcess
#from ess import elliptical_slice
#from util import nd_grid_centers, whitened_mh, x_grid_centers
#from kernel import MultiKronKernel #Kernel, SQEKernel
#from kron_util import kron_mat_vec_prod
#from mpl_toolkits.mplot3d.axes3d import Axes3D

#
# Spatio-temporal LGCP's
#
#   A static spatial component (or a few of them) are modulated by weights that
#   evolve over time (according to an exponentiated Gaussian Process).  The idea
#   is to de-couple spatial bases from temporal phenomena that modulate them, such 
#   as periodicities, bursts in intensities, etc.  
#
#   The model defines the following pieces: 
#     - B_1, ..., B_K    : non-negative basis surfaces over the 
#                          space (normalized to unit vol)
#     - w_t1, dots, w_tK : non-negative weights that modulate B_k at 
#                          each time (normalized to sum to 1)
#     - s_t              : non-negative global scaling process
#
#
#   For each space-time-bin (x_i,t_i) (with width dx, dt) we define the likelihood
#   over observations 
#     N_{x_i, t_i} ~ poisson( s_t * sum( w(t,:) * B(i) ) dVol ) 
#
#
#

class STLGCP(DiscretizedPointProcess): 

  def __init__(self, space_dim=2, grid_dim=(50,50), \
                     bbox=[(0,1), (0,1)], \
                     t_zero  = 0, t_final = 100, N_time = 100, \
                     space_kern=None, time_kern=None):
    full_bbox = bbox.append( (t_zero, t_final) )
    super(STLGCP, self).__init__(num_dim  = space_dim+1, \
                                 grid_dim = np.append(grid_dim, N_time), \
                                 bbox     = full_bbox)

    #set up spatial kernel
    if space_kernel is None: 
      self._space_kern = MultiKronKernel( ["sqeu", "sqeu"] )
    else: 
      self._space_kern = space_kern
  
    #set up time kernel
    if time_kernel is None: 
      self._time_kern = SpectralMixtureKernel()
    else: 
      self._time_kern = time_kern

  def _log_lke(self, th, data, Nz): 
    """ likelihood for z and w 
    approximate log like is just independent poissons """

    #separate out the latent surface, z, from the time proc, w
    z = th[0:Nz]
    w = th[Nz:]

    #normalize latent surface lam = exp(z) / \int exp(z)
    loglam = z[0] + z[1:]
    lam    = self._cel_vol * np.exp(loglam) 
    lam   /= self._cel_vol * np.sum(lam)

    #compute likelihood contribution from each time slice
    ll = 0
    for t in range(len(w)): 
      #pr(x | lam) = lam^x * exp(-lam) / x!
      x_t = self._grid_obs[t]
      lamt = w[t]*lam
      ll_w = np.log(lamt) * np.ravel(x_t, order='C') - lamt
      ll  += np.sum(ll_w)
    return ll

   def _log_like(self, th): 
    """ approximate log like is just independent poissons """
    #compute positive normalized basis 
    th_zs     = th[0:self._K*self._Nz]
    zs        = th_zs.reshape((self._K, self._Nz))
    logBs     = (zs[:,0] + zs[:,1:].T).T
    Bs_unnorm = self._cell_vol * np.exp(logBs) 
    row_sums  = Bs_unnorm.sum(axis=1)
    Bs        = Bs_unnorm / row_sums[:,np.newaxis]

    #make weights positive 
    th_ws = th[self._K*self._Nz:]
    ws = np.exp(th_ws.reshape((self._Nw, self._K)))

    #compute the intensity for each point process
    lams = ws.dot(Bs)

    #sum likelihood contributions 
    ll = 0
    for i in range(self._Nw): 
      #pr(x | lam) = lam^x * exp(-lam) / x!
      loglam = np.log(lams[i])
      ll += np.sum(loglam * self._pprocs[i]._grid_counts - lams[i])  # log-like
    return ll

  def fit(self, data, Nsamps=2000, prop_scale=1, \
                verbose=True, burnin=500, num_ess=10): 
    """ data should come in as a dim by N dimensional numpy array """
    #initialize a 'point process' for each example, bin data
    self._pprocs = []
    for pproc in data: 
        dpp = DiscretizedPointProcess(self._dim, self._grid_dim, self._bbox)
        dpp._count_data(pproc)
        self._pprocs.append(dpp)
    print "Num point procs: ", len(self._pprocs)

    #grab grids from one of the pprocs (they are all the same)
    self._grids = self._pprocs[0]._grids

    # number of latent z's to sample (including bias term) for each basis
    self._Nz = len(self._pprocs[0]._grid_counts) + 1
    self._Nw = len(self._pprocs) 
    # sample from the posterior w/ ESS, slice sample Hypers
    th_curr = np.zeros( self._K*self._Nz + self._K*self._Nw ) #current state of all normal prior
    h_curr  = self._kern.hypers()               # current values of hyper params
    h_curr  = np.array([1, 8, 8])
    ll_curr = None                              # current log likelihood of obs

    # keep track of all samples
    th_samps = np.zeros( (Nsamps, len(th_curr)) )
    h_samps  = np.zeros( (Nsamps, len(h_curr) ) )
    lls      = np.zeros( Nsamps )
    accept_rate = 0
    for i in range(Nsamps):
      if i%10==0 and verbose: print "  samp %d of %d"%(i,Nsamps)

      ## whitens/unwhitens 
      #def whiten(th, f): 
      #  return self._kern.whiten_process(f, th, self._grids)
      #def unwhiten(thp, nu): 
      #  return  self._kern.gen_prior(thp, self._grids, nu=nu)

      ##sample hyper params
      #h_curr, z_hyper, accepted, ll = whitened_mh( h_curr, \
      #         z_curr[1:], \
      #         whiten   = whiten, #lambda(th, f): self._kern.whiten_process(f, th, self._grids), \
      #         unwhiten = unwhiten, #lambda(thp, nu): self._kern.gen_prior(thp, self._grids, nu=nu), \
      #         Lfn      = lambda(z): self._log_like(np.append(z_curr[0], z)), \
      #         ln_prior = lambda(h): self._kern.hyper_prior_lnpdf(h), \
      #         prop_dist = lambda(th): spherical_proposal(th, prop_scale))
      #z_curr = np.append(z_curr[0], z_hyper)

      ## sample latent surface (multiple runs of ESS)

      def prior_basis_samp(): 
        return np.append( np.sqrt(10)*np.random.randn(),\
                          self._kern.gen_prior(h_curr, self._grids) ) 

      for resamp_i in range(num_ess):
        #generate from prior over all K bases, all Nw gen bias and spatial comp
        prior_zs   = np.concatenate( [ prior_basis_samp() for k in range(self._K) ] )
        prior_ws   = np.sqrt(10)*np.random.randn( self._K * self._Nw )
        prior_samp = np.concatenate((prior_zs, prior_ws))
        th_curr, log_lik = elliptical_slice( initial_theta = th_curr, \
                                             prior         = prior_samp, \
                                             lnpdf         = self._log_like )

      #store samples
      th_samps[i,] = th_curr
      #h_samps[i,] = h_curr
      lls[i] = log_lik

      #chain stats
      #accept_rate += accepted

    #print "final acceptance rate: ", float(accept_rate)/Nsamps
    self._th_samps = th_samps[burnin:,]
    self._h_samps  = h_samps[burnin:,]
    self._lls      = lls
    self._burnin   = burnin
    return th_samps, h_samps, lls

 
  

if __name__=="__main__":
  import pylab as plt
  x = np.row_stack( (np.random.beta(a=5, b=2, size=(500,2)), 
                     np.random.beta(a=2, b=5, size=(500,2)) ) )
  #x[500:,0] = 1.0 - x[500:,0]

  lgcp = LGCP()
  z_samps = lgcp.fit(x)

  plt.plot(lgcp._lls)
  plt.show()
  lam_mean = lgcp.posterior_mean_lambda()
  
  fig = plt.figure()
  plt.imshow(lam_mean.T, interpolation='none', origin='lower', extent=[0,1,0,1])
  plt.colorbar()
  plt.hold(True)
  plt.scatter(x[:,0], x[:,1])
  plt.show()
  
  #plot hyperparam posterior dists
  hsamps = lgcp._h_samps
  fig = plt.figure()
  f, axarr = plt.subplots(2, sharex=True)
  axarr[0].hist(hsamps[:,0], 20, normed=True)
  axarr[1].hist(hsamps[:,1], 20, normed=True)
  plt.show()

  #fig.savefig('lgcp_length_scale_dist_test.pdf')

  from mpl_toolkits.mplot3d import Axes3D
  from matplotlib import cm
  from matplotlib.ticker import LinearLocator, FormatStrFormatter
  import matplotlib.pyplot as plt
  import numpy as np

  fig = plt.figure()
  ax = fig.gca(projection='3d')
  X = lgcp._grids[0]
  Y = lgcp._grids[1]
  X, Y = np.meshgrid(X, Y)
  Z = lam_mean 
  surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, \
                         linewidth=0, antialiased=False)
  ax.set_zlim(np.min(lam_mean), np.max(lam_mean))
  ax.zaxis.set_major_locator(LinearLocator(10))
  ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
  fig.colorbar(surf, shrink=0.5, aspect=5)
  
  plt.show()
  
