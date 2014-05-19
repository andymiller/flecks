import numpy as np
from pproc import DiscretizedPointProcess
from mcmc.ess import elliptical_slice
from util import  whitened_mh, spherical_proposal, normalize_rows, resample_assignments
#from cutil import resample_assignments
from kernel import MultiKronKernel
from kron_util import kron_mat_vec_prod
import pylab as plt
#
# models a group of discretized log gaussian cox processes
# using a shared basis
#
#  initialize with:
#    - dimension, D
#    - number of bins in each dimension
#    - range of each dimension (bbox)
#    - covariance of each dimension
#
#  For LGCP.fit
#  input: a set of N-D points, X
#  output: posterior samples for intensity function value
#          at specified grid
#
class MixLGCP(DiscretizedPointProcess):
  def __init__(self, K=5, dim=2, grid_dim=(50,50), bbox=[(0,1), (0,1)], \
                     kernel_types=["sque", "sque"], kern=None):
    ## initalize discretized grid
    super(MixLGCP, self).__init__(dim, grid_dim, bbox)

    #set up kernel for each dimension
    if kern is None:
      self._kern = MultiKronKernel(kernel_types)
    else:
      self._kern = Kern

    #set number of LGCPs to mix
    self._K = K

    #sanity check hack
    #self._fixed_basis = np.loadtxt('/Users/acm/Code/xyhoops/andyfranks/pyscripts/nmf_basis_rank_6.txt').T

  def fit(self, data, Nsamps=2000, prop_scale=1, \
                verbose=True, burnin=500, num_ess=10, init_th=None):
    """ Mixture of LGCP main fit method:
          - data should come in as a dim by N dimensional numpy array
    """
    #initialize a 'point process' for each example, bin data
    self._pprocs = []
    for pproc in data:
        dpp = DiscretizedPointProcess(self._dim, self._grid_dim, self._bbox)
        dpp._count_data(pproc)
        self._pprocs.append(dpp)
    print "Num point procs: ", len(self._pprocs)

    #grab grids from one of the pprocs (they are all the same)
    self._grids = self._pprocs[0]._grids
    self._grid_counts = np.array([pproc._grid_counts for pproc in self._pprocs], dtype=np.int)

    # number of latent z's to sample (including bias term) for each basis
    self._Nz = len(self._pprocs[0]._grid_counts) + 1
    self._Nw = len(self._pprocs)

    #weight hyper paramater
    alpha0 = np.ones(self._K)*.1
    beta0  = np.ones(self._K)*.1

    # sample from the posterior w/ ESS, slice sample Hypers
    #if init_th is None:
    #  th_curr = np.zeros( self._K*self._Nz + self._K*self._Nw ) #current state of all normal prior
    #else:
    #  th_curr = init_th
    h_curr  = self._kern.hypers()              # current values of hyper params
    h_curr  = np.array([1, 8, 8])
    ll_curr = 0.0                              # current log likelihood of obs

    #initialize current weights and bases
    if init_th is None:
      w_curr = np.ones( self._K*self._Nw )       # current weights
      b_curr = np.zeros( self._K*self._Nz )      # current log bases
    else:
      w_curr = init_th[0:self._K*self._Nw]
      b_curr = init_th[self._K*self._Nw:]
    z_curr = np.zeros( (self._Nw, self._Nz-1, self._K), dtype=np.int ) # current assignments

    # keep track of all samples
    #th_samps = np.zeros( (Nsamps, len(th_curr)) )
    w_samps = np.zeros( (Nsamps, len(w_curr)) )
    b_samps = np.zeros( (Nsamps, len(b_curr)) )
    h_samps = np.zeros( (Nsamps, len(h_curr)) )
    lls     = np.zeros( Nsamps )
    accept_rate = 0
    for i in range(Nsamps):
      if i%5==0 and verbose: print "  samp %d of %d (ll=%2.3f)"%(i,Nsamps,ll_curr)

      ##
      ## Sample Z values (which basis each obs comes from) given B, W
      ##
      W = w_curr.reshape((self._K, self._Nw))
      B = self._beta_to_basis(b_curr)
      z_curr = resample_assignments(W, B, self._grid_counts, z_curr)

      #Lambda = W.T.dot(B)  # Npproc by Nz matrix
      #for n in range(self._Nw):
      #  Lam_n = normalize_rows(W[:,n]*B.T)
      #  for v in range(self._Nz-1):
      #    N_nv = self._grid_counts[n,v]
      #    if N_nv > 0:
      #      Lam  = W[:,n]*B[:,v]
      #      Lam  = Lam / Lam.sum()
      #      print Lam, Lam_n[v]
      #      z_curr[n,v,:] = np.random.multinomial(N_nv, Lam, size=1)[0]
      #    else:
      #      z_curr[n,v,:] = np.zeros(self._K)

      ##
      ## sample W values (conjugate)
      ##
      for n in range(self._Nw):
        for k in range(self._K):
          a_post = alpha0[k] + np.sum(z_curr[n,:,k])
          b_post = beta0[k] + 1
          W[k,n] = np.random.gamma(shape=a_post, scale=1.0/b_post)
      w_curr = W.ravel()
      w_samps[i] = w_curr

      ##
      ## Sample ESS surface given weights and Z values
      ##
      ## sample latent surface (multiple runs of ESS)
      b_mat  = b_curr.reshape((self._K, self._Nz))
      Zbasis = z_curr.sum(axis=0).T  #all shots from each basis
      for k in range(self._K):
        beta_samp = self._kern.gen_prior(h_curr, self._grids)
        prior_b = np.concatenate(([np.sqrt(10)*np.random.randn()], \
                                  beta_samp))
        def basis_log_like(beta):
            Bk   = self._single_beta_to_basis(beta)
            lams = W[k,:].sum()*Bk
            loglam = np.log(lams)
            ll = np.sum(loglam*Zbasis[k] - lams)
            return ll
        b_mat[k], log_lik = elliptical_slice( initial_theta = b_mat[k], \
                                              prior         = prior_b, \
                                              lnpdf         = basis_log_like )
      b_curr = b_mat.ravel()
      b_samps[i] = b_curr

      ### old, ess everythign ###
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


      #def prior_basis_samp():
      #  return np.append( np.sqrt(10)*np.random.randn(),\
      #                    self._kern.gen_prior(h_curr, self._grids) )
      #for resamp_i in range(num_ess):
      #  #generate from prior over all K bases, all Nw gen bias and spatial comp
      #  prior_zs   = np.concatenate( [ prior_basis_samp() for k in range(self._K) ] )
      #  prior_ws   = np.sqrt(10)*np.random.randn( self._K * self._Nw )
      #  prior_samp = np.concatenate((prior_zs, prior_ws))
      #  th_curr, log_lik = elliptical_slice( initial_theta = th_curr, \
      #                                       prior         = prior_samp, \
      #                                       lnpdf         = self._log_like )

      #store samples
      #th_samps[i,] = th_curr
      #h_samps[i,] = h_curr

      ##
      ## compute log-like
      ##
      W = w_curr.reshape((self._K, self._Nw))
      B = self._beta_to_basis(b_curr)
      lams = W.T.dot(B)
      loglam_all = np.log(lams)
      lls[i] = np.sum( loglam_all*self._grid_counts - lams )
      ll_curr = lls[i]

      #chain stats
      #accept_rate += accepted

    #print "final acceptance rate: ", float(accept_rate)/Nsamps
    self.w_samps = w_samps
    self.b_samps = b_samps
    self.h_samps = h_samps
    self._lls      = lls
    self._burnin   = burnin

  def posterior_mean_lambda(self, thin=1):
    """ return the posterior mean lambda surface """
    zvecs  = (self._z_samps[:,0] + self._z_samps[:,1:].T).T
    zvecs  = zvecs[ np.arange(0,zvecs.shape[0],thin), :]  #thin
    lamvec = self._cell_vol * np.exp(zvecs).mean(axis=0)
    return np.reshape(lamvec, self._grid_dim, order='C')

  def posterior_max_ll_lambda(self):
    """ returns the max samp lambda """
    max_i = self._lls[self._burnin:].argmax()
    zvec = self._z_samps[max_i,0] + self._z_samps[max_i,1:]
    lamvec = self._cell_vol * np.exp(zvec)
    return np.reshape(lamvec, self._grid_dim, order='C')

  def posterior_var_lambda(self):
    """ return the posterior variance of the lambda surface """
    zvecs = (self._z_samps[:,0] + self._z_samps[:,1:].T).T
    lamvec = self._cell_vol * np.exp(zvecs).var(axis=0)
    return np.reshape(lamvec, self._grid_dim, order='C')

  def plot_basis_from_samp(self, th=None):
    """  approximate log like is just independent poissons """
    #compute positive normalized basis
    if th is None:
      Bs = self._beta_to_basis(self.b_samps[-1])
    else:
      Bs, ws = self._unravel_params(th)
    f, axarr = plt.subplots(self._K,1)
    for k in range(self._K):
      axarr[k].imshow( Bs[k].reshape(self._grid_dim).T,
                       extent=self._bbox[0]+self._bbox[1] )
    return f, axarr


  #############################################
  ## private helpers
  #############################################
  def _unravel_params(self, th):
    """ take flat vector and produce basis and weight matrices """
    #compute positive normalized basis
    Bs = self._beta_to_basis(th[0:self._K*self._Nz])
    #make weights positive
    th_ws = th[self._K*self._Nz:]
    ws = np.exp(th_ws.reshape((self._Nw, self._K)))
    return Bs, ws

  def _beta_to_basis(self, bs):
    bs = bs.reshape((self._K, self._Nz))
    logBs     = (bs[:,0] + bs[:,1:].T).T
    Bs_unnorm = self._cell_vol * np.exp(logBs)
    row_sums  = Bs_unnorm.sum(axis=1)
    Bs        = Bs_unnorm / row_sums[:,np.newaxis]
    return Bs

  def _single_beta_to_basis(self, beta):
    logB = beta[0] + beta[1:]
    B_unnorm = self._cell_vol * np.exp(logB)
    return B_unnorm/B_unnorm.sum()

  def _log_like(self, th):
    """ approximate log like is just independent poissons """
    #compute the intensity for each point process
    Bs, ws = self._unravel_params(th)
    lams = ws.dot(Bs)

    #compute log probs: pr(x | lam) = lam^x * exp(-lam) / x!
    loglam_all = np.log(lams)
    ll = np.sum( loglam_all*self._grid_counts - lams )
    return ll


if __name__=="__main__":

  import numpy as np
  import sys, os, glob
  import pylab as plt

  # iterate over files, grab shots
  shot_files = glob.glob('/Users/acm/Data/nba_shots_2013/*.txt')
  player_shots = []
  pnames = []
  for i,sfile in enumerate(shot_files):
      player_name = os.path.basename(sfile).split('_')[0]

      # load shots from file
      shots = np.reshape(np.loadtxt(sfile), (-1,3))
      x = shots[:,0:2]
      if shots.shape[0] < 1:
        print "too few shots, skipping player %s"%player_name
        continue
      player_shots.append(x)
      pnames.append(player_name)

  #fit on small samp
  model = MixLGCP(dim=2, grid_dim=(12,13), bbox=[(0,47), (0,50)])
  model.fit(player_shots, Nsamps=20, burnin=2, num_ess=2)
  plt.plot(model._lls)
  model.plot_basis_from_samp()
  plt.show()



  #x = np.row_stack( (np.random.beta(a=5, b=2, size=(500,2)),
  #                   np.random.beta(a=2, b=5, size=(500,2)) ) )
  #x = np.loadtxt('/Users/acm/Data/nba_shots_2013/LeBron James_shot_attempts.txt')
  #x = x[:, 0:2]
  ##x[500:,0] = 1.0 - x[500:,0]

  ## create appropriately sized lgcp object and fit
  #lgcp = LGCP(dim=2, grid_dim=(47,50), bbox=[(0,47), (0,50)])
  #z_samps = lgcp.fit(x)

  ## plot log like trace as a sanity check
  #plt.plot(lgcp._lls)
  #plt.show()
  #lam_mean = lgcp.posterior_mean_lambda()
  #
  ##show the court with points
  #fig = plt.figure()
  #plt.imshow(lam_mean.T, interpolation='none', origin='lower', extent=[0,47,0,50])
  #plt.colorbar()
  #plt.hold(True)
  #plt.scatter(x[:,0], x[:,1])
  #plt.show()
  #
  ##plot hyperparam posterior dists
  #hnames = ('Marginal var', 'lscale1', 'lscale2')
  #fig = plt.figure()
  #for i in range( lgcp._h_samps.shape[1] ):
  #    plt.subplot(3, 1, i)
  #    plt.hist(lgcp._h_samps[:,i], 20, normed=True)
  #    plt.title(hnames[i])
  #plt.show()


  ##fig.savefig('lgcp_length_scale_dist_test.pdf')

  #from mpl_toolkits.mplot3d import Axes3D
  #from matplotlib import cm
  #from matplotlib.ticker import LinearLocator, FormatStrFormatter
  #import matplotlib.pyplot as plt
  #import numpy as np

  #fig = plt.figure()
  #ax = fig.gca(projection='3d')
  #X = lgcp._grids[0]
  #Y = lgcp._grids[1]
  #X, Y = np.meshgrid(X, Y)
  #Z = lam_mean
  #surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, \
  #                       linewidth=0, antialiased=False)
  #ax.set_zlim(np.min(lam_mean), np.max(lam_mean))
  #ax.zaxis.set_major_locator(LinearLocator(10))
  #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
  #fig.colorbar(surf, shrink=0.5, aspect=5)
  #
  #plt.show()
