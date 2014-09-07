from PyPoint.models import LGCP
import pylab as plt
import numpy as np

if __name__=="__main__":
  x = np.row_stack( (np.random.beta(a=5, b=2, size=(500,2)), 
                     np.random.beta(a=2, b=5, size=(500,2)) ) )
  x = np.loadtxt('/Users/acm/Data/nba_shots_2013/LeBron James_shot_attempts.txt')
  x = x[:, 0:2]
  #x[500:,0] = 1.0 - x[500:,0]

  # create appropriately sized lgcp object and fit
  lgcp = LGCP(dim=2, grid_dim=(47,50), bbox=[(0,47), (0,50)])
  z_samps = lgcp.fit(x)

  # plot log like trace as a sanity check
  plt.plot(lgcp._lls)
  plt.show()
  lam_mean = lgcp.posterior_mean_lambda()
  
  #show the court with points
  fig = plt.figure()
  plt.imshow(lam_mean.T, interpolation='none', origin='lower', extent=[0,47,0,50])
  plt.colorbar()
  plt.hold(True)
  plt.scatter(x[:,0], x[:,1])
  plt.show()
  
  #plot hyperparam posterior dists
  hnames = ('Marginal var', 'lscale1', 'lscale2')
  fig = plt.figure()
  for i in range( lgcp._h_samps.shape[1] ):
      plt.subplot(3, 1, i)
      plt.hist(lgcp._h_samps[:,i], 20, normed=True)
      plt.title(hnames[i])
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

