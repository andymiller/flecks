import pylab as plt
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

def create_folds(Ndata, Nfolds=10, seed=42): 
  np.random.seed(seed)
  fold_size = np.floor( Ndata/Nfolds )
  folds     = []
  inds      = np.random.permutation( Ndata )

  # gen folds 
  startI = 0
  endI   = fold_size
  for fold_idx in range(Nfolds): 
    folds.append( inds[startI:endI] )
    startI = endI
    endI   = endI + fold_size
  return folds


def draw_halfcourt(ax): 
  #rect(0, 0, 94/2, 50)
  #circle = function(x, y, r, from = 0, to = 2 * pi, lines = FALSE, ...) {
  #  theta = seq(from, to, length = 100)
  #  if (lines)
  #    lines(x + r * cos(theta), y + r * sin(theta), ...)
  #  else polygon(x + r * cos(theta), y + r * sin(theta), ...)
  #}
  #points(c(5.25, 94 - 5.25), c(25, 25), cex = 2)
  #segments(47, 0, 47, 50)
  ##circle(47, 25, 8)
  #circle(47, 25, 2, col = "lightgray")
  #theta1 = acos((25 - 35/12)/23.75)
  #circle(5.25, 25, 23.75, -pi/2 + theta1, pi/2 - theta1, TRUE)
  #circle(94 - 5.25, 25, 23.75, pi/2 + theta1, 3 * pi/2 - theta1, TRUE)
  #segments(0, 35/12, 5.25 + 23.75 * sin(theta1), 35/12)
  #segments(0, 50 - 35/12, 5.25 + 23.75 * sin(theta1), 50 - 35/12)
  #segments(94, 35/12, 94 - 5.25 - 23.75 * sin(theta1), 35/12)
  #segments(94, 50 - 35/12, 94 - 5.25 - 23.75 * sin(theta1), 50 - 35/12)
  #circle(19, 25, 6, -pi/2, pi/2, TRUE)
  #circle(19, 25, 6, pi/2, 3 * pi/2, TRUE, lty = 2)
  #circle(94 - 19, 25, 6, pi/2, 3 * pi/2, TRUE)
  #circle(94 - 19, 25, 6, -pi/2, pi/2, TRUE, lty = 2)
  #circle(5.25, 25, 4, -pi/2, pi/2, TRUE)
  #circle(94 - 5.25, 25, 4, pi/2, 3 * pi/2, TRUE)
  #rect(0, 17, 19, 33, border = "gray")
  #rect(94, 17, 94 - 19, 33, border = "gray")

  parts = {}
  parts['bounds']           = patches.Rectangle((0,0), 47, 50, fill=False, lw=2)

  #key, 
  parts['outer_key']  = patches.Rectangle((0,17), 19, 16, fill=False, lw=2)
  parts['inner_key']  = patches.Rectangle((0,19), 19, 12, fill=False, color="grey", lw=2)
  parts['jump_circle']  = patches.Circle((19, 25), radius=6, fill=False, lw=2)
  parts['restricted'] = patches.Arc( (5.25, 25), 2*4, 2*4, theta1=-90, theta2=90, lw=2)
  parts['hoop']       = patches.Circle((5.25, 25), radius=2, fill=False, lw=2)
  
  #midcourt circles
  parts['mid_circle']       = patches.Circle((47, 25), radius=8, fill=False, lw=2)
  parts['mid_small_circle'] = patches.Circle((47,25), radius=2, color="grey", lw=2)
 
  #3 point line
  break_angle = np.arccos( (25-35./12)/23.75 )      #angle between hoop->sideline and hoop->break
  break_angle_deg = break_angle / np.pi * 180
  break_length = 5.25 + 23.75*np.sin(break_angle)
  parts['arc'] = patches.Arc( (5.25, 25), 2*23.75, 2*23.75,
                              theta1 = -90+break_angle_deg, theta2 = 90-break_angle_deg, lw=2) 
  parts['break0'] = patches.Rectangle((0, 35./12), break_length, 0, lw=2)
  parts['break1'] = patches.Rectangle((0, 50-35./12), break_length, 0, lw=2) 

  #draw them
  for p in parts.itervalues():
    ax.add_patch(p) 
  
  #make sure court is drawn in proportion
  ax.set_xlim(-.1, 47.05)
  ax.set_ylim(-.1, 50.05)
  ax.set_aspect('equal')
  ax.set_axis_off()



##############################################
# NIMFA matrix factorization helper funcs
##############################################
def __fact_factor(X):
    """
    Return dense factorization factor, so that output is printed nice if factor is sparse.
    
    :param X: Factorization factor.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix`
    """
    return X.todense() if sp.isspmatrix(X) else X
    
def print_info(fit, idx=None):
    """
    Print to stdout info about the factorization.
    
    :param fit: Fitted factorization model.
    :type fit: :class:`nimfa.models.mf_fit.Mf_fit`
    :param idx: Name of the matrix (coefficient) matrix. Used only in the multiple NMF model. Therefore in factorizations 
                that follow standard or nonsmooth model, this parameter can be omitted. Currently, SNMNMF implements 
                multiple NMF model.
    :type idx: `str` with values 'coef' or 'coef1' (`int` value of 0 or 1, respectively) 
    """
    print "================================================================================================="
    print "Factorization method:", fit.fit
    print "Initialization method:", fit.fit.seed
    print "Basis matrix W: "
    print __fact_factor(fit.basis())
    print "Mixture (Coefficient) matrix H%d: " % (idx if idx != None else 0)
    print __fact_factor(fit.coef(idx))
    print "Distance (Euclidean): ", fit.distance(metric='euclidean', idx=idx)
    # We can access actual number of iteration directly through fitted model.
    # fit.fit.n_iter
    print "Actual number of iterations: ", fit.summary(idx)['n_iter']
    # We can access sparseness measure directly through fitted model.
    # fit.fit.sparseness()
    print "Sparseness basis: %7.4f, Sparseness mixture: %7.4f" % (fit.summary(idx)['sparseness'][0], fit.summary(idx)['sparseness'][1])
    # We can access explained variance directly through fitted model.
    # fit.fit.evar()
    print "Explained variance: ", fit.summary(idx)['evar']
    # We can access residual sum of squares directly through fitted model.
    # fit.fit.rss()
    print "Residual sum of squares: ", fit.summary(idx)['rss']
    # There are many more ... but just cannot print out everything =] and some measures need additional data or more runs
    # e.g. entropy, predict, purity, coph_cor, consensus, select_features, score_features, connectivity
    print "================================================================================================="


if __name__=="__main__":
  fig = plt.figure()
  ax = fig.add_subplot(111)
  draw_halfcourt(ax)
  plt.show()









