# cython: profile=True
import numpy as np
cimport numpy as np
from util import normalize_rows

def resample_assignments(np.ndarray[double, ndim=2] W, \
                         np.ndarray[double, ndim=2] B, \
                         np.ndarray[long, ndim=2] grid_counts, \
                         np.ndarray[long, ndim=3] z_curr): 
  """ W is a K by Nplayer matrix of weights (per player)
      B is the K by Vtiles matrix of (positive) bases 
      grid_counts is the Nplayer by V matrix of counts 
      z_curr is the Nplayer by V tile by K 3D array of current assignments
  """
  cdef unsigned K = W.shape[0]
  cdef unsigned N = W.shape[1]
  cdef unsigned V = B.shape[1]
  for n in range(N): 
      #Lam_n = normalize_rows(W[:,n]*B.T)
      for v in range(V):
          N_nv = grid_counts[n,v]
          if N_nv > 0: 
              Lam  = W[:,n]*B[:,v]
              Lam  = Lam / Lam.sum()
              #print Lam, Lam_n[v]
              z_curr[n,v,:] = np.random.multinomial(N_nv, Lam, size=1)[0]
          else: 
              z_curr[n,v,:] = np.zeros(K)
  return z_curr



