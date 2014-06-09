import numpy as np
import numpy.random as npr
import scipy.weave

def sample_assignments(W, B, grid_counts, z_curr, is_log=False):
  """ W is a K by Nplayer matrix of weights (per player)
      B is the K by Vtiles matrix of (positive) bases 
      grid_counts is the Nplayer by V matrix of counts 
      z_curr is the Nplayer by V tile by K 3D array of current assignments
  """
  K = W.shape[0]  # number of basis surfaces
  N = W.shape[1]  # number of players or time steps
  V = B.shape[1]  # number of spatial tiles

  # pass in log weights and basis (more numerically stable)
  if is_log: 
    for n in range(N): 
      for v in range(V): 
        N_nv = grid_counts[n,v]
        if N_nv > 0: 
          Lam = np.exp(W[:,n] + B[:,v])
          Lam = Lam / Lam.sum()
          z_curr[n,v,:] = np.random.multinomial(N_nv, Lam, size=1)[0]
        else:
          z_curr[n,v,:] = np.zeros(K)
  else: 
    for n in range(N): 
      for v in range(V):
        N_nv = grid_counts[n,v]
        if N_nv > 0: 
          Lam  = W[:,n]*B[:,v]
          Lam  = Lam / Lam.sum()
          z_curr[n,v,:] = npr.multinomial(N_nv, Lam, size=1)[0]
        else: 
          z_curr[n,v,:] = np.zeros(K)

  return z_curr
  #code = \
  #"""
  #int rcount = 0;
  #for(int n=0; n<N; ++n) {
  #    for(int v=0; v<V; ++v) {
  #        int N_nv = grid_counts[n,v]; 
  #        if(N_nv > 0) {

  #            // compute probability vector
  #            float[K] Lam; 
  #            Lam[0] = W[0,n] * B[0,v]
  #            for(int k=1; k<K; ++k) 
  #                Lam[k] = Lam[k-1] + W[k,n] * B[k,v]
  #            
  #            // count the number of RVs to fall in

  #            for(int i=0; i<N_nv; ++i) {


  #            z_curr[n,v,k]
  #        }


  #    }
  #}
  #for(int m=0; m<m; m++) {
  #  if(ns(m) > 0) {
  #    for(int trial=0; trial<ns(m); trial++) {
  #      for(int k=0; k<k; k++) {
  #        if(unif_rvs(rcount) < cumps(m,k)) {
  #          samps(m,k)++; //= samps(m,k)+1;
  #          break;
  #        }
  #      }
  #      rcount++;
  #    }
  #  }
  #}
  #"""
  #scipy.weave.inline(samp_code, ['m','k','samps', 'cumps', 'ns', 'unif_rvs'], \
  #                   type_converters=scipy.weave.converters.blitz, \
  #                   compiler='gcc')
  #return z_curr


def fast_multinomials(Ps, Ns): 
  """ given a list of M probability vectors and a list of M sizes, generate
  M draws from M different multinomials, quickly.... """

  #generate all the randomness
  cumPs    = Ps.cumsum(axis=1)
  M,K      = Ps.shape
  samps    = np.zeros( (M,K) )
  expo_rvs = npr.exponential( size = Ns.sum() + 2*M )
  samp_code = \
  """
  int rcount = 0;
  //for each probability vector, sample N times
  for(int m=0; m<M; m++) {

    //only need to do anything if N is positive
    if(Ns(m) <= 0)
      continue;

    //find max range from sum of exponentials (to generate sorted unifs) 
    float Ytot = 0.0;
    for(int n=0; n<Ns(m)+1; ++n) {
      Ytot += expo_rvs(rcount + n); 
      expo_rvs(rcount + n) = Ytot;
    }

    //count occurances in each bucket
    int k_count = 0;
    for(int n=0; n<Ns(m); ++n) {
      if(expo_rvs(rcount+n) < Ytot*cumPs(m,k_count)) {
        samps(m,k_count)++;
      } else {
        k_count++; 
        n -= 1; 
      }
      //if(k_count > K)
      //  break;
    }
    rcount += Ns(m) + 1;

  }
  """
  scipy.weave.inline(samp_code, ['M','K','samps', 'cumPs', 'Ns', 'expo_rvs'], \
                     type_converters=scipy.weave.converters.blitz, \
                     compiler='gcc')
  return samps


