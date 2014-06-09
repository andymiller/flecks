import numpy as np
#
# Functions to manipulate kronecker matrices
#

def kron_mat_vec_prod(kron_pieces, b): 
  """ matrix vector product between a kronecker matrix A (its pieces)
  and an appropriately sized vector b
  
  INPUT: 
    - kron_pieces = list of matrices such that A = kron(kron_pieces)
    - b           = N by 1 vector to be multiplied by the kronecker matrix

  OUTPUT: 
    - the N by 1 vector that results from this multiplication 
  """
  #TODO add a check here to make sure the KRON pieces and b jive
  N = len(b)
  x = b
  for Ad in reversed(kron_pieces): 
    Gd = Ad.shape[0]
    X  = np.reshape(x, (Gd, N/Gd), order='F')
    Z  = Ad.dot(X)
    x  = Z.flatten()                  #this is a row-wise stacking, which 
  return x

def kron_multi(kron_pieces): 
  """ Takes a set of matrices and returns their kron product """
  if len(kron_pieces) == 1:
      return kron_pieces[0]
  A = kron_pieces[0]
  for Ad in kron_pieces[1:]:
      A = np.kron(A, Ad)
  return A

def kron_inv(kron_pieces): 
  return [np.linalg.inv(A) for A in kron_pieces]

if __name__ == "__main__":

  # set up kronecker pieces
  As = [ np.random.random((5,5)), np.random.random((3,3)), np.random.random((2,2)) ]
  A  = kron_multi(As)

  ##
  # Test kronecker matrix vector product
  ##
  b = np.ones( A.shape[1] )
  slow_prod = A.dot(b)
  fast_prod = kron_mat_vec_prod(As, b)
  print slow_prod
  print fast_prod
  print "Kronecker Matrix - Vector products match: ", np.allclose(slow_prod, fast_prod)
  #sanity check - they all have same values
  #import pylab as plt
  #plt.subplot(2, 1, 1); plt.hist(slow_prod, bins=100);
  #plt.subplot(2, 1, 2); plt.hist(fast_prod, bins=100);
  #plt.show()

  # verify kron inverse
  print "inverse matches: ", np.allclose( np.linalg.inv(A), kron_multi(kron_inv(As)))


