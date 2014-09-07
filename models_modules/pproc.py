import numpy as np
from ..util.grid_util import nd_grid_centers, x_grid_centers

class DiscretizedPointProcess(object):
  """ Models a very generic discretized point process.
  
  This class will initialize and maintain the discretization
  of the compact space handed to it.  This factors out stuff like
  defining grids, grid centers, binning data, raveling/unraveling
  in a coherent way.  

  LGCP, STLGCP and MixLGCP will be subclasses of this.

  Note: this thing doesn't know about time, so it is the responsibility
  of the subclasses to treat that dimension differently.  

  FIELDS: 
    - self._dim      : dimensionality of the point process (space and time)
    - self._grid_dim : number of tiles for each dimension 
    - self._bbox     : range of each dimension 
    - self._data     : data passed in, which will be turned into counts
    - self._cell_vol : volume of each cell (a scalar if regular, a N array if not...)
    - self._grid_counts: counts of data in each grid cell
    - self._grids    : cell centers along each dimension of the space
  """

  def __init__(self, num_dim, grid_dim, bbox):
    """ initialize grid variables """
    self._dim      = num_dim              # dimensionality of the point process
    self._grid_dim = grid_dim         # number of tiles for each dimension
    self._bbox     = bbox             # range of each dimension
    self._data     = None
    assert self._dim == len(self._grid_dim), 'LGCP Space dimensions not coherent'

    # compute cell volume
    ## TODO insert a check to see if this is a regular grid or not
    self._cell_vol = 1.0
    for d in range(self._dim):
      self._cell_vol *= float(bbox[d][1] - bbox[d][0]) / grid_dim[d] 

  def _count_data(self, data): 
    """ bin the data into a list of counts, unravel it according to 
    'C' conventions (for the kron structure stuff) 
    """
    # grid data into tile counts
    assert self._dim == data.shape[1]
    grid_counts, edges = np.histogramdd( data, \
                                         bins=self._grid_dim, \
                                         range=self._bbox )
    self._grid_counts = np.ravel(grid_counts, order='C')  # counts in each box
    self._grids       = x_grid_centers(edges)      # list of grid centers for each dimension 
    self._grid_obs    = grid_counts                # maintain N-d structured array as well





