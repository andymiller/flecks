import numpy as np

#
# Small collection of gridding functions (for consistency across models/inf)
#

def x_grid_centers( edges ): 
    """ Given D lists of grid edges, return a list of 
    grid centers """
    x_centers = []
    for d in range(len(edges)):
      x_centers.append(edges[d][1:] - .5*(edges[d][1] - edges[d][0]))
    return x_centers
 
def nd_grid_centers( edges ): 
    """ Given D lists of grid edges (as returned by np.histogramdd), 
    returns a list of D dimensional points corresponding to each tile center """
    return nd_grid_points( x_grid_centers(edges) )

def nd_grid_points( dim_points ): 
    """ Given D grid centers along each axis, 
    enumerate all of the points in a long array """

    #hacky way to check this - any way to automate? 
    if len(dim_points)==1:
      return dim_points
    elif len(dim_points)==2:
      return np.dstack( np.meshgrid( dim_points[0], dim_points[1] ) ).reshape(-1,2)
    elif len(dim_points)==3:
      grids = np.meshgrid(dim_points[0], dim_points[1], dim_points[2])
      return np.squeeze( np.dstack( grids[0].flatten(), \
                                    grids[1].flatten(), \
                                    grids[2].flatten() ) )
    else: 
      raise NotImplementedError

def ndmesh(grids):
   # args = map(np.asarray,args)
   return np.broadcast_arrays(*[x[(slice(None),)+(None,)*i] for i, x in enumerate(grids)]) 


