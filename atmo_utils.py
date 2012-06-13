"""
The sun iradiance is based on the following data:
1) http://en.wikipedia.org/wiki/Sunlight#Composition
Compares the sunlight to blackbody at 5250C

2) http://www.vendian.org/mncharity/dir3/blackbody/UnstableURLs/bbr_color.html
Gives the RGB values for black body according to the following fields
#  K     temperature (K)
#  CMF   {" 2deg","10deg"}
#          either CIE 1931  2 degree CMFs with Judd Vos corrections
#              or CIE 1964 10 degree CMFs
#  x y   chromaticity coordinates
#  P     power in semi-arbitrary units
#  R G B {0-1}, normalized, mapped to gamut, logrithmic
#        (sRGB primaries and gamma correction)
#  r g b {0-255}
#  #rgb  {00-ff} 

  5500 K   2deg  0.3346 0.3451  1.363e+15    1.0000 0.8541 0.7277  255 238 222  #ffeede
  5500 K  10deg  0.3334 0.3413  1.479e+15    1.0000 0.8403 0.7437  255 236 224  #ffece0

Note: 5250 C = 5520 K

The RGB wavelengths are taken from
http://en.wikipedia.org/wiki/Color
and from "Retrieval of Aerosol Properties Over Land Using MISR Observations"
"""

from __future__ import division
from amitibo import memoized
import numpy as np
import itertools


#
# Some globals
#
L_SUN_RGB=(255, 236, 224)
#RGB_WAVELENGTH = (700e-3, 530e-3, 470e-3)
RGB_WAVELENGTH = (672e-3, 558e-3, 446e-3)


def calcHG(PHI, g):
    """Calculate the Henyey-Greenstein function for each voxel.
    The HG function is taken from: http://www.astro.umd.edu/~jph/HG_note.pdf
    """

    import numpy as np
    
    HG = (1 - g**2) / (1 + g**2 - 2*g*np.cos(PHI))**(3/2) / (2*np.pi)
    
    return HG


def calcTransformMatrix(src_grids, dst_coords):
    """Calculate a sparse transformation matrix.
    params:
        X_indices, Y_indices - 2D arrays containing the indices (as floats)
            from where transformation should sample values.
        src_shape - Shape of the source matrix (defaults to the destination shape)
    return:
        H - Sparse matrix representing the transform.
"""
    
    import numpy as np
    import scipy.sparse as sps

    #
    # Shape of grid
    #
    src_dims = src_grids[0].shape
    src_size = np.prod(np.array(src_dims))
    dst_dims = dst_coords[0].shape
    dst_size = np.prod(np.array(dst_dims))
    
    #
    # Calculate grid indices of coords.
    #
    indices, src_grids = coords2Indices(src_grids, dst_coords)

    #
    # Filter out coords outside of the grids.
    #
    nnz = np.ones(indices[0].shape, dtype=np.bool_)
    for ind, dim in zip(indices, src_dims):
        nnz *= (ind > 0) * (ind < dim)

    dst_indices = np.arange(dst_size)[nnz]
    nnz_indices = []
    nnz_coords = []
    for ind, coord in zip(indices, dst_coords):
        nnz_indices.append(ind[nnz])
        nnz_coords.append(coord.ravel()[nnz])
    
    #
    # Calculate the transform matrix.
    #
    diffs = []
    indices = []
    for grid, coord, ind in zip(src_grids, nnz_coords, nnz_indices):
        diffs.append([coord - grid[ind-1], grid[ind] - coord])
        indices.append([ind-1, ind])

    diffs = np.array(diffs)
    indices = np.array(indices)

    dims_range = np.arange(len(src_dims))
    strides = np.array([1] + list(src_dims[1:])).reshape((-1, 1))
    I, J, VALUES = [], [], []
    for sli in itertools.product(*[[0, 1]]*len(src_dims)):
        i = np.array(sli)
        c = indices[dims_range, sli, Ellipsis]
        v = diffs[dims_range, sli, Ellipsis]
        I.append(dst_indices)
        J.append(np.sum(c*strides, axis=0))
        VALUES.append(np.prod(v, axis=0))
        
    H = sps.coo_matrix(
        (np.array(VALUES).ravel(), np.array((np.array(I).ravel(), np.array(J).ravel()))),
        shape=(dst_size, src_size)
        ).tocsr()

    return H


def coords2Indices(grids, coords):
    """
    """

    inds = []
    slim_grids = []
    for dim, (grid, coord) in enumerate(zip(grids, coords)):
        sli = [0] * len(grid.shape)
        sli[dim] = Ellipsis
        grid = grid[sli]
        slim_grids.append(grid)
        inds.append(np.searchsorted(grid, coord.ravel()))

    return inds, slim_grids

        
@memoized
def polarTransformMatrix(X, Y, center, radius_res=None, angle_res=None):
    """(sparse) matrix representation of cartesian to polar transform.
    params:
        X, Y - 2D arrays that define the cartesian coordinates
        center - Center (in cartesian coords) of the polar coordinates.
        radius_res, angle_res - Resolution of polar coordinates.
 """

    import numpy as np

    if X.ndim == 1:
        X, Y = np.meshgrid(X, Y)

    if radius_res == None:
        radius_res = max(*X.shape)

    if angle_res == None:
        angle_res = radius_res

    #
    # Create the polar grid over which the target matrix (H) will sample.
    #
    max_R = np.max(np.sqrt((X-center[0])**2 + (Y-center[1])**2))
    T, R = np.meshgrid(np.linspace(0, np.pi, angle_res), np.linspace(0, max_R, radius_res))

    #
    # Calculate the indices of the polar grid in the Cartesian grid.
    #
    X_ = R * np.cos(T) + center[0]
    Y_ = R * np.sin(T) + center[1]

    #
    # Calculate the transform
    #
    H = calcTransformMatrix((Y, X), (Y_, X_))

    return H, T, R


@memoized
def rotationTransformMatrix(X, Y, angle, X_rot=None, Y_rot=None):
    """(sparse) matrix representation of rotation transform.
    params:
        X, Y - 2D arrays that define the cartesian coordinates
        angle - Angle of rotation [radians].
        dst_shape - Shape of the destination matrix (after rotation). Defaults
             to the shape of the full matrix after rotation (no cropping).
        X_rot, Y_rot - grid in the rotated coordinates (optional, calculated if not given). 
"""

    dy = Y[1, 0] - Y[0, 0]
    dx = X[0, 1] - X[0, 0]

    assert abs(dx) == abs(dy), "Currently not supporting non-isotropic grids (dx=%g, dy=%g)" % (dx, dy)

    H_rot = np.array(
        [[np.cos(angle), -np.sin(angle), 0],
         [np.sin(angle), np.cos(angle), 0],
         [0, 0, 1]]
        )

    m_src, n_src = X.shape

    if X_rot == None:
        coords = np.hstack((
            np.dot(H_rot, np.array([[0], [0], [1]])),
            np.dot(H_rot, np.array([[0], [m_src], [1]])),
            np.dot(H_rot, np.array([[n_src], [0], [1]])),
            np.dot(H_rot, np.array([[n_src], [m_src], [1]]))
            ))

        x0, y0, dump = np.floor(np.min(coords, axis=1)).astype(np.int)
        x1, y1, dump = np.ceil(np.max(coords, axis=1)).astype(np.int)
        dst_shape = (y1-y0, x1-x0)

        X_rot, Y_rot = np.meshgrid(np.arange(x1-x0) * dx, np.arange(y1-y0) * dy)

    m_dst, n_dst = X_rot.shape

    #
    # Calculate the rotation matrix.
    # Note:
    # The rotation is applied at the center, therefore
    # the coordinates are first centered, rotated and decentered.
    # For simplicity we calculate the rotation on 'integer' grid and
    # not on the original grid. This is why we don't support when
    # dx != dy.
    #
    H_center = np.eye(3)
    H_center[0, -1] = -n_dst/2
    H_center[1, -1] = -m_dst/2

    H_decenter = np.eye(3)
    H_decenter[0, -1] = n_src/2
    H_decenter[1, -1] = m_src/2

    H = np.dot(H_decenter, np.dot(H_rot, H_center))

    #
    # Calculate a rotated grid by applying the rotation.
    #
    X_unscaled, Y_unscaled = np.meshgrid(np.arange(n_dst), np.arange(m_dst))
    XY = np.vstack((X_unscaled.ravel(), Y_unscaled.ravel(), np.ones(X_unscaled.size)))
    XY_ = np.dot(H, XY)

    X_indices = XY_[0, :].reshape(X_unscaled.shape)
    Y_indices = XY_[1, :].reshape(X_unscaled.shape)

    H = calcTransformMatrix(X_indices, Y_indices, src_shape=X.shape)

    return H, X_rot, Y_rot

    
@memoized
def cumsumTransformMatrix(X, Y, axis=0, direction=1):
    """Calculate a (sparse) matrix representation of integration (cumsum) transform.
    params:
        X, Y - 2D arrays that define the cartesian coordinates
        axis - axis along which the integration is preformed
        direction - 1: integrate up the indices, -1: integrate down the indices.
"""
        
    import numpy as np
    import scipy.sparse as sps

    m, n = X.shape

    if axis == 0:
        dy = np.abs(Y[1:, 0] - Y[:-1, 0])
        dy = np.concatenate((dy, (dy[-1],)))
        if direction == 1:
            H = sps.spdiags(np.ones((m, m*n))*dy.reshape((-1, 1)), -n*np.arange(m), m*n, m*n)
        else:
            H = sps.spdiags(np.ones((m, m*n))*dy.reshape((-1, 1)), n*np.arange(m), m*n, m*n)                
    else:
        dx = np.abs(X[0, 1:] - X[0, :-1])
        dx = np.concatenate((dx, (dx[-1],)))
        if direction == 1:
            A = sps.csr_matrix(np.tril(np.ones((n, n))*dx))
        else:
            A = sps.csr_matrix(np.triu(np.ones((n, n))*dx))

        H = sps.kron(sps.eye(m, m), A)

    return H.tocsr()


@memoized
def integralTransformMatrix(X, Y, axis=0, direction=1):
    """Calculate a (sparse) matrix representation of integration transform.
    params:
        X, Y - 2D arrays that define the cartesian coordinates
        axis - axis along which the integration is preformed
        direction - 1: integrate up the indices, -1: integrate down the indices.
"""

    import numpy as np
    import scipy.sparse as sps

    m, n = X.shape

    if direction != 1:
        direction == -1
        
    if axis == 0:
        dy = (Y[1:, 0] - Y[:-1, 0]) * direction
        dy = np.concatenate((dy, (dy[-1],)))
        H = sps.spdiags(np.ones((m, m*n))*dy.reshape((-1, 1)), n*np.arange(m), n, m*n)
    else:
        dx = (X[0, 1:] - X[0, :-1]) * direction
        dx = np.concatenate((dx, (dx[-1],)))
        H = sps.kron(np.eye(m), np.ones((1, n))*dx)

    return H.tocsr()


def spdiag(X):
    """Return a sparse diagonal matrix. The elements of the diagonal are made of 
 the elements of the vector X."""

    import scipy.sparse as sps

    return sps.dia_matrix((X.ravel(), 0), (X.size, X.size))


if __name__ == '__main__':
    #
    # Test several of the above functions
    #
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.misc as sm
    import time

    lena = sm.lena()
    lena_ = lena.reshape((-1, 1))    
    X, Y = np.meshgrid(np.arange(lena.shape[1]), np.arange(lena.shape[0]))

    #
    # Polar transform
    #
    t0 = time.time()
    Hpol = polarTransformMatrix(X, Y, (256, 2))[0]
    lena_pol = Hpol * lena_
    print time.time() - t0
    
    plt.figure()
    plt.imshow(lena_pol.reshape(lena.shape))

    # #
    # # Rotation transform
    # #
    # Hrot1, X_rot, Y_rot = rotationTransformMatrix(X, Y, angle=-np.pi/3)
    # Hrot2 = rotationTransformMatrix(X_rot, Y_rot, np.pi/3, X, Y)[0]
    # lena_rot1 = Hrot1 * lena_
    # lena_rot2 = Hrot2 * lena_rot1

    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(lena_rot1.reshape(X_rot.shape))
    # plt.subplot(122)
    # plt.imshow(lena_rot2.reshape(lena.shape))

    # #
    # # Cumsum transform
    # #
    # Hcs1 = cumsumTransformMatrix(X, Y, axis=0, direction=1)
    # Hcs2 = cumsumTransformMatrix(X, Y, axis=1, direction=1)
    # Hcs3 = cumsumTransformMatrix(X, Y, axis=0, direction=-1)
    # Hcs4 = cumsumTransformMatrix(X, Y, axis=1, direction=-1)
    # lena_cs1 = Hcs1 * lena_
    # lena_cs2 = Hcs2 * lena_
    # lena_cs3 = Hcs3 * lena_
    # lena_cs4 = Hcs4 * lena_
    
    # plt.figure()
    # plt.subplot(221)
    # plt.imshow(lena_cs1.reshape(lena.shape))
    # plt.subplot(222)
    # plt.imshow(lena_cs2.reshape(lena.shape))
    # plt.subplot(223)
    # plt.imshow(lena_cs3.reshape(lena.shape))
    # plt.subplot(224)
    # plt.imshow(lena_cs4.reshape(lena.shape))

    plt.show()
    
    # t0 = time.time()
    # Hpol = polarTransformMatrix(X, Y, (256, 2))[0]
    # t2 = time.time() - t0

    # print 'first calculation: %g, memoized: %g' % (t1, t2)
