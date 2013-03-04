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
import numpy as np
import itertools
import pkg_resources

__all__ = ["calcHG", "L_SUN_RGB", "RGB_WAVELENGTH", "getResourcePath", "getMisrDB"]

#
# Some globals
#
L_SUN_RGB=(255, 236, 224)
#RGB_WAVELENGTH = (700e-3, 530e-3, 470e-3)
RGB_WAVELENGTH = (672e-3, 558e-3, 446e-3)

SPARSE_SIZE_LIMIT = 1e6
GRID_DIM_LIMIT = 100


def getResourcePath(name):
    """
    Return the path to a resource
    """

    return pkg_resources.resource_filename(__name__, "data/%s" % name)
    

def getMisrDB():
    """
    Return a dict with the records of the MISR particles.
    """
    
    import pickle
    
    with open(getResourcePath('misr.pkl'), 'rb') as f:
        misr = pickle.load(f)
    
    return misr

    
def viz3D(X, Y, Z, V, X_label='X', Y_label='Y', Z_label='Z', title='3D Visualization'):

    import mayavi.mlab as mlab
    
    mlab.figure()

    src = mlab.pipeline.scalar_field(X, Y, Z, V)
    src.spacing = [1, 1, 1]
    src.update_image_data = True    
    ipw_x = mlab.pipeline.image_plane_widget(src, plane_orientation='x_axes')
    ipw_y = mlab.pipeline.image_plane_widget(src, plane_orientation='y_axes')
    ipw_z = mlab.pipeline.image_plane_widget(src, plane_orientation='z_axes')
    mlab.colorbar()
    mlab.outline()
    mlab.xlabel(X_label)
    mlab.ylabel(Y_label)
    mlab.zlabel(Z_label)

    limits = []
    for grid in (X, Y, Z):
        limits += [grid.min()]
        limits += [grid.max()]
    mlab.axes(ranges=limits)
    mlab.title(title)


def viz2D(V):
    
    import matplotlib.pyplot as plt
    
    plt.figure()    
    plt.imshow(V, interpolation='nearest')
    plt.gray()
    

def calcHG(mu, g):
    """Calculate the Henyey-Greenstein function for each voxel.
    I use the modified Henyey-Greenstein function from
    'Physically reasonable analytic expression for the single-scattering phase function'
    """

    HG = 3.0/2.0*(1 - g**2)/(2 + g**2)*(1 + mu**2)/(1 + g**2 - 2*g*mu)**(3/2) / (4*np.pi)
    
    return HG


def calcHG_other(mu, g):
    """Calculate the Henyey-Greenstein function for each voxel.
    The HG function is taken from: http://www.astro.umd.edu/~jph/HG_note.pdf
    """

    HG = (1 - g**2) / (1 + g**2 - 2*g*mu)**(3/2) / (4*np.pi)
    
    return HG


def calcTransformMatrix(src_grids, dst_coords):
    """
    Calculate a sparse transformation matrix. The transform
    is represented as a mapping from the src_coords to the dst_coords.
    
    Parameters
    ----------
    src_grids : list of arrays
        Array of source grids.
        
    dst_coords : list of arrays
        Array of destination grids as points in the source grids.
        
    Returns
    -------
    H : parse matrix
        Sparse matrix, in csr format, representing the transform.
"""
    
    import numpy as np
    import scipy.sparse as sps
    import itertools

    #
    # Shape of grid
    #
    src_shape = src_grids[0].shape
    src_size = np.prod(np.array(src_shape))
    dst_shape = dst_coords[0].shape
    dst_size = np.prod(np.array(dst_shape))
    dims = len(src_shape)
    
    #
    # Calculate grid indices of coords.
    #
    indices, src_grids_slim = coords2Indices(src_grids, dst_coords)

    #
    # Filter out coords outside of the grids.
    #
    nnz = np.ones(indices[0].shape, dtype=np.bool_)
    for ind, dim in zip(indices, src_shape):
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
    for grid, coord, ind in zip(src_grids_slim, nnz_coords, nnz_indices):
        diffs.append([grid[ind] - coord, coord - grid[ind-1]])
        indices.append([ind-1, ind])

    diffs = np.array(diffs)
    diffs /= np.sum(diffs, axis=1).reshape((dims, 1, -1))
    indices = np.array(indices)

    dims_range = np.arange(dims)
    strides = np.array(src_grids[0].strides).reshape((-1, 1))
    strides /= strides[-1]
    I, J, VALUES = [], [], []
    for sli in itertools.product(*[[0, 1]]*dims):
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

    import numpy as np

    inds = []
    slim_grids = []
    for dim, (grid, coord) in enumerate(zip(grids, coords)):
        sli = [0] * len(grid.shape)
        sli[dim] = Ellipsis
        grid = grid[sli]
        slim_grids.append(grid)
        inds.append(np.searchsorted(grid, coord.ravel()))

    return inds, slim_grids

        
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


def sphericalTransformMatrix(Y, X, Z, center, radius_res=None, phi_res=None, theta_res=None, THETA_portion=0.9):
    """(sparse) matrix representation of cartesian to spherical transform.
    
    Parameters
    ----------
    Y, X, Z : array,
        List of grids. The grids are expected to be of the form created by mgrid
        and in the same order of creation. This implies that the first member has
        its changing dimension as the first dimension the second member should
        have its second dimension changing etc. It also implies that the grid should
        change only in one dimension each.
    
    center : [float, float, float]
        Center (in cartesian coords) of the spherical coordinates.
    
    radius_res, phi_res, theta_res: int, optional (default=None)
        Resolution of spherical coordinates. If None, will use the maximal
        resolution of the cartesian coords.
        
    THETA_portion : [0.0-1.0], optional (default=0.9)
        The theta range will be 0-pi/2 * THETA_portion
        
    Returns
    -------
    H : sparse matrix in CSR format,
        Transform matrix the implements the spherical transform.
        
    R, PHI, THETA : arrays
        List of grids (created using mgrid) represent the Spherical coords.
 """

    import numpy as np

    if radius_res == None:
        radius_res = max(*X.shape)

    if phi_res == None:
        phi_res = radius_res
        theta_res = radius_res

    #
    # Create the polar grid over which the target matrix (H) will sample.
    #
    max_R = np.max(np.sqrt((Y-center[0])**2 + (X-center[1])**2 + (Z-center[2])**2))
    R, PHI, THETA = np.mgrid[0:max_R:complex(0, radius_res), 0:2*np.pi:complex(0, phi_res), 0:np.pi/2*THETA_portion:complex(0, theta_res)]

    #
    # Calculate the indices of the polar grid in the Cartesian grid.
    #
    X_ = R * np.sin(THETA) * np.cos(PHI) + center[0]
    Y_ = R * np.sin(THETA) * np.sin(PHI) + center[1]
    Z_ = R * np.cos(THETA) + center[2]

    #
    # Calculate the transform
    #
    H = calcTransformMatrix((Y, X, Z), (Y_, X_, Z_))

    return H, R, PHI, THETA


def rotationTransformMatrix(X, Y, angle, X_dst=None, Y_dst=None):
    """(sparse) matrix representation of rotation transform.
    params:
        X, Y - 2D arrays that define the cartesian coordinates
        angle - Angle of rotation [radians].
        dst_shape - Shape of the destination matrix (after rotation). Defaults
             to the shape of the full matrix after rotation (no cropping).
        X_rot, Y_rot - grid in the rotated coordinates (optional, calculated if not given). 
"""

    import numpy as np
    
    H_rot = np.array(
        [[np.cos(angle), -np.sin(angle), 0],
         [np.sin(angle), np.cos(angle), 0],
         [0, 0, 1]]
        )

    if X_dst == None:
        X_slim = X[0, :]
        Y_slim = Y[:, 0]
        x0_src = np.floor(np.min(X_slim)).astype(np.int)
        y0_src = np.floor(np.min(Y_slim)).astype(np.int)
        x1_src = np.ceil(np.max(X_slim)).astype(np.int)
        y1_src = np.ceil(np.max(Y_slim)).astype(np.int)
        
        coords = np.hstack((
            np.dot(H_rot, np.array([[x0_src], [y0_src], [1]])),
            np.dot(H_rot, np.array([[x0_src], [y1_src], [1]])),
            np.dot(H_rot, np.array([[x1_src], [y0_src], [1]])),
            np.dot(H_rot, np.array([[x1_src], [y1_src], [1]]))
            ))

        x0_dst, y0_dst, dump = np.floor(np.min(coords, axis=1)).astype(np.int)
        x1_dst, y1_dst, dump = np.ceil(np.max(coords, axis=1)).astype(np.int)

        dxy_dst = min(np.min(np.abs(X_slim[1:]-X_slim[:-1])), np.min(np.abs(Y_slim[1:]-Y_slim[:-1])))
        X_dst, Y_dst = np.meshgrid(
            np.linspace(x0_dst, x1_dst, int((x1_dst-x0_dst)/dxy_dst)+1),
            np.linspace(y0_dst, y1_dst, int((y1_dst-y0_dst)/dxy_dst)+1)
        )

    #
    # Calculate a rotated grid by applying the rotation.
    #
    XY_dst = np.vstack((X_dst.ravel(), Y_dst.ravel(), np.ones(X_dst.size)))
    XY_src_ = np.dot(np.linalg.inv(H_rot), XY_dst)

    X_indices = XY_src_[0, :].reshape(X_dst.shape)
    Y_indices = XY_src_[1, :].reshape(X_dst.shape)

    H = calcTransformMatrix((Y, X), (Y_indices, X_indices))

    return H, X_dst, Y_dst


def rotation3DTransformMatrix(Y, X, Z, rotation, Y_dst=None, X_dst=None, Z_dst=None):
    """Calculate a (sparse) matrix representation of rotation transform in 3D.
    
    Parameters
    ----------
    Y, X, Z : array,
        List of grids. The grids are expected to be of the form created by mgrid
        and in the same order of creation. This implies that the first member has
        its changing dimension as the first dimension the second member should
        have its second dimension changing etc. It also implies that the grid should
        change only in one dimension each.
        
    rotation : list of floats or rotation matrix
        Either a list of floats representating the rotation in Y, X, Z axes.
        The rotations are applied separately in this order. Alternatively, rotation
        can be a 4x4 rotation matrix
    
    Y_dst, X_dst, Z_dst : array, optional (default=None)
        List of grids. The grids are expected to be of the form created by mgrid
        and in the same order of creation. The transform is calculated into these
        grids. This enables croping of the target domain after the rotation transform.
        If none, the destination grids will be calculated to contain the full transformed
        source.
    
    Returns
    -------
    H : sparse matrix in CSR format,
        Transform matrix the implements the rotation transform.
        
    H_rot : array [4x4]
        The rotation transform as calculated from the input rotation parameter.
        
    Y_dst, X_dst, Z_dst : array,
        Target grid. Either the input Y_dst, X_dst, Z_dst or the calculated grid.
"""

    import numpy as np

    if isinstance(rotation, np.ndarray) and rotation.shape == (4, 4):
        H_rot = rotation
    else:
        H_rot = calcRotationMatrix(rotation)
        
    if X_dst == None:
        Y_dst, X_dst, Z_dst = _calcRotateGrid(Y, X, Z, H_rot)

    #
    # Calculate a rotated grid by applying the rotation.
    #
    XYZ_dst = np.vstack((X_dst.ravel(), Y_dst.ravel(), Z_dst.ravel(), np.ones(X_dst.size)))
    XYZ_src_ = np.dot(np.linalg.inv(H_rot), XYZ_dst)

    Y_indices = XYZ_src_[1, :].reshape(X_dst.shape)
    X_indices = XYZ_src_[0, :].reshape(X_dst.shape)
    Z_indices = XYZ_src_[2, :].reshape(X_dst.shape)

    H = calcTransformMatrix((Y, X, Z), (Y_indices, X_indices, Z_indices))

    return H, H_rot, Y_dst, X_dst, Z_dst


def calcRotationMatrix(rotation):
    
    import numpy as np
    
    #
    # Calculate the rotation transform
    #
    theta, phi, psi = rotation

    H_rotx = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, 1]]
        )

    H_roty = np.array(
        [
            [np.cos(phi), 0, np.sin(phi), 0],
            [0, 1, 0, 0],
            [-np.sin(phi), 0, np.cos(phi), 0],
            [0, 0, 0, 1]]
        )

    H_rotz = np.array(
        [
            [np.cos(psi), -np.sin(psi), 0, 0],
            [np.sin(psi), np.cos(psi), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]
        )

    H_rot = np.dot(H_rotz, np.dot(H_roty, H_rotx))
    
    return H_rot


def _calcRotateGrid(Y, X, Z, H_rot):
    #
    # Calculate the target grid.
    # The calculation is based on calculating the minimal grid that contains
    # the transformed input grid.
    #
    Y_slim = Y[:, 0, 0]
    X_slim = X[0, :, 0]
    Z_slim = Z[0, 0, :]
    x0_src = np.floor(np.min(X_slim)).astype(np.int)
    y0_src = np.floor(np.min(Y_slim)).astype(np.int)
    z0_src = np.floor(np.min(Z_slim)).astype(np.int)
    x1_src = np.ceil(np.max(X_slim)).astype(np.int)
    y1_src = np.ceil(np.max(Y_slim)).astype(np.int)
    z1_src = np.ceil(np.max(Z_slim)).astype(np.int)

    src_coords = np.array(
        [
            [x0_src, x0_src, x1_src, x1_src, x0_src, x0_src, x1_src, x1_src],
            [y0_src, y1_src, y0_src, y1_src, y0_src, y1_src, y0_src, y1_src],
            [z0_src, z0_src, z0_src, z0_src, z1_src, z1_src, z1_src, z1_src],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ]
    )
    dst_coords = np.dot(H_rot, src_coords)

    
    x0_dst, y0_dst, z0_dst, dump = np.floor(np.min(dst_coords, axis=1)).astype(np.int)
    x1_dst, y1_dst, z1_dst, dump = np.ceil(np.max(dst_coords, axis=1)).astype(np.int)

    #
    # Calculate the grid density.
    # Note:
    # This calculation is important as having a dense grid results in a huge transform
    # matrix even if it is sparse.
    #
    dy = Y_slim[1] - Y_slim[0]
    dx = X_slim[1] - X_slim[0]
    dz = Z_slim[1] - Z_slim[0]

    delta_src_coords = np.array(
        [
            [0, dx, 0, 0, -dx, 0, 0],
            [0, 0, dy, 0, 0, -dy, 0],
            [0, 0, 0, dz, 0, 0, -dz],
            [1, 1, 1, 1, 1, 1, 1]
        ]
    )
    delta_dst_coords = np.dot(H_rot, delta_src_coords)
    delta_dst_coords.sort(axis=1)
    delta_dst_coords = delta_dst_coords[:, 1:] - delta_dst_coords[:, :-1]
    delta_dst_coords[delta_dst_coords<=0] = 10000000
    
    dx, dy, dz, dump = np.min(delta_dst_coords, axis=1)
    x_samples = min(int((x1_dst-x0_dst)/dx), GRID_DIM_LIMIT)
    y_samples = min(int((y1_dst-y0_dst)/dy), GRID_DIM_LIMIT)
    z_samples = min(int((z1_dst-z0_dst)/dz), GRID_DIM_LIMIT)
    
    dim_ratio = x_samples * y_samples * z_samples / SPARSE_SIZE_LIMIT
    if  dim_ratio > 1:
        dim_reduction = dim_ratio ** (-1/3)
        
        x_samples = int(x_samples * dim_reduction)
        y_samples = int(y_samples * dim_reduction)
        z_samples = int(z_samples * dim_reduction)
        
    Y_dst, X_dst, Z_dst = np.mgrid[
        y0_dst:y1_dst:complex(0, y_samples),
        x0_dst:x1_dst:complex(0, x_samples),
        z0_dst:z1_dst:complex(0, z_samples),
    ]
    return Y_dst, X_dst, Z_dst


def gridDerivatives(grids, forward=True):
    """
    Calculate first order partial derivatives for a list of grids.
    
    Parameters
    ----------
    grids : list,
        List of grids. The grids are expected to be of the form created by mgrid
        and in the same order of creation. This implies that the first member has
        its changing dimension as the first dimension the second member should
        have its second dimension changing etc. It also implies that the grid should
        change only in one dimension each.
        
    forward : boolean, optional (default=True)
        Forward or backward derivatives.
        
    Returns
    -------
    derivatives : list,
        List of the corresponding derivatives, as 1D arrays.
    """

    import numpy as np
    
    derivatives = []
    for dim, grid in enumerate(grids):
        sli = [0] * len(grid.shape)
        sli[dim] = Ellipsis
        grid = grid[sli]
        derivative = np.abs(grid[1:] - grid[:-1])
        if forward:
            derivative = np.concatenate((derivative, (derivative[-1],)))
        else:
            derivative = np.concatenate(((derivative[0],), derivative))
        derivatives.append(derivative)

    return derivatives
    

def cumsumTransformMatrix(grids, axis=0, direction=1, masked_rows=None):
    """
    Calculate a (sparse) matrix representation of integration (cumsum) transform.
    
    Parameters
    ----------
    grids : list,
        List of grids. The length of grids should correspond to the
        dimensions of the grid.
        
    axis : int, optional (default=0)
        Axis along which the cumsum operation is preformed.
    
    direction : {1, -1}, optional (default=1)
        Direction of integration, 1 for integrating up the indices
        -1 for integrating down the indices.
       
    masked_rows: array, optional(default=None)
        If not None, leave only the rows that are non zero in the
        masked_rows array.
    
    Returns
    -------
    H : sparse matrix in CSR format,
        Transform matrix the implements the cumsum transform.
"""
        
    import numpy as np
    import scipy.sparse as sps

    grid_shape = grids[0].shape
    strides = np.array(grids[0].strides).reshape((-1, 1))
    strides /= strides[-1]

    derivatives = gridDerivatives(grids)

    inner_stride = strides[axis]
    if direction == 1:
        inner_stride = -inner_stride
        
    inner_size = np.prod(grid_shape[axis:])

    inner_H = sps.spdiags(
        np.ones((grid_shape[axis], inner_size))*derivatives[axis].reshape((-1, 1)),
        inner_stride*np.arange(grid_shape[axis]),
        inner_size,
        inner_size)
    
    if axis == 0:
        H = inner_H
    else:
        m = np.prod(grid_shape[:axis])
        H = sps.kron(sps.eye(m, m), inner_H)

    if masked_rows != None:
        H = H.tolil()
        indices = masked_rows.ravel() == 0
        for i in indices.nonzero()[0]:
            H.rows[i] = []
            H.data[i] = []
    
    return H.tocsr()


def integralTransformMatrix(grids, jacobian=None, axis=0, direction=1):
    """
    Calculate a (sparse) matrix representation of an integration transform.
    
    Parameters
    ----------
    grids : list
        List of grids. The grids are expected to be of the form created by mgrid
        and in the same order of creation.
    
    axis : int, optional (default=0)
        The axis by which the integration is performed.
        
    direction : {1, -1}, optional (default=1)
        Direction of integration
        direction - 1: integrate up the indices, -1: integrate down the indices.
        
    Returns
    -------
    H : sparse matrix
        Sparse matrix, in csr format, representing the transform.
"""

    import numpy as np
    import scipy.sparse as sps

    grid_shape = grids[0].shape
    strides = np.array(grids[0].strides)
    strides /= strides[-1]

    derivatives = gridDerivatives(grids)

    inner_stride = strides[axis]
    
    if direction != 1:
        direction  = -1
        
    inner_height = np.abs(inner_stride)
    inner_width = np.prod(grid_shape[axis:])

    inner_H = sps.spdiags(
        np.ones((grid_shape[axis], max(inner_height, inner_width)))*derivatives[axis].reshape((-1, 1))*direction,
        inner_stride*np.arange(grid_shape[axis]),
        inner_height,
        inner_width
    )
    
    if axis == 0:
        H = inner_H
    else:
        m = np.prod(grid_shape[:axis])
        H = sps.kron(sps.eye(m, m), inner_H)

    H = H.tocsr()
    
    if jacobian != None:
        H = H * spdiag(jacobian)
        
    return H.tocsr()


def cameraTransformMatrix(PHI, THETA, focal_ratio=0.5, image_res=256, theta_compensation=False):
    """
    Calculate a sparse matrix representation of camera projection transform.
    
    Parameters
    ----------
    PHI, THETA : 3D arrays
        \phi and \theta angle grids.
    
    focal_ratio : float, optional (default=0.5)
        Ratio between the focal length of the camera and the size of the sensor.
    
    image_res : int, optional (default=256)
        Resolution of the camera image (both dimensions)
        
    theta_compensation : bool, optional (default=False)
        Compensate for angle between ray and pixel
        
    Returns
    -------
    H : sparse matrix
        Sparse matrix, in csr format, representing the transform.
"""

    import numpy as np
    import amitibo
    
    Y, X = np.mgrid[-1:1:complex(0, image_res), -1:1:complex(0, image_res)]
    PHI_ = np.arctan2(Y, X) + np.pi
    R_ = np.sqrt(X**2 + Y**2 + focal_ratio**2)
    THETA_ = np.arccos(focal_ratio / (R_ + amitibo.eps(R_)))

    #
    # Calculate the transform
    #
    H = calcTransformMatrix((PHI, THETA), (PHI_, THETA_))

    #
    # Account for cos(\theta)
    #
    if theta_compensation:
        H = spdiag(np.cos(THETA_)) * H
    
    return H


def fisheyeTransformMatrix(PHI, THETA, image_res=256, theta_compensation=False):
    """
    Calculate a sparse matrix representation of fisheye projection transform.
    
    Parameters
    ----------
    PHI, THETA : 3D arrays
        \phi and \theta angle grids.
    
    image_res : int, optional (default=256)
        Resolution of the camera image (both dimensions)
        
    Returns
    -------
    H : sparse matrix
        Sparse matrix, in csr format, representing the transform.
"""

    import numpy as np
    import amitibo
    
    Y, X = np.mgrid[-1:1:complex(0, image_res), -1:1:complex(0, image_res)]
    PHI_ = np.arctan2(Y, X) + np.pi
    R_ = np.sqrt(X**2 + Y**2)
    R_[R_ > 1] = 1
    THETA_ = np.arccos(1-R_)

    #
    # Calculate the transform
    #
    H = calcTransformMatrix((PHI, THETA), (PHI_, THETA_))

    H = spdiag(np.cos(THETA_)) * H
    
    #
    # Account for cos(\theta)
    #
    if theta_compensation:
        H = spdiag(np.cos(THETA_)**2) * H
    
    return H


def linearCameraTransformMatrix(PHI, THETA, image_res=256, theta_compensation=False):
    """
    Calculate a sparse matrix representation of a linear camera projection transform.
    
    Parameters
    ----------
    PHI, THETA : 3D arrays
        \phi and \theta angle grids.
    
    image_res : int, optional (default=256)
        Resolution of the camera image (both dimensions)
        
    theta_compensation : bool, optional (default=False)
        Compensate for angle between ray and pixel
        
    Returns
    -------
    H : sparse matrix
        Sparse matrix, in csr format, representing the transform.
"""

    import numpy as np
    import amitibo
    
    Y, X = np.mgrid[-1:1:complex(0, image_res), -1:1:complex(0, image_res)]
    PHI_ = np.arctan2(Y, X) + np.pi
    R_ = np.sqrt(X**2 + Y**2)
    THETA_ = R_ * np.pi / 2

    #
    # Calculate the transform
    #
    H = calcTransformMatrix((PHI, THETA), (PHI_, THETA_))

    #
    # Account for cos(\theta)
    #
    if theta_compensation:
        H = spdiag(np.cos(THETA_)) * H
    
    return H


def spdiag(X):
    """
    Return a sparse diagonal matrix. The elements of the diagonal are made of 
    the elements of the vector X.

    Parameters
    ----------
    X : array
        1D array to be placed on the diagonal.
        
    Returns
    -------
    H : sparse matrix
        Sparse diagonal matrix, in dia format.
"""

    import scipy.sparse as sps

    return sps.dia_matrix((X.ravel(), 0), (X.size, X.size)).tocsr()


def test2D():
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.misc as sm
    import time

    ##############################################################
    # 2D data
    ##############################################################
    lena = sm.lena()
    lena = lena[:256, ...]
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
    plt.imshow(lena_pol.reshape((512, 512)), interpolation='nearest')

    #
    # Rotation transform
    #
    Hrot1, X_rot, Y_rot = rotationTransformMatrix(X, Y, angle=-np.pi/3)
    Hrot2 = rotationTransformMatrix(X_rot, Y_rot, np.pi/3, X, Y)[0]
    lena_rot1 = Hrot1 * lena_
    lena_rot2 = Hrot2 * lena_rot1

    plt.figure()
    plt.subplot(121)
    plt.imshow(lena_rot1.reshape(X_rot.shape))
    plt.subplot(122)
    plt.imshow(lena_rot2.reshape(lena.shape))

    #
    # Cumsum transform
    #
    Hcs1 = cumsumTransformMatrix((Y, X), axis=0, direction=1)
    Hcs2 = cumsumTransformMatrix((Y, X), axis=1, direction=1)
    Hcs3 = cumsumTransformMatrix((Y, X), axis=0, direction=-1)
    Hcs4 = cumsumTransformMatrix((Y, X), axis=1, direction=-1)
    lena_cs1 = Hcs1 * lena_
    lena_cs2 = Hcs2 * lena_
    lena_cs3 = Hcs3 * lena_
    lena_cs4 = Hcs4 * lena_

    plt.figure()
    plt.subplot(221)
    plt.imshow(lena_cs1.reshape(lena.shape))
    plt.subplot(222)
    plt.imshow(lena_cs2.reshape(lena.shape))
    plt.subplot(223)
    plt.imshow(lena_cs3.reshape(lena.shape))
    plt.subplot(224)
    plt.imshow(lena_cs4.reshape(lena.shape))

    plt.show()
    
    # t0 = time.time()
    # Hpol = polarTransformMatrix(X, Y, (256, 2))[0]
    # t2 = time.time() - t0
    # print 'first calculation: %g, memoized: %g' % (t1, t2)


def test3D():
    import mayavi.mlab as mlab
    import time

    #
    # Test several of the above functions
    #
    ##############################################################
    # 3D data
    ##############################################################
    Y, X, Z = np.mgrid[-10:10:50j, -10:10:50j, -10:10:50j]
    V = np.sqrt(Y**2 + X**2 + Z**2)
    V_ = V.reshape((-1, 1))
    
    #
    # Spherical transform
    #
    t0 = time.time()
    Hsph, R, PHI, THETA = sphericalTransformMatrix(Y, X, Z, (0, 0, 0))
    Vsph = Hsph * V_
    print time.time() - t0
     
    #
    # Rotation transform
    #
    t0 = time.time()
    Hrot, rotation, Y_rot, X_rot, Z_rot = rotation3DTransformMatrix(Y, X, Z, (np.pi/4, np.pi/4, 0))
    Vrot = Hrot * V_
    Hrot2 = rotation3DTransformMatrix(Y_rot, X_rot, Z_rot, np.linalg.inv(rotation), Y, X, Z)[0]
    Vrot2 = Hrot2 * Vrot
    print time.time() - t0
     
    # #
    # # Cumsum transform
    # #
    # Hcs1 = cumsumTransformMatrix((Y, X, Z), axis=0, direction=-1)
    # Vcs1 = Hcs1 * V_

    # #
    # # Integral transform
    # #
    # Hit1 = integralTransformMatrix((Y, X, Z), axis=0, direction=-1)
    # Vit1 = Hit1 * V_

    #
    # 3D visualization
    #
    viz3D(Y, X, Z, V, title='V Rotated')
    
    viz3D(Y_rot, X_rot, Z_rot, Vrot.reshape(Y_rot.shape), title='V Rotated')
    
    viz3D(Y, X, Z, Vrot2.reshape(Y.shape), title='V Rotated Back')

    viz3D(R, PHI, THETA, Vsph.reshape(R.shape), title='V Spherical')
    
    # mlab.figure()
    # mlab.contour3d(Vcs1.reshape(V.shape), contours=[1, 2, 3], transparent=True)
    # mlab.outline()
    
    mlab.show()
    
    #
    # 2D visualization
    #
    # import matplotlib.pyplot as plt
    
    # plt.figure()
    # plt.imshow(Vit1.reshape(V.shape[:2]))
    # plt.show()

def testProjection():

    import scipy.misc as scm
    import matplotlib.pyplot as plt
    
    l = scm.lena()

    PHI, THETA = np.mgrid[0:2*np.pi:512j, 0:np.pi/2*0.9:512j]
    
    H = cameraTransformMatrix(PHI, THETA, focal_ratio=0.15)
    lp = H * l.reshape((-1, 1))

    plt.figure()
    plt.imshow(l)
    
    plt.figure()
    plt.imshow(lp.reshape((256, 256)))

    plt.show()
    

def testHG():
    
    g = 0.4
    angle = np.linspace(0, np.pi, 200)
    hg1 = calcHG(np.cos(angle), g)
    hg2 = calcHG_other(np.cos(angle), g)
    
    import matplotlib.pyplot as plt
    
    plt.figure()
    plt.plot(angle, hg1)
    plt.plot(angle, hg2)
    plt.show()


if __name__ == '__main__':
    testHG()
    