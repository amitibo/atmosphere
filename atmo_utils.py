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


def applyTransformMatrix(H, X, dst_shape=()):
    """Apply matrix H to a 2D function X. X is stacked column wise.
    """

    if dst_shape == ():
        dst_shape = X.shape
        
    m, n = dst_shape
    Z = (H * X.ravel().reshape((-1, 1))).reshape(m, n)
    
    return Z


def calcTransformMatrix(X_indices, Y_indices, src_shape=()):
    """Calculate a sparse transformation matrix.
    params:
        X_indices, Y_indices - 2D arrays containing the indices (as floats)
            from where transformation should sample values.
    return:
        H - Sparse matrix representing the transform.
"""

    import numpy as np
    import scipy.sparse as sps

    m_dst, n_dst = X_indices.shape
    
    if src_shape == ():
        src_shape = X_indices.shape
        
    m_src, n_src = src_shape
        
    #
    # Calculate the transform matrix. Based on linear interpolation
    # of neighbouring indices.
    #
    I = []
    J = []
    VALUES = []
    for index, (x, y) in enumerate(zip(X_indices.flat, Y_indices.flat)):
	i = int(y)
	j = int(x)

        if i < 1 or j < 1:
	    continue
	
	if i >= m_src-1 or j >= n_src-1:
	    continue
	
	di = y - i
	dj = x - j
	
	I.append(index)
	J.append(j + i*n_src)
	VALUES.append((1-di)*(1-dj))

	I.append(index)
	J.append(j + (i+1)*n_src)
	VALUES.append(di*(1-dj))

	I.append(index)
	J.append(j+1 + i*n_src)
	VALUES.append((1-di)*dj)

	I.append(index)
	J.append(j+1 + (i+1)*n_src)
	VALUES.append(di*dj)

    H = sps.coo_matrix(
        (np.array(VALUES), np.array((I, J))),
        shape=(m_dst*n_dst, m_src*n_src)
        ).tocsr()

    return H


@memoized
def calcPolarTransformMatrix(X, Y, center, radius_res, angle_res=None):
    """Calculate a (sparse) matrix representation of cartesian to polar
transform.
    params:
        X, Y - Are either 1D, 2D arrays that define the cartesian coordinates
        center - Center (in cartesian coords) of the polar coordinates.
        radius_res, angle_res - Resolution of polar coordinates.
    return:
        H - Sparse matrix representing the cartesian-polar transform.
            The transform is applied by multiplying the matrix by the
            1D column wise stacking of the function.
        R, T - Polar coordinates.
    """

    import numpy as np
    
    if angle_res == None:
        angle_res = radius_res

    if X.ndim == 1:
        X, Y = np.meshgrid(X, Y)
        
    m_src, n_src = X.shape
        
    #
    # Create the polar grid over which the target matrix (H) will sample.
    #
    max_R = np.max(np.sqrt((X-center[0])**2 + (Y-center[1])**2))
    T, R = np.meshgrid(np.linspace(0, np.pi, angle_res), np.linspace(0, max_R, radius_res))

    #
    # Cartesian coords of the polar grid.
    #
    X_ = R * np.cos(T) + center[0]
    Y_ = R * np.sin(T) + center[1]

    if len(X.shape) == 2:
        dx = X[0, 1] - X[0, 0]
        dy = Y[1, 0] - Y[0, 0]
        m, n = X.shape
    else:
        dx = X[1] - X[0]
        dy = Y[1] - Y[0]
        m, n = len(Y), len(X)

    #
    # Indices (in the cartesian grid) of the polar grid.
    #
    X_ = X_/dx
    Y_ = Y_/dy

    H = calcTransformMatrix(X_, Y_, src_shape=(m_src, n_src))
    
    return H, R, T


def _onewayRotationTransform(H, dst_shape, src_shape):
    
    import numpy as np

    m_dst, n_dst = dst_shape
    m_src, n_src = src_shape
    
    X, Y = np.meshgrid(np.arange(n_dst), np.arange(m_dst))

    #
    # Calculate a rotated grid by applying the rotation.
    #
    XY = np.vstack((X.ravel(), Y.ravel(), np.ones(X.size)))
    XY_ = np.dot(H, XY)
    
    X_ = XY_[0, :].reshape((m_dst, n_dst))
    Y_ = XY_[1, :].reshape((m_dst, n_dst))
    
    return calcTransformMatrix(X_, Y_, src_shape=(m_src, n_src))
    
    
@memoized
def calcPolarTransformMatrix(X, Y, center, radius_res, angle_res=None):
    """Calculate a (sparse) matrix representation of cartesian to polar
transform.
    params:
        X, Y - Are either 1D, 2D arrays that define the cartesian coordinates
        center - Center (in cartesian coords) of the polar coordinates.
        radius_res, angle_res - Resolution of polar coordinates.
    return:
        H - Sparse matrix representing the cartesian-polar transform.
            The transform is applied by multiplying the matrix by the
            1D column wise stacking of the function.
        R, T - Polar coordinates.
    """

    import numpy as np
    
    if angle_res == None:
        angle_res = radius_res

    if X.ndim == 1:
        X, Y = np.meshgrid(X, Y)
        
    m_src, n_src = X.shape
        
    #
    # Create the polar grid over which the target matrix (H) will sample.
    #
    max_R = np.max(np.sqrt((X-center[0])**2 + (Y-center[1])**2))
    T, R = np.meshgrid(np.linspace(0, np.pi, angle_res), np.linspace(0, max_R, radius_res))

    #
    # Cartesian coords of the polar grid.
    #
    X_ = R * np.cos(T) + center[0]
    Y_ = R * np.sin(T) + center[1]

    if len(X.shape) == 2:
        dx = X[0, 1] - X[0, 0]
        dy = Y[1, 0] - Y[0, 0]
        m, n = X.shape
    else:
        dx = X[1] - X[0]
        dy = Y[1] - Y[0]
        m, n = len(Y), len(X)

    #
    # Indices (in the cartesian grid) of the polar grid.
    #
    X_ = X_/dx
    Y_ = Y_/dy

    H = calcTransformMatrix(X_, Y_, src_shape=(m_src, n_src))
    
    return H, R, T


def _onewayRotationTransform(H, dst_shape, src_shape):
    
    import numpy as np

    m_dst, n_dst = dst_shape
    m_src, n_src = src_shape
    
    X, Y = np.meshgrid(np.arange(n_dst), np.arange(m_dst))

    #
    # Calculate a rotated grid by applying the rotation.
    #
    XY = np.vstack((X.ravel(), Y.ravel(), np.ones(X.size)))
    XY_ = np.dot(H, XY)
    
    X_ = XY_[0, :].reshape((m_dst, n_dst))
    Y_ = XY_[1, :].reshape((m_dst, n_dst))
    
    return calcTransformMatrix(X_, Y_, src_shape=(m_src, n_src))
    

@memoized
def calcRotationTransformMatrix(src_shape, angle, dst_shape=()):
    """Calculate a (sparse) matrix representation of rotation transform.
    params:
        src_shape - Shape of the matrix to be transformed
        angle - Angle of rotation [radians].
        dst_shape - Shape of the destination matrix (after rotation). Defaults
             to the shape of the full matrix after rotation (no cropping).
    return:
        H_forward, H_backward - Sparse matrix representing the rotation transforms.
            The transform is applied by multiplying the matrix by the
            1D column wise stacking of the function.
        dst_shape - Shape of the destination matrix (after rotation).
    """

    import numpy as np

    H_rot = np.array(
        [[np.cos(angle), -np.sin(angle), 0],
         [np.sin(angle), np.cos(angle), 0],
         [0, 0, 1]]
        )

    m_src, n_src = src_shape
    
    if dst_shape == ():
        coords = np.hstack((
            np.dot(H_rot, np.array([[0], [0], [1]])),
            np.dot(H_rot, np.array([[0], [m_src], [1]])),
            np.dot(H_rot, np.array([[n_src], [0], [1]])),
            np.dot(H_rot, np.array([[n_src], [m_src], [1]]))
            ))
        x0, y0, dump = np.floor(np.min(coords, axis=1)).astype(np.int)
        x1, y1, dump = np.ceil(np.max(coords, axis=1)).astype(np.int)
        dst_shape = (y1-y0, x1-x0)

    m_dst, n_dst = dst_shape

    #
    # Calculate the rotation matrix.
    # Note:
    # The rotation is applied at the center, therefore
    # the coordinates are first centered, rotated and decentered.
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
    H_forward = _onewayRotationTransform(H, dst_shape, src_shape)
    H_backward = _onewayRotationTransform(np.linalg.inv(H), src_shape, dst_shape)

    return H_forward, H_backward, dst_shape


@memoized
def calcIntegralTransformMatrix(src_shape, axis=0):
    """Calculate a (sparse) matrix representation of integration (cumsum) transform.
    params:
        src_shape - Shape of the matrix to be transformed
        axis - axis along which the integration is preformed
    return:
        H - Sparse matrix representing the integration transforms.
            The transform is applied by multiplying the matrix by the
            1D column wise stacking of the function.
    """

    import numpy as np
    import scipy.sparse as sps

    m, n = src_shape

    if axis == 0:
        H = sps.spdiags(np.ones((m, m*n)), -n*np.arange(m), m*n, m*n)
    else:
        A = sps.csr_matrix(np.tril(np.ones((n, n))))
        H = sps.kron(sps.eye(m, m), A)

    return H
