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
import copy


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


def calcTransformMatrix(X_indices, Y_indices, src_shape=()):
    """Calculate a sparse transformation matrix.
    params:
        X_indices, Y_indices - 2D arrays containing the indices (as floats)
            from where transformation should sample values.
        src_shape - Shape of the source matrix.
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


def coords2indices(X_grid, Y_grid, X_pts, Y_pts):
    """Calculate indices of 2D points (given in X_pts, Y_pts) in a 2D grid
(given in X_grid, Y_grid)."""

    dx = X_grid[0, 1] - X_grid[0, 0]
    dy = Y_grid[1, 0] - Y_grid[0, 0]

    X_indices = (X_pts - X_grid[0, 0])/dx
    Y_indices = (Y_pts - Y_grid[0, 0])/dy

    return X_indices, Y_indices


class baseTransform(object):
    """The base class from which all transform class inherit"""

    def __init__(self, X_src, Y_src, X_dst=None, Y_dst=None):

        if X_dst == None:
            X_dst = X_src
            Y_dst = Y_src
            
        self.X_src = X_src
        self.Y_src = Y_src
        self.X_dst = X_dst
        self.Y_dst = Y_dst

    def __call__(self, X):
        """Apply the transform"""

        import scipy.sparse as sps

        if sps.isspmatrix_dia(X):
            #
            # Assume that this is part of elementwise multiplication.
            #
            Z = self.H * X
        else:
            #
            # Assume as an application of the transform on a matrix.
            #
            if X.shape == self.X_src.shape:
                Z = (self.H * X.ravel().reshape((-1, 1))).reshape(*self.X_dst.shape)
            else:
                Z = self.H * X
            
        return Z

    def __neg__(self):
        """Overload the - operator"""

        res = copy.deepcopy(self)
        res.H = -res.H
        return res

    def __add__(self, other):
        """Overload the add operator"""

        res = copy.deepcopy(self)
        res.H = res.H + other.H
        return res

    def __iadd__(self, other):
        """Overload the iadd operator"""

        self.H = self.H + other.H
        return self
    
    def __mul__(self, other):
        """Overload the mul operator"""

        if np.isscalar(other):
            res = copy.deepcopy(self)
            res.H = other * res.H
        elif isinstance(other, baseTransform):
            res = copy.deepcopy(self)
            res.H = res.H * other.H
            res.X_src = other.X_src
            res.Y_src = other.Y_src
        else:
            res = self.__call__(other)
            
        return res

    def __rmul__(self, other):
        """Overload the mul operator"""

        if np.isscalar(other):
            res = copy.deepcopy(self)
            res.H = other*res.H
        elif sps.isspmatrix_dia(X):
            res = self.__call__(other)
        else:
            raise NotImplementedError
            
        return res

    def __getattr__(self, attr):
        if attr == 'T':
            return self.transpose()
        elif attr == 'shape':
            return self.H.shape
        elif attr == 'size':
            return self.H.size
        else:
            raise AttributeError(attr + " not found")

    def transpose(self):
        """Transpose operator"""

        res = copy.deepcopy(self)
        
        res.H = res.H.T
        X_src = res.X_src
        Y_src = res.Y_src
        res.X_src = res.X_dst
        res.Y_src = res.Y_dst
        res.X_dst = X_src
        res.Y_dst = Y_src
        
        return res


def hstack(tmat1, tmat2):
    """Concatenate two transform matrices horizontally"""

    tmat = baseTransform(
        np.vstack((tmat1.X_src, tmat2.X_src)),
        np.vstack((tmat1.Y_src, tmat2.Y_src)),
        np.vstack((tmat1.X_dst, tmat2.X_dst)),
        np.vstack((tmat1.Y_dst, tmat2.Y_dst))
        )

    tmat.H = sps.hstack((tmat1.H, tmat2.H))
    
    return tmat


class polarTransform(baseTransform):
    """(sparse) matrix representation of cartesian to polar transform.
    params:
        X, Y - 2D arrays that define the cartesian coordinates
        center - Center (in cartesian coords) of the polar coordinates.
        radius_res, angle_res - Resolution of polar coordinates.
 """

    def __init__(self, X, Y, center, radius_res=None, angle_res=None):
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

        super(polarTransform, self).__init__(X, Y, T, R)

        #
        # Calculate the indices of the polar grid in the Cartesian grid.
        #
        X_ = R * np.cos(T) + center[0]
        Y_ = R * np.sin(T) + center[1]
        X_indices, Y_indices = coords2indices(X, Y, X_, Y_)

        #
        # Calculate the transform
        #
        self.H = calcTransformMatrix(X_indices, Y_indices, src_shape=X.shape)


class rotationTransform(baseTransform):
    """(sparse) matrix representation of rotation transform.
    params:
        X, Y - 2D arrays that define the cartesian coordinates
        angle - Angle of rotation [radians].
        dst_shape - Shape of the destination matrix (after rotation). Defaults
             to the shape of the full matrix after rotation (no cropping).
        T, R - grid in the polar coordinates (optional, calculated if not given). 
"""

    def __init__(self, X, Y, angle, T=None, R=None):
        
        dy = Y[1, 0] - Y[0, 0]
        dx = X[0, 1] - X[0, 0]

        assert abs(dx) == abs(dy), "Currently not supporting non-isotropic grids (dx=%g, dy=%g)" % (dx, dy)
        
        H_rot = np.array(
            [[np.cos(angle), -np.sin(angle), 0],
             [np.sin(angle), np.cos(angle), 0],
             [0, 0, 1]]
            )

        m_src, n_src = X.shape

        if T == None:
            coords = np.hstack((
                np.dot(H_rot, np.array([[0], [0], [1]])),
                np.dot(H_rot, np.array([[0], [m_src], [1]])),
                np.dot(H_rot, np.array([[n_src], [0], [1]])),
                np.dot(H_rot, np.array([[n_src], [m_src], [1]]))
                ))

            x0, y0, dump = np.floor(np.min(coords, axis=1)).astype(np.int)
            x1, y1, dump = np.ceil(np.max(coords, axis=1)).astype(np.int)
            dst_shape = (y1-y0, x1-x0)
            
            T, R = np.meshgrid(np.arange(x1-x0), np.arange(y1-y0))
            
            R = R * dy
            T = T * dx
            
        m_dst, n_dst = R.shape
        
        super(rotationTransform, self).__init__(X, Y, T, R)
        
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
        T_unscaled, R_unscaled = np.meshgrid(np.arange(n_dst), np.arange(m_dst))
        TR = np.vstack((T_unscaled.ravel(), R_unscaled.ravel(), np.ones(T_unscaled.size)))
        TR_ = np.dot(H, TR)

        T_indices = TR_[0, :].reshape(T_unscaled.shape)
        R_indices = TR_[1, :].reshape(T_unscaled.shape)

        self.H = calcTransformMatrix(T_indices, R_indices, src_shape=X.shape)

    
class cumsumTransform(baseTransform):
    """Calculate a (sparse) matrix representation of integration (cumsum) transform.
    params:
        X, Y - 2D arrays that define the cartesian coordinates
        axis - axis along which the integration is preformed
        direction - 1: integrate up the indices, -1: integrate down the indices.
"""

    def __init__(self, X, Y, axis=0, direction=1):
        
        import numpy as np
        import scipy.sparse as sps

        super(cumsumTransform, self).__init__(X, Y)
        
        m, n = X.shape

        if axis == 0:
            dy = np.abs(Y[1, 0] - Y[0, 0])
            if direction == 1:
                self.H = sps.spdiags(np.ones((m, m*n))*dy, -n*np.arange(m), m*n, m*n)
            else:
                self.H = sps.spdiags(np.ones((m, m*n))*dy, n*np.arange(m), m*n, m*n)                
        else:
            dx = np.abs(X[0, 1] - X[0, 0])
            if direction == 1:
                A = sps.csr_matrix(np.tril(np.ones((n, n))*dx))
            else:
                A = sps.csr_matrix(np.triu(np.ones((n, n))*dx))
                
            self.H = sps.kron(sps.eye(m, m), A)


class integralTransform(baseTransform):
    """Calculate a (sparse) matrix representation of integration transform.
    params:
        X, Y - 2D arrays that define the cartesian coordinates
        axis - axis along which the integration is preformed
"""

    def __init__(self, X, Y, axis=0):
        
        import numpy as np
        import scipy.sparse as sps

        if axis == 0:
            X_dst, Y_dst = np.meshgrid(X[0, :], np.arange(1))
        else:
            X_dst, Y_dst = np.meshgrid(np.arange(1), Y[:, 0])

        super(integralTransform, self).__init__(X, Y, X_dst, Y_dst)
        
        m, n = X.shape

        if axis == 0:
            dy = np.abs(Y[1, 0] - Y[0, 0])
            self.H = sps.spdiags(np.ones((m, m*n))*dy, n*np.arange(m), n, m*n)
        else:
            dx = np.abs(X[0, 1] - X[0, 0])
            self.H = sps.kron(np.eye(m), np.ones((1, n))*dx)


def spdiag(X):
    """Return a sparse diagonal matrix. The elements of the diagonal are made of 
 the elements of the vector X."""

    import scipy.sparse as sps

    return sps.dia_matrix((X.ravel(), 0), (X.size, X.size))
