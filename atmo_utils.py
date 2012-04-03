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
    
    HG = (1 - g**2) / (1 + g**2 - 2*g*np.cos(PHI))**(3/2) / (4*np.pi)
    
    return HG


def applyTransformMatrix(H, X):
    """Apply matrix H to a 2D function X. X is stacked column wise.
    """
    
    m, n = X.shape
    Z = (H * X.ravel().reshape((-1, 1))).reshape(m, n)
    
    return Z


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
        
    max_R = np.max(np.sqrt((X-center[0])**2 + (Y-center[1])**2))
    T, R = np.meshgrid(np.linspace(0, np.pi, angle_res), np.linspace(0, max_R, radius_res))
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
        
    X_ = X_/dx
    Y_ = Y_/dy
    
    I = []
    J = []
    VALUES = []
    for index, (x, y) in enumerate(zip(X_.flat, Y_.flat)):
	i = int(y)
	j = int(x)

        if i < 1 or j < 1:
	    continue
	
	if i >= m-1 or j >= n-1:
	    continue
	
	di = y - i
	dj = x - j
	
	I.append(index)
	J.append(j + i*n)
	VALUES.append((1-di)*(1-dj))

	I.append(index)
	J.append(j + (i+1)*n)
	VALUES.append(di*(1-dj))

	I.append(index)
	J.append(j+1 + i*n)
	VALUES.append((1-di)*dj)

	I.append(index)
	J.append(j+1 + (i+1)*n)
	VALUES.append(di*dj)

    H = sps.coo_matrix((np.array(VALUES), np.array((I, J))), shape=(m*n, m*n))
    
    return H, R, T
