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
import grids
import amitibo

__all__ = ["calcHG", "L_SUN_RGB", "RGB_WAVELENGTH", "getResourcePath", "getMisrDB", "calcScatterMu"]


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


def calcScatterMu(grids, sun_angle):
    """
    Calclculate the cosine of the scattering angle at each voxel.
    """

    #
    # Rotate the grids so that the Z axis will point in the direction of the sun
    #
    grids_rotated = grids.rotate(-sun_angle, 0, 0)
    
    Z_rotated = grids_rotated[2]
    R_rotated = np.sqrt(grids_rotated[0]**2 + grids_rotated[1]**2 + grids_rotated[2]**2)

    mu = Z_rotated/(R_rotated+amitibo.eps(R_rotated))

    return mu


if __name__ == '__main__':
    testHG()
    