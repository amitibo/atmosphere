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
import amitibo
import os
import scipy.ndimage.filters as filters
from ._src import __src_path__

__all__ = [
    "calcHG",
    "calcHG_other",
    "L_SUN_RGB",
    "RGB_WAVELENGTH", 
    "getMisrDB",
    "calcScatterMu",
    "loadVadimData",
    "readConfiguration",
    "fixmat",
    "weighted_laplace"
]


#
# Some globals
#
L_SUN_RGB=(255, 236, 224)
#RGB_WAVELENGTH = (700e-3, 530e-3, 470e-3)
RGB_WAVELENGTH = (672e-3, 558e-3, 446e-3)

SPARSE_SIZE_LIMIT = 1e6
GRID_DIM_LIMIT = 100


def fixmat(mat):
    
    mat = np.array(mat, copy=True, order='C', dtype=mat.dtype.char)
    return mat


def getMisrDB():
    """
    Return a dict with the records of the MISR particles.
    """
    
    import pickle
    from amitibo import getResourcePath
    
    with open(getResourcePath('misr.pkl', package_name=__name__), 'rb') as f:
        misr = pickle.load(f)
    
    return misr

    
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


def calcScatterMu(grids, sun_angle_phi, sun_angle_theta):
    """
    Calclculate the cosine of the scattering angle at each voxel.
    """

    #
    # Rotate the grids so that the Z axis will point in the direction of the sun
    #
    grids_rotated = grids.rotate(sun_angle_theta, sun_angle_phi, 0)
    
    Z_rotated = grids_rotated[2]
    R_rotated = np.sqrt(grids_rotated[0]**2 + grids_rotated[1]**2 + grids_rotated[2]**2)

    mu = Z_rotated/(R_rotated+amitibo.eps(R_rotated))

    return mu


def loadVadimData(path, offset=(0, 0), remove_sunspot=False, FACMIN=20, scale=1.0):
    """
    Load the simulation data from the format used by Vadim: A list of folders.
    
    Parameters:
    -----------
    path : str
        Base path under which lie the folders with simulation results
    offset : (float, float)
        y, x translation to apply to the coordinates of Vadim's cameras.
        Vadim center is at 0, 0 while mine is at the center of the atmosphere.
    remove_sunspot : bool
        Remove the sunspot from the image
    FACMIN : float
        Ratio between img mean and sunspot, used to determine the sunspot.
        
    Returns:
    --------
    img_list : list of arrays
        List of images loaded from Vadim's results. The list is sorted by folder
        name
    cameras_list : list of arrays
        List of cameras centers as three element y, x, z. Given in meters, 
    """
        
    import glob
    import scipy.io as sio
    from .mcarats import Mcarats
    
    folder_list = glob.glob(os.path.join(path, "*"))
    if not folder_list:
        raise IOError("No img found in the folder")
    folder_list.sort()
    
    img_list = []
    cameras_list = []
    for i, folder in enumerate(folder_list):
        #
        # Load the image data
        #
        img_path = os.path.join(folder, "RGB_MATRIX.mat")
        try:
            data = sio.loadmat(img_path)
        except:
            print 'No image data in folder:', folder
            continue

        img = fixmat(data['Detector'])

        if remove_sunspot:
            R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            Rmax = R.max()#Mcarats.calcRmax(R, FACMIN=FACMIN)
            ys, xs = np.nonzero(R>(Rmax*0.9))
            R, G, B = [Mcarats.removeSunSpot(ch, ys, xs, MARGIN=1) for ch in (R, G, B)]
            img = np.dstack((R, G, B))
        
        img *= scale
        
        img_list.append(img)
    
        #
        # Parse cameras center file
        #
        with open(os.path.join(folder, 'params.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if parts[0] == 'CameraPosition':
                    cameras_list.append(np.array((float(parts[4])+offset[0], float(parts[2])+offset[1], float(parts[3]))))
                    break

    return img_list, cameras_list


def readConfiguration(path):
    
    from configobj import ConfigObj
    import sparse_transforms as spt
    import scipy.io as sio
    
    if os.path.isdir(path):
        base_path = path
    elif os.path.isfile(path):
        base_path, dump = os.path.split(path)
    else:
        base_path = os.path.join(__src_path__, 'data/configurations', path)

    path = os.path.join(base_path, 'configuration.ini')
        
    config = ConfigObj(path)
    
    #
    # Load atmosphere parameters
    #
    atmosphere_section = config['atmosphere']
    dy = atmosphere_section.as_float('dy')
    dx = atmosphere_section.as_float('dx')
    ny = atmosphere_section.as_int('ny')
    nx = atmosphere_section.as_int('nx')
    z_coords = np.array([float(i) for i in atmosphere_section.as_list('z_coords')])
    
    atmosphere_params = amitibo.attrClass(
        cartesian_grids=spt.Grids(
            np.arange(0, ny*dy, dy), # Y
            np.arange(0, nx*dx, dx), # X
            z_coords
            )
    )
    
    #
    # Load the sun parameters
    #
    sun_section = config['sun']

    sun_params = amitibo.attrClass(
        angle_phi=sun_section.as_float('angle_phi'),
        angle_theta=sun_section.as_float('angle_theta'),
        intensities=[float(i) for i in sun_section['intensities']],
        wavelengths=[float(i) for i in sun_section['wavelengths']]
    )
    
    #
    # Load distributions
    #
    try:
        distributions_section = config['distributions']
        air_dist_path = os.path.join(base_path, distributions_section['air_dist_path'])
        aerosols_dist_path = os.path.join(base_path, distributions_section['aerosols_dist_path'])
    except KeyError, e:
        air_dist_path = os.path.join(os.path.abspath('.'))
        aerosols_dist_path = os.path.join(os.path.abspath('.'))
        
    air_dist = fixmat(sio.loadmat(air_dist_path)['distribution'])
    aerosols_dist = fixmat(sio.loadmat(aerosols_dist_path)['distribution'])
    
    #
    # Load particle
    #
    particle_section = config['particle']

    particle = getMisrDB()[particle_section['name']]
    for attr in ('k', 'w', 'g'):
        if attr in particle_section:
            setattr(particle, attr, particle[attr])
            
    particle_params = amitibo.attrClass(
        k=np.array(particle['k']) * 10**-12,
        w=particle['w'],
        g=particle['g'],
        phase=particle_section['phase']
        )

    #
    # Load cameras
    #
    camera_section = config['cameras']
    
    camera_y =  np.array([float(i) for i in camera_section.as_list('y')])
    camera_x =  np.array([float(i) for i in camera_section.as_list('x')])
    camera_z =  np.array([float(i) for i in camera_section.as_list('z')])

    camera_params = amitibo.attrClass(
        resolution=[int(i) for i in camera_section['resolution']],
        type=camera_section['type'],
        photons_per_pixel=40000      
        )
    
    cameras=[(camera_y[i], camera_x[i], camera_z[i]) for i in range(camera_section.as_int('cameras_num'))]

    return atmosphere_params, particle_params, sun_params, camera_params, cameras, air_dist, aerosols_dist


def weighted_laplace(input, weights, output = None, mode = "reflect", cval = 0.0):
    """N-dimensional Laplace filter based on approximate second derivatives.

    Parameters
    ----------
    %(input)s
    %(output)s
    %(mode)s
    %(cval)s
    """
    
    assert input.ndim == len(weights), 'Shape of input (%d) must equal lenght of weights (%d)' % (input.ndim, len(weights))
    
    def derivative2(input, axis, output, mode, cval):
        filt = np.array((1, -2, 1)) * weights[axis]
        return filters.correlate1d(input, filt, axis, output, mode, cval, 0)
    return filters.generic_laplace(input, derivative2, output, mode, cval)


if __name__ == '__main__':
    pass
