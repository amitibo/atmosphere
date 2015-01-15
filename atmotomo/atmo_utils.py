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
from pkg_resources import resource_filename
from collections import namedtuple


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
    "weighted_laplace",
    "ColoredParam",
    "loadpds",
    "NoiseModel"
]


ColoredParam = namedtuple('ColoredParam', ['red', 'green', 'blue'])

def func_get(self, arg):
    if type(arg) == int:
        return super(ColoredParam, self).__getitem__(arg)
    return getattr(self, arg)

def func_set(self, arg, value):
    if type(arg) == int:
        super(ColoredParam, self).__setitem__(arg, value)
    setattr(self, arg, value)

setattr(ColoredParam, '__getitem__', func_get)
setattr(ColoredParam, '__setitem__', func_set)

#
# Some globals
#
L_SUN_RGB = ColoredParam(255, 236, 224)
#RGB_WAVELENGTH = (700e-3, 530e-3, 470e-3)
RGB_WAVELENGTH = ColoredParam(672e-3, 558e-3, 446e-3)

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
    
    with open(resource_filename(__name__, 'data/misr.pkl'), 'rb') as f:
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


def loadVadimData(
    path,
    base_name="RGB_MATRIX.mat",
    offset=(0, 0),
    remove_sunspot=False,
    FACMIN=20,
    scale=1.0
    ):
    """
    Load the simulation data from the format used by Vadim: A list of folders.
    
    Parameters:
    -----------
    path : str
        Base path under which lie the folders with simulation results
    base_name : str, optional (default="RGB_MATRIX.mat")
        Base name of the matrix
    offset : (float, float)
        y, x translation to apply to the coordinates of Vadim's cameras.
        Vadim center is at 0, 0 while mine is at the center of the atmosphere.
    remove_sunspot : bool, optional (default=False)
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
        img_path = os.path.join(folder, base_name)
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
        try:
            with open(os.path.join(folder, 'params.txt'), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if parts[0] == 'CameraPosition':
                        cameras_list.append(np.array((float(parts[4])+offset[0], float(parts[2])+offset[1], float(parts[3]))))
                        break
        except:
            pass

    return img_list, cameras_list


def readConfiguration(path, highten_atmosphere=False, particle_name=None):
    
    from configobj import ConfigObj
    import sparse_transforms as spt
    import scipy.io as sio
    
    if os.path.isdir(path):
        base_path = path
    elif os.path.isfile(path):
        base_path, dump = os.path.split(path)
    else:
        base_path = os.path.join(resource_filename(__name__, 'data/configurations'), path)

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
    
    #
    # Check if there is a need to extend the atmosphere up.
    #
    if highten_atmosphere:
        dz = z_coords[-1] - z_coords[-2]
        nz = len(z_coords)
        ext_nz = min(10, int(nz/2))
        ext_z_coords = np.arange(1, ext_nz+1) * dz + z_coords[-1]
        z_coords = np.concatenate((z_coords, ext_z_coords))
        
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
    # Check if there is a need to extend the atmosphere up.
    #
    if highten_atmosphere:
        air_dist = hightenDist(air_dist, target_shape=atmosphere_params.cartesian_grids.shape)
        aerosols_dist = hightenDist(aerosols_dist, target_shape=atmosphere_params.cartesian_grids.shape)
        
    #
    # Load particle
    #
    particle_section = config['particle']

    if particle_name is None:
        particle_name = particle_section['name']
        
    particle = getMisrDB()[particle_name]
    for attr in ('k', 'w', 'g'):
        if attr in particle_section:
            setattr(particle, attr, particle[attr])
            
    particle_params = amitibo.attrClass(**particle)
    particle_params.phase=particle_section['phase']

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


def hightenDist(dist, target_shape):
    temp_dist = np.zeros(target_shape)
    temp_dist[:dist.shape[0], :dist.shape[1], :dist.shape[2]] = dist
    return temp_dist


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


def loadpds(file_path):
    """Load a shdom pds file."""
    
    labels = {}
    with open(file_path, 'rb') as f:
        while True:
            line = f.readline().strip()
            if line == 'END':
                break
            label, value = [s.strip() for s in line.split('=')]
            
            if value == '"':
                value = ''
                while True:
                    line = f.readline()
                    if line.strip() == '"':
                        break
                    value += line

            try:
                labels[label] = int(value)
            except:
                labels[label] = value
            print label, value
        rest_of_file = f.read()
    
    if labels['SAMPLE_BITS'] == 8:
        dtype, sample_size = np.uint8, 1
    elif labels['SAMPLE_BITS'] == 16:
        dtype, sample_size = np.uint16, 2
    elif labels['SAMPLE_BITS'] == 32:
        dtype, sample_size = np.uint32, 4
    else:
        raise NotImplemented('Sample size: %d not implemented' % labels['SAMPLE_BITS'])
    
    h, w = labels['IMAGE_LINES'], labels['LINE_SAMPLES']
    array = np.fromstring(rest_of_file[-h*w*sample_size:], dtype=dtype).reshape((h, w))
    return array


class NoiseModel(object):
    """The data model of the client."""

    def __init__(self, QE=0.3, F=20000, B=8, DN_mean=6, DN_sigma=2, t=10):
        """
        Noise model that simulates the creation of an image in a 'real' camera.
        
        Parameters:
        -----------
        QE: float, (default=0.3)
            Quantum efficiency
        
        F:  int, (default=20000)
            Photon
        
        DN_mean - Temporal dark noise mean
        
        DN_sigma - Temporal dark noise sigma
        """
        self.QE = QE
        self.F = F
        self.B = int(B)
        self.DN_mean = int(DN_mean)
        self.DN_sigma = int(DN_sigma)
        self.t = t
        self.alpha = 2**self.B/self.F
        
    def processNoise(self, photons):
        
        #
        # Calculate the number of photons
        #
        photons = photons.astype(np.float) * self.t
        photons = np.random.poisson(photons)
        photons[photons>self.F] = self.F
        
        #
        # Convert to electrons and add the dark noise
        # based on Baer, Richard L. "A model for dark current characterization and simulation."
        # Electronic Imaging 2006. International Society for Optics and Photonics, 2006.
        #
        electrons = self.QE*photons
        DN_noise = np.random.lognormal(
            mean=np.log(self.DN_mean),
            sigma=np.log(self.DN_sigma),
            size=electrons.shape
            ).astype(np.uint)
        electrons += DN_noise
        
        #
        # Convert to gray level
        #
        g = electrons * self.alpha
        
        #
        # Quantisize
        #
        g_q = np.floor(g)
        g_q[g_q>2**self.B] = 2**self.B

        return g_q
    
    def processNoNoise(self, photons):
        
        #
        # Calculate the number of photons
        #
        photons = photons.astype(np.float) * self.t
        
        #
        # Convert to electrons and add the dark noise
        # based on Baer, Richard L. "A model for dark current characterization and simulation."
        # Electronic Imaging 2006. International Society for Optics and Photonics, 2006.
        #
        electrons = self.QE*photons
        
        #
        # Convert to gray level
        #
        g = electrons * self.alpha
        
        return g


if __name__ == '__main__':
    pass
