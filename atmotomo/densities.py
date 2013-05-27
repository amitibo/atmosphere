"""
"""

from __future__ import division
import numpy as np
from atmotomo import L_SUN_RGB, RGB_WAVELENGTH
import sparse_transforms as spt
import amitibo
import os

__all__ = [
    "density_front",
    "clouds_simulation",
    "density_clouds_vadim",
    "single_cloud_vadim",
    "calcAirMcarats",
    "single_voxel_atmosphere",
    "prepareSimulation"
]

SIMULATION_TEMPLATE_FILE_NAME = 'simulation.jinja'

clouds_atmosphere = amitibo.attrClass(
    cartesian_grids=spt.Grids(
        np.arange(0, 50000, 1000.0), # Y
        np.arange(0, 50000, 1000.0), # X
        np.arange(0, 10000, 100.0)   # H
        ),
    earth_radius=4000000,
    air_typical_h=8000,
    aerosols_typical_h=2000
)


def density_front(atmosphere_params):
    #
    # Create the sky
    #
    Y, X, H = atmosphere_params.cartesian_grids.expanded
    width = atmosphere_params.cartesian_grids.closed[0][-1]
    height = atmosphere_params.cartesian_grids.closed[2][-1]
    h = np.sqrt((X-width/2)**2 + (Y-width/2)**2 + (atmosphere_params.earth_radius+H)**2) - atmosphere_params.earth_radius
    
    #
    # Create the distributions of air
    #
    A_air = np.exp(-h/atmosphere_params.air_typical_h)
    
    #
    # Create the distributions of aerosols
    #
    A_aerosols = np.exp(-h/atmosphere_params.aerosols_typical_h)
    A_mask = np.zeros_like(A_aerosols)
    Z1 = X**2/64 + (Y-width/2)**2/1000 + (H-height/2)**2
    A_mask[Z1<4000**2] = 1
    A_aerosols *= A_mask
    
    return A_air, A_aerosols, Y, X, H, h


def clouds_simulation(
    atmosphere_params=clouds_atmosphere,
    particle_name='spherical_nonabsorbing_2.80',
    particle_phase='HG',
    camera_resolution=(128, 128),
    camera_type='linear',
    sun_angle=(0, -np.pi/4),
    visibility=100000
    ):

    #
    # Create the sky
    #
    Y, X, H = atmosphere_params.cartesian_grids.expanded
    width = atmosphere_params.cartesian_grids.closed[0][-1]
    height = atmosphere_params.cartesian_grids.closed[2][-1]
    h = np.sqrt((X-width/2)**2 + (Y-width/2)**2 + (atmosphere_params.earth_radius+H)**2) - atmosphere_params.earth_radius
    derivs = atmosphere_params.cartesian_grids.derivatives
    
    #
    # Create the distributions of air
    #
    air_dist = np.exp(-h/atmosphere_params.air_typical_h)
    
    #
    # Create the distributions of aerosols
    #
    aerosols_dist = np.exp(-h/atmosphere_params.aerosols_typical_h)
    mask = np.zeros_like(aerosols_dist)
    Z1 = (X-width/3)**2/16 + (Y-width/3)**2/16 + (H-height/2)**2*8
    Z2 = (X-width*2/3)**2/16 + (Y-width*2/3)**2/16 + (H-height/4)**2*8
    mask[Z1<3000**2] = 1
    mask[Z2<4000**2] = 1
    aerosols_dist *= mask
    aerosols_dist /= visibility
    
    #
    # Create the cameras
    #
    camera_X, camera_Y = np.meshgrid(
        np.linspace(0, width, 12)[1:-1],
        np.linspace(0, width, 12)[1:-1]
        )
    
    data = {
        'dy':derivs[0].ravel()[0],
        'dx':derivs[1].ravel()[0],
        'ny':Y.shape[0],
        'nx':Y.shape[1],
        'nz':Y.shape[2],
        'z_coords':H[0, 0, :],
        'particle_name':particle_name,
        'particle_phase':particle_phase,
        'camera_resolution':camera_resolution,
        'camera_type':camera_type,
        'cameras_num':100,
        'camera_ypos':camera_Y.ravel(),
        'camera_xpos':camera_X.ravel(),
        'camera_zpos':np.ones(camera_X.size),
        'sun_angle':sun_angle,
        'sun_intensities':L_SUN_RGB,
        'sun_wavelengths':RGB_WAVELENGTH
    }

    return data, air_dist, aerosols_dist


def density_clouds_vadim(atmosphere_params):
    #
    # Create the sky
    #
    Y, X, H = atmosphere_params.cartesian_grids.expanded
    width = atmosphere_params.cartesian_grids.closed[0][-1]
    height = atmosphere_params.cartesian_grids.closed[2][-1]
    h = np.sqrt((X-width/2)**2 + (Y-width/2)**2 + (atmosphere_params.earth_radius+H)**2) - atmosphere_params.earth_radius

    #
    # Create the distributions of air
    #
    A_air = np.exp(-h/atmosphere_params.air_typical_h)
    
    #
    # Create the distributions of aerosols
    #
    A_aerosols = np.exp(-h/atmosphere_params.aerosols_typical_h)
    A_mask = np.zeros_like(A_aerosols)
    Z1 = (-Y+width*2/3)**2/16 + (X-width/3)**2/16 + (H-height/2)**2*8
    Z2 = (-Y+width/3)**2/16 + (X-width*2/3)**2/16 + (H-height/4)**2*8
    A_mask[Z1<3000**2] = 1
    A_mask[Z2<4000**2] = 1
    A_aerosols *= A_mask

    return A_air, A_aerosols, Y, X, H, h


def single_cloud_vadim(atmosphere_params):
    #
    # Create the sky
    #
    Y, X, H = atmosphere_params.cartesian_grids.expanded
    width = atmosphere_params.cartesian_grids.closed[0][-1]
    height = atmosphere_params.cartesian_grids.closed[2][-1]
    h = np.sqrt((X-width/2)**2 + (Y-width/2)**2 + (atmosphere_params.earth_radius+H)**2) - atmosphere_params.earth_radius

    #
    # Create the distributions of air
    #
    A_air = np.exp(-h/atmosphere_params.air_typical_h)
    
    #
    # Create the distributions of aerosols
    #
    A_aerosols = np.exp(-h/atmosphere_params.aerosols_typical_h)
    A_mask = np.zeros_like(A_aerosols)
    Z = (-Y+width/3)**2/16 + (X-width*2/3)**2/16 + (H-height/4)**2*8
    A_mask[Z<4000**2] = 1
    A_aerosols *= A_mask

    return A_air, A_aerosols, Y, X, H, h


def single_voxel_atmosphere(atmosphere_params, indices_list=[(0, 0, 0)], density=0.001, decay=False):
    #
    # Create the sky
    #
    Y, X, H = atmosphere_params.cartesian_grids.expanded
    width = atmosphere_params.cartesian_grids.closed[0][-1]
    
    A_aerosol_base = density
    if decay:
        h = np.sqrt((X-width/2)**2 + (Y-width/2)**2 + H**2)
        A_aerosol_base = A_aerosol_base * np.exp(-h/atmosphere_params.aerosols_typical_h)

    #
    # Create the distributions of aerosols
    #
    A_aerosols = []
    for voxel_indices in indices_list:
        tmp = np.zeros_like(H)
        tmp[voxel_indices] = 1
        A_aerosols.append(tmp * A_aerosol_base)
        
    return A_aerosols, Y, X, H


def fitExp(Z, z0, e0):
    """Interpolate new values to an exponent"""
    
    c = np.polyfit(z0, np.log(e0), 1)
    e = np.exp(c[0]*Z + c[1])
    
    return e


def calcAirMcarats(Z):
    """Calculate the air molecules density according to the mcarats files conf_ci0045-67"""
    
    z_mcarats = np.array((0.000000E+00,  3.000000E+01,  6.000000E+01,  9.000000E+01,  1.200000E+02,  1.500000E+02,  1.800000E+02,  2.100000E+02,  2.300000E+02,  2.500000E+02,  2.600000E+02,  2.700000E+02,  2.800000E+02,  2.900000E+02,  3.000000E+02,  3.100000E+02,  3.200000E+02,  3.300000E+02,  3.400000E+02,  3.500000E+02,  3.600000E+02,  3.700000E+02,  3.800000E+02,  3.900000E+02,  4.000000E+02,  4.100000E+02,  4.200000E+02,  4.300000E+02,  4.400000E+02,  4.500000E+02,  4.600000E+02,  6.000000E+02,  1.000000E+03,  2.000000E+03,  3.000000E+03,  4.000000E+03,  6.000000E+03,  8.000000E+03,  1.000000E+04,  1.500000E+04,  2.000000E+04,  3.000000E+04))
    z_mid = (z_mcarats[1:] + z_mcarats[:-1])/2
    ext_0045 = np.array((2.45113E-05,  2.44561E-05,  2.43958E-05,  2.43346E-05,  2.42734E-05,  2.42117E-05,  2.41498E-05,  2.40980E-05,  2.40568E-05,  2.40256E-05,  2.40046E-05,  2.39831E-05,  2.39609E-05,  2.39383E-05,  2.39154E-05,  2.38922E-05,  2.38690E-05,  2.38456E-05,  2.38222E-05,  2.37988E-05,  2.37755E-05,  2.37520E-05,  2.37279E-05,  2.37015E-05,  2.36700E-05,  2.36279E-05,  2.35715E-05,  2.35065E-05,  2.34441E-05,  2.33933E-05,  2.31618E-05,  2.24001E-05,  2.07798E-05,  1.87793E-05,  1.69788E-05,  1.45587E-05,  1.17577E-05,  9.41149E-06,  6.07198E-06,  2.91507E-06,  9.61213E-07))
    ext_0055 = np.array((1.09831E-05,  1.09583E-05,  1.09313E-05,  1.09039E-05,  1.08765E-05,  1.08488E-05,  1.08211E-05,  1.07979E-05,  1.07794E-05,  1.07655E-05,  1.07561E-05,  1.07464E-05,  1.07365E-05,  1.07263E-05,  1.07161E-05,  1.07057E-05,  1.06953E-05,  1.06848E-05,  1.06743E-05,  1.06639E-05,  1.06534E-05,  1.06429E-05,  1.06321E-05,  1.06202E-05,  1.06061E-05,  1.05873E-05,  1.05620E-05,  1.05329E-05,  1.05049E-05,  1.04821E-05,  1.03784E-05,  1.00371E-05,  9.31107E-06,  8.41467E-06,  7.60792E-06,  6.52350E-06,  5.26843E-06,  4.21713E-06,  2.72075E-06,  1.30620E-06,  4.30703E-07))
    ext_0067 = np.array((4.98723E-06,  4.97600E-06,  4.96373E-06,  4.95129E-06,  4.93883E-06,  4.92627E-06,  4.91369E-06,  4.90315E-06,  4.89475E-06,  4.88841E-06,  4.88414E-06,  4.87976E-06,  4.87525E-06,  4.87065E-06,  4.86598E-06,  4.86127E-06,  4.85654E-06,  4.85179E-06,  4.84703E-06,  4.84227E-06,  4.83752E-06,  4.83275E-06,  4.82783E-06,  4.82246E-06,  4.81606E-06,  4.80749E-06,  4.79602E-06,  4.78279E-06,  4.77009E-06,  4.75975E-06,  4.71266E-06,  4.55768E-06,  4.22799E-06,  3.82096E-06,  3.45462E-06,  2.96221E-06,  2.39230E-06,  1.91492E-06,  1.23545E-06,  5.93121E-07,  1.95575E-07))

    e45 = fitExp(Z, z_mid, ext_0045)
    e55 = fitExp(Z, z_mid, ext_0055)
    e67 = fitExp(Z, z_mid, ext_0067)
    
    return (e67, e55, e45)


def prepareSimulation(path, func, *params, **kwrds):
    import jinja2
    from amitibo import getResourcePath, BaseData
    import scipy.io as sio
    
    tpl_loader = jinja2.FileSystemLoader(searchpath=getResourcePath('.', package_name=__name__))
    tpl_env = jinja2.Environment(loader=tpl_loader)
    
    BaseData._tpl_env = tpl_env
    simulation = BaseData(template_path=SIMULATION_TEMPLATE_FILE_NAME)
    
    data, air_dist, aerosols_dist = func(*params, **kwrds)
    
    sio.savemat(
        os.path.join(path, 'air_dist.mat'),
        {'distribution': air_dist},
        do_compression=True
    )
    simulation.addData(air_dist_path=os.path.join(path, 'air_dist.mat'))
    
    sio.savemat(
        os.path.join(path, 'aerosols_dist.mat'),
        {'distribution': aerosols_dist},
        do_compression=True
    )
    simulation.addData(aerosols_dist_path=os.path.join(path, 'aerosols_dist.mat'))
    
    simulation.addData(**data)
    
    with open(os.path.join(path, 'configuration.ini'), 'w') as f:
        f.write(simulation.render())
    

if __name__ == '__main__':
    pass

    
    