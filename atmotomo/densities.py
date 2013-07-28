"""
"""

from __future__ import division
import numpy as np
from atmotomo import L_SUN_RGB, RGB_WAVELENGTH
import amitibo
import os
import scipy.ndimage.filters as filters

__all__ = [
    "single_voxel_simulation",
    "front_simulation",
    "clouds_simulation",
    "calcAirMcarats",
    "prepareSimulation",
]

SIMULATION_TEMPLATE_FILE_NAME = 'simulation.jinja'


def exp_dists(
    atmosphere_params,
    aerosols_typical_density
    ):
    """Create exponentially decaying atmospheres"""
    
    #
    # Create the sky
    #
    Y, X, H = atmosphere_params.cartesian_grids.expanded
    width = atmosphere_params.cartesian_grids.closed[0][-1]
    h = np.sqrt((X-width/2)**2 + (Y-width/2)**2 + (atmosphere_params.earth_radius+H)**2) - atmosphere_params.earth_radius
     
    #
    # Create the distributions of air
    #
    air_dist = np.exp(-h/atmosphere_params.air_typical_h)
    
    #
    # Create the distributions of aerosols
    #
    aerosols_dist = aerosols_typical_density*np.exp(-h/atmosphere_params.aerosols_typical_h)
    
    return air_dist, aerosols_dist


def grid_cameras(
    atmosphere_params,
    camera_grid_size
    ):
    """Distribute cameras on a 'noised' grid"""
    
    Y, X, H = atmosphere_params.cartesian_grids.expanded
    width = atmosphere_params.cartesian_grids.closed[0][-1]

    #
    # Create the cameras
    #
    derivs = atmosphere_params.cartesian_grids.derivatives
    dy = derivs[0].ravel()[0]
    dx = derivs[1].ravel()[0]
    dz = derivs[2].ravel()[0]
    camera_X, camera_Y = np.meshgrid(
        np.linspace(0, width, camera_grid_size[0]+2)[1:-1],
        np.linspace(0, width, camera_grid_size[1]+2)[1:-1]
        )
    camera_Y = camera_Y.ravel()
    camera_X = camera_X.ravel()
    
    np.random.seed(0)
    camera_Y += (2*np.random.rand(camera_X.size)-1) * dy/2
    camera_X += (2*np.random.rand(camera_X.size)-1) * dx/2
    camera_Z = np.random.rand(camera_X.size) * dz/2
    
    return camera_Y, camera_X, camera_Z


def base_data(
    atmosphere_params,
    particle_name,
    particle_phase,
    camera_resolution,
    camera_coords,
    camera_type,
    sun_angle_phi,
    sun_angle_theta
    ):
    """Create the data for a basic simulation configuration"""
    
    Y, X, H = atmosphere_params.cartesian_grids.expanded
    width = atmosphere_params.cartesian_grids.closed[0][-1]
    derivs = atmosphere_params.cartesian_grids.derivatives
    dy = derivs[0].ravel()[0]
    dx = derivs[1].ravel()[0]
    dz = derivs[2].ravel()[0]
    
    camera_Y, camera_X, camera_Z = camera_coords
    
    #
    # Store the data
    #
    data = {
        'dy':dy,
        'dx':dx,
        'ny':Y.shape[0],
        'nx':Y.shape[1],
        'nz':Y.shape[2],
        'z_coords':H[0, 0, :],
        'particle_name':particle_name,
        'particle_phase':particle_phase,
        'camera_resolution':camera_resolution,
        'camera_type':camera_type,
        'cameras_num':len(camera_Y),
        'camera_ypos':camera_Y,
        'camera_xpos':camera_X,
        'camera_zpos':camera_Z,
        'sun_angle_phi':sun_angle_phi,
        'sun_angle_theta':sun_angle_theta,
        'sun_intensities':L_SUN_RGB,
        'sun_wavelengths':RGB_WAVELENGTH
    }
    return data


def clouds_simulation(
    atmosphere_params,
    particle_name='spherical_nonabsorbing_2.80',
    particle_phase='HG',
    camera_resolution=(128, 128),
    camera_grid_size=(10, 10),
    camera_type='linear',
    sun_angle_phi=0,
    sun_angle_theta=np.pi/4,
    aerosols_typical_density=10**12,
    cloud_coords=((50000/3, 50000/3, 5000), (50000*2/3, 50000*2/3, 2500),),
    cloud_radiuses=((12000, 12000, 1500/np.sqrt(2)),(16000, 16000, 2000/np.sqrt(2)),)
    ):
    """Simulation made of ellipsoid clouds"""
    
    air_dist, aerosols_dist = exp_dists(atmosphere_params, aerosols_typical_density)
    
    camera_coords = grid_cameras(atmosphere_params, camera_grid_size)
    
    data = base_data(
        atmosphere_params,
        particle_name,
        particle_phase,
        camera_resolution,
        camera_coords,
        camera_type,
        sun_angle_phi,
        sun_angle_theta
    )

    #
    # Create the aerosols dist.
    #
    Y, X, H = atmosphere_params.cartesian_grids.expanded
    width = atmosphere_params.cartesian_grids.closed[0][-1]
    height = atmosphere_params.cartesian_grids.closed[2][-1]
    
    mask = np.zeros_like(aerosols_dist)
    for coords, radiuses in zip(cloud_coords, cloud_radiuses):
        Z = ((Y-coords[0])/radiuses[0])**2 + ((X-coords[1])/radiuses[1])**2 + ((H-coords[2])/radiuses[2])**2
        mask[Z<1] = 1
    
    aerosols_dist *= mask
    
    return data, air_dist, aerosols_dist


def front_simulation(
    atmosphere_params,
    particle_name='spherical_nonabsorbing_2.80',
    particle_phase='HG',
    camera_resolution=(128, 128),
    camera_grid_size=(10, 10),
    camera_type='linear',
    sun_angle_phi=0,
    sun_angle_theta=np.pi/4,
    aerosols_typical_density=10**12,
    cloud1_radius=3000,
    cloud2_radius=4000
    ):
    """A simulation of a cloud front"""
    
    air_dist, aerosols_dist = exp_dists(atmosphere_params, aerosols_typical_density)
    
    camera_coords = grid_cameras(atmosphere_params, camera_grid_size)
    
    data = base_data(
        atmosphere_params,
        particle_name,
        particle_phase,
        camera_resolution,
        camera_coords,
        camera_type,
        sun_angle_phi,
        sun_angle_theta
    )

    #
    # Create the aerosols dist.
    #
    Y, X, H = atmosphere_params.cartesian_grids.expanded
    width = atmosphere_params.cartesian_grids.closed[0][-1]
    height = atmosphere_params.cartesian_grids.closed[2][-1]
    
    mask = np.zeros_like(aerosols_dist)
    Z1 = X**2/64 + (Y-width/2)**2/1000 + (H-height/2)**2
    mask[Z1<4000**2] = 1
    
    aerosols_dist *= mask
    
    return data, air_dist, aerosols_dist


def single_voxel_simulation(
    atmosphere_params,
    particle_name='spherical_nonabsorbing_2.80',
    particle_phase='HG',
    camera_resolution=(128, 128),
    camera_grid_size=(10, 10),
    camera_type='linear',
    sun_angle_phi=0,
    sun_angle_theta=np.pi/4,
    voxel_indices=(0, 0, 0),
    aerosols_typical_density=1,
    ):
    """A simulation of an empty atmosphere with single voxel of aerosols"""
    
    #
    # Create the camera coords
    #
    camera_coords = grid_cameras(atmosphere_params, camera_grid_size)
    
    data = base_data(
        atmosphere_params,
        particle_name,
        particle_phase,
        camera_resolution,
        camera_coords,
        camera_type,
        sun_angle_phi,
        sun_angle_theta
    )

    #
    # Create the aerosols dist.
    #
    air_dist = np.zeros(atmosphere_params.cartesian_grids.shape)
    aerosols_dist = np.zeros(atmosphere_params.cartesian_grids.shape)    
    aerosols_dist[voxel_indices] = aerosols_typical_density

    return data, air_dist, aerosols_dist


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


def prepareSimulation(
    path,
    func,
    smoothing_sigma=0.0,
    *params,
    **kwrds
    ):
    import jinja2
    from amitibo import getResourcePath, BaseData
    import scipy.io as sio
    
    tpl_loader = jinja2.FileSystemLoader(searchpath=getResourcePath('.', package_name=__name__))
    tpl_env = jinja2.Environment(loader=tpl_loader)
    
    BaseData._tpl_env = tpl_env
    simulation = BaseData(template_path=SIMULATION_TEMPLATE_FILE_NAME)
    
    #
    # Create the path if necessary
    #
    if not os.path.isdir(path):
        os.makedirs(path)
        
    #
    # Create the simulation data using the func function.
    #
    data, air_dist, aerosols_dist = func(*params, **kwrds)
    
    if smoothing_sigma > 0.0:
        aerosols_dist = filters.gaussian_filter(aerosols_dist, sigma=smoothing_sigma)
        
    sio.savemat(
        os.path.join(path, 'air_dist.mat'),
        {'distribution': air_dist},
        do_compression=True
    )
    simulation.addData(air_dist_path='./air_dist.mat')
    
    sio.savemat(
        os.path.join(path, 'aerosols_dist.mat'),
        {'distribution': aerosols_dist},
        do_compression=True
    )
    simulation.addData(aerosols_dist_path='./aerosols_dist.mat')
    
    simulation.addData(**data)
    
    with open(os.path.join(path, 'configuration.ini'), 'w') as f:
        f.write(simulation.render())
    

if __name__ == '__main__':
    pass

    
    
