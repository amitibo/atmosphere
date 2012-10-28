"""
Simulate the scattering of the sky where the aerosols have a general distribution.
"""

from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from atmo_utils import calcHG, L_SUN_RGB, RGB_WAVELENGTH, spdiag, viz2D, viz3D
import atmo_utils
from camera import Camera
import amitibo
import mayavi.mlab as mlab
import scipy.io as sio
import os
import warnings
import time


#
# Global settings
#
atmosphere_params = amitibo.attrClass(
    cartesian_grids=(
        slice(0, 400, 4),
        slice(0, 400, 4),
        slice(0, 80, 1)
        ),
    earth_radius=4000,
    L_SUN_RGB=L_SUN_RGB,
    RGB_WAVELENGTH=RGB_WAVELENGTH,
    air_typical_h=8,
    aerosols_typical_h=1.2
)

camera_params = amitibo.attrClass(
    radius_res=40,
    phi_res=80,
    theta_res=80,
    focal_ratio=0.15,
    image_res=512,
    theta_compensation=False
)

profile = False


def calcRadianceGradientHelper(
    ATMO_aerosols_,
    ATMO_air_,
    Y,
    X,
    H,
    aerosol_params,
    sky_params,
    camera_center
    ):
    """
    Helper function that does the actual calculation of the radiance gradient.
    """
    
    ATMO_aerosols_ = ATMO_aerosols_.reshape((-1, 1))
    ATMO_air_ = ATMO_air_.reshape((-1, 1))
    
    H_pol, R, PHI, THETA = atmo_utils.sphericalTransformMatrix(
        Y,
        X,
        H,
        camera_center,
        radius_res=sky_params['radius_res'],
        phi_res=sky_params['phi_res'],
        theta_res=sky_params['theta_res']
        )
    
    #
    # Calculate the distance matrices and scattering angle
    #
    H_distances, scatter_angle = \
        calcOpticalDistancesMatrix(
            Y,
            X,
            H,
            sky_params['sun_angle'],
            H_pol,
            R,
            PHI,
            THETA
            )

    scatter_angle.shape = (-1, 1)
    
    H_int = atmo_utils.integralTransformMatrix((R, PHI, THETA), axis=0)
    H_camera = atmo_utils.cameraTransformMatrix(PHI[0, :, :], THETA[0, :, :], focal_ratio=sky_params['focal_ratio'], image_res=sky_params['image_res'])

    #vizTransforms2(
        #H_pol,
        #H_int,
        #H_camera,
        #ATMO_air_,
        #Y,
        #X,
        #H,
        #R,
        #PHI,
        #THETA
    #)
    
    mu = numpy.cos(scatter_angle)

    #
    # Calculate scattering for each channel (in case of the railey scattering)
    #
    gimg = []
    for L_sun, lambda_, k, w, g in zip(
            sky_params["L_SUN_RGB"],
            sky_params["RGB_WAVELENGTH"],
            aerosol_params["k_RGB"],
            aerosol_params["w_RGB"],
            aerosol_params["g_RGB"]
            ):
        #
        # Calculate scattering and extiniction for aerosols
        #
        extinction_aerosols = k / aerosol_params["visibility"]
        scatter_aerosols_partial = w * extinction_aerosols * calcHG(mu, g)
        scatter_aerosols = spdiag(scatter_aerosols_partial * (H_pol * ATMO_aerosols_))
        scatter_aerosols_grad = H_pol.T * spdiag(scatter_aerosols_partial)
        exp_aerosols = spdiag(numpy.exp(-extinction_aerosols * H_distances * ATMO_aerosols_))
        exp_aerosols_grad = -extinction_aerosols * H_distances.T * exp_aerosols

        #
        # Calculate scattering and extiniction for air (wave length dependent)
        #
        extinction_air = 1.09e-3 * lambda_**-4.05
        scatter_air = spdiag(extinction_air * (1 + mu**2) * 3 / (16*numpy.pi) * (H_pol * ATMO_air_))
        exp_air = spdiag(numpy.exp(-extinction_air * H_distances * ATMO_air_))

        #
        # Calculate the gradient of the radiance
        #
        temp1 = scatter_aerosols_grad * exp_aerosols + exp_aerosols_grad * scatter_aerosols
        temp2 = exp_aerosols_grad * scatter_air
        radiance_gradient = (temp1 + temp2) * exp_air

        #
        # Calculate projection on camera
        #
        gimg.append(L_sun * radiance_gradient * H_int.T * H_camera.T)
        
    return gimg


def calcRadianceGradient(ATMO_aerosols, ATMO_air, aerosol_params, sky_params):
    """
    Calculate the gradient of the radiance at some atmospheric distribution.
    """

    #
    # Rowwise representation of the atmosphere 
    #
    ATMO_aerosols_ = ATMO_aerosols.reshape((-1, 1))
    ATMO_air_ = ATMO_air.reshape((-1, 1))

    #
    # Create the sky
    #
    X, H = numpy.meshgrid(
        numpy.arange(0, sky_params['width'], sky_params['dx']),
        numpy.arange(0, sky_params['height'], sky_params['dh'])[::-1]
        )

    img = calcRadianceGradientHelper(
        ATMO_aerosols_,
        ATMO_air_,
        X,
        H,
        aerosol_params,
        sky_params,
        sky_params['camera_center']
        )

    return img


def main_parallel(aerosol_params, sky_params, results_path=''):
    """Run the calculation in parallel on a space of parameters"""

    sun_angle_range = numpy.linspace(0, numpy.pi/2, 32)
    
    job_server = amitibo.start_jobServer()
    jobs = []

    for sun_angle in sun_angle_range:
        sky_params['sun_angle'] = sun_angle
        jobs.append(
            job_server.submit(
                calcRadiance,
                args=(aerosol_params, sky_params, results_path),
                depfuncs=amitibo.depfuncs(globals()),
                modules=(
                    'numpy',
                    'math',
                    'atmo_utils',
                    'amitibo'
                )
            )
        )

    images = []
    max_values = []
    for job in jobs:
        img = job()
        max_values.append(numpy.max(img))
        images.append(img)

    max_value = numpy.max(numpy.array(max_values))
    figures = []
    for img, sun_angle in zip(images, sun_angle_range):
        #
        # Account for gamma correction
        #
        img /= max_value
        img = img**0.45
    
        #
        # Plot results
        #
        fig = draw_image(img, sky_params['focal_ratio'], sun_angle)
        amitibo.saveFigures(results_path, (fig, ), bbox_inches='tight')
        figures.append(fig)

    pdf = amitibo.figuresPdf(os.path.join(results_path, 'report.pdf'))
    pdf.saveFigures(figures)


def serial(particle_params, results_path):
    
    #
    # Create some aerosols distribution
    #
    Y, X, H = np.mgrid[atmosphere_params.cartesian_grids]
    width = atmosphere_params.cartesian_grids[0].stop
    height = atmosphere_params.cartesian_grids[2].stop
    h = np.sqrt((X-width/2)**2 + (Y-width/2)**2 + (atmosphere_params.earth_radius+H)**2) - atmosphere_params.earth_radius
    ATMO_aerosols = np.exp(-h/atmosphere_params.aerosols_typical_h)
    ATMO_air = np.exp(-h/atmosphere_params.air_typical_h)


    for i, sun_angle in enumerate((0,)):#numpy.linspace(0, numpy.pi/2, 4)):
        #
        # Instantiating the camera
        #
        cam = Camera(
            sun_angle,
            atmosphere_params=atmosphere_params,
            camera_params=camera_params,
            camera_position=(width/2, width/2, 0.2)
        )
        
        #
        # Calculating the image
        #
        img = cam.calcImage(A_air=ATMO_air, A_aerosols=ATMO_aerosols, particle_params=particle_params)
        
        sio.savemat(os.path.join(results_path, 'img%d.mat' % i), {'img':img}, do_compression=True)
        

if __name__ == '__main__':

    #
    # Load the MISR database.
    #
    import pickle
    
    with open('misr.pkl', 'rb') as f:
        misr = pickle.load(f)
    
    particles_list = misr.keys()
    particle = misr[particles_list[0]]
    particle_params = amitibo.attrClass(
        k_RGB=np.array(particle['k']) / np.max(np.array(particle['k'])),#* 10**-12,
        w_RGB=particle['w'],
        g_RGB=(particle['g']),
        visibility=10
        )

    #
    # Create afolder for results
    #
    results_path = amitibo.createResultFolder(params=[atmosphere_params, particle_params, camera_params])

    if profile:
        import cProfile    
        cmd = "serial(SKY_PARAMS, aerosol_params, results_path)"
        cProfile.runctx(cmd, globals(), locals(), filename="atmosphere_camera.profile")
    else:
        serial(particle_params, results_path)