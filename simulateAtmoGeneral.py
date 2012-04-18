"""
Simulate the scattering of the sky where the aerosols have a general distribution.
"""

from __future__ import division
from mpl_toolkits.mplot3d import axes3d
import scipy.interpolate
import matplotlib.pyplot as plt
import numpy
from atmo_utils import calcHG, L_SUN_RGB, RGB_WAVELENGTH
import atmo_utils
import os.path
import pickle
import math
import amitibo
import IPython


SKY_PARAMS = {
    'width': 200,
    'height': 50,
    'dxh': 1,
    'camera_center': (80, 2),
    'sun_angle': -45/180*math.pi,
    'L_SUN_RGB': L_SUN_RGB,
    'RGB_WAVELENGTH': RGB_WAVELENGTH
}

VISIBILITY = 10


def calcOpticalDistancesMatrix(X, Y, ATMO, sun_angle, Hpol):

    #
    # Prepare transformation matrices
    #
    src_shape = ATMO.shape
    Hrot_forward = atmo_utils.rotationTransform(X, Y, angle=sun_angle)
    Hrot_backward = \
      atmo_utils.rotationTransform(Hrot_forward.X_dst, Hrot_forward.Y_dst, -sun_angle, X, Y)
    
    Hint1 = atmo_utils.integralTransform(Hrot_forward.X_dst, Hrot_forward.Y_dst)
    Hint2 = atmo_utils.integralTransform(Hpol.X_dst, Hpol.Y_dst, direction=-1)

    #
    # Apply transform matrices to calculate the path up to the
    # pixel
    #
    ATMO_rotated = Hrot_forward(ATMO)
    temp1 = Hint1(ATMO_rotated)
    ATMO_to = Hrot_backward(temp1)
    ATMO_to_polar = Hpol(ATMO_to)

    #
    # Apply transform matrices to calculate the path from the
    # pixel
    #
    ATMO_polar = Hpol(ATMO)
    ATMO_from_polar = Hint2(ATMO_polar)
    
    return ATMO_polar, ATMO_to_polar, ATMO_from_polar


def calcRadiance(aerosol_params, sky_params, results_path='', plot_results=False):

    #
    # Create the sky
    #
    X, H = numpy.meshgrid(
        numpy.arange(0, sky_params['width'], sky_params['dxh']),
        numpy.arange(0, sky_params['height'], sky_params['dxh'])[::-1]
        )

    #
    # Create the distributions of air and aerosols
    #
    #ATMO_aerosols = numpy.zeros_like(H, dtype=numpy.float64)
    ATMO_aerosols = numpy.exp(-H/aerosol_params["aerosols_typical_h"])
    ATMO_aerosols[:, :int(H.shape[1]/2)] = 0

    #ATMO_air = numpy.zeros_like(H, dtype=numpy.float64)
    ATMO_air = numpy.exp(-H/aerosol_params["air_typical_h"])

    #
    # Calculate a mask over the atmosphere
    # Note:
    # The mask is used to maskout in the polar axis,
    # pixels that are not in the cartesian axis.
    # I set the boundary rows and columns to 0 so that when converting
    # from cartisian to polar coords the interpolation will not 'create'
    # atmosphere above the sky.
    #
    mask = numpy.ones(X.shape)
    mask[:4, :] = 0
    mask[:, :4] = 0
    mask[:, -4:] = 0

    Hpol = atmo_utils.polarTransform(X, H, sky_params['camera_center'])
    mask_polar = Hpol(mask)
    
    ATMO_aerosols *= mask
    ATMO_air *= mask
    
    #
    # Calculate the distances
    #
    ATMO_aerosols_polar, ATMO_aerosols_to_polar, ATMO_aerosols_from_polar = \
        calcOpticalDistancesMatrix(
            X,
            H,
            ATMO_aerosols,
            sky_params['sun_angle'],
            Hpol
            )
    ATMO_air_polar, ATMO_air_to_polar, ATMO_air_from_polar = \
        calcOpticalDistancesMatrix(
            X,
            H,
            ATMO_air,
            sky_params['sun_angle'],
            Hpol
            )
    
    #
    # Calculate scattering angle
    #
    grid_PHI = Hpol.X_dst
    scatter_angle = sky_params['sun_angle'] + grid_PHI + numpy.pi/2

    #
    # Calculate scattering for each channel (in case of the railey scattering)
    #
    img = []
    for L_sun, lambda_, k, w, g in zip(
            sky_params["L_SUN_RGB"],
            sky_params["RGB_WAVELENGTH"],
            aerosol_params["k_RGB"],
            aerosol_params["w_RGB"],
            aerosol_params["g_RGB"]
            ):
        #
        # Calculate scattering and extiniction for air (wave length dependent)
        #
        extinction_aerosol = k / aerosol_params["visibility"]
        scatter_aerosol = extinction_aerosol * calcHG(scatter_angle, g) * ATMO_aerosols_polar * w
        
        extinction_air = 1.09e-3 * lambda_**-4.05
        scatter_air = extinction_air * (1 + numpy.cos(scatter_angle)**2) / (2*numpy.pi) * ATMO_air_polar
        
        #
        # Calculate the radiance
        #
        temp = extinction_aerosol*(ATMO_aerosols_to_polar + ATMO_aerosols_from_polar) + \
          extinction_air*(ATMO_air_to_polar + ATMO_air_from_polar)
        radiance = (scatter_aerosol + scatter_air) * numpy.exp(-temp) * mask_polar

        #
        # Calculate projection on camera
        #
        img.append(L_sun * numpy.sum(radiance, axis=0))

    if plot_results:
        #
        # Create the image
        #
        IMG = numpy.transpose(numpy.array(img, ndmin=3), (2, 0, 1))
        IMG = numpy.tile(IMG, (1, IMG.shape[0], 1))

        #
        # Account for gamma correction
        #
        IMG **= 0.45

        #
        # Plot results
        #
        fig1 = plt.figure()
        plt.subplot(331)
        plt.imshow(ATMO_air, interpolation='nearest', cmap='gray')
        plt.title('ATMO_air\nmax:%g' % numpy.max(ATMO_air))
        plt.subplot(332)
        plt.imshow(ATMO_aerosols, interpolation='nearest', cmap='gray')
        plt.title('ATMO_aerosols\nmax:%g' % numpy.max(ATMO_aerosols))
        plt.subplot(333)
        plt.imshow(mask_polar, interpolation='nearest', cmap='gray')
        plt.title('mask_polar\nmax:%g' % numpy.max(mask_polar))
        plt.subplot(334)
        plt.imshow(ATMO_air_to_polar, interpolation='nearest', cmap='gray')
        plt.title('ATMO_air_to_polar\nmax:%g' % numpy.max(ATMO_air_to_polar))
        plt.subplot(335)
        plt.imshow(ATMO_air_from_polar, interpolation='nearest', cmap='gray')
        plt.title('ATMO_air_from_polar\nmax:%g' % numpy.max(ATMO_air_from_polar))
        plt.subplot(336)
        plt.imshow(ATMO_aerosols_to_polar, interpolation='nearest', cmap='gray')
        plt.title('ATMO_aerosols_to_polar\nmax:%g' % numpy.max(ATMO_aerosols_to_polar))
        plt.subplot(337)
        plt.imshow(ATMO_aerosols_from_polar, interpolation='nearest', cmap='gray')
        plt.title('ATMO_aerosols_from_polar\nmax:%g' % numpy.max(ATMO_aerosols_from_polar))
        plt.subplot(338)
        plt.imshow(radiance, interpolation='nearest', cmap='gray')
        plt.title('radiance\nmax:%g' % numpy.max(radiance))
        plt.subplot(339)
        plt.imshow(IMG/numpy.max(IMG), interpolation='nearest')
        plt.title('IMG\nmax:%g' % numpy.max(IMG))

        IMG_scaled = IMG / numpy.max(IMG)
        h = int(IMG_scaled.shape[0] / 2)

        fig2 = plt.figure()
        plt.subplot(211)
        extent = (0, 1, 90, 0)
        plt.imshow(IMG_scaled[h:0:-1, ...], aspect=1/270, extent=extent, interpolation='nearest')
        plt.xticks([0, 0.5, 1.0])
        plt.yticks([0, 30, 60, 90])
        plt.title('Visibility Parameter %d' % aerosol_params["visibility"])

        plt.subplot(212)
        extent = (0, 1, -90, 0)
        plt.imshow(IMG_scaled[h:, ...], aspect=1/270, extent=extent, interpolation='nearest')
        plt.xticks([0, 0.5, 1.0])
        plt.yticks([0, -30, -60, -90])

        amitibo.saveFigures(results_path, (fig1, fig2), bbox_inches='tight')

        plt.show()

    return img
    
    
def main_parallel(aerosol_params, sky_params, results_path=''):
    """Run the calculation in parallel on a space of parameters"""

    vis_range = numpy.logspace(-10, 10, 20)
    h_range = numpy.logspace(-10, 10, 20)

    job_server = amitibo.start_jobServer()
    jobs = []

    aerosol_params.pop('aerosols_typical_h')
    for visibility in vis_range:
        aerosol_params['visibility'] = visibility
        temp_jobs = [job_server.submit(
                    calcRadiance,
                    args=(dict(aerosols_typical_h=h, **aerosol_params), sky_params),
                    depfuncs=amitibo.depfuncs(globals()),
                    modules=(
                        'numpy',
                        'scipy.interpolate',
                        'skimage.transform',
                        'math'
                        )
                    ) for h in h_range]
        jobs.append(temp_jobs)

    results = []
    for temp_jobs in jobs:
        results.append([numpy.max(job()) for job in temp_jobs])

 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = numpy.meshgrid(numpy.log10(h_range), numpy.log10(vis_range))
    Z = numpy.array(results)
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
    
    amitibo.saveFigures(results_path, bbox_inches='tight')

    plt.show()


def main_serial(aerosol_params, sky_params, results_path=''):
    """Run the calculation on a single parameters set."""

    calcRadiance(aerosol_params, sky_params, results_path, True)
    

if __name__ == '__main__':

    #
    # Load the MISR database.
    #
    with open('misr.pkl', 'rb') as f:
        misr = pickle.load(f)
    
    particles_list = misr.keys()
    particle = misr[particles_list[0]]
    aerosol_params = {
        "k_RGB": numpy.array(particle['k']) / numpy.max(numpy.array(particle['k'])),#* 10**-12,
        "w_RGB": particle['w'],
        "g_RGB": (particle['g']),
        "visibility": VISIBILITY,
        "air_typical_h": 8,
        "aerosols_typical_h": 8,        
    }

    #
    # Create afolder for results
    #
    results_path = amitibo.createResultFolder(params=[aerosol_params, SKY_PARAMS])

    #main_parallel(aerosol_params, SKY_PARAMS)
    main_serial(aerosol_params, SKY_PARAMS, results_path)
