"""
Simulate the scattering of the sky where the aerosols have a general distribution.
"""

from __future__ import division
from mpl_toolkits.mplot3d import axes3d
import scipy.interpolate
import matplotlib.pyplot as plt
import numpy
from atmo_utils import calcHG, L_SUN_RGB, RGB_WAVELENGTH, spdiag
import atmo_utils
import os.path
import pickle
import math
import amitibo
import IPython


SKY_PARAMS = {
    'width': 50,
    'height': 20,
    'dxh': 1,
    'camera_center': (80, 2),
    'camera_dist_res': 100,
    'camera_angle_res': 100,
    'sun_angle': -45/180*math.pi,
    'L_SUN_RGB': L_SUN_RGB,
    'RGB_WAVELENGTH': RGB_WAVELENGTH
}

VISIBILITY = 10


def calcOpticalDistancesTransform(X, Y, ATMO, sun_angle, H_pol, T, R):

    ATMO_ = ATMO.reshape((-1, 1))
    
    #
    # Prepare transformation matrices
    #
    Hrot_forward, X_dst, Y_dst = atmo_utils.rotationTransformMatrix(X, Y, angle=sun_angle)
    Hrot_backward = atmo_utils.rotationTransformMatrix(X_dst, Y_dst, -sun_angle, X, Y)[0]
    
    Hint1 = atmo_utils.cumsumTransformMatrix(X_dst, Y_dst)
    Hint2 = atmo_utils.cumsumTransformMatrix(T, R, direction=-1)

    #
    # Apply transform matrices to calculate the path up to the
    # pixel
    #
    ATMO_rotated = Hrot_forward * ATMO_
    temp1 = Hint1 * ATMO_rotated
    ATMO_to = Hrot_backward * temp1
    ATMO_to_polar = H_pol * ATMO_to

    #
    # Apply transform matrices to calculate the path from the
    # pixel
    #
    ATMO_polar = H_pol * ATMO_
    ATMO_from_polar = Hint2 * ATMO_polar
    
    return ATMO_polar.reshape(T.shape), ATMO_to_polar.reshape(T.shape), ATMO_from_polar.reshape(T.shape)


def calcOpticalDistancesMatrix(X, Y, sun_angle, H_pol, T, R):

    #
    # Prepare transformation matrices
    #
    Hrot_forward, X_dst, Y_dst = atmo_utils.rotationTransformMatrix(X, Y, angle=sun_angle)
    Hrot_backward = atmo_utils.rotationTransformMatrix(X_dst, Y_dst, -sun_angle, X, Y)[0]
    
    Hint1 = atmo_utils.cumsumTransformMatrix(X_dst, Y_dst)
    Hint2 = atmo_utils.cumsumTransformMatrix(T, R, direction=-1)

    temp1 = H_pol * Hrot_backward * Hint1 * Hrot_forward
    temp2 = Hint2 * H_pol

    return temp2 + temp1


def calcRadianceHelper(ATMO_aerosols_, ATMO_air_, X, H, aerosol_params, sky_params):
    
    ATMO_aerosols_ = ATMO_aerosols_.reshape((-1, 1))
    
    #
    # Calculate a mask over the atmosphere
    # Note:
    # The mask is used to maskout in the polar axis,
    # pixels that are not in the cartesian axis.
    # I set the boundary rows and columns to 0 so that when converting
    # from cartisian to polar coords the interpolation will not 'create'
    # atmosphere above the sky.
    #
    H_pol, T, R = atmo_utils.polarTransformMatrix(
        X,
        H,
        sky_params['camera_center'],
        radius_res=sky_params['camera_dist_res'],
        angle_res=sky_params['camera_angle_res']
        )
    
    #
    # Calculate the distance matrices
    #
    H_aerosols = \
        calcOpticalDistancesMatrix(
            X,
            H,
            sky_params['sun_angle'],
            H_pol,
            T,
            R
            )
    
    H_air = \
        calcOpticalDistancesMatrix(
            X,
            H,
            sky_params['sun_angle'],
            H_pol,
            T,
            R
            )

    H_int = atmo_utils.integralTransformMatrix(T, R, axis=0)
        
    #
    # Calculate scattering angle
    #
    scatter_angle = sky_params['sun_angle'] + T.reshape((-1, 1)) + numpy.pi/2

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
        extinction_aerosols = k / aerosol_params["visibility"]
        scatter_aerosols = w * extinction_aerosols * calcHG(scatter_angle, g) * (H_pol * ATMO_aerosols_)
        exp_aerosols = numpy.exp(-extinction_aerosols * H_aerosols * ATMO_aerosols_)
        
        #
        # Calculate the radiance
        #
        radiance = exp_aerosols

        #
        # Calculate projection on camera
        #
        img.append(L_sun * H_int * radiance)
        
    return img


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
    ATMO_aerosols_ = ATMO_aerosols.reshape((-1, 1))
    
    #ATMO_air = numpy.zeros_like(H, dtype=numpy.float64)
    ATMO_air = numpy.exp(-H/aerosol_params["air_typical_h"])
    ATMO_air_ = ATMO_air.reshape((-1, 1))

    #
    # Using the helper to do the actual calculations.
    #
    img = calcRadianceHelper(ATMO_aerosols_, ATMO_air_, X, H, aerosol_params, sky_params)
    
    if plot_results:
        #
        # Create the image
        #
        IMG = numpy.transpose(numpy.array(img, ndmin=3), (1, 2, 0))
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
    

def calcRadianceGradientHelper(ATMO_aerosols_, ATMO_air_, X, H, aerosol_params, sky_params):

    ATMO_aerosols_ = ATMO_aerosols_.reshape((-1, 1))
    
    #
    # Calculate a mask over the atmosphere
    # Note:
    # The mask is used to maskout in the polar axis,
    # pixels that are not in the cartesian axis.
    # I set the boundary rows and columns to 0 so that when converting
    # from cartisian to polar coords the interpolation will not 'create'
    # atmosphere above the sky.
    #
    H_pol, T, R = atmo_utils.polarTransformMatrix(
        X,
        H,
        sky_params['camera_center'],
        radius_res=sky_params['camera_dist_res'],
        angle_res=sky_params['camera_angle_res']
        )
    
    #
    # Calculate the distances
    #
    #
    # Calculate the distance matrices
    #
    H_aerosols = \
        calcOpticalDistancesMatrix(
            X,
            H,
            sky_params['sun_angle'],
            H_pol,
            T,
            R
            )
    
    H_air = \
        calcOpticalDistancesMatrix(
            X,
            H,
            sky_params['sun_angle'],
            H_pol,
            T,
            R
            )

    H_int = atmo_utils.integralTransformMatrix(T, R, axis=0)
        
    #
    # Calculate scattering angle
    #
    scatter_angle = sky_params['sun_angle'] + T + numpy.pi/2

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
        # Calculate scattering and extiniction for aerosols
        #
        extinction_aerosols = k / aerosol_params["visibility"]
        exp_aerosols = numpy.exp(-extinction_aerosols * H_aerosols * ATMO_aerosols_)
        exp_aerosols_grad = -extinction_aerosols * H_aerosols.T * spdiag(exp_aerosols)

        #
        # Calculate the radiance
        #
        radiance = exp_aerosols_grad

        #
        # Calculate projection on camera
        #
        img.append(L_sun * radiance * H_int.T)
        
    return img


def calcRadianceGradient(ATMO_aerosols, ATMO_air, aerosol_params, sky_params):

    #
    # Rowwise representation of the atmosphere 
    #
    ATMO_aerosols_ = ATMO_aerosols.reshape((-1, 1))
    ATMO_air_ = ATMO_air.reshape((-1, 1))

    #
    # Create the sky
    #
    X, H = numpy.meshgrid(
        numpy.arange(0, sky_params['width'], sky_params['dxh']),
        numpy.arange(0, sky_params['height'], sky_params['dxh'])[::-1]
        )

    img = calcRadianceGradientHelper(ATMO_aerosols_, ATMO_air_, X, H, aerosol_params, sky_params)

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
