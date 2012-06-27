"""
Simulate the scattering of the sky where the aerosols have a general distribution.
"""

from __future__ import division
import matplotlib.pyplot as plt
import numpy
from atmo_utils import calcHG, L_SUN_RGB, RGB_WAVELENGTH, spdiag
import atmo_utils
import os.path
import pickle
import amitibo


SKY_PARAMS = {
    'width': 80,
    'height': 40,
    'dxh': 10,
    'camera_center': (40, 40, 2),
    'camera_dist_res': 20,
    'camera_angle_res': 20,
    'sun_angle': 0/180*numpy.pi,
    'L_SUN_RGB': L_SUN_RGB,
    'RGB_WAVELENGTH': RGB_WAVELENGTH
}

VISIBILITY = 10


def calcScatterAngle(R, PHI, THETA, rotation):
    """
    """

    X_ = R * numpy.sin(THETA) * numpy.cos(PHI)
    Y_ = R * numpy.sin(THETA) * numpy.sin(PHI)
    Z_ = R * numpy.cos(THETA)

    XYZ_dst = numpy.vstack((X_.ravel(), Y_.ravel(), Z_.ravel(), numpy.ones(R.size)))
    XYZ_src_ = numpy.dot(rotation, XYZ_dst)

    Z_ = XYZ_src_[2, :]
    R_ = numpy.sqrt(numpy.sum(XYZ_src_[:3, :]**2, axis=0))

    angle = numpy.arccos(Z_/(R_+amitibo.eps(R_)))

    return numpy.pi-angle

    
def calcOpticalDistancesMatrix(Y, X, Z, sun_angle, H_pol, R, PHI, THETA):

    #
    # Prepare transformation matrices
    #
    Hrot_forward, rotation, Y_rot, X_rot, Z_rot = atmo_utils.rotation3DTransformMatrix(Y, X, Z, rotation=(0, sun_angle, 0))
    Hrot_backward = atmo_utils.rotation3DTransformMatrix(Y_rot, X_rot, Z_rot, numpy.linalg.inv(rotation), Y, X, Z)[0]
    
    Hint1 = atmo_utils.cumsumTransformMatrix((Y_rot, X_rot, Z_rot), axis=2)
    Hint2 = atmo_utils.cumsumTransformMatrix((R, PHI, THETA), axis=0, direction=-1)

    temp1 = H_pol * Hrot_backward * Hint1 * Hrot_forward
    temp2 = Hint2 * H_pol

    scatter_angle = calcScatterAngle(R, PHI, THETA, rotation)
    
    return temp2 + temp1, scatter_angle


def calcRadianceHelper(
        ATMO_aerosols_,
        ATMO_air_,
        Y,
        X,
        H,
        aerosol_params,
        sky_params,
        camera_center,
        added_noise=0
        ):
    
    H_pol, R, PHI, THETA = atmo_utils.sphericalTransformMatrix(
        Y,
        X,
        H,
        camera_center,
        radius_res=sky_params['camera_dist_res'],
        angle_res=sky_params['camera_angle_res']
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
    H_camera = atmo_utils.cameraTransformMatrix(PHI[0, :, :], THETA[0, :, :], focal_ratio=0.15)
    
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
        exp_aerosols = numpy.exp(-extinction_aerosols * H_distances * ATMO_aerosols_)
        
        extinction_air = 1.09e-3 * lambda_**-4.05
        scatter_air = extinction_air * (1 + numpy.cos(scatter_angle)**2) / (2*numpy.pi) * (H_pol * ATMO_air_)
        exp_air = numpy.exp(-extinction_air * H_distances * ATMO_air_)
        
        #
        # Calculate the radiance
        #
        radiance = (scatter_air + scatter_aerosols) * exp_air * exp_aerosols

        #
        # Calculate projection on camera
        #
        temp_img = L_sun * H_camera * H_int * radiance
        temp_img = temp_img + added_noise*temp_img.std()*numpy.random.randn(*temp_img.shape)
        temp_img[temp_img<0] = 0
        img.append(temp_img.reshape(256, 256))
        
    return img


def calcRadiance(aerosol_params, sky_params, results_path='', plot_results=False):

    #
    # Create the sky
    #
    Y, X, H = numpy.mgrid[0:sky_params['width']:sky_params['dxh'], 0:sky_params['width']:sky_params['dxh'], 0:sky_params['height']:sky_params['dxh']]

    #
    # Create the distributions of air and aerosols
    #
    #ATMO_aerosols = numpy.exp(-H/aerosol_params["aerosols_typical_h"])
    ATMO_aerosols = numpy.zeros(H.shape)
    ATMO_aerosols_ = ATMO_aerosols.reshape((-1, 1))
    
    ATMO_air = numpy.exp(-H/aerosol_params["air_typical_h"])
    ATMO_air_ = ATMO_air.reshape((-1, 1))

    #
    # Using the helper to do the actual calculations.
    #
    img = calcRadianceHelper(
        ATMO_aerosols_,
        ATMO_air_,
        Y,
        X,
        H,
        aerosol_params,
        sky_params,
        sky_params['camera_center']
        )

    #
    # Create the image
    #
    img = numpy.transpose(numpy.array(img), (1, 2, 0))

    if plot_results:
        #
        # Account for gamma correction
        #
        IMG = img**0.45
        IMG /= numpy.max(IMG)
    
        #
        # Plot results
        #
        fig = plt.figure()
        plt.imshow(IMG, interpolation='nearest')

        amitibo.saveFigures(results_path, (fig, ), bbox_inches='tight')

        plt.show()

    return img
    

def calcRadianceGradientHelper(ATMO_aerosols_, ATMO_air_, X, H, aerosol_params, sky_params, camera_center):

    ATMO_aerosols_ = ATMO_aerosols_.reshape((-1, 1))
    
    H_pol, T, R = atmo_utils.polarTransformMatrix(
        X,
        H,
        camera_center,
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

    H_int = atmo_utils.integralTransformMatrix((R, T), axis=0)
        
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
        scatter_aerosols = spdiag(w * extinction_aerosols * calcHG(scatter_angle, g))
        exp_aerosols = numpy.exp(-extinction_aerosols * H_aerosols * ATMO_aerosols_)
        exp_aerosols_grad = -extinction_aerosols * H_aerosols.T * spdiag(exp_aerosols)

        #
        # Calculate scattering and extiniction for air (wave length dependent)
        #
        extinction_air = 1.09e-3 * lambda_**-4.05
        scatter_air = spdiag(extinction_air * (1 + numpy.cos(scatter_angle)**2) / (2*numpy.pi))
        exp_air = numpy.exp(-extinction_air * H_air * ATMO_air_)

        #
        # Calculate the radiance
        #
        temp1 = (H_pol.T * spdiag(exp_aerosols) + exp_aerosols_grad * spdiag(H_pol * ATMO_aerosols_)) * scatter_aerosols
        temp2 = exp_aerosols_grad * spdiag(H_pol * ATMO_air_) * scatter_air
        radiance = (temp1 + temp2) * spdiag(exp_air)

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

    sun_angle_range = numpy.linspace(0, numpy.pi/2, 16)
    
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

    for job in jobs:
        img = job()
        #
        # Account for gamma correction
        #
        IMG = img**0.45
        IMG /= numpy.max(IMG)
    
        #
        # Plot results
        #
        fig = plt.figure()
        plt.imshow(IMG, interpolation='nearest')

        amitibo.saveFigures(results_path, (fig, ), bbox_inches='tight')



def main_serial(aerosol_params, sky_params, results_path=''):
    """Run the calculation on a single parameters set."""

    calcRadiance(aerosol_params, sky_params, results_path, plot_results=True)
    

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

    main_parallel(aerosol_params, SKY_PARAMS, results_path)
    #main_serial(aerosol_params, SKY_PARAMS, results_path)

