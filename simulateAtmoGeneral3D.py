"""
Simulate the scattering of the sky where the aerosols have a general distribution.
"""

from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy
from atmo_utils import calcHG, L_SUN_RGB, RGB_WAVELENGTH, spdiag
import atmo_utils
import os.path
import pickle
import amitibo
import mayavi.mlab as mlab


SKY_PARAMS = {
    'width': 400,
    'height': 20,
    'earth_radius': 4000,
    'dx': 8,
    'dh': 1,
    'camera_center': (200, 200, 0.2),
    'radius_res': 40,
    'phi_res': 80,
    'theta_res': 40,
    'image_res': 512,
    'focal_ratio': 0.15,
    'sun_angle': 45/180*numpy.pi,
    'L_SUN_RGB': L_SUN_RGB,
    'RGB_WAVELENGTH': RGB_WAVELENGTH
}

VISIBILITY = 50


def viz3D(X, Y, Z, V):

    mlab.figure()
    
    s = mlab.pipeline.scalar_field(X, Y, Z, V)
    ipw_x = mlab.pipeline.image_plane_widget(s, plane_orientation='x_axes')
    ipw_y = mlab.pipeline.image_plane_widget(s, plane_orientation='y_axes')
    ipw_z = mlab.pipeline.image_plane_widget(s, plane_orientation='z_axes')
    mlab.colorbar()
    mlab.outline()
    mlab.axes()    


def viz2D(V):
    
    plt.figure()    
    plt.imshow(V, interpolation='nearest')
    plt.gray()
    

def vizTransforms(Y, X, Z, H_pol, Hrot_forward, Hrot_backward, Hint1, Hint2, R, Y_rot, scatter_angle):

    ATMO = numpy.exp(-Z/8)
    ATMO_ = ATMO.reshape((-1, 1))
    
    ATMO_pol_= H_pol * ATMO_
    ATMO_rot_ = Hrot_forward * ATMO_
    ATMO_back_ = Hrot_backward * ATMO_rot_
    ATMO_int1_ = Hint1 * ATMO_rot_
    ATMO_int2_ = Hint2 * ATMO_pol_
     
    viz3D(X, Y, Z, ATMO)
    viz3D(ATMO_pol_.reshape(R.shape))
    viz3D(ATMO_rot_.reshape(Y_rot.shape))
    viz3D(ATMO_back_.reshape(Y.shape))
    viz3D(ATMO_int1_.reshape(Y_rot.shape))
    viz3D(ATMO_int2_.reshape(R.shape))
    viz3D(scatter_angle.reshape(R.shape))

    mlab.show()


def vizTransforms2(
    H_pol,
    H_int,
    H_camera,
    ATMO_,
    Y,
    X,
    H,
    R,
    PHI,
    THETA
    ):
    
    ATMO_pol_= H_pol * ATMO_
    ATMO_int_ = H_int * ATMO_pol_
    ATMO_img_ = H_camera * ATMO_int_
     
    viz3D(Y, X, H, ATMO_.reshape(X.shape))
    viz3D(R, PHI, THETA, ATMO_pol_.reshape(R.shape))
    mlab.show()

    viz2D(ATMO_int_.reshape(R.shape[1:]))
    viz2D(ATMO_img_.reshape((SKY_PARAMS['image_res'], SKY_PARAMS['image_res'])))

    plt.show()

    
def calcScatterAngle(R, PHI, THETA, rotation):
    """
    Calclculate the scattering angle at each voxel.
    """

    X_ = R * numpy.sin(THETA) * numpy.cos(PHI)
    Y_ = R * numpy.sin(THETA) * numpy.sin(PHI)
    Z_ = R * numpy.cos(THETA)

    XYZ_dst = numpy.vstack((X_.ravel(), Y_.ravel(), Z_.ravel(), numpy.ones(R.size)))
    XYZ_src_ = numpy.dot(rotation, XYZ_dst)

    Z_rotated = XYZ_src_[2, :]
    R_rotated = numpy.sqrt(numpy.sum(XYZ_src_[:3, :]**2, axis=0))

    angle = numpy.arccos(Z_rotated/(R_rotated+amitibo.eps(R_rotated)))

    return angle


def calcOpticalDistancesMatrix(Y, X, Z, sun_angle, H_pol, R, PHI, THETA):
    """
    Calculate the optical distances from the sun through any voxel in the atmosphere and
    from there to the camera.
    """
    
    #
    # Prepare transformation matrices
    #
    Hrot_forward, rotation, Y_rot, X_rot, Z_rot = atmo_utils.rotation3DTransformMatrix(Y, X, Z, rotation=(0, sun_angle, 0))
    Hrot_backward = atmo_utils.rotation3DTransformMatrix(Y_rot, X_rot, Z_rot, numpy.linalg.inv(rotation), Y, X, Z)[0]
    
    #
    # Calculate a mask for Hint2
    #
    mask = numpy.ones_like(X)
    mask_pol = H_pol * mask.reshape((-1, 1))

    Hint1 = atmo_utils.cumsumTransformMatrix((Y_rot, X_rot, Z_rot), axis=2, direction=-1)
    Hint2 = atmo_utils.cumsumTransformMatrix((R, PHI, THETA), axis=0, masked_rows=mask_pol)

    temp1 = H_pol * Hrot_backward * Hint1 * Hrot_forward
    temp2 = Hint2 * H_pol

    scatter_angle = calcScatterAngle(R, PHI, THETA, rotation)
    
    #vizTransforms(Y, X, Z, H_pol, Hrot_forward, Hrot_backward, Hint1, Hint2, R, Y_rot, scatter_angle)
    
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

    vizTransforms2(
        H_pol,
        H_int,
        H_camera,
        ATMO_air_,
        Y,
        X,
        H,
        R,
        PHI,
        THETA
    )
    
    mu = numpy.cos(scatter_angle)

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
        scatter_aerosols = w * extinction_aerosols * calcHG(mu, g) * (H_pol * ATMO_aerosols_)
        exp_aerosols = numpy.exp(-extinction_aerosols * H_distances * ATMO_aerosols_)
        
        extinction_air = 1.09e-3 * lambda_**-4.05
        scatter_air = extinction_air * (1 + mu**2) * 3 / (16*numpy.pi) * (H_pol * ATMO_air_)
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
        img.append(temp_img.reshape(sky_params['image_res'], sky_params['image_res']))
        
    return img


def draw_image(img, focal_ratio, sun_angle):
    """
    """
    
    c = img.shape[0]/2
    sun_x = c * (1 + focal_ratio * numpy.tan(sun_angle))

    fig = plt.figure()
    ax = plt.axes([0, 0, 1, 1])

    #
    # Draw the sky
    #
    plt.imshow(img)

    #
    # Draw the sun
    #
    sun_patch = mpatches.Circle((sun_x, c), 3, ec='r', fc='r')
    ax.add_patch(sun_patch)

    #
    # Draw angle arcs
    #
    for arc_angle in range(0, 90, 15)[1:]:
        d = c * focal_ratio * numpy.tan(arc_angle/180*numpy.pi)
        arc_patch = mpatches.Arc(
            (c, c),
            2*d,
            2*d,
            90,
            15,
            345,
            ec='r'
        )
        ax.add_patch(arc_patch)
        plt.text(c, c+d, str(arc_angle), ha="center", va="center", size=10, color='r')

    return fig
    
    
def calcRadiance(aerosol_params, sky_params, results_path='', plot_results=False):
    """
    Calculate the radiance of the atmosphere according to aerosol and sky parameters.
    """
    
    #
    # Create the sky
    #
    height = sky_params['height']
    width = sky_params['width']
    dx = sky_params['dx']
    dh = sky_params['dh']
    earth_radius = sky_params['earth_radius']
    
    Y, X, H = numpy.mgrid[0:width:dx, 0:width:dx, 0:height:dh]

    #
    # Create the distributions of air and aerosols
    #
    h = numpy.sqrt((X-width/2)**2 + (Y-width/2)**2 + (earth_radius+H)**2) - earth_radius
    
    ATMO_aerosols = numpy.exp(-h/aerosol_params["aerosols_typical_h"])
    ATMO_aerosols_ = ATMO_aerosols.reshape((-1, 1))
    
    ATMO_air = numpy.exp(-h/aerosol_params["air_typical_h"])
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
        fig = draw_image(IMG, sky_params['focal_ratio'], sky_params['sun_angle'])
        amitibo.saveFigures(results_path, (fig, ), bbox_inches='tight')
        plt.show()

    return img
    

def calcRadianceGradientHelper(ATMO_aerosols_, ATMO_air_, X, H, aerosol_params, sky_params, camera_center):
    """
    Helper function that does the actual calculation of the radiance gradient.
    """
    
    ATMO_aerosols_ = ATMO_aerosols_.reshape((-1, 1))
    
    H_pol, T, R = atmo_utils.polarTransformMatrix(
        X,
        H,
        camera_center,
        radius_res=sky_params['radius_res'],
        angle_res=sky_params['phi_res']
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

    #main_parallel(aerosol_params, SKY_PARAMS, results_path)
    main_serial(aerosol_params, SKY_PARAMS, results_path)

