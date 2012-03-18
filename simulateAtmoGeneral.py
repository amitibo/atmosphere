"""
Simulate the scattering of the sky where the aerosols have a general distribution.
"""

from __future__ import division
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
from atmo_utils import calcHG, L_SUN_RGB, RGB_WAVELENGTH
import pickle
import math


SKY_PARAMS = {
    'width': 200,
    'height': 20,
    'dxh': 0.1,
    'camera_center': (50, 0.1)
}

SUN_ANGLE = 0



def polarCoords(X, Y, center):
    """Convert cartesian coords to polar coords around some center"""
    
    PHI = np.arctan2(Y-center[1], X-center[0])
    R = np.sqrt((X-center[0])**2 + (Y-center[1])**2)

    return R, PHI


def polarTransform(
        values,
        values_R,
        values_PHI,
        radius_res=None,
        angle_res=None,
        interp_method='linear'
        ):
    """Cartesian to polar transform"""

    if radius_res == None:
        radius_res = angle_res = max(*values.shape)
        
    values_points = np.vstack((values_R.flatten(), values_PHI.flatten())).T
    
    max_R = np.max(values_R)
    
    angle_range = \
        np.linspace(0, np.pi, angle_res+1)[:-1]
    radius_range = np.linspace(0, max_R, radius_res+1)[:-1]
    
    grid_PHI, grid_R = np.meshgrid(angle_range, radius_range)
    
    polar_values = \
        griddata(
            values_points,
            values.flatten(),
            (grid_R, grid_PHI),
            method=interp_method,
            fill_value=0
            )
    polar_values[polar_values<0] = 0
    
    return polar_values, grid_R, grid_PHI


def rotationTransform(
        values,
        angle,
        fit_image=True,
        final_size=None
        ):
    """Transform by rotation"""

    if np.isscalar(angle):
        H = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    else:
        H = angle

    x0 = y0 = 0
    y1, x1 = values.shape
    
    if fit_image:
        coords = np.hstack((
            np.dot(H, np.array([[0], [0], [1]])),
            np.dot(H, np.array([[0], [y1], [1]])),
            np.dot(H, np.array([[x1], [0], [1]])),
            np.dot(H, np.array([[x1], [y1], [1]]))
            ))
        x0, y0, dump = np.floor(np.min(coords, axis=1)).astype(np.int)
        x1, y1, dump = np.ceil(np.max(coords, axis=1)).astype(np.int)

    H[0, -1] = -x0
    H[1, -1] = -y0

    values = skimage.transform.fast_homography(
        values,
        H,
        output_shape=(y1-y0, x1-x0)
        )

    if final_size != None:
        x0 = int((values.shape[1]-final_size[1])/2)
        y0 = int((values.shape[0]-final_size[0])/2)
        values = values[y0:y0+final_size[0], x0:x0+final_size[1]]
        
    return values


def calcOpticalDistances(ATMO, SUN_ANGLE, R, PHI, dXY):
    #
    # Calculate the effect of the path up to the pixel
    #
    ATMO_rotated = rotationTransform(ATMO, SUN_ANGLE)
    temp1 = np.cumsum(ATMO_rotated, axis=0)*dXY/math.cos(SUN_ANGLE)
    ATMO_to = rotationTransform(temp1, -SUN_ANGLE, final_size=ATMO.shape)
    ATMO_to_polar = polarTransform(ATMO_to, R, PHI)[0]

    #
    # Calculate the effect of the path from the pixel
    #
    temp1, grid_R = polarTransform(ATMO, R, PHI)[:2]
    dR = abs(grid_R[1, 0] - grid_R[0, 0])
    ATMO_from_polar = np.cumsum(temp1, axis=0)*dR

    return ATMO_to_polar, ATMO_from_polar


def main():
    """Main function"""

    #
    # Load the MISR database.
    #
    with open('misr.pkl', 'rb') as f:
        misr = pickle.load(f)
    
    particles_list = misr.keys()
    particle = misr[particles_list[0]]
    aerosol_params = {
        "k_RGB": np.array(particle['k']) / np.max(np.array(particle['k'])),#* 10**-12,
        "w_RGB": particle['w'],
        "g_RGB": (particle['g']),
        "visibility": 1
    }
    
    #
    # Create the sky
    #
    X, H = np.meshgrid(
        np.arange(0, SKY_PARAMS['width'], SKY_PARAMS['dxh']),
        np.arange(0, SKY_PARAMS['height'], SKY_PARAMS['dxh'])[::-1]
        )

    R, PHI = polarCoords(X, H, center=SKY_PARAMS['camera_center'])

    #
    # Create the distributions of air and aerosols
    # Note:
    # I set the topmost row to 0 so that when converting from cartisian to polar
    # coords the interpolation will not 'create' atmosphere above the sky.
    #
    ATMO_aerosols = np.exp(-H/1.2)
    ATMO_air = np.exp(-H/8)

    #
    # Calculate a mask over the atmosphere
    # Note:
    # The mask is used to maskout in the polar axis,
    # pixels that are not in the cartesian axis.
    #
    mask = np.ones(X.shape)
    mask[0, :] = 0
    mask[:, 0] = 0
    mask[:, -1] = 0
    
    mask_polar, grid_R, grid_PHI = polarTransform(mask, R, PHI)
    ATMO_aerosols *= mask
    ATMO_air *= mask
    
    #
    # Calculate the distances
    #
    ATMO_aerosols_to_polar, ATMO_aerosols_from_polar = \
        calcOpticalDistances(ATMO_aerosols, SUN_ANGLE, R, PHI, SKY_PARAMS['dxh'])
    ATMO_air_to_polar, ATMO_air_from_polar = \
        calcOpticalDistances(ATMO_air, SUN_ANGLE, R, PHI, SKY_PARAMS['dxh'])
    
    #
    # Calculate scattering angle
    #
    scatter_angle = SUN_ANGLE + grid_PHI + np.pi/2

    #
    # Calculate scattering for each channel (in case of the railey scattering)
    #
    img = []
    for L_sun, lambda_, k, w, g in zip(L_SUN_RGB, RGB_WAVELENGTH, aerosol_params["k_RGB"], aerosol_params["w_RGB"], aerosol_params["g_RGB"]):
        #
        # Calculate scattering and extiniction for air (wave length dependent)
        #
        extinction_aerosol = k
        scatter_aerosol = calcHG(scatter_angle, g)
        extinction_air = 1.09e-3 * lambda_**-4.05
        scatter_air = extinction_air*(1 + np.cos(scatter_angle)**2)
        
        #
        # Calculate total attenuation
        #
        temp = extinction_aerosol*(ATMO_aerosols_to_polar + ATMO_aerosols_from_polar) + \
          extinction_air*(ATMO_air_to_polar + ATMO_air_from_polar)
        attenuation = (scatter_aerosol + scatter_air) * np.exp(-temp) * mask_polar

        #
        # Calculate projection on camera
        #
        img.append(L_sun * np.sum(attenuation, axis=0))

    IMG = np.tile(np.transpose(np.array(img, ndmin=3), (0, 2, 1)), (100, 1, 1))

    #
    # Plot results
    #
    plt.figure()
    plt.subplot(321)
    plt.imshow(ATMO_air, interpolation='nearest', cmap='gray')
    plt.subplot(322)
    plt.imshow(mask_polar, interpolation='nearest', cmap='gray')
    plt.subplot(323)
    plt.imshow(ATMO_air_to_polar, interpolation='nearest', cmap='gray')
    plt.subplot(324)
    plt.imshow(ATMO_air_from_polar, interpolation='nearest', cmap='gray')
    plt.subplot(325)
    plt.imshow(attenuation, interpolation='nearest', cmap='gray')
    plt.subplot(326)
    plt.imshow(IMG/np.max(IMG), interpolation='nearest')
    
    plt.show()


if __name__ == '__main__':
    main()
