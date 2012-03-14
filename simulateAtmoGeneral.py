"""
Simulate the scattering of the sky where the aerosols have a general distribution.
"""

from __future__ import division
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
from atmo_utils import calcHG
import math


SKY_PARAMS = {
    'width': 10,
    'height': 2,
    'dxh': 0.02,
    'camera_center': (5, 10),
    'angle_res': 180,
    'dist_res': 50,
    'interp_method': 'cubic'
}

SUN_ANGLE = np.pi/8
EXTINCTION = 0.001
G = 0.0

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
    min_PHI = np.min(values_PHI)
    max_PHI = np.max(values_PHI)

    angle_range = \
        np.linspace(min_PHI, max_PHI, angle_res+1)[:-1]
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
    
    return polar_values


def rotationTransform(
        values,
        angle,
        fit_image=True,
        final_size=None
        ):
    """Transform by rotation """

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


def project():
    pass


def main():
    """Main function"""

    #
    # Create the sky
    #
    X, H = np.meshgrid(
        np.arange(0, SKY_PARAMS['width'], SKY_PARAMS['dxh']),
        np.arange(0, SKY_PARAMS['height'], SKY_PARAMS['dxh'])
        )

    ATMO = np.random.rand(*X.shape)
    ATMO[:1, :] = 0

    #
    # Calculate the effect of the path up to the pixel
    #
    R, PHI = polarCoords(X, H, center=SKY_PARAMS['camera_center'])
    
    ATMO_rotated = rotationTransform(ATMO, SUN_ANGLE)
    temp1 = np.cumsum(ATMO_rotated, axis=0)
    ATMO_to = rotationTransform(temp1, -SUN_ANGLE, final_size=ATMO.shape)
    ATMO_to_polar = polarTransform(ATMO_to, R, PHI)

    #
    # Calculate the effect of the path from the pixel
    #
    temp1 = polarTransform(ATMO, R, PHI)
    ATMO_from_polar = np.cumsum(temp1, axis=0)

    #
    # Calculate a mask over the atmosphere
    #
    mask = np.ones(ATMO.shape)
    mask[:1, :] = 0
    mask_polar = polarTransform(mask, R, PHI)

    #
    # Calculate scattering
    #
    scatter_angle = SUN_ANGLE + PHI
    scatter = calcHG(scatter_angle, G)
    scatter_polar = polarTransform(scatter, R, PHI)

    #
    # Calculate projection
    #
    atten = scatter_polar * np.exp(-EXTINCTION*(ATMO_to_polar + ATMO_from_polar)) * mask_polar
    
    img = np.sum(atten, axis=0)
    IMG = np.tile(img, (100, 1))

    #
    # Plot results
    #
    plt.figure()
    plt.subplot(321)
    plt.imshow(ATMO, interpolation='nearest', cmap='gray')
    plt.subplot(322)
    plt.imshow(mask_polar, interpolation='nearest', cmap='gray')
    plt.subplot(323)
    plt.imshow(ATMO_to_polar, interpolation='nearest', cmap='gray')
    plt.subplot(324)
    plt.imshow(ATMO_from_polar, interpolation='nearest', cmap='gray')
    plt.subplot(325)
    plt.imshow(atten, interpolation='nearest', cmap='gray')
    plt.subplot(326)
    plt.imshow(IMG, interpolation='nearest', cmap='gray')
    
    plt.show()


if __name__ == '__main__':
    main()
