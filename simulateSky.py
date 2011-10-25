# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 18:21:25 2011

@author: amitibo

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

"""
from __future__ import division
from utils import createResultFolder, attrClass
import numpy as np
import argparse
import math
import matplotlib.pyplot as plt
import pickle
import os


def calcOP(x0, y0, angle, width, height):
    """Calculate the optical path between a point and the sky"""
    
    if angle == 0:
        assert(0)
    elif angle > 0:
        X0 = np.concatenate(([[x0], np.arange(math.ceil(x0), width+1)]))
        Y1 = np.concatenate(([[y0], np.arange(math.ceil(y0), height+1)]))
    else:
        X0 = np.concatenate(([[x0], np.arange(math.floor(x0), -1, -1)]))
        Y1 = np.concatenate(([[y0], np.arange(math.ceil(y0), height+1)]))
    
    Y0 = 1/math.tan(angle) * (X0 - x0) + y0
    indices = Y0 <= height
    X0 = X0[indices]
    Y0 = Y0[indices]
    
    X1 = math.tan(angle) * (Y1 - y0) + x0
    indices = (X1 <= width) & (X1 >= 0)
    Y1 = Y1[indices]
    X1 = X1[indices]
    
    XY = np.vstack((np.hstack((X0, X1)), np.hstack((Y0, Y1))))

    #
    # Note: I round the x index before using unique due to overcome small
    # inaccuricies.
    # 
    u, indices = np.unique(np.round(XY[0, :], decimals=5), return_index=True)
    XY = XY[:, indices] 
    
    lengths = np.sum((XY[:, 1:]-XY[:, :-1])**2, axis=0)
    XY = np.floor(XY)    
    indices = np.sum(XY * np.array([[1], [width]]), axis=0)
    
    if angle > 0:
        indices = indices[:-1]
    else:
        indices = indices[1:]

    return indices.astype(np.int), lengths


def createAtmosphere(width, height, lower_visibiliy=10000, upper_visibiliy=50000, K=0.0005*10**-12, noise=0):
    """Create the particle density distribution in the atmosphere"""
    
    upper_n = 1/upper_visibiliy/K
    lower_n = 1/lower_visibiliy/K
    
    log_scale = np.logspace(np.log(upper_n), np.log(lower_n), height, base=np.e)
    return np.ones((height, width)) * log_scale.reshape((-1, 1))


def calcHG(dangles, g):
    """Calculate the Henyey-Greenstein function for each voxel.
    The HG function is taken from: http://www.astro.umd.edu/~jph/HG_note.pdf
    """
    
    HG = (1 - g**2) / (1 + g**2 - 2*g*np.cos(dangles))**(3/2) / (4*np.pi)
    return HG


def calcSkyOP(params, results_folder):
    """Calc the optical path for each voxel in the sky."""
    
    #
    # Create the distribution of particles in the atmosphere.
    #
    print 'Calculate the Atmosphere...\n'
    n_ATM = createAtmosphere(params.width, params.height)
    
    #
    # First calcualte the SR optical path to each voxel.
    #
    print 'Calculate the SR optical path to each voxel...\n'
    l_SR = np.zeros((params.height, params.width))
    for i in range(params.height):
        for j in range(params.width):
            indices, lengths = calcOP(j+0.5, i+0.5, params.sun_angle, params.width, params.height)
            l_SR[i, j] = np.sum(n_ATM.flatten()[indices] * lengths)*params.dz

    #
    # Calculate the optical distance from each voxel to each
    # Note:
    # I reverse the order of the tan2 parameters.
    #
    print 'Calculate the LOS optical path to each voxel...\n'
    vox_angles = np.zeros_like(l_SR)
    l_LOS = np.zeros_like(l_SR)
    for i in range(params.height):
        for j in range(params.width):
            cam_angle = -math.atan2(j+0.5 - params.camera_x, params.height-(i+0.5))
            vox_angles[i, j] = cam_angle
            indices, lengths = calcOP(j+0.5, i+0.5, cam_angle, params.width, params.height)
            l_LOS[i, j] = np.sum(n_ATM.flatten()[indices] * lengths)*params.dz

    sky = {
        'n_ATM': n_ATM,
        'l_SR': l_SR,
        'l_LOS': l_LOS,
        'vox_angles': vox_angles
    }
    
    file_path = os.path.join(results_folder, 'sky.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(sky, f)
    
    return sky


def calcCamIR(sun_angle, sky, params, results_folder):

    n_ATM = sky['n_ATM']
    l_SR = sky['l_SR']
    l_LOS = sky['l_LOS']
    vox_angles = sky['vox_angles']
    
    #
    # Calculate the radiance from each voxel.
    # I calculate the accuracy 
    print 'Calculate the radiance for each voxel...\n'
    cam_resolution = 10**params.ANGLE_ACCURACY
    dangles = sun_angle+np.pi-vox_angles 
    angle_indices = np.round((vox_angles/np.pi+0.5)*cam_resolution)    
    cam_radiance = np.zeros((1, cam_resolution+1, 3))

    for ch_ind, (k, w, g, F_sol) in enumerate(zip(params.k_RGB, params.w_RGB, params.g_RGB, params.F_sol_RGB)):
        #
        # Calculate the Heney-Greenstein function.
        #
        print 'Calculate the Heney-Greenstein function for each voxel...\n'
        
        p = calcHG(dangles, g)
        vox_iradiance = F_sol * k * w * n_ATM * p * np.exp(-k * l_SR) * np.exp(-k * l_LOS)
    
        for i, v in zip(angle_indices.flatten(), vox_iradiance.flatten()):
            cam_radiance[0, i, ch_ind] += v
    
    #
    # Tile the 2D camera to 3D camera.
    #
    cam_radiance = np.tile(cam_radiance, (cam_resolution, 1, 1))  
    np.save('cam.npy', cam_radiance)
    
    #
    # Stretch the values.
    #
    cam_radiance = (cam_radiance/np.max(cam_radiance)*255).astype(np.uint8)
    
    #
    # Plot the results
    #
    plt.figure()
    plt.imshow(cam_radiance)
    plt.title('Sky Simulation')
    plt.savefig(os.path.join(results_folder, 'sky.png'))
    plt.show()

    
def main():
    #
    # Parse the command line
    #
    parser = argparse.ArgumentParser(description='Simulate the sky.')
    parser.add_argument('--folder', type=str, default='', help='Load previously calculated sky optical paths.')
    args = parser.parse_args()

    #
    # Set the params of the run
    #
    width = 20
    sky_params = attrClass(
                dz=100,
                width=width,
                height=20,
                sun_angle=np.pi/6,
                camera_x=width/2
                )

    aerosol_params = attrClass(
                ANGLE_ACCURACY=2,
                k_RGB=np.array((0.000772, 0.000396, 0.000217)) * 10**-12,
                w_RGB=(1.0, 1.0, 1.0),
                g_RGB=(0.432, 0.352, 0.287),
                F_sol_RGB=(255, 236, 224)
                )
                
    united_params = attrClass()
    united_params.__dict__.update(sky_params.__dict__)
    united_params.__dict__.update(aerosol_params.__dict__)
    
    #
    # Run the simulation
    #
    results_folder = createResultFolder('simulateSky_results', params=united_params)
    if not args.folder:
        sky = calcSkyOP(sky_params, results_folder)
    else:
        file_name = os.path.join(args.folder, 'sky.pkl')
        with open(file_name, 'rb') as f:
            sky = pickle.load(f)
            
    calcCamIR(sky_params.sun_angle, sky, aerosol_params, results_folder)


if __name__ == '__main__':
    main()
    
