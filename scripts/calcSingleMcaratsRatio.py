"""
Simulate the scattering of the sky where the aerosols have a single voxel distribution (Non parallel code).
"""

from __future__ import division
import numpy as np
from atmotomo import calcHG, L_SUN_RGB, RGB_WAVELENGTH, getResourcePath, getMisrDB
from atmotomo import Camera
from atmotomo import single_voxel_atmosphere, calcAirMcarats
import atmotomo
import amitibo
import scipy.io as sio
import os
import warnings
import itertools
import time
import socket
import sys
import argparse
import glob


#
# Global settings
#
atmosphere_params = amitibo.attrClass(
    cartesian_grids=(
        slice(0, 50000, 1000.0), # Y
        slice(0, 50000, 1000.0), # X
        slice(0, 10000, 100.0)   # H
        ),
    earth_radius=-1,
    L_SUN_RGB=L_SUN_RGB,
    RGB_WAVELENGTH=RGB_WAVELENGTH,
    air_typical_h=8000,
    aerosols_typical_h=2000
)

camera_params = amitibo.attrClass(
    image_res=128,
    photons_per_pixel=40000
)

camera_position = np.array((25001.0, 25001.0, 1.0))
SUN_ANGLE = -np.pi/4
    
VISIBILITY = 10000

#VOXEL_INDICES= [
    #(24, 24, 7), 
    #(24, 22, 5)]
    
#
# The cluster values
#
#VOXEL_INDICES= [
    #(16, 16, 10), 
    #(16, 16, 30), 
    #(16, 16, 50), 
    #(16, 16, 80)]

#
# Vadim values
#
#VOXEL_INDICES= [
    #(24, 24, 5),
    #(24, 24, 7,),
    #(24, 24, 10),
    #(24, 24, 15),
    #(24, 24, 20),
    #(24, 24, 25),
    #(24, 24, 30),
    #(24, 24, 35),
    #(24, 24, 40),
    #(24, 24, 45),
    #(24, 24, 50),
#]
VOXEL_INDICES = [(24, 24, i-1) for i in (5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100)]

def serial(particle_params, save_path=None, load_path=None):
    
    results_path = amitibo.createResultFolder(params=[atmosphere_params, particle_params, camera_params], src_path=atmotomo.__src_path__)
   
    #
    # Create the distributions
    #
    A_aerosols, Y, X, Z = single_voxel_atmosphere(atmosphere_params, indices_list=VOXEL_INDICES, density=1/VISIBILITY, decay=False)
    
    #
    # Instantiating the camera
    #
    cam = Camera()
    
    if load_path != None:
        cam.load(os.path.abspath(load_path))
    else:
        cam.create(
            SUN_ANGLE,
            atmosphere_params=atmosphere_params,
            camera_params=camera_params,
            camera_position=camera_position
        )
        
        if save_path != None:
            cam.save(os.path.abspath(save_path))
            
    cam.setA_air(np.zeros_like(A_aerosols[0]))

    #
    # Calculating the image
    #
    for i, aero_dist in enumerate(A_aerosols):
        img = cam.calcImage(A_aerosols=aero_dist, particle_params=particle_params, add_noise=False)
    
        sio.savemat(os.path.join(results_path, 'img%d.mat' % i), {'img':img}, do_compression=True)
    

if __name__ == '__main__':

    #
    # Parse the input
    #
    parser = argparse.ArgumentParser(description='Simulate single voxel atmosphere')
    parser.add_argument('--save_path', type=str, default=None, help='Path for storing the camera.')
    parser.add_argument('--load_path', type=str, default=None, help='Path for loading the camera.')
    args = parser.parse_args()
    
    #
    # Load the MISR database.
    #
    particle = getMisrDB()['spherical_nonabsorbing_2.80']

    particle_params = amitibo.attrClass(
        k_RGB=np.array(particle['k']) / np.max(np.array(particle['k'])),
        w_RGB=particle['w'],
        g_RGB=(particle['g'])
        )

    serial(particle_params, args.save_path, args.load_path)