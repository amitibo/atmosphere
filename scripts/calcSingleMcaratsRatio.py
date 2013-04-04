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
    subgrid_res=(100, 100, 10),
    grid_noise=1.,
    photons_per_pixel=40000
)

camera_position = np.array((25001.0, 25001.0, 1.0))
SUN_ANGLE = -np.pi/4

profile = False
    
VISIBILITY = 10000

VOXEL_INDICES= [
    (25, 25, 15), 
    (25, 25, 10),
    (25, 25, 5),
    (24, 24, 1),
    (24, 24, 5),
    (24, 24, 10),
    (20, 26, 15),
    (20, 20, 10)]

def serial(particle_params):
    
    results_path = amitibo.createResultFolder(params=[atmosphere_params, particle_params, camera_params], src_path=atmotomo.__src_path__)
   
    #
    # Create the distributions
    #
    A_aerosols, Y, X, Z = single_voxel_atmosphere(atmosphere_params, indices_list=VOXEL_INDICES, density=1/VISIBILITY)
    
    #
    # Instantiating the camera
    #
    cam = Camera()
    cam.create(
        SUN_ANGLE,
        atmosphere_params=atmosphere_params,
        camera_params=camera_params,
        camera_position=camera_position
    )
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
    parser.add_argument('--subgrid', type=float, default=8, help='Ratio of subgrid (multiplied by 100, 100, 10) (default=8).')
    args = parser.parse_args()
    
    camera_params.subgrid_res = [int(args.subgrid*i) for i in camera_params.subgrid_res]
    
    #
    # Load the MISR database.
    #
    particle = getMisrDB()['spherical_nonabsorbing_2.80']

    particle_params = amitibo.attrClass(
        k_RGB=np.array(particle['k']) / np.max(np.array(particle['k'])),
        w_RGB=particle['w'],
        g_RGB=(particle['g'])
        )

    serial(particle_params)