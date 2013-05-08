"""
Simulate the scattering of the sky where the aerosols have a general distribution.
"""

from __future__ import division
import numpy as np
from atmotomo import calcHG, L_SUN_RGB, RGB_WAVELENGTH, getResourcePath, getMisrDB
from atmotomo import Camera
from atmotomo import density_clouds1, density_clouds_vadim, calcAirMcarats, single_voxel_atmosphere
import sparse_transforms as spt
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
    cartesian_grids=spt.Grids(
        np.arange(0, 50000, 1000.0), # Y
        np.arange(0, 50000, 1000.0), # X
        np.arange(0, 10000, 100.0)   # H
        ),
    earth_radius=4000000,
    L_SUN_RGB=L_SUN_RGB,
    RGB_WAVELENGTH=RGB_WAVELENGTH,
    air_typical_h=8000,
    aerosols_typical_h=2000
)

camera_params = amitibo.attrClass(
    image_res=128,
    radius_res=100,
    photons_per_pixel=40000
)

camera_position = np.array((9507, 22815.9, 84.431))
SUN_ANGLE = -np.pi/4
CAMERA_CENTERS = [np.array((i, j, 0.)) + 0.1*np.random.rand(3) for i, j in itertools.product(np.linspace(5000., 45000, 5), np.linspace(5000., 45000, 5))]

VISIBILITY = 10000
KM_TO_METER = 1000
profile = False
    

def parallel(particle_params, cameras):
    
    from mpi4py import MPI
    
    comm = MPI.COMM_WORLD

    #
    # override excepthook so that an exception in one of the childs will cause mpi to abort execution.
    #
    def abortrun(type, value, tb):
        import traceback
        traceback.print_exception(type, value, tb)
        MPI.COMM_WORLD.Abort(1)
        
    sys.excepthook = abortrun
        
    #
    # Create the distributions
    #
    #A_air, A_aerosols, Y, X, H, h = density_clouds1(atmosphere_params)
    #A_aerosols = A_aerosols / VISIBILITY
    A_aerosols, Y, X, Z = single_voxel_atmosphere(atmosphere_params, indices_list=[(24, 24, 19)], density=1/VISIBILITY, decay=False)
    
    #
    # Instantiating the camera
    #
    if comm.rank < len(cameras):
        cam = Camera()
        cam.create(
            SUN_ANGLE,
            atmosphere_params=atmosphere_params,
            camera_params=camera_params,
            camera_position=cameras[comm.rank]
        )
        
        #z_coords = H[0, 0, :]
        #air_exts = calcAirMcarats(z_coords)
        #cam.set_air_extinction(air_exts)
        cam.setA_air(np.zeros_like(A_aerosols[0]))
        
        #
        # Calculating the image
        #
        img = cam.calcImage(A_aerosols=A_aerosols[0], particle_params=particle_params, add_noise=True)
    else:
        img = []
        
    result = comm.gather(img, root=0)
    if comm.rank == 0:
        results_path = amitibo.createResultFolder(params=[atmosphere_params, particle_params, camera_params], src_path=atmotomo.__src_path__)
        
        for i, img in enumerate(result):
            if img != []:
                sio.savemat(os.path.join(results_path, 'img%d.mat' % i), {'img':img}, do_compression=True)
    
    
def serial(particle_params):
    
    results_path = amitibo.createResultFolder(params=[atmosphere_params, particle_params, camera_params], src_path=atmotomo.__src_path__)
   
    #
    # Create the distributions
    #
    A_air, A_aerosols, Y, X, H, h = density_clouds1(atmosphere_params)
    A_aerosols = A_aerosols / VISIBILITY
    
    for i, sun_angle in enumerate([-np.pi/4]):#np.linspace(0, np.pi/2, 12)):
        #
        # Instantiating the camera
        #
        cam = Camera()
        cam.create(
            sun_angle,
            atmosphere_params=atmosphere_params,
            camera_params=camera_params,
            camera_position=camera_position
        )
        cam.setA_air(A_air)
        #camera_path = amitibo.createResultFolder(base_path='d:/amit/tmp')
        #cam.save(camera_path)
        
        #
        # Calculating the image
        #
        img = cam.calcImage(A_aerosols=A_aerosols, particle_params=particle_params, add_noise=True)
        
        sio.savemat(os.path.join(results_path, 'img%d.mat' % i), {'img':img}, do_compression=True)
        

if __name__ == '__main__':

    #
    # Parse the input
    #
    parser = argparse.ArgumentParser(description='Simulate atmosphere')
    parser.add_argument('--cameras', action='store_true', help='load the cameras from the cameras file')
    parser.add_argument('--ref_images', help='path to reference images')
    parser.add_argument('--parallel', action='store_true', help='run the parallel mode')
    parser.add_argument('--profile', action='store_true', help='run the profiler (will use serial mode)')
    args = parser.parse_args()
    
    cameras = []
    if args.cameras:
        #
        # Get the camera positions from the camera positions file
        #
        with open(getResourcePath('CamerasPositions.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                cameras.append(np.array([float(i) for i in line.strip().split()])*KM_TO_METER)
                
    elif args.ref_images:
        #
        # Get the camera positions from the ref images folders
        #
        
        for path in glob.glob(os.path.join(args.ref_images, "*")):
            #
            # Parse cameras center file
            #
            with open(os.path.join(path, 'params.txt'), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if parts[0] == 'CameraPosition':
                        cameras.append(np.array((float(parts[4])+25000, float(parts[2])+25000, float(parts[3]))))
                        break
    
    else:
        cameras = CAMERA_CENTERS
        
    #
    # Load the MISR database.
    #
    particle = getMisrDB()['spherical_nonabsorbing_2.80']

    particle_params = amitibo.attrClass(
        k_RGB=np.array(particle['k']) / np.max(np.array(particle['k'])),#* 10**-12,
        w_RGB=particle['w'],
        g_RGB=(particle['g'])
        )

    if args.profile:
        import cProfile    
        cmd = "serial(particle_params)"
        cProfile.runctx(cmd, globals(), locals(), filename="atmosphere_camera.profile")
    else:
        if args.parallel:
            parallel(particle_params, cameras)
        else:
            serial(particle_params)