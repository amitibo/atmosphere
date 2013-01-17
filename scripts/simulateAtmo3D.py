"""
Simulate the scattering of the sky where the aerosols have a general distribution.
"""

from __future__ import division
import numpy as np
from atmotomo import calcHG, L_SUN_RGB, RGB_WAVELENGTH, getResourcePath
from atmotomo import Camera
from atmotomo import density_clouds1, density_clouds_vadim
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


#
# Global settings
#
atmosphere_params = amitibo.attrClass(
    cartesian_grids=(
        slice(0, 50, 1.0), # Y
        slice(0, 50, 1.0), # X
        slice(0, 10, 0.1)   # H
        ),
    earth_radius=4000,
    L_SUN_RGB=L_SUN_RGB,
    RGB_WAVELENGTH=RGB_WAVELENGTH,
    air_typical_h=8,
    aerosols_typical_h=2
)

camera_params = amitibo.attrClass(
    image_res=128,
    subgrid_res=(10, 10, 1),
    grid_noise=0.05,
    photons_per_pixel=40000
)

camera_position = np.array((9.507, 22.8159, 0.084431))
SUN_ANGLE = -np.pi/4
CAMERA_CENTERS = [np.array((i, j, 0.)) + 0.1*np.random.rand(3) for i, j in itertools.product(np.linspace(5., 45, 5), np.linspace(5., 45, 5))]

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
    A_air, A_aerosols, Y, X, H, h = density_clouds1(atmosphere_params)
        
    #
    # Instantiating the camera
    #
    cam = Camera()
    cam.create(
        SUN_ANGLE,
        atmosphere_params=atmosphere_params,
        camera_params=camera_params,
        camera_position=cameras[comm.rank]
    )
    
    cam.setA_air(A_air)
    
    #
    # Calculating the image
    #
    img = cam.calcImage(A_aerosols=A_aerosols, particle_params=particle_params, add_noise=True)
        
    result = comm.gather(img, root=0)
    if comm.rank == 0:
        results_path = amitibo.createResultFolder(params=[atmosphere_params, particle_params, camera_params], src_path=atmotomo.__src_path__)
        
        for i, img in enumerate(result):
            sio.savemat(os.path.join(results_path, 'img%d.mat' % i), {'img':img}, do_compression=True)
    
    
def serial(particle_params):
    
    results_path = amitibo.createResultFolder(params=[atmosphere_params, particle_params, camera_params], src_path=atmotomo.__src_path__)
   
    #
    # Create the distributions
    #
    A_air, A_aerosols, Y, X, H, h = density_clouds1(atmosphere_params)
    
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
    parser.add_argument('--cameras', help='path to cameras file')
    parser.add_argument('--parallel', action='store_true', help='run the parallel mode')
    parser.add_argument('--profile', action='store_true', help='run the profiler (will use serial mode)')
    args = parser.parse_args()
    
    
    if args.cameras:
        cameras = []
        with open(os.path.abspath(args.cameras), 'r') as f:
            lines = f.readlines()
            for line in lines:
                cameras.append(np.array([float(i) for i in line.strip().split(',')]))
    else:
        cameras = CAMERA_CENTERS
        
    #
    # Load the MISR database.
    #
    import pickle
    
    with open(getResourcePath('misr.pkl'), 'rb') as f:
        misr = pickle.load(f)
    
    particles_list = misr.keys()
    particle = misr['spherical_nonabsorbing_2.80']
    particle_params = amitibo.attrClass(
        k_RGB=np.array(particle['k']) / np.max(np.array(particle['k'])),#* 10**-12,
        w_RGB=particle['w'],
        g_RGB=(particle['g']),
        visibility=5
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