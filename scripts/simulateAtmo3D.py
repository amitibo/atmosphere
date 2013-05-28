"""
Simulate the scattering of the sky where the aerosols have a general distribution.
"""

from __future__ import division
import numpy as np
from atmotomo import Camera
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


def parallel(params_path, add_noise, job_id=None, ref_mc=None):
    
    #import wingdbstub

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
    # Load the simulation params
    #
    atmosphere_params, particle_params, sun_params, camera_params, cameras, air_dist, aerosols_dist = atmotomo.readConfiguration(params_path)
    
    #
    # Load the reference mc camera positions
    #
    if ref_mc != None:
        closed_grids = atmosphere_params.cartesian_grids.closed
        dump, cameras = atmotomo.loadVadimData(os.path.abspath(ref_mc), offset=(closed_grids[0][-1]/2, closed_grids[1][-1]/2))
        
    #
    # Create the results path
    #
    if comm.rank == 0:
        results_path = amitibo.createResultFolder(
            params=[atmosphere_params, particle_params, sun_params, camera_params],
            src_path=atmotomo.__src_path__,
            job_id=job_id
        )
        
    #
    # Instantiating the camera
    #
    if comm.rank < len(cameras):
        cam = Camera()
        cam.create(
            sun_params=sun_params,
            atmosphere_params=atmosphere_params,
            camera_params=camera_params,
            camera_position=cameras[comm.rank]
        )
        
        cam.setA_air(air_dist)
        
        #
        # Calculating the image
        #
        img = cam.calcImage(
            A_aerosols=aerosols_dist,
            particle_params=particle_params,
            add_noise=add_noise
        )
    else:
        img = []
        
    result = comm.gather(img, root=0)
    
    if comm.rank == 0:
        for i, img in enumerate(result):
            if img != []:
                sio.savemat(
                    os.path.join(results_path, 'img%d.mat' % i),
                    {'img':img},
                    do_compression=True
                )
    
    
def serial(params_path, add_noise):
    
    #
    # Load the simulation params
    #
    atmosphere_params, particle_params, sun_params, camera_params, cameras, air_dist, aerosols_dist = atmotomo.readConfiguration(params_path)
    
    results_path = amitibo.createResultFolder(
        params=[atmosphere_params, particle_params, sun_params, camera_params],
        src_path=atmotomo.__src_path__
    )
   
    #
    # Instantiating the camera
    #
    cam = Camera()
    cam.create(
        sun_params=sun_params,
        atmosphere_params=atmosphere_params,
        camera_params=camera_params,
        camera_position=cameras[0],
        radius_resolution=100
    )
    
    cam.setA_air(air_dist)
    
    #
    # Calculating the image
    #
    img = cam.calcImage(
        A_aerosols=aerosols_dist,
        particle_params=particle_params,
        add_noise=add_noise
    )
    
    sio.savemat(
        os.path.join(results_path, 'img%d.mat' % 0),
        {'img':img},
        do_compression=True
    )
    

if __name__ == '__main__':

    #
    # Parse the input
    #
    parser = argparse.ArgumentParser(description='Simulate atmosphere')
    parser.add_argument('--parallel', action='store_true', help='run the parallel mode')
    parser.add_argument('--profile', action='store_true', help='run the profiler (will use serial mode)')
    parser.add_argument('--noise', action='store_true', help='Add noise to the image creation')
    parser.add_argument('--job_id', default=None, help='pbs job ID (set automatically by the PBS script)')
    parser.add_argument('--ref_mc', help='path to reference mc images. Used only for getting the camera position (this option should be removed once we start using the unified configuration files).')
    parser.add_argument('params_path', help='Path to simulation parameters')
    args = parser.parse_args()
    
    if args.profile:
        import cProfile    
        cmd = "serial(args.params_path, args.noise)"
        cProfile.runctx(cmd, globals(), locals(), filename="atmosphere_camera.profile")
    else:
        if args.parallel:
            parallel(args.params_path, args.noise, args.job_id, args.ref_mc)
        else:
            serial(args.params_path, args.noise)
