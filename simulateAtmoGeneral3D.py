"""
Simulate the scattering of the sky where the aerosols have a general distribution.
"""

from __future__ import division
import numpy as np
from atmo_utils import calcHG, L_SUN_RGB, RGB_WAVELENGTH
import atmo_utils
from camera import Camera
import amitibo
import scipy.io as sio
import os
import warnings
import time
import socket
import sys


#
# Global settings
#
atmosphere_params = amitibo.attrClass(
    cartesian_grids=(
        slice(0, 400, 1), # Y
        slice(0, 400, 1), # X
        slice(0, 10, 0.1)   # H
        ),
    earth_radius=4000,
    L_SUN_RGB=L_SUN_RGB,
    RGB_WAVELENGTH=RGB_WAVELENGTH,
    air_typical_h=8,
    aerosols_typical_h=1.2
)

camera_params = amitibo.attrClass(
    radius_res=100,
    phi_res=100,
    theta_res=100,
    focal_ratio=0.15,
    image_res=128,
    theta_compensation=False
)

profile = False



def parallel(particle_params):
    
    from mpi4py import MPI
    
    comm = MPI.COMM_WORLD

    Y, X, H = np.mgrid[atmosphere_params.cartesian_grids]
    width = atmosphere_params.cartesian_grids[0].stop
    height = atmosphere_params.cartesian_grids[2].stop
    h = np.sqrt((X-width/2)**2 + (Y-width/2)**2 + (atmosphere_params.earth_radius+H)**2) - atmosphere_params.earth_radius
    ATMO_aerosols = np.exp(-h/atmosphere_params.aerosols_typical_h)
    ATMO_air = np.exp(-h/atmosphere_params.air_typical_h)

    sun_angles = np.linspace(0, np.pi/2, comm.size)

    #
    # Instantiating the camera
    #
    cam = Camera(
        sun_angles[comm.rank],
        atmosphere_params=atmosphere_params,
        camera_params=camera_params,
        camera_position=(width/2, width/2, 0.2)
    )
    
    #
    # Calculating the image
    #
    img = cam.calcImage(A_air=ATMO_air, A_aerosols=ATMO_aerosols, particle_params=particle_params)
        
    result = comm.gather(img, root=0)
    if comm.rank == 0:
        results_path = amitibo.createResultFolder(params=[atmosphere_params, particle_params, camera_params])
        
        for i, img in enumerate(result):
            sio.savemat(os.path.join(results_path, 'img%d.mat' % i), {'img':img}, do_compression=True)
    
    
def serial(particle_params):
    
    #
    # Create some aerosols distribution
    #
    Y, X, H = np.mgrid[atmosphere_params.cartesian_grids]
    width = atmosphere_params.cartesian_grids[0].stop
    height = atmosphere_params.cartesian_grids[2].stop
    h = np.sqrt((X-width/2)**2 + (Y-width/2)**2 + (atmosphere_params.earth_radius+H)**2) - atmosphere_params.earth_radius
    ATMO_aerosols = np.exp(-h/atmosphere_params.aerosols_typical_h)
    ATMO_air = np.exp(-h/atmosphere_params.air_typical_h)

    results_path = amitibo.createResultFolder(params=[atmosphere_params, particle_params, camera_params])

    for i, sun_angle in enumerate([0]):#np.linspace(0, np.pi/2, 4)):
        #
        # Instantiating the camera
        #
        cam = Camera(
            sun_angle,
            atmosphere_params=atmosphere_params,
            camera_params=camera_params,
            camera_position=(width/2, width/2, 0.2)
        )
        
        cam.setA_air(ATMO_air)
        
        #
        # Calculating the image
        #
        img = cam.calcImage(A_aerosols=ATMO_aerosols, particle_params=particle_params)
        
        sio.savemat(os.path.join(results_path, 'img%d.mat' % i), {'img':img}, do_compression=True)
        

if __name__ == '__main__':

    #
    # Load the MISR database.
    #
    import pickle
    
    with open('misr.pkl', 'rb') as f:
        misr = pickle.load(f)
    
    particles_list = misr.keys()
    particle = misr[particles_list[0]]
    particle_params = amitibo.attrClass(
        k_RGB=np.array(particle['k']) / np.max(np.array(particle['k'])),#* 10**-12,
        w_RGB=particle['w'],
        g_RGB=(particle['g']),
        visibility=100
        )

    #if profile:
        #import cProfile    
        #cmd = "serial(particle_params)"
        #cProfile.runctx(cmd, globals(), locals(), filename="atmosphere_camera.profile")
    #else:
        #parallel(particle_params)
        
    serial(particle_params)