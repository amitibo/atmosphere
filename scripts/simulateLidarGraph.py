"""
Simulate the scattering of the sky where the aerosols have a general distribution.
"""

from __future__ import division
import numpy as np
import atmotomo
import amitibo
import scipy.io as sio
from scipy import ndimage
import matplotlib.pyplot as plt
import os
import time
import sys
import argparse
import glob


ds = 10.0
wavelength = 558e-3
RGB_CHANNEL = 1
air_height_constant = 8000


def interpolate(grids, values, coords):

    ratios = [bins/length[-1] for bins, length in zip(grids.shape, grids.closed)]
    coords = [coord * ratio for coord, ratio in zip(coords, ratios)]
    
    samples = ndimage.map_coordinates(
        values,
        coordinates=coords
    )
     
    return samples


def main(params_path, add_noise, run_arguments):
    
    #
    # Load the simulation params
    #
    atmosphere_params, particle_params, sun_params, camera_params, cameras, air_dist, aerosols_dist = \
        atmotomo.readConfiguration(params_path)
    aerosols_dist = np.transpose(aerosols_dist, (1, 0, 2))
    
    grids = atmosphere_params.cartesian_grids
    width, height = grids.closed[0][-1], grids.closed[2][-1]
    lidar_position = np.array((width/2, width/2, 0))
    
    z_coords = np.linspace(0, height, height/ds)
    y_coords = np.ones_like(z_coords) * lidar_position[0]
    x_coords = np.ones_like(z_coords) * lidar_position[1]
    
    aerosols_samples = interpolate(
        grids,
        aerosols_dist,
        coords=[y_coords, x_coords, z_coords]
    )
    air_samples = interpolate(
        grids,
        air_dist,
        coords=[y_coords, x_coords, z_coords]
    )
    
    Oz = 1 - np.exp(-z_coords/100)
    
    #
    # 
    #
    air_ext_coef = 1.09e-6 * wavelength**-4.05 * air_samples * ds
    air_scatter = 3 / (16*np.pi) * (1 + np.cos(np.pi)**2) * air_ext_coef
    
    aerosol_ext_coef = particle_params.k[RGB_CHANNEL] * aerosols_samples * ds
    aerosol_scatter = atmotomo.calcHG_other(np.cos(np.pi), particle_params.g[RGB_CHANNEL]) * aerosol_ext_coef

    i_lidar = Oz * (air_scatter + aerosol_scatter) * np.exp(-2*(np.cumsum(air_ext_coef) + np.cumsum(aerosol_ext_coef))) / z_coords**2
    
    #
    #
    #
    plt.figure()
    plt.plot(np.log(i_lidar), z_coords)
    
    #
    # Camera from the side
    #
    angles = np.linspace(0, np.pi/2, 100)
    angles = angles[1:-5]
    i_cameras = []
    cameras_distances = np.concatenate((np.arange(-40, 0), np.arange(1, 40))) * 500
    for dx in cameras_distances:
        print dx
        camera_position = np.array((width/2+dx, width/2, 0))
        z_crossings = abs(dx) * np.tan(angles)
        
        i_camera = []
        for z_cross, angle in zip(z_crossings, angles):
            
            #
            # Path along the Laser Beam
            #
            LB_z_coords = ds*np.arange(0, int(z_cross/ds), dtype=np.float)
            if len(LB_z_coords) == 0:
                LB_z_coords = np.array((ds,))
            LB_x_coords = np.ones_like(LB_z_coords) * lidar_position[0]
            LB_y_coords = np.ones_like(LB_z_coords) * lidar_position[0]
        
            LB_aerosols_samples = interpolate(
                grids,
                aerosols_dist,
                coords=[LB_y_coords, LB_x_coords, LB_z_coords]
            )
            LB_air_samples_old = interpolate(
                grids,
                air_dist,
                coords=[LB_y_coords, LB_x_coords, LB_z_coords]
            )
            LB_air_samples = np.exp(-LB_z_coords/air_height_constant)
            
            LB_air_ext_coef = 1.09e-6 * wavelength**-4.05 * LB_air_samples * ds
            LB_aerosol_ext_coef = particle_params.k[RGB_CHANNEL] * LB_aerosols_samples * ds
            
            #
            # Scatter
            #
            LB_air_scatter = 3 / (16*np.pi) * (1 + np.cos(np.pi/2+angle)**2) * LB_air_ext_coef[-1]
            LB_aerosol_scatter = atmotomo.calcHG_other(np.cos(np.pi/2+angle), particle_params.g[RGB_CHANNEL]) * particle_params.w[RGB_CHANNEL] * LB_aerosol_ext_coef[-1]
    
            #
            # Path along the Line Of Sight
            #
            LOS_z_coords = np.sin(angle)*ds*np.arange(0, int(z_cross/ds/np.sin(angle)), dtype=np.float)
            LOS_x_coords = lidar_position[0] + dx - np.sign(dx)*np.cos(angle)*ds*np.arange(0, int(abs(dx)/ds/np.cos(angle)), dtype=np.float)
            LOS_y_coords = np.ones_like(LOS_z_coords) * lidar_position[0]
            
            LOS_aerosols_samples = interpolate(
                grids,
                aerosols_dist,
                coords=[LOS_y_coords, LOS_x_coords, LOS_z_coords]
            )
            LOS_air_samples = np.exp(-LOS_z_coords/air_height_constant)
            
            LOS_air_ext_coef = 1.09e-6 * wavelength**-4.05 * LOS_air_samples * ds
            LOS_aerosol_ext_coef = particle_params.k[RGB_CHANNEL] * LOS_aerosols_samples * ds
            
            #
            # Intensity
            #
            i_camera.append((LB_air_scatter + LB_aerosol_scatter)*np.exp(-np.sum(LB_air_ext_coef+LB_aerosol_ext_coef))*np.exp(-np.sum(LOS_air_ext_coef+LOS_aerosol_ext_coef)))
            
        i_cameras.append(i_camera)
        
    plt.figure()
    plt.imshow(np.array(i_cameras).T, origin='lower', interpolation='nearest')
    plt.xticks(np.array((0, int(len(cameras_distances)/2), len(cameras_distances))), (-20, 0, 20))
    plt.xlabel('Distance from LIDAR [km]')
    plt.yticks(np.array((0, int(len(angles)/3), int(len(angles)/3*2), len(angles))), (0, 30, 60, 90))
    plt.ylabel('Elevation angle [degrees]')
    plt.colorbar()
    plt.show()
            
    
if __name__ == '__main__':

    #
    # Parse the input
    #
    parser = argparse.ArgumentParser(description='Simulate atmosphere')
    parser.add_argument('--noise', action='store_true', help='Add noise to the image creation')
    parser.add_argument('params_path', help='Path to simulation parameters')
    args = parser.parse_args()
    
    main(args.params_path, args.noise, run_arguments=args)
