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

ds = 1.0
wavelength = 558e-3
RGB_CHANNEL = 1

def interpolate(grids, values, coords):

    ratios = [bins/length[-1] for bins, length in zip(grids.shape, grids.closed)]
    coords = [coord * ratio for coord, ratio in zip(coords, ratios)]
    
    samples = ndimage.map_coordinates(
        values,
        coordinates=coords
    )
     
    return samples


def serial(params_path, add_noise, run_arguments):
    
    #
    # Load the simulation params
    #
    atmosphere_params, particle_params, sun_params, camera_params, cameras, air_dist, aerosols_dist = \
        atmotomo.readConfiguration(params_path)
    
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

    i_lidar = Oz * (air_scatter + aerosol_scatter) * np.exp(-2*(np.cumsum(air_ext_coef) + np.cumsum(aerosol_ext_coef)))
    
    #
    #
    #
    plt.subplot(251)
    plt.plot(z_coords, aerosols_samples)
    plt.subplot(252)
    plt.plot(z_coords, air_ext_coef)
    plt.subplot(253)
    plt.plot(z_coords, air_scatter)
    plt.subplot(254)
    plt.plot(z_coords, aerosol_ext_coef)
    plt.subplot(255)
    plt.plot(z_coords, aerosol_scatter)
    plt.subplot(256)
    plt.plot(z_coords, Oz)
    plt.subplot(257)
    plt.plot(i_lidar, z_coords)
    plt.show()
    
    
if __name__ == '__main__':

    #
    # Parse the input
    #
    parser = argparse.ArgumentParser(description='Simulate atmosphere')
    parser.add_argument('--noise', action='store_true', help='Add noise to the image creation')
    parser.add_argument('params_path', help='Path to simulation parameters')
    args = parser.parse_args()
    
    serial(args.params_path, args.noise, run_arguments=args)
