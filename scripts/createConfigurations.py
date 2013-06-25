"""
Create configuration files for a atmosphere simulations.
"""
from __future__ import division
import sparse_transforms as spt
import numpy as np
import atmotomo 
import scipy.io as sio
import os
import argparse
import amitibo


clouds_atmosphere = amitibo.attrClass(
    cartesian_grids=spt.Grids(
        np.arange(0, 50, 1.0), # Y
        np.arange(0, 50, 1.0), # X
        np.arange(0, 10, 0.1)   # H
        ),
    earth_radius=4000,
    air_typical_h=8,
    aerosols_typical_h=2
)

clouds_atmosphere_low_resolution = amitibo.attrClass(
    cartesian_grids=spt.Grids(
        np.arange(0, 50, 5.0), # Y
        np.arange(0, 50, 5.0), # X
        np.arange(0, 10, 1.0)   # H
        ),
    earth_radius=4000,
    air_typical_h=8,
    aerosols_typical_h=2
)

def main(output_path):
    #
    # Create the outputpath if it doesn't already exist
    #
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    atmotomo.prepareSimulation(
        path=os.path.join(output_path, 'two_clouds_low_density'),
        func=atmotomo.clouds_simulation,
        atmosphere_params=clouds_atmosphere,
        visibility=100
    )
    atmotomo.prepareSimulation(
        path=os.path.join(output_path, 'two_clouds_high_density'),
        func=atmotomo.clouds_simulation,
        atmosphere_params=clouds_atmosphere,
        visibility=10
    )
    atmotomo.prepareSimulation(
        path=os.path.join(output_path, 'two_clouds_high_density_low_resolution'),
        func=atmotomo.clouds_simulation,
        atmosphere_params=clouds_atmosphere_low_resolution,
        camera_resolution=(32, 32),
        camera_grid_size=(4, 3),
        visibility=10000
    )


if __name__ == '__main__':
    #
    # Parse the input
    #
    parser = argparse.ArgumentParser(description='Create configuration files for the atmosphere simulations')
    parser.add_argument('--output_path', default=None, help='path to configuration files')
    args = parser.parse_args()
    
    if args.output_path == None:
        output_path = amitibo.getResourcePath('configurations', package_name='atmotomo')
    else:
        output_path = os.path.abspath(args.output_path)

    main(output_path)
