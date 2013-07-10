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
        np.arange(0, 50000, 1000.0), # Y
        np.arange(0, 50000, 1000.0), # X
        np.arange(0, 10000, 100.0)   # H
        ),
    earth_radius=4000000,
    air_typical_h=8000,
    aerosols_typical_h=2000
)

clouds_atmosphere_medium_resolution = amitibo.attrClass(
    cartesian_grids=spt.Grids(
        np.arange(0, 50000, 5000.0), # Y
        np.arange(0, 50000, 5000.0), # X
        np.arange(0, 10000, 500.0)   # H
        ),
    earth_radius=4000000,
    air_typical_h=8000,
    aerosols_typical_h=2000
)

clouds_atmosphere_low_resolution = amitibo.attrClass(
    cartesian_grids=spt.Grids(
        np.arange(0, 50000, 10000.0), # Y
        np.arange(0, 50000, 10000.0), # X
        np.arange(0, 10000, 2000.0)   # H
        ),
    earth_radius=4000000,
    air_typical_h=8000,
    aerosols_typical_h=2000
)

def main(output_path):
    #
    # Create the outputpath if it doesn't already exist
    #
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    atmotomo.prepareSimulation(
        path=os.path.join(output_path, 'two_clouds_low_density'),
        func=atmotomo.two_layer_clouds_simulation,
        atmosphere_params=clouds_atmosphere,
        aerosols_typical_density=10**6
    )
    atmotomo.prepareSimulation(
        path=os.path.join(output_path, 'two_clouds_high_density'),
        func=atmotomo.two_layer_clouds_simulation,
        atmosphere_params=clouds_atmosphere,
        aerosols_typical_density=10**7
    )
    atmotomo.prepareSimulation(
        path=os.path.join(output_path, 'two_clouds_low_density_medium_resolution'),
        func=atmotomo.two_layer_clouds_simulation,
        atmosphere_params=clouds_atmosphere_medium_resolution,
        camera_resolution=(32, 32),
        camera_grid_size=(4, 3),
        aerosols_typical_density=10**6
    )
    atmotomo.prepareSimulation(
        path=os.path.join(output_path, 'two_clouds_high_density_medium_resolution'),
        func=atmotomo.two_layer_clouds_simulation,
        atmosphere_params=clouds_atmosphere_medium_resolution,
        camera_resolution=(32, 32),
        camera_grid_size=(4, 3),
        aerosols_typical_density=10**7
    )
    atmotomo.prepareSimulation(
        path=os.path.join(output_path, 'two_clouds_low_density_low_resolution'),
        func=atmotomo.two_layer_clouds_simulation,
        atmosphere_params=clouds_atmosphere_low_resolution,
        camera_resolution=(32, 32),
        camera_grid_size=(4, 3),
        aerosols_typical_density=10**6
    )
    atmotomo.prepareSimulation(
        path=os.path.join(output_path, 'two_clouds_high_density_low_resolution'),
        func=atmotomo.two_layer_clouds_simulation,
        atmosphere_params=clouds_atmosphere_low_resolution,
        camera_resolution=(32, 32),
        camera_grid_size=(4, 3),
        aerosols_typical_density=10**7
    )
    atmotomo.prepareSimulation(
        path=os.path.join(output_path, 'front_low_density_medium_resolution'),
        func=atmotomo.front_simulation,
        atmosphere_params=clouds_atmosphere_medium_resolution,
        camera_resolution=(32, 32),
        camera_grid_size=(4, 3),
        aerosols_typical_density=10**6
    )
    atmotomo.prepareSimulation(
        path=os.path.join(output_path, 'front_high_density_medium_resolution'),
        func=atmotomo.front_simulation,
        atmosphere_params=clouds_atmosphere_medium_resolution,
        camera_resolution=(32, 32),
        camera_grid_size=(4, 3),
        aerosols_typical_density=10**7
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
