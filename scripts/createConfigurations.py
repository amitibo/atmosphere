"""
Create configuration files for a atmosphere simulations.
"""
from __future__ import division
import numpy as np
import atmotomo 
import scipy.io as sio
import os
import argparse


def main(output_path):
    #
    # Create the outputpath if it doesn't already exist
    #
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    atmotomo.prepareSimulation(path=output_path, func=atmotomo.clouds_simulation)
    
if __name__ == '__main__':
    #
    # Parse the input
    #
    parser = argparse.ArgumentParser(description='Create configuration files for the atmosphere simulations')
    parser.add_argument('output_path', help='path to configuration files')
    args = parser.parse_args()
    
    output_path = os.path.abspath(args.output_path)

    main(output_path)
