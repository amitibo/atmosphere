"""
Simulate the scattering of the sky where the aerosols have a general distribution.
"""

from __future__ import division
import numpy as np
from atmotomo import SHDOM
import os
import argparse


if __name__ == '__main__':

    #
    # Parse the input
    #
    parser = argparse.ArgumentParser(description='Simulate atmosphere')
    parser.add_argument('--parallel', action='store_true', help='run the parallel mode')
    parser.add_argument('--job_id', default=None, help='pbs job ID (set automatically by the PBS script)')
    parser.add_argument('--cameras_limit', type=int, default=-1, help='Limit the number of cameras simulated.')
    parser.add_argument('params_path', help='Path to simulation parameters')
    args = parser.parse_args()
    
    shdom = SHDOM(parallel=args.parallel)
    shdom.load_configuration(
        config_name=args.params_path,
        particle_name='spherical_absorbing_0.57_ssa_green_0.94'
    )
    
    shdom.forward(cameras_limit=args.cameras_limit)
