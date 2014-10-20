"""
"""

from __future__ import division
import numpy as np
from amitibo import tamnun
import argparse


def main():
    """Main doc """
    
    parser = argparse.ArgumentParser(description='Run analysis with varying number of cameras')
    parser.add_argument('--queue_name', default='all_l_p', help='Queue name. (default=all_l_p)')
    parser.add_argument('--repetitions', type=int, default=10, help='Number of repetitions per camera number. (default=10).')
    parser.add_argument('--ref_ratio', type=float, help='Reference ratio between mc and single scattering images.')
    parser.add_argument('configuration', help='Name of configuration to use')
    
    args = parser.parse_args()
        
    #
    # Submit the separate jobs 
    #
    jobs_id = []
    for i in range(args.repetitions):
        for camera_num in cameras_number_list:
            params = "--highten_atmosphere --regularization_decay 0.0005 --ref_ratio %g --job_id $PBS_JOBID --mask_sun manual %s --weights 1.0 1.0 0.1 --camera_num %d" % \
                (args.ref_ratio, args.configuration, camera_num)
            
            jobs_id.append(
                tamnun.qsub(
                    cmd='$HOME/anaconda/bin/analyzeAtmo3D.py',
                    params=params,
                    M=camera_num+1,
                    N=1,
                    arrangement='free',
                    work_directory='$HOME/code/atmosphere',
                    queue_name=args.queue_name
                )
            )
        
    
if __name__ == '__main__':
    main()

    
    