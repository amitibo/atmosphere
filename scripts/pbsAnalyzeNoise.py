"""
"""

from __future__ import division
import numpy as np
from amitibo import tamnun
import itertools
import argparse

BASE_ARGS = "--highten_atmosphere --regularization_decay 0.0005 --ref_ratio {ref_ratio} --job_id $PBS_JOBID --mask_sun manual {configuration} --weights 1.0 1.0 0.1 --camera_num {camera_num} --noise_params {QE} {F} {B} {DN_mean} {DN_sigma} {t}"

def main():
    """Main doc """
    
    parser = argparse.ArgumentParser(description='Run analysis with varying number of cameras')
    parser.add_argument('--queue_name', default='all_l_p', help='Queue name. (default=all_l_p)')
    parser.add_argument('--repetitions', type=int, default=10, help='Number of repetitions per camera number. (default=10).')
    parser.add_argument('--ref_ratio', type=float, default=41.29, help='Reference ratio between mc and single scattering images.')
    parser.add_argument('--camera_num', type=int, default=-1, help='number of cameras.')
    parser.add_argument('configuration', help='Name of configuration to use')
    
    args = parser.parse_args()
        
    #
    # Submit the separate jobs 
    #
    jobs_id = []
    for QE, F, B, DN_mean, DN_sigma, t in itertools.product((0.15, 0.3), (20000,), (8, 16), (6,), (2,), (10, 50)):
        params =  BASE_ARGS.format(
            ref_ratio=args.ref_ratio,
            configuration=args.configuration, args.camera_num,
            camera_num=args.camera_num,
            QE=QE,
            F=F,
            B=B,
            DN_mean=DN_mean,
            DN_sigma=DN_sigma,
            t=t
        )
        
        jobs_id.append(
            tamnun.qsub(
                cmd='$HOME/anaconda/bin/analyzeAtmo3D.py',
                params=params,
                M=camera_num,
                N=1,
                arrangement='free',
                work_directory='$HOME/code/atmosphere',
                queue_name=args.queue_name
            )
        )
        
    
if __name__ == '__main__':
    main()

    
    