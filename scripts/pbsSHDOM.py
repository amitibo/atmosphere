"""
"""

from __future__ import division
import numpy as np
from amitibo import tamnun
import argparse


def main():
    """Main doc """
    
    parser = argparse.ArgumentParser(
        description='Run SHDOM tasks using the tamnun.'
    )
    parser.add_argument('--queue_name', default='all_l_p', help='Queue name. (default=all_l_p)')
    parser.add_argument('--extinction', default=None, help='Path to extinction file')
    parser.add_argument('--cores_num', default=36, type=int, help='Number of cores to use')
    parser.add_argument('configuration', help='Name of configuration to use')
    
    args = parser.parse_args()
        
    #
    # Submit the separate jobs 
    #
    params = "--parallel --extinction {extinction} --job_id $PBS_JOBID {configuration}".format(
        extinction=args.extinction,
        configuration=args.configuration
    )
    
    tamnun.qsub(
        cmd='$HOME/anaconda/bin/simulateSHDOM.py',
        params=params,
        M=args.cores_num,
        N=1,
        arrangement='free',
        work_directory='$HOME/code/atmosphere',
        queue_name=args.queue_name
    )

    
if __name__ == '__main__':
    main()

    
    
