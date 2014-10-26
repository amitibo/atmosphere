"""
"""

from __future__ import division
import numpy as np
from amitibo import tamnun
from pkg_resources import resource_filename
import itertools
import argparse
import atmotomo
import glob
from pkg_resources import resource_filename
import os

BASE_ARGS = "--highten_atmosphere --regularization_decay 0.0005 --ref_ratio {ref_ratio} --job_id $PBS_JOBID --mask_sun manual {configuration} --weights 1.0 1.0 0.1 --camera_num {camera_num} --noise_params {QE} {F} {B} {DN_mean} {DN_sigma} {t}"

QUEUE_NAMES = [
    'minerva_h_p',
    'all_l_p',
    'np_weekend',
    'gpu_l_p',
    'general_ld'
]


def list_prompt(vals):
    prompt = ['[{i}]: {l}'.format(i=i, l=l) for i, l in enumerate(vals)]
    prompt.append('-->')

    while True:
        selection = raw_input('\n'.join(prompt))
        try:
            selection = int(selection)

            if selection in range(len(vals)):
                break
        except:
            pass

        print 'Wrong selction.'

    print '\nSelected:%s\n' % vals[selection]

    return vals[selection]


def main():
    """Main doc """
    
    parser = argparse.ArgumentParser(description='Run analysis with varying number of cameras')
    parser.add_argument('--queue_name', default=None, help='Queue name. (default=all_l_p)')
    parser.add_argument('--ref_ratio', type=float, default=41.29, help='Reference ratio between mc and single scattering images.')
    parser.add_argument('--camera_num', type=int, default=-1, help='number of cameras.')
    parser.add_argument('--config', default=None, help='Name of configuration to use')
    
    args = parser.parse_args()
        
    #
    # List all configurations
    #
    if args.config is None:
        base_path = resource_filename('atmotomo', 'data/configurations')
        folder_list = [os.path.split(folder)[-1] for folder in glob.glob(os.path.join(base_path, '*'))]
        folder_list.sort()
 
        configuration = list_prompt(folder_list)
    else:
        configuration = args.config

    #
    # Check the required number of cores
    #
    if args.camera_num < 1:
        import atmotomo
        atmosphere_params, particle_params, sun_params, camera_params, cameras, air_dist, aerosols_dist = \
                atmotomo.readConfiguration(configuration)

        camera_num = len(cameras)
    else:
        camera_num = args.camera_num

    #
    #
    #
    if args.queue_name is None:
        os.system('FreeNodes-all')
        os.system('FreeNodes-minerva')
        os.system('FreeNodes-gpu')

        queue_name = list_prompt(QUEUE_NAMES)
    else:
        queue_name = args.queue_name

    #
    # Submit the separate jobs 
    #
    jobs_id = []
    for QE, F, B, DN_mean, DN_sigma, t in itertools.product((0.15, 0.3), (10000, 20000, 40000), (8, 16), (6, 12), (2, 4), (10, 50, 100)):
        params =  BASE_ARGS.format(
            ref_ratio=args.ref_ratio,
            configuration=configuration,
            camera_num=camera_num,
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
                queue_name=queue_name
            )
        )
        
    
if __name__ == '__main__':
    main()

    
    
