"""
"""

from __future__ import division
import numpy as np
from amitibo import tamnun
from pkg_resources import resource_filename
import argparse
import glob
import os

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
    
    parser = argparse.ArgumentParser(
        description='Run SHDOM tasks using the tamnun.'
    )
    parser.add_argument('--queue_name', default=None, help='Queue name.')
    parser.add_argument('--extinction', default=None, help='Path to extinction file')
    parser.add_argument('--cores_num', default=-1, type=int, help='Number of cores to use')
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
    if args.cores_num < 1:
        import atmotomo
        atmosphere_params, particle_params, sun_params, camera_params, cameras, air_dist, aerosols_dist = \
                atmotomo.readConfiguration(configuration)

        cores_num = len(cameras)
    else:
        cores_num = args.cores_num

    #
    # Check the type of run
    #
    cmds = ['simulateSHDOM.py', 'analyzeSHDOM.py']
    cmd = list_prompt(cmds)

    #
    # Create the parameters of the run
    #
    if args.extinction is None:
        params = "--parallel --job_id $PBS_JOBID {configuration}".format(
            configuration=configuration
        )
    else:
        params = "--parallel --extinction {extinction} --job_id $PBS_JOBID {configuration}".format(
            extinction=args.extinction,
            configuration=configuration
        )

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
    tamnun.qsub(
        cmd=os.path.join('$HOME/anaconda/bin', cmd),
        params=params,
        M=cores_num,
        N=1,
        arrangement='free',
        work_directory='$HOME/code/atmosphere',
        queue_name=queue_name
    )

    
if __name__ == '__main__':
    main()

