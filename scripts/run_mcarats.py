"""
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from atmotomo import RGB_WAVELENGTH, getResourcePath
import amitibo
import jinja2
import os
import subprocess as sub


GRADS_TEMPLATE_FILE_NAME = 'grads.jinja'
CONF_TEMPLATE_FILE_NAME = 'conf.jinja'
SOLVER_F3D = 0
COLOR_BALANCE = (1.28, 1.0, 0.8)


def storeGRADS(file_name, *params):
    
    array_tuple = []
    for arr in params:
        array_tuple.append(arr.astype(np.float32).transpose(range(arr.ndim-1, -1, -1)).ravel())
    
    stored_array = np.hstack(array_tuple)
    stored_array.tofile(file_name)
    
    
def loadGRADS(file_name):
    pass


def createConfFile(file_name, tpl_env):

    tpl = tpl_env.get_template(CONF_TEMPLATE_FILE_NAME)
    
    with open(file_name, 'w') as f:
        f.write(
            tpl.render(
            )            
        )
        

def createAtmFile(file_name, shape, tpl_env):

    tpl = tpl_env.get_template(GRADS_TEMPLATE_FILE_NAME)
    
    ctl_file_name = '%s.ctl' % file_name
    with open(ctl_file_name, 'w') as f:
        f.write(
            tpl.render(
            file_name=file_name,
            x_axis=shape[1],
            y_axis=shape[0],
            z_axis=shape[2]
            )            
        )
        
    tmpa3d = np.ones(shape) # Temperature perturbation (K)
    abst3d = np.ones(shape) # Absorption coefficient perturbation (/m)
    extp3d = np.ones(shape) # Extinction coefficient (/m)
    omgp3d = np.ones(shape) # Single scattering albedo
    apfp3d = np.ones(shape) # Phase function specification parameter
    
    storeGRADS(file_name, tmpa3d, abst3d, extp3d, omgp3d, apfp3d)
    

def calcExposure(file_name):
    
    os.system('bin_exposure')
    
    
def main(photon_num=1e4, solver=SOLVER_F3D, ):
    """Main doc"""
    
    #
    # Create the template environment
    #
    tpl_loader = jinja2.FileSystemLoader(searchpath=getResourcePath('.'))
    tpl_env = jinja2.Environment(loader=tpl_loader)
    
    #
    # Create the results folder
    #
    results_path = amitibo.createResultFolder()
    
    conf_file_name = os.path.join(results_path, 'conf_base')
    atmo_file_name = os.path.join(results_path, 'base.atm')
    output_file_name = os.path.join(results_path, 'out')
    img_file_name = os.path.join(results_path, 'img')
    
    #
    # Loop on different color channels
    #
    img_channels = []
    for wlen_i, wlen in enumerate(RGB_WAVELENGTH):
        #
        # Prepare the init files
        #
        createConfFile(file_name=conf_file_name, tpl_env=tpl_env)
        createAtmFile(file_name=atmo_file_name, shape=(50, 50, 41), tpl_env=tpl_env)
    
        #
        # Run the simulation
        #
        cmd = 'mcarats %(photon_num)d %(solver)d %(conf_file)s %(output_file)s' % {
            'photon_num': photon_num,
            'solver': solver,
            'conf_file': conf_file_name,
            'output_file': output_file_name
        }
        prc_ret = sub.Popen(cmd, shell=True, stdin=sub.PIPE, stdout=sub.PIPE, stderr=sub.PIPE)
        #print prc_ret.stdout.read()
        prc_ret.wait()
        
        #
        # Create the channel image
        #
        cmd = 'bin_exposure %(time_lag)d %(time_width)d %(fmax)g %(power)g %(ctl_file)s.ctl' % {
            'time_lag': 0,
            'time_width': 1,
            'fmax': 2,
            'power': 0.6,
            'ctl_file': output_file_name
        }
        prc_ret = sub.Popen(cmd, shell=True, stdin=sub.PIPE, stdout=sub.PIPE, stderr=sub.PIPE)
        ret_txt = prc_ret.stdout.read()
        
        Rmax = float(ret_txt.split('\n')[1].split()[1])
        print Rmax
        
        cmd = 'bin_gray %(factor)g %(Rmax)g %(power)g %(ctl_file)s.ctl %(img_file)s' % {
            'factor': COLOR_BALANCE[wlen_i],
            'Rmax': Rmax,
            'ctl_file': output_file_name,
            'power': 0.6,
            'img_file': img_file_name
        }

        prc_ret = sub.Popen(cmd, shell=True, stdin=sub.PIPE, stdout=sub.PIPE, stderr=sub.PIPE)
        ret_split = prc_ret.stdout.read().split()
        img_width = int(ret_split[0])
        img_height = int(ret_split[1])
        gray_file_name = ret_split[3]
        img_channels.append(np.fromfile(gray_file_name, dtype=np.uint8).reshape((img_height, img_width)))
        
    #
    # Create the final image
    #
    img = np.dstack(img_channels)
    plt.imshow(img)
    plt.show()
    
    
if __name__ == '__main__':
    main()

    
    