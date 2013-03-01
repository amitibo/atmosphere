"""
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from atmotomo import RGB_WAVELENGTH, getResourcePath, density_clouds1, calcAirMcarats
import amitibo
import jinja2
import os
import subprocess as sub


GRADS_TEMPLATE_FILE_NAME = 'grads.jinja'
CONF_TEMPLATE_FILE_NAME = 'conf.jinja'
SOLVER_F3D = 0
COLOR_BALANCE = (1.28, 1.0, 0.8)
SEA_LEVEL_TEMP = 290

def storeGRADS(file_name, *params):
    
    array_tuple = []
    for arr in params:
        array_tuple.append(arr.astype(np.float32).transpose(range(arr.ndim-1, -1, -1)).ravel())
    
    stored_array = np.hstack(array_tuple)
    stored_array.tofile(file_name)
    
    
def loadGRADS(file_name):
    pass


def arr2str(arr):
    
    val = ', '.join(['%g' % i for i in arr])
    return val


class Mcarats(object):
    
    def __init__(self, base_folder):
        #
        # Create the template environment
        #
        tpl_loader = jinja2.FileSystemLoader(searchpath=getResourcePath('.'))
        self._tpl_env = jinja2.Environment(loader=tpl_loader)
    
        self._atmo_file_name = os.path.join(base_folder, 'base.atm')
        self._conf_file_name = os.path.join(base_folder, 'conf_base')
        self._out_file_name = os.path.join(base_folder, 'out')
        self._img_file_name = os.path.join(base_folder, 'img')

        self._ext1D = []
        self._omg1D = []
        self._apf1D = []
        self._ext3D = []
        self._omg3D = []
        self._apf3D = []

        self._sun_theta = 0.0
        self._sun_phi = 0.0
        
    def setAtmosphereDims(self, shape, dx, dy, z_coords):
        
        self._shape = shape
        self._dx = dx
        self._dy = dy
        self._z_coords = z_coords

    def add1Ddistribution(self, ext1d, omg1d, apf1d):
        
        self._ext1D.append(ext1d)
        self._omg1D.append(omg1d)
        self._apf1D.append(apf1d)
        
    def add3Ddistribution(self, ext3d, omg3d, apf3d):
        
        self._ext3D.append(ext3d)
        self._omg3D.append(omg3d)
        self._apf3D.append(apf3d)

    def setSolarSource(self, theta=0.0, phi=0.0):
        
        self._sun_theta = theta
        self._sun_phi = phi
        
    def addCamera(self, camera_params):
        
        self._camera_params = camera_params
        
    def _createConfFile(self):
        """Create the configuration file for the simulation"""
        
        tpl = self._tpl_env.get_template(CONF_TEMPLATE_FILE_NAME)
        
        #
        # The 1-D data relates to Air distribution
        #
        with open(self._conf_file_name, 'w') as f:
            f.write(
                tpl.render(
                    atmo_file_name=os.path.split(self._atmo_file_name)[-1],
                    x_axis=self._shape[0],
                    y_axis=self._shape[1],
                    z_axis=self._shape[2],
                    cameras_num=1,
                    img_size=self._camera_params.image_res,
                    dx=self._dx,
                    dy=self._dy,
                    z_coords=arr2str(self._z_coords),
                    tmp1d=arr2str(SEA_LEVEL_TEMP*np.ones_like(self._z_coords)),
                    ext1d=arr2str(self._ext1D[0]),
                    omg1d=arr2str(self._omg1D[0]),
                    apf1d=arr2str(self._apf1D[0]),
                    abs1d=arr2str(np.zeros_like(self._z_coords)),
                    sun_theta=self._sun_theta,
                    sun_phi=self._sun_phi
                )            
            )
            
    def _createAtmFile(self):
        """Create the atmosphere file"""
        
        tpl = self._tpl_env.get_template(GRADS_TEMPLATE_FILE_NAME)

        ctl_file_name = '%s.ctl' % self._atmo_file_name
        with open(ctl_file_name, 'w') as f:
            f.write(
                tpl.render(
                file_name=os.path.split(self._atmo_file_name)[-1],
                x_axis=self._shape[1],
                y_axis=self._shape[0],
                z_axis=self._shape[2]
                )            
            )
        
        #
        # tmpa3d - Temperature perturbation (K)
        # abst3d - Absorption coefficient perturbation (/m)
        # extp3d - Extinction coefficient (/m)
        # omgp3d - Single scattering albedo
        # apfp3d - Phase function specification parameter
        #
        tmpa3d = np.zeros(self._shape)
        abst3d = np.zeros(self._shape)
        extp3d = self._ext3D[0]
        omgp3d = self._omg3D[0]
        apfp3d = self._apf3D[0]
        
        storeGRADS(self._atmo_file_name, tmpa3d, abst3d, extp3d, omgp3d, apfp3d)

    def _run_simulation(self, photon_num, solver):
        """Run the simulation"""
        
        cmd = 'mcarats %(photon_num)d %(solver)d %(conf_file)s %(output_file)s' % {
            'photon_num': photon_num,
            'solver': solver,
            'conf_file': self._conf_file_name,
            'output_file': self._out_file_name
        }
        prc_ret = sub.Popen(cmd, shell=True, stdin=sub.PIPE, stdout=sub.PIPE, stderr=sub.PIPE)
        print prc_ret.stdout.read()
        prc_ret.wait()
        
    def _calcImgChannel(self):
        """"""
        #
        # Calculate exposure
        #
        cmd = 'bin_exposure %(time_lag)d %(time_width)d %(fmax)g %(power)g %(ctl_file)s.ctl' % {
            'time_lag': 0,
            'time_width': 1,
            'fmax': 2,
            'power': 0.6,
            'ctl_file': self._out_file_name
        }
        print cmd
        prc_ret = sub.Popen(cmd, shell=True, stdin=sub.PIPE, stdout=sub.PIPE, stderr=sub.PIPE)
        ret_txt = prc_ret.stdout.read()
        print ret_txt
        Rmax = float(ret_txt.split('\n')[1].split()[1])
        print Rmax
        
        #
        # Create gray image
        #
        cmd = 'bin_gray %(factor)g %(Rmax)g %(power)g %(ctl_file)s.ctl %(img_file)s' % {
            'factor': COLOR_BALANCE[0],
            'Rmax': Rmax,
            'ctl_file': self._out_file_name,
            'power': 0.6,
            'img_file': self._img_file_name
        }
        print cmd
        prc_ret = sub.Popen(cmd, shell=True, stdin=sub.PIPE, stdout=sub.PIPE, stderr=sub.PIPE)

        ret_split = prc_ret.stdout.read().split()
        img_width = int(ret_split[0])
        img_height = int(ret_split[1])
        gray_file_name = ret_split[3]
        img = np.fromfile(gray_file_name, dtype=np.uint8).reshape((img_height, img_width))
        
        return img

    def run(self, photon_num=1e4, solver=SOLVER_F3D):
        #
        # Prepare the init files
        #
        self._createConfFile()
        self._createAtmFile()

        #
        # Run the simulation
        #
        self._run_simulation(photon_num, solver)
        
        #
        # Create the channel image
        #
        img = self._calcImgChannel()
        
        return img
         
    
def main(photon_num=1e6, solver=SOLVER_F3D):
    """Main doc"""
    
    #
    # Load the MISR database.
    #
    import pickle
    
    with open(getResourcePath('misr.pkl'), 'rb') as f:
        misr = pickle.load(f)
    
    particles_list = misr.keys()
    particle = misr['spherical_nonabsorbing_2.80']

    #
    # Simulation parameters
    #
    atmosphere_params = amitibo.attrClass(
        cartesian_grids=(
            slice(0, 50, 1.0), # Y
            slice(0, 50, 1.0), # X
            slice(0, 10, 0.1)   # H
            ),
        earth_radius=4000,
        RGB_WAVELENGTH=RGB_WAVELENGTH,
        air_typical_h=8,
        aerosols_typical_h=2,
        sun_angle=30
    )

    camera_params = amitibo.attrClass(
        image_res=128
    )
    
    A_air, A_aerosols, Y, X, Z, h = density_clouds1(atmosphere_params)
    dx = abs(X[0, 1, 0] - X[0, 0, 0])*1000
    dy = abs(Y[1, 0, 0] - Y[0, 0, 0])*1000
    dz = abs(Z[0, 0, 1] - Z[0, 0, 0])*1000
    z_coords = Z[0, 0, :]*1000
    z_coords = np.concatenate((z_coords, [z_coords[-1]+dz]))
    e45, e55, e67 = calcAirMcarats(z_coords)

    #
    # Create the results folder
    #
    results_path = amitibo.createResultFolder()

    #
    # Create the test
    #
    mc = Mcarats(results_path)
    mc.setAtmosphereDims(shape=A_aerosols.shape, dx=dx, dy=dy, z_coords=z_coords)
    mc.add1Ddistribution(
        ext1d=e45,
        omg1d=np.ones_like(e45),
        apf1d=-1*np.ones_like(e45)
    )
    mc.add3Ddistribution(
        ext3d=particle['k'][0]*A_aerosols,
        omg3d=particle['w'][0]*np.ones_like(A_aerosols),
        apf3d=particle['g'][0]*np.ones_like(A_aerosols)
    )
    mc.addCamera(camera_params)
    mc.setSolarSource(theta=30.0)
    
    #
    # Run the test
    #
    img = mc.run()
    
    #
    # Show the results
    #
    plt.imshow(img)
    plt.show()
    
    
if __name__ == '__main__':
    main()

    
    