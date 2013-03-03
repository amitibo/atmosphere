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
    
    
def loadGRADS(file_name, dtype=np.float32):
    """Load a grads file with a strict syntax as used by mcarats"""
    
    with open(file_name, 'r') as fin:
        lines = fin.readlines()
        
    lines = [line.strip() for line in lines]
    
    dset_file = lines[0].split()[1][1:]
    undef = float(lines[5].split()[1])
    xdef, ydef, zdef, tdef, vars_num = [int(line.split()[1]) for line in lines[6:11]]
    
    vars_names = []
    vars_z = []
    for i in range(11, 11+vars_num):
        parts = lines[i].split()
        vars_names.append(parts[0])
        vars_z.append(int(parts[1]))

    dset_path = os.path.join(os.path.split(file_name)[0], dset_file)
    values = np.fromfile(dset_path, dtype=dtype)
    values[values==undef] = np.nan
    
    vars_dict = {}
    start_i = 0
    for name, z in zip(vars_names, vars_z):
        size = xdef * ydef * z
        end_i = start_i + size
        vars_dict[name] = values[start_i:end_i].reshape(xdef, ydef, z).transpose((2, 1, 0))
        start_i = end_i
    
    return vars_dict


def arr2str(arr):
    
    val = ', '.join(['%g' % i for i in arr])
    return val


class Mcarats(object):
    
    def __init__(self, base_folder, base_name='base'):
        #
        # Create the template environment
        #
        tpl_loader = jinja2.FileSystemLoader(searchpath=getResourcePath('.'))
        self._tpl_env = jinja2.Environment(loader=tpl_loader)
    
        self._atmo_file_name = os.path.join(base_folder, '%s.atm' % base_name)
        self._conf_file_name = os.path.join(base_folder, '%s_conf' % base_name)
        self._out_file_name = os.path.join(base_folder, '%s_out' % base_name)

        self._ext1D = []
        self._omg1D = []
        self._apf1D = []
        self._abs1D = []
        self._ext3D = []
        self._omg3D = []
        self._apf3D = []
        self._abs3D = []

        self._sun_theta = 0.0
        self._sun_phi = 0.0
        
    def setAtmosphereDims(self, shape, dx, dy, z_coords, iz3l=None, nz3=None):
        
        self._shape = shape
        self._dx = dx
        self._dy = dy
        self._z_coords = z_coords
        
        if iz3l == None:
            self._iz3l = 1
            self._nz3 = shape[2]
        else:
            self._iz3l = iz3l
            self._nz3 = nz3

    def add1Ddistribution(self, ext1d, omg1d, apf1d, abs1d=None):
        
        self._ext1D.append(ext1d)
        self._omg1D.append(omg1d)
        self._apf1D.append(apf1d)
        if abs1d == None:
            self._abs1D.append(np.zeros_like(ext1d))
        else:
            self._abs1D.append(abs1d)

    def add3Ddistribution(self, ext3d, omg3d, apf3d, abs3d=None):
        
        self._ext3D.append(ext3d)
        self._omg3D.append(omg3d)
        self._apf3D.append(apf3d)
        if abs3d == None:
            self._abs3D.append(np.zeros_like(ext3d))
        else:
            self._abs3D.append(abs3d)

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
                    iz3l=self._iz3l,
                    nz3=self._nz3,
                    cameras_num=1,
                    img_size=self._camera_params.image_res,
                    dx=self._dx,
                    dy=self._dy,
                    z_coords=arr2str(self._z_coords),
                    tmp1d=arr2str(SEA_LEVEL_TEMP*np.ones_like(self._z_coords)),
                    ext1d=arr2str(self._ext1D[0]),
                    omg1d=arr2str(self._omg1D[0]),
                    apf1d=arr2str(self._apf1D[0]),
                    abs1d=arr2str(self._abs1D[0]),
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
        tmpa3d = SEA_LEVEL_TEMP*np.ones(self._shape)
        abst3d = self._abs3D[0]
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
    
    @staticmethod
    def calcImg(out_file_names):
        """"""
        #
        # Calculate average exposure
        #
        ctl_files = ' '.join(['%s.ctl' % ctl_file for ctl_file in out_file_names])
        cmd = 'bin_exposure %(time_lag)d %(time_width)d %(fmax)g %(power)g %(ctl_files)s' % {
            'time_lag': 0,
            'time_width': 1,
            'fmax': 2,
            'power': 0.6,
            'ctl_files': ctl_files
        }
        print cmd
        prc_ret = sub.Popen(cmd, shell=True, stdin=sub.PIPE, stdout=sub.PIPE, stderr=sub.PIPE)
        ret_txt = prc_ret.stdout.read()
        print ret_txt
        Rmax = float(ret_txt.split('\n')[1].split()[1])
        print Rmax

        imgs = []
        for out_file_name in out_file_names:
            #
            # Create gray image
            #
            img_file_name = os.path.join(os.path.split(out_file_name)[0], 'img')
            cmd = 'bin_gray %(factor)g %(Rmax)g %(power)g %(ctl_file)s.ctl %(img_file)s' % {
                'factor': COLOR_BALANCE[0],
                'Rmax': Rmax,
                'ctl_file': out_file_name,
                'power': 0.6,
                'img_file': img_file_name
            }
            print cmd
            prc_ret = sub.Popen(cmd, shell=True, stdin=sub.PIPE, stdout=sub.PIPE, stderr=sub.PIPE)
    
            ret_split = prc_ret.stdout.read().split()
            img_width = int(ret_split[0])
            img_height = int(ret_split[1])
            gray_file_name = ret_split[3]
            imgs.append(np.fromfile(gray_file_name, dtype=np.uint8).reshape((img_height, img_width)))
        
        return imgs

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
        
        return self._out_file_name
         
    
def main(photon_num=1e7, solver=SOLVER_F3D):
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
        image_res=512
    )
    
    A_air, A_aerosols, Y, X, Z, h = density_clouds1(atmosphere_params)
    dx = abs(X[0, 1, 0] - X[0, 0, 0])*1000
    dy = abs(Y[1, 0, 0] - Y[0, 0, 0])*1000
    dz = abs(Z[0, 0, 1] - Z[0, 0, 0])*1000
    z_coords = Z[0, 0, :]*1000
    z_coords = np.concatenate((z_coords, [z_coords[-1]+dz]))
    air_ext = calcAirMcarats(z_coords)

    #
    # Create the results folder
    #
    results_path = amitibo.createResultFolder()

    #
    # Create the test
    #
    out_files = []
    for ch in range(3):
        mc = Mcarats(results_path)
        mc.setAtmosphereDims(shape=A_aerosols.shape, dx=dx, dy=dy, z_coords=z_coords)
        mc.add1Ddistribution(
            ext1d=air_ext[ch],
            omg1d=np.ones_like(air_ext[ch]),
            apf1d=-1*np.ones_like(air_ext[ch])
        )
        mc.add3Ddistribution(
            ext3d=particle['k'][ch]*A_aerosols,
            omg3d=particle['w'][ch]*np.ones_like(A_aerosols),
            apf3d=particle['g'][ch]*np.ones_like(A_aerosols)
        )
        mc.addCamera(camera_params)
        mc.setSolarSource(theta=120.0, phi=180.0)
        
        #
        # Run the test
        #
        out_files.append(mc.run())
    
    img = np.dstack(Mcarats.calcImg(out_files))
    
    #
    # Show the results
    #
    plt.imshow(img)
    plt.show()
    
    
def main_0045(photon_num=1e7, solver=SOLVER_F3D):
    """Main doc"""
    
    camera_params = amitibo.attrClass(
        image_res=128
    )
    
    dx = 30
    dy = 30
    z_coords = np.array((0.000000E+00,  3.000000E+01,  6.000000E+01,  9.000000E+01,  1.200000E+02,  1.500000E+02,  1.800000E+02,  2.100000E+02,  2.300000E+02,  2.500000E+02,  2.600000E+02,  2.700000E+02,  2.800000E+02,  2.900000E+02,  3.000000E+02,  3.100000E+02,  3.200000E+02,  3.300000E+02,  3.400000E+02,  3.500000E+02,  3.600000E+02,  3.700000E+02,  3.800000E+02,  3.900000E+02,  4.000000E+02,  4.100000E+02,  4.200000E+02,  4.300000E+02,  4.400000E+02,  4.500000E+02,  4.600000E+02,  6.000000E+02,  1.000000E+03,  2.000000E+03,  3.000000E+03,  4.000000E+03,  6.000000E+03,  8.000000E+03,  1.000000E+04,  1.500000E+04,  2.000000E+04,  3.000000E+04))
    Atm_ext1d = np.array((2.45113E-05,  2.44561E-05,  2.43958E-05,  2.43346E-05,  2.42734E-05,  2.42117E-05,  2.41498E-05,  2.40980E-05,  2.40568E-05,  2.40256E-05,  2.40046E-05,  2.39831E-05,  2.39609E-05,  2.39383E-05,  2.39154E-05,  2.38922E-05,  2.38690E-05,  2.38456E-05,  2.38222E-05,  2.37988E-05,  2.37755E-05,  2.37520E-05,  2.37279E-05,  2.37015E-05,  2.36700E-05,  2.36279E-05,  2.35715E-05,  2.35065E-05,  2.34441E-05,  2.33933E-05,  2.31618E-05,  2.24001E-05,  2.07798E-05,  1.87793E-05,  1.69788E-05,  1.45587E-05,  1.17577E-05,  9.41149E-06,  6.07198E-06,  2.91507E-06,  9.61213E-07))
    Atm_omg1d = np.array((1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ))
    Atm_apf1d = -Atm_omg1d
    Atm_abs1d = np.array((7.94729E-09,  7.94729E-09,  7.94729E-09,  7.94729E-09,  7.94729E-09,  7.94729E-09,  7.94729E-09,  5.96047E-09,  5.96047E-09,  5.96045E-09,  5.96047E-09,  5.96047E-09,  5.96045E-09,  5.96047E-09,  5.96047E-09,  5.96045E-09,  5.96047E-09,  5.96047E-09,  5.96045E-09,  5.96047E-09,  5.96047E-09,  5.96045E-09,  5.96047E-09,  5.96047E-09,  5.96045E-09,  5.96047E-09,  5.96047E-09,  5.96045E-09,  5.96047E-09,  5.96047E-09,  8.51495E-09,  8.94071E-09,  8.94074E-09,  9.05995E-09,  9.35797E-09,  9.95407E-09,  1.11164E-08,  1.27258E-08,  2.29610E-08,  4.34327E-08,  4.78620E-08))
    atm_file = getResourcePath('les05_0045.atm.ctl')
    atm_dict = loadGRADS(atm_file)
    
    #
    # Create the results folder
    #
    results_path = amitibo.createResultFolder()

    #
    # Create the test
    #
    mc = Mcarats(results_path)
    mc.setAtmosphereDims(shape=(60, 60, 41), dx=dx, dy=dy, z_coords=z_coords, iz3l=10, nz3=21)
    mc.add1Ddistribution(
        ext1d=Atm_ext1d,
        omg1d=Atm_omg1d,
        apf1d=Atm_apf1d,
        abs1d=Atm_abs1d
    )
    mc.add3Ddistribution(
        ext3d=atm_dict['extp3d'],
        omg3d=atm_dict['omgp3d'],
        apf3d=atm_dict['apfp3d'],
        abs3d=atm_dict['abst3d']
    )
    mc.addCamera(camera_params)
    mc.setSolarSource(theta=120.0, phi=180.0)
    
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

    
    