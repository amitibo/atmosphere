"""
"""

from __future__ import division
import numpy as np
from atmotomo import getResourcePath
import jinja2
import os
import subprocess as sub

__all__ = ["storeGRADS", "loadGRADS", "Mcarats", "SOLVER_F3D"]

GRADS_TEMPLATE_FILE_NAME = 'grads.jinja'
CONF_TEMPLATE_FILE_NAME = 'conf.jinja'
CAMERA_TEMPLATE_FILE_NAME = 'camera.jinja'
SOLVER_F3D = 0
COLOR_BALANCE = (1.28, 1.0, 0.8)
SEA_LEVEL_TEMP = 290
MCARATS_BIN = 'mcarats'
MCARATS_MPI_BIN = 'mcarats_mpi'


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
        vars_dict[name] = values[start_i:end_i].reshape(z, xdef, ydef).transpose((2, 1, 0))
        start_i = end_i
    
    return vars_dict


def arr2str(arr):
    
    val = ', '.join(['%g' % i for i in arr])
    return val


class BaseData(object):
    def __init__(self, template_path, tpl_env):
        self._tpl = tpl_env.get_template(template_path)
        self._data = {}
    
    def _appendDataField(self, field_name, value):
        if not field_name in self._data.keys():
            self._data[field_name] = []
        self._data[field_name].append(value)
    
    def addData(self, **kwrds):
        for k, v in kwrds.items():
            self._appendDataField(k ,v)
            
    def render(self):
        
        data = {}
        for k, v in self._data.items():
            if isinstance(v, list) or isinstance(v, tuple) or isinstance(v, np.ndarray):
                data[k] = arr2str(v)
            else:
                data[k] = v
                
        return self._tpl.render(data)
       

class Mcarats(object):
    
    def __init__(self, base_folder, base_name='base', use_mpi=True):
        #
        # Create the template environment
        #
        tpl_loader = jinja2.FileSystemLoader(searchpath=getResourcePath('.'))
        self._tpl_env = jinja2.Environment(loader=tpl_loader)
    
        self._atmo_file_name = os.path.join(base_folder, '%s.atm' % base_name)
        self._conf_file_name = os.path.join(base_folder, '%s_conf' % base_name)
        self._out_file_name = os.path.join(base_folder, '%s_out' % base_name)

        self._tmp1D = []
        self._ext1D = []
        self._omg1D = []
        self._apf1D = []
        self._abs1D = []
        
        self._tmp3D = []
        self._ext3D = []
        self._omg3D = []
        self._apf3D = []
        self._abs3D = []

        self._camera_data = BaseData(CAMERA_TEMPLATE_FILE_NAME, self._tpl_env)
        self._camera_num = 0
        
        self._sun_theta = 0.0
        self._sun_phi = 0.0
        
        if use_mpi:
            self._mcarats_bin = MCARATS_MPI_BIN
        else:
            self._mcarats_bin = MCARATS_BIN
            
    def configure(
        self,
        shape,
        dx,
        dy,
        z_coords,
        tmp_prof=0,
        iz3l=None,
        nz3=None,
        img_width=512,
        img_height=512
        ):
        """Configure the simulation"""
        
        self._shape = shape
        self._dx = dx
        self._dy = dy
        self._z_coords = z_coords
        self._tmp_prof = tmp_prof
        
        if iz3l == None:
            self._iz3l = 1
            self._nz3 = shape[2]
        else:
            self._iz3l = iz3l
            self._nz3 = nz3

        self._img_width = img_width
        self._img_height = img_height
        
    def add1Ddistribution(self, ext1d, omg1d, apf1d, tmp1d=None, abs1d=None):
        """Add a 1D distribution"""
        
        self._ext1D.append(ext1d)
        self._omg1D.append(omg1d)
        self._apf1D.append(apf1d)
        if tmp1d == None:
            self._tmp1D.append(SEA_LEVEL_TEMP*np.ones_like(self._z_coords))
        else:
            self._tmp1D.append(tmp1d)
        if abs1d == None:
            self._abs1D.append(np.zeros_like(ext1d))
        else:
            self._abs1D.append(abs1d)

    def add3Ddistribution(self, ext3d, omg3d, apf3d, tmp3d=None, abs3d=None):
        """Add a 3D distribution"""
        
        self._ext3D.append(ext3d)
        self._omg3D.append(omg3d)
        self._apf3D.append(apf3d)
        if tmp3d == None:
            self._tmp3D.append(SEA_LEVEL_TEMP*np.ones_like(ext3d))
        else:
            self._tmp3D.append(tmp3d)
        if abs3d == None:
            self._abs3D.append(np.zeros_like(ext3d))
        else:
            self._abs3D.append(abs3d)

    def setSolarSource(self, theta=0.0, phi=0.0):
        """Set the sun source"""
        
        self._sun_theta = theta
        self._sun_phi = phi
        
    def addCamera(
        self,
        xpos,
        ypos,
        zloc,
        rmin0=10.0,
        rmax0=18000.0,
        theta=0.0,
        phi=0.0,
        psi=0.0,
        umax=180,
        vmax=180,
        qmax=180,
        apsize=0.1
        ):
        """
        Add a camera to the simulation

        Parameters
        ----------
        xpos, ypos : float 
            Relative position of the camera in the x, y axes (in the range 0.0-1.0)
        
        zloc : float
            Location of the camera in the z axis (in meters)

        rmin0, rmax0 : float
            Minimum and maximum distance between emission and camera (in meters)
        
         theta, phi, psi : float
            Rotation angles of the camera (in degrees)
        
        umax, vmax : float
            Maximum angles of projection image coordinates (in degress)
        
        qmax : float
            Maximum FOV cone angle (in degress)
        
        apsize : float
            Diameter of the camera lens (in meters)
            
        Returns
        -------
        """
        
        self._camera_data.addData(
            xpos=xpos,
            ypos=ypos,
            zloc=zloc,
            rmin0=rmin0,
            rmax0=rmax0,
            theta=theta,
            phi=phi,
            psi=psi,
            umax=umax,
            vmax=vmax,
            qmax=qmax,
            apsize=apsize
        )
        self._camera_num += 1
        
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
                    cameras_num=self._camera_num,
                    img_x=self._img_width,
                    img_y=self._img_height,
                    Rad_job=self._camera_data.render(),
                    dx=self._dx,
                    dy=self._dy,
                    z_coords=arr2str(self._z_coords),
                    tmp_prof=self._tmp_prof,
                    tmp1d=arr2str(self._tmp1D[0]),
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
        shape = self._tmp3D[0].shape
        z_axis = np.max([arr[0].shape[2] for arr in (self._tmp3D, self._abs3D, self._ext3D, self._omg3D, self._apf3D)])
        ctl_file_name = '%s.ctl' % self._atmo_file_name
        with open(ctl_file_name, 'w') as f:
            f.write(
                tpl.render(
                file_name=os.path.split(self._atmo_file_name)[-1],
                x_axis=shape[1],
                y_axis=shape[0],
                z_axis=z_axis,
                tmp_z_axis=self._tmp3D[0].shape[2],
                abs_z_axis=self._abs3D[0].shape[2],
                ext_z_axis=self._ext3D[0].shape[2],
                omg_z_axis=self._omg3D[0].shape[2],
                apf_z_axis=self._apf3D[0].shape[2]
                )            
            )
        
        #
        # tmpa3d - Temperature perturbation (K)
        # abst3d - Absorption coefficient perturbation (/m)
        # extp3d - Extinction coefficient (/m)
        # omgp3d - Single scattering albedo
        # apfp3d - Phase function specification parameter
        #
        tmpa3d = self._tmp3D[0]
        abst3d = self._abs3D[0]
        extp3d = self._ext3D[0]
        omgp3d = self._omg3D[0]
        apfp3d = self._apf3D[0]
        
        storeGRADS(self._atmo_file_name, tmpa3d, abst3d, extp3d, omgp3d, apfp3d)

    def _run_simulation(self, photon_num, solver):
        """Run the simulation"""
        
        cmd = '%(mcarats_bin)s %(photon_num)d %(solver)d %(conf_file)s %(output_file)s' % {
            'mcarats_bin': self._mcarats_bin,
            'photon_num': photon_num,
            'solver': solver,
            'conf_file': self._conf_file_name,
            'output_file': self._out_file_name
        }
        print cmd
        prc_ret = sub.Popen(cmd, shell=True, stdin=sub.PIPE, stdout=sub.PIPE, stderr=sub.PIPE)
        print prc_ret.stdout.read()
        prc_ret.wait()
    
    @staticmethod
    def calcRGBImg(Rout_path, Gout_path, Bout_path, time_lag=0, time_width=1, fmax=2, power=0.6):
        """"""
        #
        # Calculate average exposure
        #
        ctl_files = ' '.join(['%s.ctl' % ctl_file for ctl_file in (Rout_path, Gout_path, Bout_path)])
        cmd = 'bin_exposure %(time_lag)d %(time_width)d %(fmax)g %(power)g %(ctl_files)s' % {
            'time_lag': time_lag,
            'time_width': time_width,
            'fmax': fmax,
            'power': power,
            'ctl_files': ctl_files
        }
        print cmd
        prc_ret = sub.Popen(cmd, shell=True, stdin=sub.PIPE, stdout=sub.PIPE, stderr=sub.PIPE)
        ret_txt = prc_ret.stdout.read()
        print ret_txt
        Rmax = float(ret_txt.split('\n')[1].split()[1])
        print Rmax

        imgs_matrix = []
        for out_file_name in (Rout_path, Gout_path, Bout_path):
            #
            # Create gray image
            #
            img_file_name = os.path.join(os.path.split(out_file_name)[0], 'img')
            cmd = 'bin_gray %(factor)g %(Rmax)g %(power)g %(ctl_file)s.ctl %(img_file)s' % {
                'factor': COLOR_BALANCE[0],
                'Rmax': Rmax,
                'ctl_file': out_file_name,
                'power': power,
                'img_file': img_file_name
            }
            print cmd
            prc_ret = sub.Popen(cmd, shell=True, stdin=sub.PIPE, stdout=sub.PIPE, stderr=sub.PIPE)
    
            imgs = []
            ret_split = prc_ret.stdout.read().split()
            for w, h, file_name in zip(ret_split[::4], ret_split[1::4], ret_split[3::4]):
                imgs.append(np.fromfile(file_name, dtype=np.uint8).reshape((int(h), int(w))))
        
            imgs_matrix.append(imgs)
        
        RGB_imgs = []
        for r, g, b in zip(*imgs_matrix):
            RGB_imgs.append(np.dstack((r, g, b)))
            
        return RGB_imgs

    def run(self, photon_num, solver=SOLVER_F3D):
        """
        Run the mcarats model.
        
        Parameters
        ----------
        photon_num : int
            Number of photons to use
            
        solver : int, optional(default=SOLVER_F3D)
            Type of solver to use for the monte carlo simulation

        Returns:
        --------
        out_file_name : str
            Path to the output of the simulation
        """
        
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
         
    
if __name__ == '__main__':
    pass

    
    