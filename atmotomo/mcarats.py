"""
"""

from __future__ import division
import numpy as np
import scipy.stats as stats
import jinja2
import os
import subprocess as sub
from amitibo import BaseData, getResourcePath

__all__ = ["storeGRADS", "loadGRADS", "Mcarats", "SOLVER_F3D", "Job"]

GRADS_TEMPLATE_FILE_NAME = 'grads.jinja'
CONF_TEMPLATE_FILE_NAME = 'conf.jinja'
CAMERA_TEMPLATE_FILE_NAME = 'camera.jinja'
JOB_TEMPLATE_FILE_NAME = 'job.jinja'
SOLVER_F3D = 0
COLOR_BALANCE = (1.28, 1.0, 0.8)
SEA_LEVEL_TEMP = 290
MCARATS_BIN = 'mcarats'
MCARATS_MPI_BIN = 'mcarats_mpi'


def storeGRADS(file_name, *arrays):
    
    array_tuple = []
    for arr in arrays:
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


class Job(BaseData):
    
    #
    # count3D is used for counting the jobs defining the 3D atmospheres
    # it used for setting the dataset id that the data is read from
    # during the simulation.
    #
    count3D = 0
    
    def __init__(self):
        
        super(Job, self).__init__(template_path=JOB_TEMPLATE_FILE_NAME)        
        self._camera_data = BaseData(template_path=CAMERA_TEMPLATE_FILE_NAME)
        self._atmosphere3D = {}

    def set1Ddistribution(self, ext, omg, apf, tmp=None, abst=None):
        """
        Add a 1D distribution
        ---------------------
        
        ext: 1D array like.
             Extinction coefficients [1/m]
        omg: 1D array like,
             Single scattering albedo
        apf: 1D array like.
             Phase function specification parameter.
             apf <= -2 : Isotropic phase function.
             -2 < apf <= -1 : Rayleigh scattering.
             -1 < apf < 1 : Henyey-Greenstein phase function
             1 < apf : Tabulated phase function given by database file (Not supported yet in this code).
        """
        
        if ext.ndim != 1:
            raise NotImplementedError("Multiple 1D particle distributions not yet supported")
        
        if tmp == None:
            tmp = SEA_LEVEL_TEMP*np.ones_like(ext)

        if abst == None:
            abst = np.zeros_like(ext)

        self.addData(
            tmp1d=tmp,
            abs1d=abst,
            ext1d=ext,
            omg1d=omg,
            apf1d=apf
            )
        
    def set3Ddistribution(self, ext, omg, apf, tmp=None, abst=None):
        """
        Add a 3D distribution
        ---------------------
        
        ext: 3D array like.
             Extinction coefficients [1/m]
        omg: 3D array like,
             Single scattering albedo
        apf: 3D array like.
             Phase function specification parameter.
             apf <= -2 : Isotropic phase function.
             -2 < apf <= -1 : Rayleigh scattering.
             -1 < apf < 1 : Henyey-Greenstein phase function
             1 < apf : Tabulated phase function given by database file (Not supported yet in this code).
"""
        
        if ext.ndim != 3:
            raise NotImplementedError("Multiple 3D particle distributions not yet supported")
        
        self.__class__.count3D += 1
        self.addData(idread=self.count3D)
        
        self._atmosphere3D['ext'] = ext
        self._atmosphere3D['omg'] = omg
        self._atmosphere3D['apf'] = apf
        if tmp == None:
            self._atmosphere3D['tmp'] = SEA_LEVEL_TEMP*np.ones_like(ext)
        else:
            self._atmosphere3D['tmp'] = tmp
        if abst == None:
            self._atmosphere3D['abst'] = np.zeros_like(ext)
        else:
            self._atmosphere3D['abst'] = abst

    def setSolarSource(self, theta=0.0, phi=0.0):
        """Set the sun source"""
        
        self.addData(
            sun_theta=theta,
            sun_phi=phi
        )
        
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
        
    @property
    def details(self):
        
        self.addData(
            rad_job=self._camera_data.render()
            )
            
        return self.render()
    
    @property
    def atmosphere3D(self):
        
        return self._atmosphere3D
    
    
class Mcarats(object):
    
    def __init__(self, base_folder, base_name='base', use_mpi=True):
        #
        # Create the template environment
        #
        tpl_loader = jinja2.FileSystemLoader(searchpath=getResourcePath('.', package_name=__name__))
        self._tpl_env = jinja2.Environment(loader=tpl_loader)
        
        BaseData._tpl_env = self._tpl_env
        Job.count3D = 0
        
        self._atmo_file_name = os.path.join(base_folder, '%s.atm' % base_name)
        self._conf_file_name = os.path.join(base_folder, '%s_conf' % base_name)
        self._out_file_name = os.path.join(base_folder, '%s_out' % base_name)
        
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
        target=2,
        tmp_prof=0,
        iz3l=None,
        nz3=None,
        np1d=1,
        np3d=1,
        camera_num=1,
        img_width=128,
        img_height=128
        ):
        """
        Configure the MCARaTS simulation

        Parameters
        ----------
        shape : (int, int, int)
            Atmosphere grid shape.
        
        dx, dy : float
            Voxel size in the X, Y axes (in meters)
            
        z_coords : array
            z layers locations (in meters)

        target : {2, 3} optional (default=2)
            Target mode (2=radiance, 3=volume rendering)
        
        tmp_prof : {0, 1} optional (default=0)
             Flag for temperature profile (0=temp data are given for each layer)
        
        iz3l : int, optional (default=None)
            Starting Z index of 3-D distribution (None=first index)
        
        nz3 : int, optional (default=None)
            Ending Z index of 3-D distribution (None=Last index)
        
        np1d : int, optional (default=1)
            Number of 1D particle distributions.
        
        np3d : int, optional (default=1)
            Number of 3D particle distributions.
        
        camera_num : int, optional (default=1)
            Number of sensors.
        
        img_width : int, optional (default=128)
            Width of image.
            
        img_height : int, optional (default=128)
            Width of image.
            
        Returns
        -------
        """

        self._target = target
        self._shape = shape
        self._dx = dx
        self._dy = dy
        self._z_coords = z_coords
        
        if tmp_prof != 0:
            raise NotImplementedError("Temperature profile at layer boundry is not supported (at the job class).")
        self._tmp_prof = tmp_prof
        
        if iz3l == None:
            self._iz3l = 1
            self._nz3 = shape[2]
        else:
            self._iz3l = iz3l
            self._nz3 = nz3

        self._np1d = np1d
        self._np3d = np3d
        
        self._camera_num = camera_num
        self._img_width = img_width
        self._img_width = img_width
        self._img_height = img_height
        
    def _createConfFile(self, jobs):
        """Create the configuration file for the simulation"""
        
        tpl = self._tpl_env.get_template(CONF_TEMPLATE_FILE_NAME)
        
        for job in jobs:
            job.addData(
                dx=self._dx,
                dy=self._dy,
                z_coords=arr2str(self._z_coords),
            )
            
        #
        # The 1-D data relates to Air distribution
        #
        with open(self._conf_file_name, 'w') as f:
            f.write(
                tpl.render(
                    target=self._target,
                    atmo_file_name=os.path.split(self._atmo_file_name)[-1],
                    x_axis=self._shape[0],
                    y_axis=self._shape[1],
                    z_axis=self._shape[2],
                    iz3l=self._iz3l,
                    nz3=self._nz3,
                    cameras_num=self._camera_num,
                    img_x=self._img_width,
                    img_y=self._img_height,
                    tmp_prof=self._tmp_prof,
                    np1d=self._np1d,
                    np3d=self._np3d,
                    jobs=jobs,
                    njob=len(jobs)
                )            
            )
        
        return self._conf_file_name
    
    def _createAtmFile(self, jobs):
        """Create the atmosphere file"""

        atmo3D = [job.atmosphere3D for job in jobs if job.atmosphere3D != {}]
        
        if atmo3D == []:
            return
        
        tpl = self._tpl_env.get_template(GRADS_TEMPLATE_FILE_NAME)
        
        shape = atmo3D[0]['tmp'].shape
        
        ctl_file_name = '%s.ctl' % self._atmo_file_name
        with open(ctl_file_name, 'w') as f:
            f.write(
                tpl.render(
                file_name=os.path.split(self._atmo_file_name)[-1],
                x_axis=shape[1],
                y_axis=shape[0],
                z_axis=shape[2],
                tmp_z_axis=shape[2],
                abs_z_axis=shape[2],
                ext_z_axis=shape[2],
                omg_z_axis=shape[2],
                apf_z_axis=shape[2]
                )            
            )
        
        #
        # tmpa3d - Temperature perturbation (K)
        # abst3d - Absorption coefficient perturbation (/m)
        # extp3d - Extinction coefficient (/m)
        # omgp3d - Single scattering albedo
        # apfp3d - Phase function specification parameter
        #
        array = []
        for atm in atmo3D:
            array += [atm['tmp'], atm['abst'], atm['ext'], atm['omg'], atm['apf']]
        
        storeGRADS(self._atmo_file_name, *array)

    def runSimulation(self, photon_num, solver=SOLVER_F3D, conf_file_name=None, out_file_name=None):
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
        
        if conf_file_name == None:
            conf_file_name = self._conf_file_name
        if out_file_name == None:
            out_file_name = self._out_file_name
            
        cmd = '%(mcarats_bin)s %(photon_num)d %(solver)d %(conf_file)s %(output_file)s' % {
            'mcarats_bin': self._mcarats_bin,
            'photon_num': photon_num,
            'solver': solver,
            'conf_file': conf_file_name,
            'output_file': out_file_name
        }
        print cmd
        prc_ret = sub.Popen(cmd, shell=True, stdin=sub.PIPE, stdout=sub.PIPE, stderr=sub.PIPE)
        print prc_ret.stdout.read()
        prc_ret.wait()
    
        return out_file_name
    
    @staticmethod
    def calcExposure(out_paths, time_lag=0, time_width=1, fmax=1.2, power=0.6):
        """
        Calculate the average exposure for a set of output radiance files (of the MCARaTS bin).

        Parameters
        ----------
        out_paths : list of str 
            Paths to the radiance output files.
        
        time_lag : integer, optional (default=0)
            The time lag is useful to realistically simulate that real optical instrument
            and human eyes adjust their aperture size with a time delay.

        time_width : integer, optional (default=1)
            Temporal filter width (integer, 1 means no filtering).
        
        fmax : float, optional (default=1.2)
            Factor for automatic determination of Rmax (usually around 1.2), used for
            eliminating the sun hot spots from the calculation of the exposure
        
        power : float, optional (default=0.6)
            Scaling exponent for nonliner conversion
        
        Returns
        -------
        
        Rmax_txt : str
            Recommended exposure data, in the text form returned by the bin_exposure utility..
            
        Rmaxs : array
            Recommended exposure data, which can be used as Rmax values input for bin_gray utility.
        
        """
        
        ctl_files = ' '.join(['%s.ctl' % ctl_file for ctl_file in out_paths])
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
        Rmaxs = []
        for line in ret_txt.strip().split('\n'):
            if line.strip().startswith('#'):
                continue
            Rmaxs.append(float(line.split()[1]))

        return ret_txt, np.array(Rmaxs)
    
    @staticmethod
    def calcRGBImg(Rout_path, Gout_path, Bout_path, time_lag=0, time_width=1, fmax=2, power=0.6):
        """
        Calculate the RGB images (of the MCARaTS bin files).

        Parameters
        ----------
        Rout_path, Gout_path, Bout_path : list of str 
            Paths to the radiance output files relating to the Red, Green and Blue channels
            respectively.
        
        time_lag : integer, optional (default=0)
            The time lag is useful to realistically simulate that real optical instrument
            and human eyes adjust their aperture size with a time delay.

        time_width : integer, optional (default=1)
            Temporal filter width (integer, 1 means no filtering).
        
        fmax : float, optional (default=1.2)
            Factor for automatic determination of Rmax (usually around 1.2), used for
            eliminating the sun hot spots from the calculation of the exposure
        
        power : float, optional (default=0.6)
            Scaling exponent for nonliner conversion.
        
        Returns
        -------
        
        Images : List of arrays.
             List of RGB images.
        
        """
        
        Rmax_txt, Rmaxs = Mcarats.calcExposure(
            (Rout_path, Gout_path, Bout_path),
            time_lag=time_lag,
            time_width=time_width,
            fmax=fmax, power=power
        )
        
        imgs_matrix = []
        for out_file_name in (Rout_path, Gout_path, Bout_path):
            #
            # Create gray image
            #
            img_file_name = os.path.join(os.path.split(out_file_name)[0], 'img')
            cmd = 'bin_gray %(factor)g %(Rmax)g %(power)g %(ctl_file)s.ctl %(img_file)s' % {
                'factor': COLOR_BALANCE[0],
                'Rmax': 0,
                'ctl_file': out_file_name,
                'power': power,
                'img_file': img_file_name
            }
            print cmd
            prc_ret = sub.Popen(cmd, shell=True, stdin=sub.PIPE, stdout=sub.PIPE, stderr=sub.PIPE)
    
            imgs = []
            prc_ret.stdin.write(Rmax_txt)
            out = prc_ret.stdout.read()
            
            #
            # Ugly hack in case the command breaks line in the middle of an image path
            #
            out = ''.join([s[1:] if s[0] == ' ' else s for s in out.strip().split('\n')])
            
            ret_split = out.split()
            for w, h, file_name in zip(ret_split[::4], ret_split[1::4], ret_split[3::4]):
                imgs.append(np.fromfile(file_name, dtype=np.uint8).reshape((int(h), int(w))))
        
            imgs_matrix.append(imgs)
        
        RGB_imgs = []
        for r, g, b in zip(*imgs_matrix):
            RGB_imgs.append(np.dstack((r, g, b)))
            
        return RGB_imgs

    @staticmethod
    def calcRmax(x, fmax=1.2, FACMIN=1.3):
        xmn = x.mean()
        xst = x.std()
        
        return max(xmn*FACMIN, xmn + xst*fmax)
    
    @staticmethod
    def removeSunSpot(ch, ys, xs, MARGIN=2):
        ymin = ys.min()-MARGIN
        ymax = ys.max()+MARGIN
        xmin = xs.min()-MARGIN
        xmax = xs.max()+MARGIN
        
        ch_part = ch[ymin:ymax, xmin:xmax].copy()
        ch_part[ys-ymin, xs-xmin] = np.nan

        ch[ymin:ymax, xmin:xmax] = np.mean(stats.nanmean(ch_part))
        
        return ch
    
    @staticmethod
    def calcMcaratsImg(R_ch, G_ch, B_ch, slc, IMG_SHAPE, remove_sun=True):
        R, G, B = [ch[slc].reshape(IMG_SHAPE) for ch in (R_ch, G_ch, B_ch)]
        Rmax = Mcarats.calcRmax(R)
        ys, xs = np.nonzero(R>Rmax)
        
        if remove_sun:
            R, G, B = [Mcarats.removeSunSpot(ch, ys, xs) for ch in (R, G, B)]
                   
        img = np.dstack((R, G, B))
        
        return img
    
    def prepareSimulation(self, jobs):
        """
        Create the configuration files needed for the simulation run.
        """
        
        #
        # Prepare the init files
        #
        conf_file = self._createConfFile(jobs)
        self._createAtmFile(jobs)

        return conf_file
    
    
if __name__ == '__main__':
    pass

    
    