"""
"""

from __future__ import division
import numpy as np
import subprocess as sbp
from . import loadpds, readConfiguration, RGB_WAVELENGTH, ColoredParam, L_SUN_RGB
from pkg_resources import resource_filename
import matplotlib.pyplot as plt
import tempfile
import amitibo
import Image
import os

__all__ = (
    'calcTemperature',
    'createMassContentFile',
    'createOpticalPropertyFile',
    'createMieTable',
    'solveRTE',
    'createImage',
    'SHDOM'
)

debug = True

def runCmd(cmd, *args):
    """
    Run a cmd as a subprocess and pass a list of args on stdin.
    """

    p = sbp.Popen([cmd], stdout=sbp.PIPE, stdin=sbp.PIPE, stderr=sbp.STDOUT)
    
    #input_string = "\n".join(['"'+os.path.abspath(arg)+'"' if (type(arg)==str and ('/' in arg)) else str(arg) for arg in args ])+"\n"
    input_string = "\n".join([str(arg) for arg in args])+"\n"
    
    res = p.communicate(input=input_string)
    
    if debug:
        print '='*70
        print 'CMD:', cmd
        print 'STDOUT:\n', res[0]
        print 'STDERR:\n', res[1]

    return res
    

def calcTemperature(z_levels, T0=295):
    """
    Calculate temperature for given z levels
    Taken from http://home.anadolu.edu.tr/~mcavcar/common/ISAweb.pdf
    
    T0 - temperature at see level [kelvin]
    """
    
    temperature = T0 - 6.5 * z_levels
    
    return temperature


def createMassContentFile(
    outfile,
    atmosphere_params,
    effective_radius,
    mass_content=None,
    particle_dist=None,
    cross_section=None
    ):
    """
    This function creates a new .part file 
    If mass_content is given it is used as the extinction matrix. If not
    the particle properties are taken from the MISR table and the given distribution.
    """

    if mass_content is None:
        
        #
        # TODO: figure out what is this magic number?
        # from aviad code: beta - extinction per 1 g/m^2 density of material taken from twoClouds_blue_Mie.scat
        particle_mass = 1.8885e-3
        density2number_ratio = cross_section / particle_mass
        mass_content = particle_dist * density2number_ratio

    grids = atmosphere_params.cartesian_grids
    z_levels = grids[2]
    nx, ny, nz = grids.shape
    
    derivs = grids.derivatives
    dy = derivs[0].ravel()[0]
    dx = derivs[1].ravel()[0]

    temperature = calcTemperature(z_levels)
    
    i_ind, j_ind, k_ind = np.unravel_index(range(grids.size), (nx, ny, nz))
    
    with open(outfile, 'wb') as f:
        f.write('3\n')
        np.savetxt(f, ((nx, ny, nz),), fmt='%d', delimiter=' ')
        np.savetxt(f, ((dx, dy),), fmt='%.4f', delimiter=' ')
        np.savetxt(f, z_levels.reshape(1, -1), fmt='%.4f', delimiter='\t')
        np.savetxt(f, temperature.reshape(1, -1), fmt='%.4f', delimiter='\t')
        
        for i, j, k, m in zip(i_ind, j_ind, k_ind, mass_content.ravel()):
            f.write('%d\t%d\t%d\t1\t1\t%.8f\t%.5f\n' % (i+1, j+1, k+1, m, effective_radius))


def createMieTable(
    outfile,
    wavelen,
    refindex,
    density,
    effective_radius,
    max_radius=50,
    partype="A",
    distflag="L",
    sigma=0.7
    ):
    """
    
    """
    
    #
    # "A" for aerosol particle
    # "L" for lognormal size distribution
    # maxradius truncates the inifinite lognormal distribution
    #
    wavelen1 = wavelen
    wavelen2 = wavelen
    rindex = "({real}, {imag})".format(real=refindex.real, imag=refindex.imag)
    pardens = density
    nretab = 1
    sretab=effective_radius
    eretab=effective_radius
    miefile=outfile

    runCmd(
        'make_mie_table',
        wavelen1,
        wavelen2,
        partype,
        rindex,
        pardens,
        distflag,
        sigma,
        nretab,
        sretab,
        eretab,
        max_radius,
        outfile
    )
    

def createOpticalPropertyFile(
    outfile,
    scat_file,
    part_file,
    wavelen,
    maxnewphase=50,
    asymtol=0.1,
    fracphasetol=0.1,
    ):
    """
    This function creates a new .part file 
    If mass_content is given it is used as the extinction matrix. If not
    the particle properties are taken from the MISR table and the given distribution.
    """

    #
    # Wavelength dependent rayleigh coefficient at sea level
    # calculated by k=(2.97e-4)*lambda^(-4.15+0.2*lambda)
    #
    rayl_coef = (2.97e-4)*wavelen**(-4.15+0.2*wavelen)
    
    nscattab=1
    scatnums=1
    nzo=0

    runCmd(
        'propgen',
        nscattab,
        scat_file,
        scatnums,
        part_file,
        maxnewphase,
        asymtol,
        fracphasetol,
        rayl_coef,
        nzo,
        outfile
    )


def solveRTE(
    nx, ny, nz,
    propfile,
    wavelen,
    maxiter,
    outfile,
    Nmu=8,
    Nphi=16,
    solarmu = 0.5,
    solarphi=0.0,
    solarflux=1.0,
    sfcalb=0.05,
    IPflag=0,
    BCflag=3,
    deltaM='T',
    splitacc=-1,
    shacc=0.003,
    solacc=1.0E-4,
    accel='T'
    ):
    """
    Solve the Radiative Transfer using SHDOM.
    
    Parameters:
    ===========
    Nmu=8                 # number of zenith angles in both hemispheres
    Nphi=16               # number of azimuth angles
    mu0 = 0.5             # solar cosine zenith angle
    phi0=0.0              # solar beam azimuth (degrees)
    flux0=1.0             # solar flux (relative)
    sfcalb=0.05           # surface Lambertian albedo
    IPflag=0              # independent pixel flag (0 = 3D, 3 = IP)
    BCflag=3              # horizontal boundary conditions (0 = periodic)
    deltaM=T              # use delta-M scaling for solar problems
    splitacc=-1           # adaptive grid cell splitting accuracy (was 0.10) negative for no splitting 
    shacc=0.003           # adaptive spherical harmonic accuracy
    solacc=1.0E-4         # solution accuracy
    accel=T               # solution acceleration flag

    """
    
    runname = 'Solve'
    sfcfile = 'NONE'
    ckdfile = 'NONE'
    infile = 'NONE'
    gridtype = 'P'
    srctype = 'S'
    noutfiles = 0
    ncdffile = 'NONE'
    max_memory = 1000
    grid_factor = 2.2
    spheric_factor = 0.6
    point_ratio = 1.5
    skyrad = 0.0
    
    runCmd(
        'shdom90',
        runname,
        propfile,
        sfcfile,
        ckdfile,
        infile,
        outfile,
        nx, ny, nz,
        Nmu, Nphi,
        BCflag,
        IPflag,
        deltaM,
        gridtype,
        srctype,
        solarflux,
        solarmu,
        solarphi,
        skyrad,
        sfcalb,
        wavelen,
        splitacc,
        shacc,
        accel,
        solacc,
        maxiter,
        noutfiles,
        ncdffile,
        max_memory,
        grid_factor,
        spheric_factor,
        point_ratio
    )
    

def createImage(
    nx, ny, nz,
    propfile,
    solvefile,
    wavelen,
    imgbinfile,
    imgfile,
    camX,
    camY,
    camZ,
    camtheta=0,
    camphi=0,
    camrotang=0,
    camnlines=128,
    camnsamps=128,
    nbytes=4,        
    maxiter=0,
    scale = 4,    
    Nmu=8,
    Nphi=16,
    solarmu = -0.5,
    solarphi=0.0,
    solarflux=1.0,
    sfcalb=0.05,
    IPflag=0,
    BCflag=3,
    deltaM='T',
    splitacc=-1,
    shacc=0.003,
    solacc=1.0E-4,
    accel='T'
    ):
    """
    Create the images from a SHDOM solve run.
    
    Parameters:
    ===========
    Nmu=8                 # number of zenith angles in both hemispheres
    Nphi=16               # number of azimuth angles
    mu0 = 0.5             # solar cosine zenith angle
    phi0=0.0              # solar beam azimuth (degrees)
    flux0=1.0             # solar flux (relative)
    sfcalb=0.05           # surface Lambertian albedo
    IPflag=0              # independent pixel flag (0 = 3D, 3 = IP)
    BCflag=3              # horizontal boundary conditions (0 = periodic)
    deltaM=T              # use delta-M scaling for solar problems
    splitacc=-1           # adaptive grid cell splitting accuracy (was 0.10) negative for no splitting 
    shacc=0.003           # adaptive spherical harmonic accuracy
    solacc=1.0E-4         # solution accuracy
    accel=T               # solution acceleration flag

    """
    
    runname = 'Visualize'
    sfcfile = 'NONE'
    ckdfile = 'NONE'
    outfile = 'NONE'
    gridtype = 'P'
    srctype = 'S'
    noutfiles = 1
    outtype = 'V'
    cammode = 1
    binfile = 'temp.bin'
    ncdffile = 'NONE'
    max_memory = 120
    grid_factor = 2.2
    spheric_factor = 0.6
    point_ratio = 1.5
    skyrad = 0.0
    camdelline=179.99/camnlines
    camdelsamp=179.99/camnsamps
    
    runCmd(
        'gradient',
        runname,
        propfile,
        sfcfile,
        ckdfile,
        solvefile,
        outfile,
        nx, ny, nz,
        Nmu, Nphi,
        BCflag,
        IPflag,
        deltaM,
        gridtype,
        srctype,
        solarflux,
        solarmu,
        solarphi,
        skyrad,
        sfcalb,
        wavelen,
        splitacc,
        shacc,
        accel,
        solacc,
        maxiter,
        noutfiles,
        outtype,
        cammode,
        nbytes,
        scale,
        camX, 
        camY,
        camZ,
        camtheta, 
        camphi, 
        camrotang, 
        camnlines, 
        camnsamps,
        camdelline,
        camdelsamp,
        imgbinfile,
        imgfile,
        ncdffile,
        max_memory,
        grid_factor,
        spheric_factor,
        point_ratio
    )


########################################################################
class SHDOM(object):
    """Wrapper for the SHDOM RTE solver"""

    #----------------------------------------------------------------------
    def __init__(self, maxiter=100, splitacc=-1, nbytes=1, scale=4, parallel=False):
        """Constructor"""
        
        self.maxiter = maxiter
        self.splitacc = splitacc
        self.nbytes = nbytes
        self.scale = scale
    
        if parallel:
            from mpi4py import MPI
            import sys
            import traceback
            self.comm = MPI.COMM_WORLD

            #
            # override excepthook so that an exception in one of the childs will cause mpi to abort execution.
            #
            def abortrun(type, value, tb):
                traceback.print_exception(type, value, tb)
                MPI.COMM_WORLD.Abort(1)
                
            sys.excepthook = abortrun
        else:
            stam = collections.namedtuple('stam', ['rank', 'size', 'Barrier'])
            def void():
                pass

            self.comm = stam(0, 1, void)

    def load_configuration(self, config_name, particle_name):
        """Load atmosphere configuration"""
    
        self.atmosphere_params, self.particle_params, self.sun_params, self.camera_params, cameras, air_dist, self.particle_dist = \
            readConfiguration(config_name, particle_name=particle_name)
        
        self.cameras = []
        for cam in cameras:
            self.cameras.append([0.001*coord for coord in cam])
 
        #
        # Convert the grid to KM
        #
        self.atmosphere_params.cartesian_grids = self.atmosphere_params.cartesian_grids.scale(0.001)
    
    def forward(self, gamma=True, imshow=False, camera_limit=None):
        """Run the SHDOM algorithm in the forward direction."""
        
        if self.comm.rank == 0:
            results_path = amitibo.createResultFolder(
                base_path=os.path.expanduser("~/results"),
                params=[self.atmosphere_params, self.particle_params, self.sun_params, self.camera_params],
                src_path=resource_filename(__name__, '')
            )
        else:
            results_path = None

        results_path = self.comm.bcast(results_path, root=0)

        if camera_limit is not None:
            self.cameras = self.cameras[:camera_limit]
            
        #
        # Create the particle file
        #
        if self.comm.rank == 0:
            part_file = os.path.join(results_path, 'part_file.part')
            createMassContentFile(
                part_file,
                self.atmosphere_params,
                effective_radius=self.particle_params.effective_radius,
                particle_dist=self.particle_dist,
                cross_section=self.particle_params.k[0]
            )
        
        grids = self.atmosphere_params.cartesian_grids
        nx, ny, nz = grids.shape

        #
        # 
        #
        imgs_names = ColoredParam([], [], []) 
        for color in ('red', 'green', 'blue'):
            scat_file = os.path.join(results_path, 'mie_table_{color}.scat'.format(color=color))
            prp_file = os.path.join(results_path, 'prop_{color}.prp'.format(color=color))
            solve_file = os.path.join(results_path, 'sol_{color}.bin'.format(color=color))
            
            if self.comm.rank == 0:
                #
                # Create the Mie tables.
                #
                createMieTable(
                    scat_file,
                    wavelen=RGB_WAVELENGTH[color],
                    refindex=self.particle_params.refractive_index[color],
                    density=self.particle_params.density,
                    effective_radius=self.particle_params.effective_radius
                )
             
                #
                # Create the properties file
                #
                createOpticalPropertyFile(
                    outfile=prp_file,
                    scat_file=scat_file,
                    part_file=part_file,
                    wavelen=RGB_WAVELENGTH[color],
                )
            
                #
                # Solve the RTE problem
                #
                solveRTE(
                    nx, ny, nz,
                    prp_file,
                    wavelen=RGB_WAVELENGTH[color],                
                    maxiter=self.maxiter,
                    solarflux=L_SUN_RGB[color],
                    splitacc=self.splitacc,
                    outfile=solve_file,
                    )

            self.comm.Barrier()

            #
            # Calculate the images.
            #
            for i, (camX, camY, camZ) in enumerate(self.cameras):
                if i % self.comm.size != self.comm.rank:
                    continue

                imgbin_file = os.path.join(results_path, 'img_{color}_{i}.bin'.format(color=color, i=i))
                img_file = os.path.join(results_path, 'img_{color}_{i}.pds'.format(color=color, i=i))
                imgs_names[color].append((i, img_file))
                print 'rank:{rank} is about to make image:{i}-{color}'.format(rank=self.comm.rank, i=i, color=color)
                createImage(
                    nx, ny, nz,
                    prp_file,
                    solve_file,
                    wavelen=RGB_WAVELENGTH[color],                
                    imgbinfile=imgbin_file,
                    imgfile=img_file,
                    camX=camX,
                    camY=camY,
                    camZ=camZ,
                    camnlines=self.camera_params.resolution[0], 
                    camnsamps=self.camera_params.resolution[1],
                    solarflux=L_SUN_RGB[color],
                    splitacc=self.splitacc,                    
                    nbytes=self.nbytes,
                    scale=self.scale,
                )
                print 'rank:{rank} finished making image:{i}-{color}'.format(rank=self.comm.rank, i=i, color=color)

        for chan_names in zip(imgs_names.red, imgs_names.green, imgs_names.blue):

            img = []
            for i, ch_name in chan_names:
                img.append(loadpds(ch_name))
        
            img = np.transpose(np.array(img), (1, 2, 0))
            
            if gamma:
                img = (20*img.astype(np.float)**0.4).astype(np.uint8)
            
            im = Image.fromarray(img)
            img_file = os.path.join(results_path, 'img_{i}.jpg'.format(i=i))
            im.save(img_file)
            
            if imshow:
                plt.figure()
                plt.imshow(img)
        
    def reverse(self):
        """Run the SHDOM algorithm to solve the inverse problem."""
        
        pass
    

if __name__ == '__main__':
    pass
    
