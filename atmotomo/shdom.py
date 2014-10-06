"""
"""

from __future__ import division
import numpy as np
import subprocess as sbp
import os

__all__ = (
    'calcTemperature',
    'createMassContentFile',
    'createOpticalPropertyFile',
    'createMieTable',
    'solveRTE',
    'createImage'
)

debug = True

def runCmd(cmd, *args):
    """
    Run a cmd as a subprocess and pass a list of args on stdin.
    """

    p = sbp.Popen([cmd], stdout=sbp.PIPE, stdin=sbp.PIPE, stderr=sbp.STDOUT)

    res = p.communicate(input="\n".join([str(arg) for arg in args])+"\n")
    
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
    char_radius,
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

    grids =  atmosphere_params.cartesian_grids
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
            f.write('%d\t%d\t%d\t1\t1\t%.5f\t%.5f\n' % (i+1, j+1, k+1, m, char_radius))


def createMieTable(
    outfile,
    wavelen,
    refindex,
    density,
    char_radius,
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
    sretab=char_radius
    eretab=char_radius
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
    imgfile,
    camX,
    camY,
    camZ,
    camtheta=0,
    camphi=0,
    camrotang=0,
    camnlines=128,
    camnsamps=128,
    camdelline=179.0/128,
    camdelsamp=179.0/128,
    nbytes=4,        
    maxiter=0,
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
    scale = 4
    binfile = 'temp.bin'
    ncdffile = 'NONE'
    max_memory = 120
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
        imgfile,
        ncdffile,
        max_memory,
        grid_factor,
        spheric_factor,
        point_ratio
    )


if __name__ == '__main__':
    pass
    