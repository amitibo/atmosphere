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
    'createMieTable'
)


def runCmd(cmd, *args):
    """
    Run a cmd as a subprocess and pass a list of args on stdin.
    """

    p = sbp.Popen([cmd], stdout=sbp.PIPE, stdin=sbp.PIPE, stderr=sbp.STDOUT)

    res = p.communicate(input="\n".join([repr(arg) for arg in args]))
    )
    

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
        density2number_ratio = cross_section * 1e-12 / particle_mass
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
    
    Nzother = 0
    
    #
    # Wavelength dependent rayleigh coefficient at sea level
    # calculated by k=(2.97e-4)*lambda^(-4.15+0.2*lambda)
    #
    rayl_coef = (2.97e-4)*wavelen**(-4.15+0.2*wavelen)
    
    p = sbp.Popen(['propgen'], stdout=sbp.PIPE, stdin=sbp.PIPE, stderr=sbp.STDOUT)

    res = p.communicate(
        input="{nscattab}\n{scattabfiles}\n{scatnums}\n{parfile}\n{maxnewphase}\n{asymtol}\n{fracphasetol}\n{raylcoef}\n{nzo}\n{propfile}\n".format(
            nscattab=1,
            scattabfiles='({scat_file})'.format(scat_file=scat_file),
            scatnums='(1)',
            parfile=part_file,
            maxnewphase=maxnewphase,
            asymtol=asymtol,
            fracphasetol=fracphasetol,
            raylcoef=rayl_coef,
            nzo=0,
            propfile=outfile
        )
    )
    
    print res


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
    particle_num = 1
    maxradius = 50 
    
    p = sbp.Popen(['make_mie_table'], stdout=sbp.PIPE, stdin=sbp.PIPE, stderr=sbp.STDOUT)

    res = p.communicate(
        input="{wavelen1}\n{wavelen2}\n{partype}\n{rindex}\n{pardens}\n{distflag}\n{alpha}\n{nretab}\n{sretab}\n{eretab}\n{maxradius}\n{miefile}\n".format(
            wavelen1=wavelen,
            wavelen2=wavelen,
            partype=partype,
            rindex="({real}, {imag})".format(real=refindex.real, imag=refindex.imag),
            pardens=density,
            distflag=distflag,
            alpha=sigma,
            nretab=particle_num,
            sretab=char_radius,
            eretab=char_radius,
            maxradius=max_radius,
            miefile=outfile
        )
    )
    

def solveRTE(
    Nmu=8,
    Nphi=16,
    mu0 = 0.5,
    phi0=0.0,
    flux0=1.0,
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
    
    p = sbp.Popen(['shdom90'], stdout=sbp.PIPE, stdin=sbp.PIPE, stderr=sbp.STDOUT)

    res = p.communicate(
        input="{runname}\n{propfile}\n{sfcfile}\n{ckdfile}\n{infile}\n{outfile}\n{nx}\n{ny}\n{nz}".format(
            wavelen1=wavelen,
            wavelen2=wavelen,
            partype=partype,
            rindex="({real}, {imag})".format(real=refindex.real, imag=refindex.imag),
            pardens=density,
            distflag=distflag,
            alpha=sigma,
            nretab=particle_num,
            sretab=char_radius,
            eretab=char_radius,
            maxradius=max_radius,
            miefile=outfile
        )
    )
    
    
if __name__ == '__main__':
    pass
    