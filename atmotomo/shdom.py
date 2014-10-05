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
    rayl_coef,
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
    

if __name__ == '__main__':
    pass
    