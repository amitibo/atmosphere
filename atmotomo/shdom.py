"""
"""

from __future__ import division
import numpy as np
import subprocess as sbp

__all__ = (
    'calcTemperature',
    'createMassContentFile'
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
    file_path,
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
    
    with open(file_path, 'wb') as f:
        f.write('3\n')
        np.savetxt(f, ((nx, ny, nz),), fmt='%d', delimiter=' ')
        np.savetxt(f, ((dx, dy),), fmt='%.4f', delimiter=' ')
        np.savetxt(f, z_levels.reshape(1, -1), fmt='%.4f', delimiter='\t')
        np.savetxt(f, temperature.reshape(1, -1), fmt='%.4f', delimiter='\t')
        
        for i, j, k, m in zip(i_ind, j_ind, k_ind, mass_content.ravel()):
            f.write('%d\t%d\t%d\t1\t1\t%.5f\t%.5f\n' % (i+1, j+1, k+1, m, char_radius))


def createParticleMixtureFile(
    file_path,
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
    
    wavenames = ('red', 'green', 'blue')
    wavelengths = (0.672, 0.558, 0.446)
    rayl_coef_array = (0.00148, 0.0031, 0.0079)

    # Wavelength dependent rayleigh coefficient at sea level 
    # calculated by k=(2.97e-4)*lambda^(-4.15+0.2*lambda)
    
    # parfile = ${basefile}.part
    maxnewphase = 50
    asymtol = 0.1
    fracphasetol = 0.1
    Nzother = 0
    
    for wavename, wavelen, raylcoef in ():
        prpfile = """${basefile}_${wavename}.prp"""
        scattable = """("${basefile}_${wavename}_Mie.scat"""        

        
        put $#scattable $scattable "$scattypes" $parfile \
            $maxnewphase $asymtol $fracphasetol $raylcoef \
            $Nzother $prpfile  | propgen   

        p = sbp.Popen(['propgen'], stdout=sbp.PIPE, stdin=sbp.PIPE, stderr=sbp.STDOUT)'
        p.communicate(
            input="1 "
        )


def createScatFile(base_path):
    """
    """
    
    wavenames = ('red', 'green', 'blue')
    wavelengths = (0.672, 0.558, 0.446)
    rayl_coef_array = (0.00148, 0.0031, 0.0079)

    # Wavelength dependent rayleigh coefficient at sea level 
    # calculated by k=(2.97e-4)*lambda^(-4.15+0.2*lambda)
    
    partype = "A"                # W for water
    refindex="(1.45,-0.0006)"    # aerosol complex index of refraction 
    pardens=1.91                 # particle bulk denisty (g/cm^3)
    distflag=L                   # G=gamma, L=lognormal size distribution
    sigma = 0.7                  # lognormal dist shape parameter
    Nretab=1                     # number of effective radius in table
    Sretab=0.57; Eretab=0.57     # starting, ending effective radius (micron)
    maxradius=50 
    
    for wavename, wavelen, raylcoef in zip(wavenames, wavelengths, rayl_coef_array):
        outfile = "{base_file}_{wavename}_Mie.scat".format(base_file=base_path, wavename=wavename)
        p = sbp.Popen(['make_mie_table'], stdout=sbp.PIPE, stdin=sbp.PIPE, stderr=sbp.STDOUT)'
        p.communicate(
            input="{wavelen1} {wavelen2} {partype} {rindex} {pardens} {distflag} {alpha} {nretab} {sretab} {eretab} {maxradius} {miefile}".format(
                wavelen1=wavelen,
                wavelen2=wavelen,
                partype=partype,
                rindex=refindex,
                pardens=pardens,
                distflag=distflag,
                alpha=sigma,
                nretab=Nretab,
                sretab=Sretab,
                eretab=Eretab,
                maxradius=maxradius,
                miefile=outfile
            )
        )


        if __name__ == '__main__':
    pass
    