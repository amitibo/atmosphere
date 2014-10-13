"""
"""

from __future__ import division
import unittest
import atmotomo
import sparse_transforms as spt
import itertools
import numpy as np
import amitibo
import os


class Test(unittest.TestCase):
    
    def setUp(self):
        
        self.atmosphere_params, self.particle_params, sun_params, camera_params, cameras, air_dist, self.particle_dist = \
            atmotomo.readConfiguration('two_clouds_high_density_mediumhigh_resolution_smooth', particle_name='spherical_absorbing_0.57_ssa_green_0.94')
        
        self.atmosphere_params.cartesian_grids = self.atmosphere_params.cartesian_grids.scale(0.001)
    
    #@unittest.skip('')
    def test_1createPartFile(self):
        
        outfile = 'part_file.part'
        
        atmotomo.createMassContentFile(
            outfile,
            self.atmosphere_params,
            effective_radius=self.particle_params.effective_radius,
            particle_dist=self.particle_dist,
            cross_section=self.particle_params.k[0]
            )

    #@unittest.skip('')
    def test_2createMieTable(self):
        
        for color in ('red', 'green', 'blue'):
            outfile = 'mie_table_{color}.scat'.format(color=color)
            atmotomo.createMieTable(
                outfile,
                wavelen=getattr(atmotomo.RGB_WAVELENGTH, color),
                refindex=getattr(self.particle_params.refractive_index, color),
                density=self.particle_params.density,
                effective_radius=self.particle_params.effective_radius
            )
        
    #@unittest.skip('')
    def test_3createOpticalPorpFile(self):
        
        part_file = 'part_file.part'
        
        for color in ('red', 'green', 'blue'):
            scat_file = 'mie_table_{color}.scat'.format(color=color)
            outfile = 'prop_{color}.prp'.format(color=color)
            
            atmotomo.createOpticalPropertyFile(
                outfile,
                scat_file=scat_file,
                part_file=part_file,
                wavelen=getattr(atmotomo.RGB_WAVELENGTH, color),
            )
            
    #@unittest.skip('')
    def test_4solveRTE(self):
        
        outfile = 'part_file.part'
        grids =  self.atmosphere_params.cartesian_grids
        nx, ny, nz = grids.shape
        
        for color, flux in zip(('red', 'green', 'blue'), (255, 236, 224)):
            propfile = 'prop_{color}.prp'.format(color=color)
            outfile = 'sol_{color}.bin'.format(color=color)
            
            atmotomo.solveRTE(
                nx, ny, nz,
                propfile,
                wavelen=getattr(atmotomo.RGB_WAVELENGTH, color),                
                maxiter=100,
                solarflux=flux,
                splitacc=0.1,
                outfile=outfile,
                )

    #@unittest.skip('')
    def test_5createImage(self):
        
        outfile = 'part_file.part'
        grids =  self.atmosphere_params.cartesian_grids
        nx, ny, nz = grids.shape
        
        for i, (camX, camY) in enumerate(itertools.product((9.01, 39.01), repeat=2)):
            for color, flux in zip(('red', 'green', 'blue'), (255, 236, 224)):
                propfile = 'prop_{color}.prp'.format(color=color)
                solvefile = 'sol_{color}.bin'.format(color=color)
                imgbinfile = 'img_{color}_{i}.bin'.format(color=color, i=i)
                imgfile = 'img_{color}_{i}.pds'.format(color=color, i=i)
                
                atmotomo.createImage(
                    nx, ny, nz,
                    propfile,
                    solvefile,
                    wavelen=getattr(atmotomo.RGB_WAVELENGTH, color),                
                    imgbinfile=imgbinfile,
                    imgfile=imgfile,
                    camX=camX,
                    camY=camY,
                    camZ=0.1,
                    solarflux=flux,
                    splitacc=-0.1,                    
                    nbytes=1,
                    scale=4,
                )

    #@unittest.skip('')
    def test_6showImages(self):

        import matplotlib.pyplot as plt
        
        for i in range(4):
            img = []
            for color in ('red', 'green', 'blue'):
                imgfile = 'img_{color}_{i}.pds'.format(color=color, i=i)

                img.append(atmotomo.loadpds(imgfile))
        
            img = np.transpose(np.array(img), (1, 2, 0))
            
            img = (20*img.astype(np.float)**0.4).astype(np.uint8)
            
            plt.figure()
            plt.imshow(img)
            
        plt.show()
            
        

if __name__ == '__main__':
    unittest.main()
