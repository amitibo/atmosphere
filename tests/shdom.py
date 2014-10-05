"""
"""

from __future__ import division
import unittest
import atmotomo
import sparse_transforms as spt
import numpy as np
import amitibo
import os


class Test(unittest.TestCase):
    
    def setUp(self):
        
        self.atmosphere_params = amitibo.attrClass(
            cartesian_grids=spt.Grids(
                np.arange(0, 400, 10.0), # Y
                np.arange(0, 400, 10.0), # X
                np.arange(0, 80, 10.0)   # H
                ),
        )

        self.particle = atmotomo.getMisrDB()['spherical_nonabsorbing_2.80']
        self.particle_dist = np.random.rand(*self.atmosphere_params.cartesian_grids.shape) * 10**12

    def test_createPartFile(self):
        
        outfile = 'part_file.part'
        
        atmotomo.createMassContentFile(
            outfile,
            self.atmosphere_params,
            char_radius=self.particle['char radius'],
            particle_dist=self.particle_dist,
            cross_section=self.particle['k'][0]
            )
        
    def test_createMieTable(self):
        
        for color in ('red', 'green', 'blue'):
            outfile = 'mie_table_{color}.scat'.format(color=color)
            atmotomo.createMieTable(
                outfile,
                wavelen=getattr(atmotomo.RGB_WAVELENGTH, color),
                refindex=getattr(self.particle['refractive index'], color),
                density=self.particle['density'],
                char_radius=self.particle['char radius']
            )
        
    def test_createOpticalPorpFile(self):
        
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
        

if __name__ == '__main__':
    unittest.main()
