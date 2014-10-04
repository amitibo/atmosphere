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
        
        pass

    def test_createPartFile(self):
        
        atmosphere_params = amitibo.attrClass(
            cartesian_grids=spt.Grids(
                np.arange(0, 400, 10.0), # Y
                np.arange(0, 400, 10.0), # X
                np.arange(0, 80, 10.0)   # H
                ),
        )
        
        particle = atmotomo.getMisrDB()['spherical_nonabsorbing_2.80']
        particle_dist = np.random.rand(*atmosphere_params.cartesian_grids.shape) * 10**12
        path = os.path.abspath('part_file.part')
        
        atmotomo.createMassContentFile(
            path,
            atmosphere_params,
            char_radius=particle['char radius'],
            particle_dist=particle_dist,
            cross_section=particle['k'][0]
            )
        
    def test_createMieTable(self):
        
        atmotomo.createScatFile('mie_table')
        

if __name__ == '__main__':
    unittest.main()
