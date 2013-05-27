import unittest

from amitibo import getResourcePath
import atmotomo
import sparse_transforms as spt
import numpy as np
import amitibo
from configobj import ConfigObj
import os


class Test(unittest.TestCase):
    
    #def setUp(self):
        #import scipy.io as sio
        
        #atmosphere_params = amitibo.attrClass(
            #cartesian_grids=spt.Grids(
                #np.arange(0, 50000, 1000.0), # Y
                #np.arange(0, 50000, 1000.0), # X
                #np.arange(0, 10000, 100.0)   # H
                #),
            #earth_radius=4000000,
            #air_typical_h=8000,
            #aerosols_typical_h=2000
        #)

        #A_air, A_aerosols, Y, X, H, h = atmotomo.density_clouds1(atmosphere_params)

        #sio.savemat('./air_dist.mat', {'A_air': A_air})
        #sio.savemat('./aerosols_dist.mat', {'A_aerosols': A_aerosols})
        
    def tearDown(self):
        
        os.remove('./air_dist.mat')
        os.remove('./aerosols_dist.mat')
        
    def test01(self):
        """Test loading the configuration file"""
        
        #config = atmotomo.readConfiguration(getResourcePath('configuration.ini'))
        config = atmotomo.readConfiguration('/u/amitibo/data/configurations/two_clouds_low_density/configuration.ini')
        
        print config
        

if __name__ == '__main__':
    unittest.main()
