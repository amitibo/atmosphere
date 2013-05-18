from __future__ import division
import unittest
from atmotomo import calcHG, calcHG_other
import numpy as np


class Test(unittest.TestCase):
    
    def setUp(self):
        
        pass
        
        
    def testHG(self):
        
        g = 0.4
        angle = np.linspace(0, np.pi, 200)
        hg1 = calcHG(np.cos(angle), g)
        hg2 = calcHG_other(np.cos(angle), g)
        
        import matplotlib.pyplot as plt
        
        plt.figure()
        plt.plot(angle, hg1)
        plt.plot(angle, hg2)
        plt.show()
    

if __name__ == '__main__':
    unittest.main()
