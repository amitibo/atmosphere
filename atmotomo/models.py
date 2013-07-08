"""
Parameteric models
"""

from __future__ import division
import numpy as np

__all__ = ['SphereCloudsModel']


class SphereCloudsModel(object):
    def __init__(
        self,
        atmosphere_params,        
        clouds_num=2,
        aerosols_typical_h=2000,
        aerosols_typical_density=10**12,
        earth_radius=4000000
        ):
        
        self.atmosphere_params = atmosphere_params
        self.lower_bounds = np.zeros(clouds_num * 6)
        self.upper_bounds = 50000*np.ones(clouds_num * 6)

        Y, X, H = self.atmosphere_params.cartesian_grids.expanded
        width = self.atmosphere_params.cartesian_grids.closed[0][-1]
        height = self.atmosphere_params.cartesian_grids.closed[2][-1]
        h = np.sqrt((X-width/2)**2 + (Y-width/2)**2 + (earth_radius+H)**2) - earth_radius
        
        #
        # Create the distributions of aerosols
        #
        self.aerosols_dist = aerosols_typical_density*np.exp(-h/aerosols_typical_h)
         
    def __call__(self, x):
        
        #
        # Split x to the spheres coords
        #
        sc = [np.array(x[i:i+6]) for i in range(0, len(x), 6)]
        
        Y, X, H = self.atmosphere_params.cartesian_grids.expanded
        mask = np.zeros_like(self.aerosols_dist)

        for s in sc:
            R = (X-s[0])**2/s[1]**2 + (Y-s[2])**2/s[3]**2 + (H-s[4])**2/s[5]**2
            mask[R<1] = 1

        return self.aerosols_dist * mask


def main():
    """Main doc """
    
    pass

    
if __name__ == '__main__':
    main()

    
    