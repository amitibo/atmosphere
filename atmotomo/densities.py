"""
"""

from __future__ import division
import numpy as np

__all__ = ["density_front", "density_clouds1", "density_clouds_vadim"]


def density_front(atmosphere_params):
    #
    # Create the sky
    #
    Y, X, H = np.mgrid[atmosphere_params.cartesian_grids]
    width = atmosphere_params.cartesian_grids[0].stop
    height = atmosphere_params.cartesian_grids[2].stop
    h = np.sqrt((X-width/2)**2 + (Y-width/2)**2 + (atmosphere_params.earth_radius+H)**2) - atmosphere_params.earth_radius

    #
    # Create the distributions of air
    #
    A_air = np.exp(-h/atmosphere_params.air_typical_h)
    
    #
    # Create the distributions of aerosols
    #
    A_aerosols = np.exp(-h/atmosphere_params.aerosols_typical_h)
    A_mask = np.zeros_like(A_aerosols)
    Z1 = (X)**2/64 + (Y-width/2)**2/1000 + (H-height/2)**2
    A_mask[Z1<4**2] = 1
    A_aerosols *= A_mask
    
    return A_air, A_aerosols, Y, X, H, h


def density_clouds1(atmosphere_params):
    #
    # Create the sky
    #
    Y, X, H = np.mgrid[atmosphere_params.cartesian_grids]
    width = atmosphere_params.cartesian_grids[0].stop
    height = atmosphere_params.cartesian_grids[2].stop
    h = np.sqrt((X-width/2)**2 + (Y-width/2)**2 + (atmosphere_params.earth_radius+H)**2) - atmosphere_params.earth_radius

    #
    # Create the distributions of air
    #
    A_air = np.exp(-h/atmosphere_params.air_typical_h)
    
    #
    # Create the distributions of aerosols
    #
    A_aerosols = np.exp(-h/atmosphere_params.aerosols_typical_h)
    A_mask = np.zeros_like(A_aerosols)
    Z1 = (X-width/3)**2/16 + (Y-width/3)**2/16 + (H-height/2)**2*8
    Z2 = (X-width*2/3)**2/16 + (Y-width*2/3)**2/16 + (H-height/4)**2*8
    A_mask[Z1<3**2] = 1
    A_mask[Z2<4**2] = 1
    A_aerosols *= A_mask

    return A_air, A_aerosols, Y, X, H, h


def density_clouds_vadim(atmosphere_params):
    #
    # Create the sky
    #
    Y, X, H = np.mgrid[atmosphere_params.cartesian_grids]
    width = atmosphere_params.cartesian_grids[0].stop
    height = atmosphere_params.cartesian_grids[2].stop
    h = np.sqrt((X-width/2)**2 + (Y-width/2)**2 + (atmosphere_params.earth_radius+H)**2) - atmosphere_params.earth_radius

    #
    # Create the distributions of air
    #
    A_air = np.exp(-h/atmosphere_params.air_typical_h)
    
    #
    # Create the distributions of aerosols
    #
    A_aerosols = np.exp(-h/atmosphere_params.aerosols_typical_h)
    A_mask = np.zeros_like(A_aerosols)
    Z1 = (-Y+width*2/3)**2/16 + (X-width/3)**2/16 + (H-height/2)**2*8
    Z2 = (-Y+width/3)**2/16 + (X-width*2/3)**2/16 + (H-height/4)**2*8
    A_mask[Z1<3**2] = 1
    A_mask[Z2<4**2] = 1
    A_aerosols *= A_mask

    return A_air, A_aerosols, Y, X, H, h


def main():
    """Main doc """
    
    import amitibo
    import mayavi.mlab as mlab
    
    atmosphere_params = amitibo.attrClass(
        cartesian_grids=(
            slice(0, 50, 1.0), # Y
            slice(0, 50, 1.0), # X
            slice(0, 10, 0.1)   # H
            ),
        earth_radius=4000,
        air_typical_h=8,
        aerosols_typical_h=2
    )

    A_air, A_aerosols1, Y, X, H, h = density_clouds1(atmosphere_params)
    A_air, A_aerosols2, Y, X, H, h = density_clouds_vadim(atmosphere_params)
    
    amitibo.viz3D(Y, X, H, A_aerosols1)
    amitibo.viz3D(Y, X, H, A_aerosols2)

    mlab.show()
    
    
if __name__ == '__main__':
    main()

    
    