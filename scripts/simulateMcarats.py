"""
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from atmotomo import RGB_WAVELENGTH, getResourcePath, getMisrDB, density_clouds1, calcAirMcarats, Mcarats, SOLVER_F3D
import amitibo
import os


def main(photon_num=1e7, solver=SOLVER_F3D):
    """Main doc"""
    
    particle = getMisrDB()['spherical_nonabsorbing_2.80']

    #
    # Simulation parameters
    #
    atmosphere_params = amitibo.attrClass(
        cartesian_grids=(
            slice(0, 50, 1.0), # Y
            slice(0, 50, 1.0), # X
            slice(0, 10, 0.1)   # H
            ),
        earth_radius=4000,
        RGB_WAVELENGTH=RGB_WAVELENGTH,
        air_typical_h=8,
        aerosols_typical_h=2,
        sun_angle=30
    )

    camera_params = amitibo.attrClass(
        img_x=400,
        img_y=300,
        theta=90,
        phi=0
    )
    
    A_air, A_aerosols, Y, X, Z, h = density_clouds1(atmosphere_params)
    dx = abs(X[0, 1, 0] - X[0, 0, 0])*1000
    dy = abs(Y[1, 0, 0] - Y[0, 0, 0])*1000
    dz = abs(Z[0, 0, 1] - Z[0, 0, 0])*1000
    z_coords = Z[0, 0, :]*1000
    z_coords = np.concatenate((z_coords, [z_coords[-1]+dz]))
    air_ext = calcAirMcarats(z_coords)

    #
    # Create the results folder
    #
    results_path = amitibo.createResultFolder()

    #
    # Create the test
    #
    out_files = []
    for ch in range(3):
        mc = Mcarats(results_path, base_name='base%d'%ch)
        mc.setAtmosphereDims(shape=A_aerosols.shape, dx=dx, dy=dy, z_coords=z_coords, tmp_prof=0)
        mc.add1Ddistribution(
            ext1d=air_ext[ch],
            omg1d=np.ones_like(air_ext[ch]),
            apf1d=-1*np.ones_like(air_ext[ch])
        )
        mc.add3Ddistribution(
            ext3d=particle['k'][ch]*A_aerosols * 10**-5,
            omg3d=particle['w'][ch]*np.ones_like(A_aerosols),
            apf3d=particle['g'][ch]*np.ones_like(A_aerosols)
        )
        mc.addCamera(camera_params)
        mc.setSolarSource(theta=120.0, phi=180.0)
        
        #
        # Run the test
        #
        out_files.append(mc.run(photon_num=photon_num, solver=solver))
    
    img = np.dstack(Mcarats.calcImg(out_files)[::-1])
    
    #
    # Show the results
    #
    plt.imshow(img)
    plt.show()
    
    
if __name__ == '__main__':
    main()

    
    