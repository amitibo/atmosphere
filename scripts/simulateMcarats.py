"""
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from atmotomo import RGB_WAVELENGTH, getResourcePath, getMisrDB, density_clouds1, calcAirMcarats, Mcarats, SOLVER_F3D
import itertools
import argparse
import amitibo
import os


def main(photon_num=1e6, solver=SOLVER_F3D, use_mpi=True):
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
        mc = Mcarats(results_path, base_name='base%d'%ch, use_mpi=use_mpi)
        mc.configure(
            shape=A_aerosols.shape,
            dx=dx,
            dy=dy,
            z_coords=z_coords,
            tmp_prof=0,
            img_width=512,
            img_height=512
        )
        mc.add1Ddistribution(
            ext1d=air_ext[ch],
            omg1d=np.ones_like(air_ext[ch]),
            apf1d=-1*np.ones_like(air_ext[ch])
        )
        mc.add3Ddistribution(
            ext3d=particle['k'][2-ch]*A_aerosols * 10**-12*1000*1000*100,
            omg3d=particle['w'][2-ch]*np.ones_like(A_aerosols),
            apf3d=particle['g'][2-ch]*np.ones_like(A_aerosols)
        )
        
        for xpos, ypos in itertools.product(np.arange(0.0, 1.0, 0.3), np.arange(0.0, 1.0, 0.3)):
            mc.addCamera(
                xpos=xpos,
                ypos=ypos,
                zloc=0,
                theta=0,
                phi=0,
                psi=270,
            )
            
        mc.setSolarSource(theta=120.0, phi=180.0)
        
        #
        # Run the test
        #
        out_files.append(mc.run(photon_num=photon_num, solver=solver))
    
    imgs = Mcarats.calcRGBImg(*out_files)
    
    #
    # Show the results
    #
    figures = []
    for img in imgs:
        figures.append(plt.figure())
        plt.imshow(img)
    
    amitibo.saveFigures(results_path, bbox_inches='tight', figures=figures)
    
    
if __name__ == '__main__':
    #
    # Parse the input
    #
    parser = argparse.ArgumentParser(description='Use MC MCarats for simulating the sky')
    parser.add_argument('--use_mpi', action='store_true', help='run using mpi')
    parser.add_argument('--photons', type=int, default=1e3, help='Number of photons')
    args = parser.parse_args()
    
    main(photon_num=args.photons, use_mpi=args.use_mpi)

    
    