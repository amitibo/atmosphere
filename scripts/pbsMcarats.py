"""
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from atmotomo import RGB_WAVELENGTH, getResourcePath, getMisrDB, density_clouds1,\
     calcAirMcarats, Mcarats, Job
from amitibo import tamnun
import itertools
import argparse
import amitibo
import time
import os

SLEEP_PERIOD = 10

ATMOSPHERE_WIDTH = 50000
ATMOSPHERE_HEIGHT = 10000

KM_TO_METERS = 1000
VISIBILITY = 20 * KM_TO_METERS


def prepareSimulationFiles(results_path, cameras_file, img_size, target):
    """Main doc"""
    
    particle = getMisrDB()['spherical_nonabsorbing_2.80']

    #
    # Simulation parameters
    #
    atmosphere_params = amitibo.attrClass(
        cartesian_grids=(
            slice(0, ATMOSPHERE_WIDTH, 1000.0), # Y
            slice(0, ATMOSPHERE_WIDTH, 1000.0), # X
            slice(0, ATMOSPHERE_HEIGHT, 100.)   # H
            ),
        earth_radius=4000000,
        RGB_WAVELENGTH=RGB_WAVELENGTH,
        air_typical_h=8000,
        aerosols_typical_h=2000,
        sun_angle=30
    )

    A_air, A_aerosols, Y, X, Z, h = density_clouds1(atmosphere_params)
    dx = abs(X[0, 1, 0] - X[0, 0, 0])
    dy = abs(Y[1, 0, 0] - Y[0, 0, 0])
    dz = abs(Z[0, 0, 1] - Z[0, 0, 0])
    z_coords = Z[0, 0, :]
    z_coords = np.concatenate((z_coords, [z_coords[-1]+dz]))
    air_ext = calcAirMcarats(z_coords)

    #
    # Calculate the cameras position
    #
    cameras_position = []
    if cameras_file:
        #
        # Get the camera positions from the camera positions file
        # Note:
        # -----
        # The pos_ratio is used for converting the position [km] to ratio (in x, y)
        # and height [in meters]
        #
        pos_ratio = np.array((KM_TO_METERS/ATMOSPHERE_WIDTH, KM_TO_METERS/ATMOSPHERE_WIDTH, KM_TO_METERS))
        with open(getResourcePath('CamerasPositions.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                cameras_position.append(
                    np.array([float(i) for i in line.strip().split()])*pos_ratio
                    )
    else:
        cameras_position = itertools.product(np.linspace(0.1, 0.9, 10), np.linspace(0.1, 0.9, 10), [0])
        
    #
    # Create the test
    #
    k_RGB=np.array(particle['k']) / np.max(np.array(particle['k']))
    
    conf_files = []
    for ch in range(3):
        mc = Mcarats(results_path, base_name='base%d'%ch, use_mpi=True)
        mc.configure(
            shape=A_aerosols.shape,
            dx=dx,
            dy=dy,
            z_coords=z_coords,
            target=target,
            np1d=1,
            np3d=1,
            img_width=img_size,
            img_height=img_size
        )

        job = Job()
        
        job.setSolarSource(theta=120.0, phi=180.0)
        
        job.set1Ddistribution(
            ext=air_ext[ch],
            omg=np.ones_like(air_ext[ch]),
            apf=-1*np.ones_like(air_ext[ch])
        )
        
        job.set3Ddistribution(
            ext=k_RGB[ch]*A_aerosols / VISIBILITY,
            omg=particle['w'][ch]*np.ones_like(A_aerosols),
            apf=particle['g'][ch]*np.ones_like(A_aerosols)
        )

        for ypos, xpos, zloc in cameras_position:
            job.addCamera(
                xpos=xpos,
                ypos=ypos,
                zloc=zloc,
                theta=0,
                phi=0,
                psi=0,
            )
            
        #
        # Store the configuration files
        #
        conf_files.append(mc.prepareSimulation([job]))
    
    return conf_files


def main():
    """Main doc """
    
    parser = argparse.ArgumentParser(description='Simulate atmosphere using MCARaTS on the tamnun cluster')
    parser.add_argument('--cameras_file', action='store_true', help='load the cameras from the cameras file')
    parser.add_argument('--photons', type=int, default=1e9, help='Number of photos to simulate (default=1e9).')
    parser.add_argument('--img_size', type=int, default=128, help='Image size, used for width and height (default=128).')
    parser.add_argument('--target', type=int, choices=range(2,4), default=2, help='Target mode 2=radiance, 3=volume rendering. (default=2).')
    args = parser.parse_args()
    
    #
    # Create the results folder
    #
    results_path = os.path.abspath(amitibo.createResultFolder())

    #
    # Create the different atmosphere data for the different channels.
    #
    conf_files = prepareSimulationFiles(
        results_path,
        cameras_file=args.cameras_file,
        img_size=args.img_size,
        target=args.target
    )
    out_files = ['%s_out' % name for name in conf_files]
    
    #
    # Submit the separate jobs 
    #
    jobs_id = []
    for conf_file, out_file in zip(conf_files, out_files):
        jobs_id.append(
            qsub(
                cmd='$HOME/.local/bin/mcarats_mpi',
                params='%d 0 %s %s' % (args.photons, conf_file, out_file),
                M=4,
                N=12,
                work_directory='$HOME/code/atmosphere'
            )
        )
        
    #
    # Sleep-Wait checking the status of the jobs
    #
    while len(jobs_id) > 0:
        time.sleep(SLEEP_PERIOD)
        jobs_id = [id for id in jobs_id if tamnun.qstat(id)]
        
    #
    # Process the results
    #
    imgs = Mcarats.calcRGBImg(*out_files)
    
    #
    # Show the results
    #
    figures = []
    for img in imgs:
        figures.append(plt.figure())
        plt.imshow(img)
    
    amitibo.saveFigures(results_path, bbox_inches='tight', figures=figures)
    
    #
    # Notify by email
    #
    tamnun.sendmail(subject="finished run", content="Finished run")
    
    
if __name__ == '__main__':
    main()

    
    