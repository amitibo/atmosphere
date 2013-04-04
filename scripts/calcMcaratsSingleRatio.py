"""
Simulate single voxel atmospheres using MCARaTS on the tamnun cluster
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from atmotomo import RGB_WAVELENGTH, getMisrDB,\
     single_voxel_atmosphere, Mcarats, Job
import itertools
import argparse
import amitibo
from amitibo import tamnun
import time
import os

SLEEP_PERIOD = 10

ATMOSPHERE_WIDTH = 50000
ATMOSPHERE_HEIGHT = 10000

VISIBILITY = 10000

VOXEL_INDICES= [
    (25, 25, 15), 
    (25, 25, 10),
    (25, 25, 5),
    (24, 24, 1),
    (24, 24, 5),
    (24, 24, 10),
    (20, 26, 15),
    (20, 20, 10)]


def prepareSimulationFiles(results_path, img_size, target):
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
        RGB_WAVELENGTH=RGB_WAVELENGTH,
        air_typical_h=8000,
        aerosols_typical_h=2000,
        sun_angle=30
    )

    A_aerosols, Y, X, Z = single_voxel_atmosphere(atmosphere_params, indices_list=VOXEL_INDICES, density=1/VISIBILITY)

    dx = abs(X[0, 1, 0] - X[0, 0, 0])
    dy = abs(Y[1, 0, 0] - Y[0, 0, 0])
    dz = abs(Z[0, 0, -1] - Z[0, 0, -2])
    z_coords = Z[0, 0, :]
    z_coords = np.concatenate((z_coords, [z_coords[-1]+dz]))

    #
    # Set the camera position
    #
    cameras_position = (np.array([0.5, 0.5, 1.0]), )
        
    #
    # Create the test
    #
    k_RGB=np.array(particle['k']) / np.max(np.array(particle['k']))
    
    conf_files = []
    for ch in range(len(RGB_WAVELENGTH)):
        mc = Mcarats(results_path, base_name='base%d'%ch, use_mpi=True)
        mc.configure(
            shape=A_aerosols[0].shape,
            dx=dx,
            dy=dy,
            z_coords=z_coords,
            target=target,
            np1d=0,
            np3d=1,
            img_width=img_size,
            img_height=img_size
        )
        
        jobs = []
        for aero_dist in A_aerosols:
            job = Job()
            job.setSolarSource(theta=120.0, phi=180.0)
            job.set3Ddistribution(
                ext=k_RGB[ch]*aero_dist,
                omg=particle['w'][ch]*np.ones_like(aero_dist),
                apf=particle['g'][ch]*np.ones_like(aero_dist)
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
            
            jobs.append(job)
            
        #
        # Store the configuration files
        #
        conf_files.append(mc.prepareSimulation(jobs))
    
    return conf_files


def main():
    """Main doc """
    
    parser = argparse.ArgumentParser(description='Simulate single voxel atmospheres using MCARaTS on the tamnun cluster')
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
            tamnun.qsub(
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

    
    