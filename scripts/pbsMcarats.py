"""
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from atmotomo import RGB_WAVELENGTH, getResourcePath, getMisrDB, density_clouds1,\
     calcAirMcarats, Mcarats, Job, single_voxel_atmosphere
import sparse_transforms as spt
from amitibo import tamnun
import itertools
import argparse
import amitibo
import time
import glob
import os

SLEEP_PERIOD = 10

KM_TO_METERS = 1000
VISIBILITY = 10000

CAMERA_CENTERS = [np.array((i, j, 0.)) + 0.1*np.random.rand(3) for i, j in itertools.product(np.linspace(5000., 45000, 5), np.linspace(5000., 45000, 5))]


def createConfiguationFiles(
    particle,
    grids,
    results_path,
    img_size,
    target,
    A_aerosols,
    cameras_position,
    no_air=False
    ):
    
    k_RGB=np.array(particle['k']) / np.max(np.array(particle['k']))
    dx, dy, dz = [grid_deriv.ravel()[0] for grid_deriv in grids.derivatives]
    z_coords = grids.closed[-1]

    air_ext = calcAirMcarats(z_coords)
    if no_air:
        air_ext = [np.zeros_like(z_coords) for i in range(len(air_ext))]
    
    #
    # Create the configuration files
    #
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
            camera_num=len(cameras_position),
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


def prepareSimulation(results_path, cameras_position, img_size, target):
    """Main doc"""
    
    particle = getMisrDB()['spherical_nonabsorbing_2.80']

    #
    # Simulation parameters
    #
    atmosphere_params = amitibo.attrClass(
        cartesian_grids=spt.Grids(
            np.arange(0, 50000, 1000.0), # Y
            np.arange(0, 50000, 1000.0), # X
            np.arange(0, 10000, 100.0)   # H
            ),
        earth_radius=4000000,
        RGB_WAVELENGTH=RGB_WAVELENGTH,
        air_typical_h=8000,
        aerosols_typical_h=2000,
        sun_angle=30
    )

    #NOT_USED, A_aerosols, Y, X, Z, h = density_clouds1(atmosphere_params)
    #no_air=False
    A_aerosols, Y, X, Z = single_voxel_atmosphere(atmosphere_params, indices_list=[(24, 24, 19)], density=1/VISIBILITY, decay=False)
    A_aerosols = A_aerosols[0]
    no_air=True
    
    #
    # Convert the camera positions to ratio 
    #
    width = atmosphere_params.cartesian_grids.closed[0][-1]
    pos_ratio = np.array((1/width, 1/width, 1))
    cameras_position = [np.array(pos)*pos_ratio for pos in cameras_position]
    
    #
    # Create the configuration files.
    #
    conf_files = createConfiguationFiles(
        particle,
        atmosphere_params.cartesian_grids,
        results_path,
        img_size,
        target,
        A_aerosols,
        cameras_position,
        no_air=no_air
    )

    return conf_files


def main():
    """Main doc """
    
    parser = argparse.ArgumentParser(description='Simulate atmosphere using MCARaTS on the tamnun cluster')
    parser.add_argument('--cameras', action='store_true', help='load the cameras from the cameras file')
    parser.add_argument('--ref_images', help='path to reference images')
    parser.add_argument('--photons', type=int, default=1e9, help='Number of photos to simulate (default=1e9).')
    parser.add_argument('--img_size', type=int, default=128, help='Image size, used for width and height (default=128).')
    parser.add_argument('--target', type=int, choices=range(2,4), default=2, help='Target mode 2=radiance, 3=volume rendering. (default=2).')
    parser.add_argument('--queue_name', default='all_l_p', help='Queue name. (default=all_l_p)')
    parser.add_argument('--node_num', type=int, default=2, help='Number of nodes to use. (default=2).')
    
    args = parser.parse_args()
    
    #
    # Create the results folder
    #
    results_path = os.path.abspath(amitibo.createResultFolder())

    #
    # Cameras positions
    #
    cameras = []
    if args.cameras:
        #
        # Get the camera positions from the camera positions file
        #
        with open(getResourcePath('CamerasPositions.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                cameras.append(np.array([float(i) for i in line.strip().split()])*KM_TO_METER)
                
    elif args.ref_images:
        #
        # Get the camera positions from the ref images folders
        #
        
        for path in glob.glob(os.path.join(args.ref_images, "*")):
            #
            # Parse cameras center file
            #
            with open(os.path.join(path, 'params.txt'), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if parts[0] == 'CameraPosition':
                        cameras.append(np.array((float(parts[4])+25000, float(parts[2])+25000, float(parts[3]))))
                        break
    
    else:
        cameras = CAMERA_CENTERS
        
    #
    # Create the different atmosphere data for the different channels.
    #
    conf_files = prepareSimulation(
        results_path,
        cameras_position=cameras,
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
                M=args.node_num,
                N=12,
                work_directory='$HOME/code/atmosphere',
                queue_name=args.queue_name
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

    
    