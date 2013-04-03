"""
Simulate single voxel atmospheres using MCARaTS on the tamnun cluster
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from atmotomo import RGB_WAVELENGTH, getResourcePath, getMisrDB,\
     single_voxel_atmosphere, calcAirMcarats, Mcarats, SOLVER_F3D, Job
import subprocess as sub
import itertools
import argparse
import amitibo
import jinja2
import time
import os

SLEEP_PERIOD = 10
SENDMAIL = "/usr/sbin/sendmail" # sendmail location
PBS_TEMPLATE_FILE_NAME = 'pbs.jinja'

ATMOSPHERE_WIDTH = 50000
ATMOSPHERE_HEIGHT = 10000

KM_TO_METERS = 1000
VISIBILITY = 20 * KM_TO_METERS


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
        earth_radius=4000000,
        RGB_WAVELENGTH=RGB_WAVELENGTH,
        air_typical_h=8000,
        aerosols_typical_h=2000,
        sun_angle=30
    )

    A_aerosols, Y, X, Z = single_voxel_atmosphere(atmosphere_params, heights=[10, 20, 30])
    dx = abs(X[0, 1, 0] - X[0, 0, 0])
    dy = abs(Y[1, 0, 0] - Y[0, 0, 0])
    dz = abs(Z[0, 0, -1] - Z[0, 0, -2])
    z_coords = Z[0, 0, :]
    z_coords = np.concatenate((z_coords, [z_coords[-1]+dz]))

    #
    # Set the camera position
    #
    cameras_position = (np.array([0.5, 0.5, 0.0]), )
        
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


def qsub(pbs_tpl, results_path, conf_file, out_file, photons_num):
    """Submit a job"""
    
    prc_ret = sub.Popen('qsub', shell=True, stdin=sub.PIPE, stdout=sub.PIPE, stderr=sub.PIPE)
    pbs_script = pbs_tpl.render(
        queue_name='minerva_h_p',
        M=4,
        N=12,
        work_directory='$HOME/code/atmosphere',
        cmd='$HOME/.local/bin/mcarats_mpi',
        params='%d 0 %s %s' % (photons_num, conf_file, out_file)
    )
    prc_ret.stdin.write(pbs_script)
    prc_ret.stdin.close()
    id = prc_ret.stdout.read()
    return id


def qstat(id):
    """Check the status of a job"""

    prc_ret = sub.Popen('qstat %s' % id, shell=True, stdin=sub.PIPE, stdout=sub.PIPE, stderr=sub.PIPE)
    out = prc_ret.stdout.read().strip()
    err = prc_ret.stderr.read().strip()
    
    #
    # Check if job finished
    #
    if 'Unknown Job Id' in err:
        return False
    
    return True


def main():
    """Main doc """
    
    parser = argparse.ArgumentParser(description='Simulate single voxel atmospheres using MCARaTS on the tamnun cluster')
    parser.add_argument('--photons', type=int, default=1e9, help='Number of photos to simulate (default=1e9).')
    parser.add_argument('--img_size', type=int, default=128, help='Image size, used for width and height (default=128).')
    parser.add_argument('--target', type=int, choices=range(2,4), default=2, help='Target mode 2=radiance, 3=volume rendering. (default=2).')
    args = parser.parse_args()
    
    #
    # Create template loader
    #
    tpl_loader = jinja2.FileSystemLoader(searchpath=getResourcePath('.'))
    tpl_env = jinja2.Environment(loader=tpl_loader)
    pbs_tpl = tpl_env.get_template(PBS_TEMPLATE_FILE_NAME)

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
            qsub(
                pbs_tpl,
                results_path,
                conf_file, 
                out_file,
                photons_num=args.photons
            )
        )
        
    #
    # Sleep-Wait checking the status of the jobs
    #
    while len(jobs_id) > 0:
        time.sleep(SLEEP_PERIOD)
        jobs_id = [id for id in jobs_id if qstat(id)]
        
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
    p = os.popen("%s -t" % SENDMAIL, "w")
    p.write("To: amitibo@tx.technion.ac.il\n")
    p.write("Subject: finished run\n")
    p.write("\n") # blank line separating headers from body
    p.write("Finished run\n")
    sts = p.close()
    
    
if __name__ == '__main__':
    main()

    
    