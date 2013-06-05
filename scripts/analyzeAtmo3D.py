"""
Reconstruct a general distribution of aerosols in the atmosphere.
"""
from __future__ import division
import numpy as np
from atmotomo import calcHG, L_SUN_RGB, RGB_WAVELENGTH
from atmotomo import Camera
from atmotomo import calcAirMcarats, getMisrDB, Mcarats, single_voxel_atmosphere, density_clouds_vadim
import sparse_transforms as spt
import atmotomo
import scipy.io as sio
import scipy.ndimage as ndimage
import amitibo
import itertools
import os
import sys
import argparse
import glob
import tempfile
import shutil

#
# Set logging level
#
import logging

#
# Initialize the mpi system
#
from mpi4py import MPI
comm = MPI.COMM_WORLD

mpi_size = MPI.COMM_WORLD.Get_size()
mpi_rank = MPI.COMM_WORLD.Get_rank()

IMGTAG = 1
OBJTAG = 2
GRADTAG = 3
DIETAG = 4


MAX_ITERATIONS = 1000
MCARATS_IMG_SCALE = 10.0**9.7
VADIM_IMG_SCALE = 503.166

profile = False


#
# override excepthook so that an exception in one of the childs will cause mpi to abort execution.
#
def abortrun(type, value, tb):
    import traceback
    traceback.print_exception(type, value, tb)
    MPI.COMM_WORLD.Abort(1)
    
sys.excepthook = abortrun


def split_lists(items, n):
    """"""
    
    k = len(items) % n
    p = int(len(items) / n)

    indices = np.zeros(n+1, dtype=np.int)
    indices[1:] = p
    indices[:k] += 1
    indices = indices.cumsum()
    
    return [items[s:e] for s, e in zip(indices[:-1], indices[1:])]


class RadianceProblem(object):
    def __init__(self, A_aerosols, A_air, results_path):

        #
        # Send the real atmospheric distribution to all childs so as to create the measurement.
        #
        for i in range(1, mpi_size):
            comm.send([A_air, A_aerosols, results_path], dest=i, tag=IMGTAG)

        self._objective_values = []
        self._intermediate_values = []
        self._atmo_shape = A_aerosols.shape
        self._results_path = results_path
        
    def objective(self, x):
        """Calculate the objective"""

        x = x.reshape((-1, 1))
        
        #
        # Distribute 
        #
        for i in range(1, mpi_size):
            comm.Send(x, dest=i, tag=OBJTAG)

        #
        # Store temporary x in case the simulation is stoped in the middle.
        #
        sio.savemat(
            os.path.join(self._results_path, 'temp_rad.mat'),
            {
                'estimated': x.reshape(self._atmo_shape),
            },
            do_compression=True
        )
        
        #
        # Query the slaves for the calculate objective.
        #
        sts = MPI.Status()

        obj = 0
        temp = np.empty(1)
        for i in range(1, mpi_size):
            comm.Recv(temp, source=MPI.ANY_SOURCE, status=sts)
            obj += temp[0]
        
        self._objective_values.append(obj)

        #
        # Save temporary objective values in case the simulation is stopped
        # in the middle.
        #
        sio.savemat(
            os.path.join(self._results_path, 'temp_obj.mat'),
            {
                'objective': np.array(self.obj_values)
            },
            do_compression=True
        )
        
        return obj
    
    def gradient(self, x):
        """The callback for calculating the gradient"""

        x = x.reshape((-1, 1))

        for i in range(1, mpi_size):
            comm.Send(x, dest=i, tag=GRADTAG)

        sts = MPI.Status()

        temp = np.empty((x.size, 1))
        grad = np.zeros_like(temp)
        for i in range(1, mpi_size):
            comm.Recv(temp, source=MPI.ANY_SOURCE, status=sts)
            grad += temp
            
        return grad.flatten()

    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):

        self._intermediate_values.append(obj_value)
        logging.log(logging.INFO, 'iteration: %d, objective: %g' % (iter_count, obj_value))
        
        return True

    @property
    def obj_values(self):
        
        if self._intermediate_values:
            return self._intermediate_values
        else:
            return self._objective_values
        

def master(air_dist, aerosols_dist, results_path, solver='ipopt', job_id=None):
    #import rpdb2; rpdb2.start_embedded_debugger('pep')
    
    logging.basicConfig(
        filename=os.path.join(results_path, 'run.log'),
        level=logging.DEBUG
    )

    #
    # Initial distribution for optimization
    # Note:
    # I don't use zeros_like because the dtype of aerosols_dist
    # is '<d' (probably because it originates in a matlab matrix)
    # and mpi4py doesn't like to comm.Send it (produces KeyError).
    # Using np.zeros produces byteorder '=' which stands for
    # native ('<' means little endian).
    #
    x0 = np.zeros(aerosols_dist.shape)

    #
    # Create the optimization problem object
    #
    radiance_problem = RadianceProblem(
        A_aerosols=aerosols_dist,
        A_air=air_dist,
        results_path=results_path
    )

    if solver == 'ipopt':
        import ipopt
        
        #
        # Define the problem
        #
        lb = np.zeros(aerosols_dist.size)
        
        ipopt.setLoggingLevel(logging.DEBUG)

        problem = ipopt.problem(
            n=aerosols_dist.size,
            m=0,
            problem_obj=radiance_problem,
            lb=lb
        )
    
        #
        # Set solver options
        #
        #problem.addOption('derivative_test', 'first-order')
        #problem.addOption('derivative_test_print_all', 'yes')
        #problem.addOption('derivative_test_tol', 5e-3)
        #problem.addOption('derivative_test_perturbation', 1e-8)
        problem.addOption('hessian_approximation', 'limited-memory')
        problem.addOption('mu_strategy', 'adaptive')
        problem.addOption('tol', 1e-9)
        problem.addOption('max_iter', MAX_ITERATIONS)
    
        #
        # Solve the problem
        #
        x, info = problem.solve(x0)
    else:
        import scipy.optimize as sop
        
        x, obj, d = sop.fmin_l_bfgs_b(
            func=radiance_problem.objective,
            x0=x0,
            fprime=radiance_problem.gradient,
            bounds=[(0, None)]*x0.size,
            maxfun=MAX_ITERATIONS
        )

    #
    # Kill all slaves
    #
    for i in range(1, mpi_size):
        comm.Send(x, dest=i, tag=DIETAG)
    
    #
    # Store the estimated distribution
    #
    sio.savemat(
        os.path.join(results_path, 'radiance.mat'),
        {
            'true': aerosols_dist,
            'estimated': x.reshape(aerosols_dist.shape),
            'objective': np.array(radiance_problem.obj_values)
        },
        do_compression=True
    )


def slave(
    atmosphere_params,
    particle_params,
    sun_params,
    camera_params,
    camera_positions,
    ref_images,
    switch_cams_period=10,
    simulate_imgs=False
    ):
    #import rpdb2; rpdb2.start_embedded_debugger('pep')

    assert len(camera_positions) == len(ref_images), 'The number of cameras positions and reference images should be equal'
    camera_num = len(camera_positions)
    
    #
    # Instatiate the camera slave
    #
    cam_paths = []
    for camera_position in camera_positions:
        
        cam = Camera()
        cam.create(
            sun_params=sun_params,
            atmosphere_params=atmosphere_params,
            camera_params=camera_params,
            camera_position=camera_position
        )
        
        cam_path = tempfile.mkdtemp(prefix='/gtmp/')
        cam.save(cam_path)
        cam_paths.append(cam_path)
    
    #
    # Set the camera_index to point to the last camera created
    # which is the currently loaded camera.
    #
    camera_index = camera_num-1
    switch_counter = 0
    
    sts = MPI.Status()

    #
    # The first data should be for creating the measured images.
    #
    data = comm.recv(source=0, tag=MPI.ANY_TAG, status=sts)

    tag = sts.Get_tag()
    if tag != IMGTAG:
        raise Exception('The first data transaction should be for calculting the meeasure images')

    A_air = data[0]
    A_aerosols = data[1]
    results_path = data[2]
    
    if simulate_imgs:
        for i, cam_path in enumerate(cam_paths):
            cam.load(cam_path)        
            cam.setA_air(A_air)
    
            sim_img = cam.calcImage(
                A_aerosols=A_aerosols,
                particle_params=particle_params
            )
            
            sio.savemat(
                os.path.join(results_path, 'sim_img' + ('0000%d%d.mat' % (mpi_rank, i))[-9:]),
                {'img': sim_img},
                do_compression=True
            )

    for i, ref_img in enumerate(ref_images):

        sio.savemat(
            os.path.join(results_path, 'ref_img' + ('0000%d%d.mat' % (mpi_rank, i))[-9:]),
            {'img': ref_img},
            do_compression=True
        )
        
    cam.setA_air(A_air)

    ref_img = ref_images[camera_index]
    
    
    #
    # Loop the messages of the master
    #
    while 1:
        comm.Recv(A_aerosols, source=0, tag=MPI.ANY_TAG, status=sts)
        
        tag = sts.Get_tag()
        if tag == DIETAG:
            break

        if tag == OBJTAG:
            print 'slave %d calculating objective' % mpi_rank
            
            img = cam.calcImage(
                A_aerosols=A_aerosols,
                particle_params=particle_params
            )
            
            temp = ref_img.reshape((-1, 1)) - img.reshape((-1, 1))
            obj = np.dot(temp.T, temp)
            
            comm.Send(np.array(obj), dest=0)
            
            #
            # Check if there is a need to switch the cams
            #
            switch_counter += 1
            if switch_counter % camera_num == 0:
                camera_index = (camera_index + 1) % camera_num
                
                print 'slave %d switching to camera index %d' % (mpi_rank, camera_index)
                
                cam.load(cam_paths[camera_index])        
                cam.setA_air(A_air)

                ref_img = ref_images[camera_index]
                print 'slave %d successfully created camera' % mpi_rank
                
                
        elif tag == GRADTAG:
            print 'slave %d calculating gradient' % mpi_rank
            
            img = cam.calcImage(
                A_aerosols=A_aerosols,
                particle_params=particle_params
            )

            grad = cam.calcImageGradient(
                img_err=ref_img-img,
                A_aerosols=A_aerosols,
                particle_params=particle_params
            )
                    
            comm.Send(grad, dest=0)
            
        else:
            raise Exception('Unexpected tag %d' % tag)

    #
    # Save the image the relates to the calculated aerosol distribution
    #
    for i, cam_path in enumerate(cam_paths):
        cam.load(cam_path)        
        cam.setA_air(A_air)

        final_img = cam.calcImage(
            A_aerosols=A_aerosols,
            particle_params=particle_params
        )
        
        sio.savemat(
            os.path.join(results_path, 'final_img' + ('0000%d%d.mat' % (mpi_rank, i))[-9:]),
            {'img': final_img},
            do_compression=True
        )
        
        try:
            shutil.rmtree(cam_path)
        except Exception, e:
            print 'Failed to remove folder %s:\n%s\n' % (cam_path, repr(e))
        

def loadSlaveData(atmosphere_params, ref_images, mcarats, sigma, remove_sunspot):
    """"""
    
    global mpi_size
    
    if mcarats:
        raise NotImplemented('The mcarats code is not yet adapted to the new configuration files')
    
        mpi_size = min(mpi_size, len(lines)+1)
        
        if mpi_rank >= mpi_size:
            sys.exit()
                
        path = os.path.abspath(mcarats)

        R_ch, G_ch, B_ch = [np.fromfile(os.path.join(path, 'base%d_conf_out' % i), dtype=np.float32) for i in range(3)]
        IMG_SHAPE = (128, 128)
        IMG_SIZE = IMG_SHAPE[0] * IMG_SHAPE[1]
        
        if mpi_rank > 0:
            slc = slice((mpi_rank-1)*IMG_SIZE, mpi_rank*IMG_SIZE)
            ref_img = Mcarats.calcMcaratsImg(R_ch, G_ch, B_ch, slc, IMG_SHAPE)
            ref_img = ref_img.astype(np.float) / MCARATS_IMG_SCALE
        
    elif ref_images:
        #
        # Load the reference images
        #
        closed_grids = atmosphere_params.cartesian_grids.closed
        ref_images_list, camera_positions_list = atmotomo.loadVadimData(
            ref_images,
            (closed_grids[0][-1]/2, closed_grids[1][-1]/2),
            remove_sunspot=remove_sunspot
        )
        
        #
        # Limit the number of mpi processes used.
        #
        mpi_size = min(mpi_size, len(camera_positions_list)+1)
        
        if mpi_rank >= mpi_size:
            sys.exit()
        
        #
        # Smooth the reference images if necessary
        #
        for ref_img in ref_images_list:
            if sigma > 0.0:
                for channel in range(ref_img.shape[2]):
                    ref_img[:, :, channel] = \
                        ndimage.filters.gaussian_filter(ref_img[:, :, channel], sigma=sigma)
            
            
    else:
        raise Exception('No reference images given')

    return ref_images_list, camera_positions_list

    

def main(
    params_path,
    ref_mc=None,
    mcarats=None,
    simulate=False,
    sigma=0.0,
    remove_sunspot=False,
    job_id=None
    ):
    
    #import wingdbstub

    #
    # Load the simulation params
    #
    atmosphere_params, particle_params, sun_params, camera_params, cameras, air_dist, aerosols_dist = atmotomo.readConfiguration(params_path)
        
    #
    # Calculate the camera position (this is relevant only for the slave
    # but it also calculated the mpi_size which important for the master
    # also)
    #
    ref_images_list, camera_positions_list = loadSlaveData(
        atmosphere_params,
        ref_mc,
        mcarats,
        sigma,
        remove_sunspot
    )
    
    if mpi_rank == 0:
        #
        # Create the results path
        #
        results_path = amitibo.createResultFolder(
            params=[atmosphere_params, particle_params, sun_params, camera_params],
            src_path=atmotomo.__src_path__,
            job_id=job_id
        )
        
        #
        # Set up the solver server.
        #
        master(
            air_dist=air_dist,
            aerosols_dist=aerosols_dist,
            results_path=results_path,
            solver='bfgs',
            job_id=job_id
        )
    else:
        ref_images = split_lists(ref_images_list, mpi_size-1)[mpi_rank-1]
        camera_positions = split_lists(camera_positions_list, mpi_size-1)[mpi_rank-1]
        slave(
            atmosphere_params=atmosphere_params,
            particle_params=particle_params,
            sun_params=sun_params,
            camera_params=camera_params,
            camera_positions=camera_positions,
            ref_images=ref_images,
            simulate_imgs=simulate
        )


if __name__ == '__main__':    
    #
    # Parse the input
    #
    parser = argparse.ArgumentParser(description='Analyze atmosphere')
    parser.add_argument('--mcarats', help='path to reference mcarats results folder')
    parser.add_argument('--ref_mc', help='path to reference images of vadims code')
    parser.add_argument('--sigma', type=float, default=0.0, help='smooth the reference image by sigma')
    parser.add_argument('--simulate', action='store_true', help='Simulate images from given distributions (useful for debug).')
    parser.add_argument('--remove_sunspot', action='store_true', help='Remove sunspot from reference images.')
    parser.add_argument('--job_id', default=None, help='pbs job ID (set automatically by the PBS script)')
    parser.add_argument('params_path', help='Path to simulation parameters')
    args = parser.parse_args()

    main(
        params_path=args.params_path,
        ref_mc=args.ref_mc,
        mcarats=args.mcarats,
        simulate=args.simulate,
        sigma=args.sigma,
        remove_sunspot=args.remove_sunspot,
        job_id=args.job_id
    )
