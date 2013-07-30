"""
Reconstruct a general distribution of aerosols in the atmosphere.
"""
from __future__ import division
import numpy as np
from atmotomo import Camera, Mcarats
import sparse_transforms as spt
import atmotomo
import scipy.io as sio
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import amitibo
import itertools
import os
import sys
import argparse
import glob
import tempfile
import shutil
import time

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
READYTAG = 5
RATIOTAG = 6


MAX_ITERATIONS = 4000
MAX_ITERATIONS_STEP = 200
MCARATS_IMG_SCALE = 10.0**9.7

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
    indices[1:k+1] += 1
    indices = indices.cumsum()
    
    return [items[s:e] for s, e in zip(indices[:-1], indices[1:])]


def calcRatio(ref_img, single_img, erode=False):
    #
    # Calc a joint mask
    #
    mask = (ref_img > 0) * (single_img > 0)
    if erode:
        for i in range(3):
            mask[:, :, i] = morph.greyscale_erode(mask[:, :, i].astype(np.uint8) , morph.disk(1))
        mask = mask>0
    
    ratio = ref_img[mask].mean() / single_img[mask].mean()

    return ratio


def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""

    return lambda x,y: height*np.exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)


class RadianceProblem(object):
    def __init__(
        self,
        atmosphere_params,
        A_aerosols,
        A_air,
        results_path,
        ref_imgs,
        laplace_weights,
        regularization_decay=1.0,
        tau=0.0,
        ref_ratio=0.0,
        use_simulated=False        
        ):

        #
        # Send the real atmospheric distribution to all childs so as to create the measurement.
        #
        for i in range(1, mpi_size):
            comm.send([A_air, A_aerosols, results_path], dest=i, tag=IMGTAG)

        if not use_simulated:
            #
            # Recieve the simulated images and sort according to rank
            #
            sts = MPI.Status()
    
            unsort_sim_imgs = []
            sim_imgs_src = []
            for i in range(1, mpi_size):
                unsort_sim_imgs.append(comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=sts))
                assert sts.tag == RATIOTAG, 'Expecting the RATIO tag'            
                sim_imgs_src.append(sts.source)
            
            sim_imgs = []
            for i in np.argsort(sim_imgs_src):
                sim_imgs += unsort_sim_imgs[i]
    
            #
            # if ratio is not given it is calculated automaticall
            #
            if ref_ratio == 0.0:
                means = []
                for i, (ref_img, sim_img) in enumerate(zip(ref_imgs, sim_imgs)):
                    means.append(calcRatio(ref_img, sim_img))
                    
                ref_ratio = np.mean(means)
    
            #
            # Calculate the mask around the sun
            #
            sun_mask_auto = calcAutoMask(sim_imgs, ref_imgs, ref_ratio)
            
            sun_mask_manual = calcManualMask(ref_imgs)
            
            sio.savemat(
                os.path.join(results_path, 'sun_mask.mat'),
                {
                    'sun_mask_auto': sun_mask_auto,
                    'sun_mask_manual': sun_mask_manual
                },
                do_compression=True
            )
        
            #
            # Send back the averaged calculated ratio
            #
            for i in range(1, mpi_size):
                comm.send([ref_ratio, sun_mask_auto, sun_mask_manual], dest=i, tag=RATIOTAG)
        
        #
        # Calculate a height dependant weight map for the regularization
        #
        Y, X, Z = atmosphere_params.cartesian_grids.expanded
        self._regu_mask = np.exp(-Z * regularization_decay)

        self.atmosphere_params = atmosphere_params
        self.laplace_weights = laplace_weights
        self._objective_values = []
        self._intermediate_values = []
        self._atmo_shape = A_aerosols.shape
        self._true_dist = A_aerosols
        self._results_path = results_path
        self._objective_cnt = 0
        self._tau = tau
        
    @property
    def tau(self):
        
        return self._tau
    
    @tau.setter
    def tau(self, tau):
        
        self._tau = tau
    
    
    def objective(self, x):
        """Calculate the objective"""

        logging.log(logging.INFO, 'Objective called.')
        x = x.reshape((-1, 1))
        
        #
        # Distribute 
        #
        for i in range(1, mpi_size):
            comm.Send([x, x.dtype.char], dest=i, tag=OBJTAG)

        #
        # Query the slaves for the calculate objective.
        #
        sts = MPI.Status()

        obj = 0
        temp = np.empty(1)
        for i in range(1, mpi_size):
            comm.Recv(temp, source=MPI.ANY_SOURCE, status=sts)
            obj += temp[0]
        
        #
        # Add regularization
        #
        x_laplace = atmotomo.weighted_laplace(x.reshape(self._atmo_shape), weights=self.laplace_weights)
        
        #
        # Apply a height dependant masking to the regularization
        #
        x_laplace *= self._regu_mask
        
        obj += self._tau * np.linalg.norm(x_laplace)**2
        
        self._objective_values.append(obj)

        if self._objective_cnt % 10 == 0:
            #
            # Store temporary radiance and objective values in case the simulation is
            # stoped in the middle.
            #
            Y, X, Z = self.atmosphere_params.cartesian_grids.expanded
            limits = np.array([int(l) for l in self.atmosphere_params.cartesian_grids.limits])
            
            sio.savemat(
                os.path.join(self._results_path, 'temp_radiance_tau_%g.mat' % self.tau),
                {
                    'limits': limits,
                    'Y': Y,
                    'X': X,
                    'Z': Z,
                    'true': self._true_dist,
                    'estimated': x.reshape(self._atmo_shape),
                    'objective': np.array(self.obj_values)
                },
                do_compression=True
            )           
        
        self._objective_cnt += 1
        
        logging.log(logging.INFO, '....objective finished.')
        return obj
    
    def gradient(self, x):
        """The callback for calculating the gradient"""

        logging.log(logging.INFO, 'Gradient called.')
        x = x.reshape((-1, 1))

        for i in range(1, mpi_size):
            comm.Send([x, x.dtype.char], dest=i, tag=GRADTAG)

        sts = MPI.Status()

        temp = np.empty((x.size, 1))
        grad = np.zeros_like(temp)
        for i in range(1, mpi_size):
            comm.Recv(temp, source=MPI.ANY_SOURCE, status=sts)
            grad += temp
            
        #grad_numerical = approx_fprime(x.ravel(), self.objective, epsilon=1e-8)
        #np.save('grad.npy', grad)
        #np.save('grad_numerical.npy', grad_numerical)
        
        #
        # For some reason the gradient is transposed. I found it out by comparing
        # with the numberical gradient. A possible reason is a mismatch between
        # Fortran and C order representation possibly due to loading of the configuration
        # files.
        #
        #grad = np.transpose(grad.reshape(self._atmo_shape))

        #
        # Add regularization
        #
        x_laplace = atmotomo.weighted_laplace(
            x.reshape(self._atmo_shape),
            weights=self.laplace_weights
        )
        #
        # Apply height dependant map
        #
        x_laplace *= self._regu_mask * self._regu_mask
        
        grad_x_laplace = atmotomo.weighted_laplace(
            x_laplace,
            weights=self.laplace_weights
        )
        
        logging.log(logging.INFO, '....gradient finished.')
        return grad.flatten() + 2 * self._tau * grad_x_laplace.flatten()

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


def calcManualMask(ref_imgs):
    img_shape = ref_imgs[0].shape
    
    #
    # Create a gaussian at the center of the sun
    #
    X, Y = np.meshgrid(np.linspace(0, 1, img_shape[0]), np.linspace(0, 1, img_shape[1]))
    mask = gaussian(4.5, 0.8, 0.50, 0.05, 0.05)(X, Y)
    
    #
    # Mask the horizon pixel
    #
    Y_sensor, step = np.linspace(-1.0, 1.0, img_shape[0], endpoint=False, retstep=True)
    X_sensor = np.linspace(-1.0, 1.0, img_shape[1], endpoint=False)
    X_sensor, Y_sensor = np.meshgrid(X_sensor+step/2, Y_sensor+step/2)
    R_sensor = np.sqrt(X_sensor**2 + Y_sensor**2)
    THETA = R_sensor * np.pi / 2
    
    mask[THETA>(np.pi/2*80/90)] = 4
    
    #
    # Calculate the actual mask
    #
    sun_mask_manual = np.tile(np.exp(-mask)[:, :, np.newaxis], (1, 1, 3))
    
    return sun_mask_manual


def calcAutoMask(sim_imgs, ref_imgs, ref_ratio):
    err_imgs = []
    for i, (ref_img, sim_img) in enumerate(zip(ref_imgs, sim_imgs)):
        err_imgs.append(ref_img/ref_ratio - sim_img)
    err_mean = np.dstack(err_imgs).mean(axis=2)
    sun_mask_auto = np.tile(np.exp(-err_mean)[:, :, np.newaxis], (1, 1, 3))
    return sun_mask_auto
        

class ParametericRadianceProblem(RadianceProblem):
    
    def __init__(self, model, *params, **kwds):
        
        super(ParametericRadianceProblem, self).__init__(*params, **kwds)
        self._model = model
        
    def objective(self, x, userdata):
        
        _x = self._model(x)
        
        return super(ParametericRadianceProblem, self).objective(_x)

        
def master(
    atmosphere_params,
    air_dist,
    aerosols_dist,
    results_path,
    ref_imgs,
    laplace_weights,
    tau=0.0,
    regularization_decay=0.0,
    ref_ratio=0.0,
    solver='ipopt',
    use_simulated=False,    
    init_with_solution=False,
    ):
    
    #import rpdb2; rpdb2.start_embedded_debugger('pep')
    #import wingdbstub
    
    logging.basicConfig(
        filename=os.path.join(results_path, 'run.log'),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
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
    atmo_shape = aerosols_dist.shape
    if init_with_solution:
        x0 = aerosols_dist.copy()
    else:
        x0 = np.zeros(atmo_shape)

    #
    # Create the optimization problem object
    #
    radiance_problem = RadianceProblem(
        atmosphere_params=atmosphere_params,
        A_aerosols=aerosols_dist,
        A_air=air_dist,
        results_path=results_path,
        ref_imgs=ref_imgs,
        laplace_weights=laplace_weights,
        tau=tau,
        regularization_decay=regularization_decay,
        ref_ratio=ref_ratio,
        use_simulated=use_simulated
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
        
    elif solver == 'global':
        
        import DIRECT
        
        #
        # Create the model
        #
        model = atmotomo.SphereCloudsModel(
            atmosphere_params,
            clouds_num=2
        )
        
        #
        # Create the optimization problem object
        #
        radiance_problem = ParametericRadianceProblem(
            model=model,
            atmosphere_params=atmosphere_params,
            A_aerosols=aerosols_dist,
            A_air=air_dist,
            results_path=results_path,
            ref_imgs=ref_imgs,
            laplace_weights=laplace_weights,
            tau=tau,
            regularization_decay=regularization_decay,
            ref_ratio=ref_ratio,
            use_simulated=use_simulated
        )

        #
        # Solve the problem
        #
        x, fmin, ierror = DIRECT.solve(
            objective=radiance_problem.objective,
            l=model.lower_bounds,
            u=model.upper_bounds
        )
        
        #
        # Calculate the 
        #
        x = model(x)
        
    else:
        import scipy.optimize as sop
        
        bounds = calcBounds(grids=atmosphere_params.cartesian_grids)
        
        zero_voxels = min(3, int(atmo_shape[2]/8 + 0.5))
            
        for tau, factr, pgtol in zip(np.logspace(-8, -10, num=3), [1e7, 5e6, 1e6], [1e-5, 1e-6, 1e-7]):
            print 'Running optimization using tau=%g, factr=%g' % (tau, factr)
            radiance_problem.tau = tau
            for j in range(int(MAX_ITERATIONS/MAX_ITERATIONS_STEP)):
                x, obj, info = sop.fmin_l_bfgs_b(
                    func=radiance_problem.objective,
                    x0=x0,
                    fprime=radiance_problem.gradient,
                    bounds=bounds,
                    factr=factr,
                    pgtol=pgtol,
                    maxfun=MAX_ITERATIONS_STEP
                )
                
                #
                # Prepare the next iteration zeroing the top of the atmosphere
                #
                x0 = x.reshape(atmo_shape).copy()
                x0[:, :, -zero_voxels:] = 0

                #
                # Check if convereged
                #
                if info['warnflag'] != 1:
                    break
                

            #
            # store optimization info
            #
            import pickle
            with open(os.path.join(results_path, 'optimization_info_tau_%g.pkl' % tau), 'w') as f:
                pickle.dump(info, f)    


    #
    # Kill all slaves
    #
    for i in range(1, mpi_size):
        comm.Send([x, x.dtype.char], dest=i, tag=DIETAG)
    
    #
    # Store the estimated distribution
    #
    Y, X, Z = atmosphere_params.cartesian_grids.expanded
    limits = np.array([int(l) for l in atmosphere_params.cartesian_grids.limits])
    
    sio.savemat(
        os.path.join(results_path, 'radiance.mat'),
        {
            'limits': limits,
            'Y': Y,
            'X': X,
            'Z': Z,
            'true': aerosols_dist,
            'estimated': x.reshape(atmo_shape),
            'objective': np.array(radiance_problem.obj_values)
        },
        do_compression=True
    )
    
    #
    # store optimization info
    #
    import pickle
    with open(os.path.join(results_path, 'optimization_info.pkl'), 'w') as f:
        pickle.dump(info, f)    


def calcBounds(grids):
    
    bounds = [[0, None]] * grids.size
    
    return bounds

    #temp = np.ones(grids.shape)
    #temp[:, :, -5:] = 0
    #bounds = []
    #for val in temp.flat:
        #if val == 0:
            #bounds.append([0, 0])
        #else:
            #bounds.append([0, None])

    #return bounds


def slave(
    atmosphere_params,
    particle_params,
    sun_params,
    camera_params,
    camera_positions,
    ref_images,
    use_simulated=False,
    mask_sun=None,
    save_cams=False
    ):
    
    #import rpdb2; rpdb2.start_embedded_debugger('pep')
    #import wingdbstub
    
    assert len(camera_positions) == len(ref_images), 'Slave_%d: The number of cameras positions, %d, and reference images, %d, should be equal' % (mpi_rank, len(camera_positions), len(ref_images))
    camera_num = len(camera_positions)
    
    #
    # Instantiate the camera slave
    #
    cams_or_paths = []
    for camera_position in camera_positions:
        
        if save_cams:
            cam_path = tempfile.mkdtemp(prefix='/gtmp/')
        else:
            cam_path = None
            
        cam = Camera()
        cam.create(
            sun_params=sun_params,
            atmosphere_params=atmosphere_params,
            camera_params=camera_params,
            camera_position=camera_position,
            save_path=cam_path
        )
    
        if save_cams:
            cams_or_paths.append(cam_path)
        else:
            cams_or_paths.append(cam)
            
    
    #
    # The first data should be for creating the measured images.
    #
    sts = MPI.Status()
    data = comm.recv(source=0, tag=MPI.ANY_TAG, status=sts)

    tag = sts.Get_tag()
    if tag != IMGTAG:
        raise Exception('The first data transaction should be for calculting the meeasure images')

    A_air = data[0]
    A_aerosols = data[1]
    results_path = data[2]
    
    if not save_cams:
        for cam in cams_or_paths:
            cam.setA_air(A_air)
        
    #
    # Calculate and save simulated images
    #
    sim_imgs = []
    for i, cam in enumerate(cams_or_paths):
        if save_cams:
            cam = Camera().load(cam)
            cam.setA_air(A_air)
        
        sim_img = cam.calcImage(
            A_aerosols=A_aerosols,
            particle_params=particle_params
        )
        sim_imgs.append(sim_img)
        
        sio.savemat(
            os.path.join(results_path, 'sim_img' + ('0000%d%d.mat' % (mpi_rank, i))[-9:]),
            {'img': sim_img},
            do_compression=True
        )
    
    #
    # Use simulated images as reference
    #
    if use_simulated:
        ref_images = []
            
        for i, cam in enumerate(cams_or_paths):
            if save_cams:
                cam = Camera().load(cam)                
                cam.setA_air(A_air)
    
            ref_img = cam.calcImage(
                A_aerosols=A_aerosols,
                particle_params=particle_params,
                add_noise=True
            )
            
            ref_images.append(ref_img)

        sun_mask = 1
    else:
        #
        # Send back the simulated images receive the global ratio and std image
        #
        comm.send(sim_imgs, dest=0, tag=RATIOTAG)
        
        ref_ratio, sun_mask_auto, sun_mask_manual = comm.recv(source=0, tag=MPI.ANY_TAG, status=sts)
        assert sts.tag == RATIOTAG, 'Expecting the RATIO tag'
        
        #
        # Create a mask around the sun center.
        #
        if mask_sun == 'auto':
            sun_mask = sun_mask_auto
        elif mask_sun == 'manual':
            sun_mask = sun_mask_manual
        else:
            sun_mask = 1
    
        #
        # Update the reference images according to the ref_ratio
        #
        for i, ref_img in enumerate(ref_images):
            #
            # Note, I change ref_images in place so that ref_img is also effected.
            #
            ref_images[i] /= ref_ratio
                
    #
    # Save the ref images
    #
    for i, ref_img in enumerate(ref_images):
        sio.savemat(
            os.path.join(results_path, 'ref_img' + ('0000%d%d.mat' % (mpi_rank, i))[-9:]),
            {'img': ref_img},
            do_compression=True
        )

    #
    # Set the camera_index to point to the last camera created
    # which is the currently loaded camera.
    #
    camera_index = camera_num-1
    switch_counter = 0
                
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

            obj = 0
            for cam, ref_img in zip(cams_or_paths, ref_images):
                if save_cams:
                    cam = Camera().load(cam)
                    cam.setA_air(A_air)

                img = cam.calcImage(
                    A_aerosols=A_aerosols,
                    particle_params=particle_params
                )
                
                temp = ((ref_img - img) * sun_mask).reshape((-1, 1))
                obj += np.dot(temp.T, temp)
            
            comm.Send(np.array(obj), dest=0)                
                
        elif tag == GRADTAG:
            
            grad = None
            for cam, ref_img in zip(cams_or_paths, ref_images):
                if save_cams:
                    cam = Camera().load(cam)
                    cam.setA_air(A_air)
                    
                img = cam.calcImage(
                    A_aerosols=A_aerosols,
                    particle_params=particle_params
                )
    
                temp = cam.calcImageGradient(
                    img_err=(ref_img-img)*sun_mask**2,
                    A_aerosols=A_aerosols,
                    particle_params=particle_params
                )
                if grad == None:
                    grad = temp
                else:
                    grad += temp
                    
            comm.Send([grad, grad.dtype.char], dest=0)
            
        else:
            raise Exception('Unexpected tag %d' % tag)

    #
    # Save the image the relates to the calculated aerosol distribution
    #
    for i, cam in enumerate(cams_or_paths):
        if save_cams:
            cam_path = cam
            cam = Camera().load(cam_path)
            cam.setA_air(A_air)

            try:
                shutil.rmtree(cam_path)
            except Exception, e:
                print 'Failed to remove folder %s:\n%s\n' % (cam_path, repr(e))

        final_img = cam.calcImage(
            A_aerosols=A_aerosols,
            particle_params=particle_params
        )
        
        sio.savemat(
            os.path.join(results_path, 'final_img' + ('0000%d%d.mat' % (mpi_rank, i))[-9:]),
            {'img': final_img},
            do_compression=True
        )
        

def loadSlaveData(
    atmosphere_params,
    params_path,
    ref_mc_path,
    mcarats,
    sigma,
    remove_sunspot
    ):
    """"""
    
    if mcarats:
        raise NotImplemented('The mcarats code is not yet adapted to the new configuration files')

    if not ref_mc_path:
        ref_mc_path = os.path.join(atmotomo.__src_path__, 'data/monte_carlo_simulations', params_path)

    #
    # Load the reference images
    #
    closed_grids = atmosphere_params.cartesian_grids.closed
    ref_images_list, camera_positions_list = atmotomo.loadVadimData(
        ref_mc_path,
        (closed_grids[0][-1]/2, closed_grids[1][-1]/2),
        remove_sunspot=remove_sunspot
    )
    
    #
    # Smooth the reference images if necessary
    #
    for ref_img in ref_images_list:
        if sigma > 0.0:
            for channel in range(ref_img.shape[2]):
                ref_img[:, :, channel] = \
                    ndimage.filters.gaussian_filter(ref_img[:, :, channel], sigma=sigma)
            
    return ref_images_list, camera_positions_list


def main(
    params_path,
    ref_mc_path=None,
    ref_ratio=0.0,
    save_cams=False,
    mcarats=None,
    use_simulated=False,
    mask_sun=None,
    laplace_weights=(1.0, 1.0, 1.0),
    sigma=0.0,
    init_with_solution=False,
    solver='bfgs',
    tau=0.0,
    regularization_decay=0.0,
    remove_sunspot=False,
    highten_atmosphere=False,
    run_arguments=None
    ):
    
    global mpi_size
    
    #import wingdbstub

    #
    # Load the simulation params
    #
    atmosphere_params, particle_params, sun_params, camera_params, camera_positions_list, air_dist, aerosols_dist = atmotomo.readConfiguration(params_path, highten_atmosphere)
    
    #
    # Limit the number of mpi processes used.
    #
    mpi_size = min(mpi_size, len(camera_positions_list)+1)
    
    if mpi_rank >= mpi_size:
        sys.exit()
        
    if use_simulated:
        ref_images_list = [None] * len(camera_positions_list)
    else:
        ref_images_list, camera_positions_list_temp = loadSlaveData(
            atmosphere_params,
            params_path,
            ref_mc_path,
            mcarats,
            sigma,
            remove_sunspot
        )
        
    if mpi_rank == 0:
        #
        # Create the results path
        #
        results_path = amitibo.createResultFolder(
            params=[atmosphere_params, particle_params, sun_params, camera_params, run_arguments],
            src_path=atmotomo.__src_path__
        )
        
        #
        # Set up the solver server.
        #
        master(
            atmosphere_params=atmosphere_params,
            air_dist=air_dist,
            aerosols_dist=aerosols_dist,
            results_path=results_path,
            ref_imgs=ref_images_list,
            laplace_weights=laplace_weights,
            tau=tau,
            regularization_decay=regularization_decay,
            ref_ratio=ref_ratio,
            solver=solver,
            use_simulated=use_simulated,
            init_with_solution=init_with_solution
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
            use_simulated=use_simulated,
            mask_sun=mask_sun,
            save_cams=save_cams
        )


if __name__ == '__main__':    
    #
    # Parse the input
    #
    parser = argparse.ArgumentParser(description='Analyze atmosphere')
    parser.add_argument('--mcarats', help='path to reference mcarats results folder')
    parser.add_argument('--ref_mc', default=None, help='path to reference images of vadims code')
    parser.add_argument('--ref_ratio', type=float, default=0.0, help='intensity ratio between reference images and the images of the single algorithm.')
    parser.add_argument('--regularization_decay', type=float, default=0.0, help='Ratio of decay of the regularization')
    parser.add_argument('--save_cams', action='store_true', help='Save the cameras to temp file instead of storing them in the memory.')
    parser.add_argument('--sigma', type=float, default=0.0, help='smooth the reference image by sigma')
    parser.add_argument('--use_simulated', action='store_true', help='Use simulated images for reconstruction.')
    parser.add_argument('--remove_sunspot', action='store_true', help='Remove sunspot from reference images.')
    parser.add_argument('--mask_sun', default=None, help='Mask the area of the sun in the reference images [manual-use a predefined mask/auto-calculate mask based on error between montecarlo and single simulations].')
    parser.add_argument('--init_with_solution', action='store_true', help='Initialize the solver with the correct solution.')
    parser.add_argument('--job_id', default=None, help='pbs job ID (set automatically by the PBS script)')
    parser.add_argument('--solver', default='bfgs', help='type of solver to use [bfgs (default), global (DIRECT algorithm), ipopt]')
    parser.add_argument('--tau', type=float, default=0.0, help='regularization coefficient')
    parser.add_argument('params_path', help='Path to simulation parameters')
    parser.add_argument('--weights', type=float, nargs='+', default=(1.0, 1.0, 1.0), help='Weight of laplacian smoothing')
    parser.add_argument('--highten_atmosphere', action='store_true', help='Extend the atmosphere up with empty voxels.')
    args = parser.parse_args()

    main(
        params_path=args.params_path,
        ref_mc_path=args.ref_mc,
        ref_ratio=args.ref_ratio,
        save_cams=args.save_cams,
        mcarats=args.mcarats,
        use_simulated=args.use_simulated,
        mask_sun=args.mask_sun,
        laplace_weights=args.weights,
        sigma=args.sigma,
        init_with_solution=args.init_with_solution,
        solver=args.solver,
        tau=args.tau,
        regularization_decay=args.regularization_decay,
        remove_sunspot=args.remove_sunspot,
        highten_atmosphere=args.highten_atmosphere,
        run_arguments=args
    )
