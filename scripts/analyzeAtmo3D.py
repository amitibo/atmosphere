"""
Reconstruct a general distribution of aerosols in the atmosphere.
"""
from __future__ import division
import numpy as np
from atmotomo import calcHG, L_SUN_RGB, RGB_WAVELENGTH, getResourcePath
from atmotomo import Camera
from atmotomo import density_clouds1, calcAirMcarats, getMisrDB, Mcarats
import atmotomo
import scipy.io as sio
import scipy.ndimage as ndimage
import amitibo
import itertools
import os
import sys
import argparse
import glob


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

MAX_ITERATIONS = 4000
KM_TO_METER = 1000

#
# Global settings
#
atmosphere_params = amitibo.attrClass(
    cartesian_grids=(
        slice(0, 50000., 1000.), # Y
        slice(0, 50000., 1000.), # X
        slice(0, 10000., 100.)  # H
        ),
    earth_radius=4000000,
    L_SUN_RGB=L_SUN_RGB,
    RGB_WAVELENGTH=RGB_WAVELENGTH,
    air_typical_h=8000,
    aerosols_typical_h=2000
)

camera_params = amitibo.attrClass(
    image_res=128,
    subgrid_res=(800, 800, 80),
    grid_noise=1.,
    photons_per_pixel=40000
)

##
## node*cores = 2*12 = 25 = 5*5 - 2 (cameras) + 1 (master)
##
#CAMERA_CENTERS = [np.array((i, j, 0.)) + 0.1*np.random.rand(3) for i, j in itertools.product(np.linspace(1.5, 9.5, 5), np.linspace(1.5, 9.5, 5))]
#CAMERA_CENTERS = CAMERA_CENTERS[:-2]

#
# node*cores = 6*12 = 72 = 8*9 - 1 (cameras) + 1 (master)
#
#CAMERA_CENTERS = [np.array((i, j, 0.)) + 0.1*np.random.rand(3) for i, j in itertools.product(np.linspace(10, 190, 8), np.linspace(10, 190, 9))]
#CAMERA_CENTERS = CAMERA_CENTERS[:-1]

#
# node*cores = 8*12 = 96 = 10*10 - 5 (cameras) + 1 (master)
#
CAMERA_CENTERS = [np.array((i, j, 0.)) + 0.1*np.random.rand(3) for i, j in itertools.product(np.linspace(5., 45, 10), np.linspace(5., 45, 10))]
CAMERA_CENTERS = CAMERA_CENTERS[:-5]

SUN_ANGLE = -np.pi/4
REF_IMG_SCALE = 10.0**4
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


class RadianceProblem(object):
    def __init__(self, A_aerosols, air_exts, results_path):

        #
        # Send the real atmospheric distribution to all childs so as to create the measurement.
        #
        for i in range(1, mpi_size):
            comm.send([air_exts, A_aerosols, results_path], dest=i, tag=IMGTAG)

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
        

def master(particle_params, solver='ipopt'):
    #import rpdb2; rpdb2.start_embedded_debugger('pep')
    
    results_path = amitibo.createResultFolder(params=[atmosphere_params, particle_params, camera_params], src_path=atmotomo.__src_path__)
    logging.basicConfig(filename=os.path.join(results_path, 'run.log'), level=logging.DEBUG)

    #
    # Create the distributions
    #
    A_air, A_aerosols, Y, X, H, h = density_clouds1(atmosphere_params)
    
    z_coords = H[0, 0, :]
    air_exts = calcAirMcarats(z_coords)
    
    #
    # Initial distribution for optimization
    #
    x0 = np.zeros_like(A_aerosols)

    #
    # Create the optimization problem object
    #
    radiance_problem = RadianceProblem(
        A_aerosols=A_aerosols,
        air_exts=air_exts,
        results_path=results_path
    )

    if solver == 'ipopt':
        import ipopt
        
        #
        # Define the problem
        #
        lb = np.zeros(A_aerosols.size)
        
        ipopt.setLoggingLevel(logging.DEBUG)

        problem = ipopt.problem(
            n=A_aerosols.size,
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
            'true': A_aerosols,
            'estimated': x.reshape(A_aerosols.shape),
            'objective': np.array(radiance_problem.obj_values)
        },
        do_compression=True
    )


def slave(particle_params, camera_position, ref_img):
    #import rpdb2; rpdb2.start_embedded_debugger('pep')

    #
    # Instatiate the camera slave
    #
    cam = Camera()
    cam.create(
        SUN_ANGLE,
        atmosphere_params=atmosphere_params,
        camera_params=camera_params,
        camera_position=camera_position
    )
    
    sts = MPI.Status()

    #
    # The first data should be for creating the measured images.
    #
    data = comm.recv(source=0, tag=MPI.ANY_TAG, status=sts)

    tag = sts.Get_tag()
    if tag != IMGTAG:
        raise Exception('The first data transaction should be for calculting the meeasure images')

    air_exts = data[0]
    A_aerosols = data[1]
    results_path = data[2]
    
    cam.set_air_extinction(air_exts)

    if ref_img is None:
        ref_img = cam.calcImage(
            A_aerosols=A_aerosols,
            particle_params=particle_params,
            add_noise=True
        )

    sio.savemat(
        os.path.join(results_path, 'ref_img%d.mat' % mpi_rank),
        {'img': ref_img},
        do_compression=True
    )

    #
    # Loop the messages of the master
    #
    while 1:
        comm.Recv(A_aerosols, source=0, tag=MPI.ANY_TAG, status=sts)

        tag = sts.Get_tag()
        if tag == DIETAG:
            break

        if tag == OBJTAG:
            img = cam.calcImage(
                A_aerosols=A_aerosols,
                particle_params=particle_params
            )
            
            temp = ref_img.reshape((-1, 1)) - img.reshape((-1, 1))
            obj = np.dot(temp.T, temp)
            
            comm.Send(np.array(obj), dest=0)
            
        elif tag == GRADTAG:
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
    final_img = cam.calcImage(
        A_aerosols=A_aerosols,
        particle_params=particle_params
    )

    sio.savemat(
        os.path.join(results_path, 'final_img%d.mat' % mpi_rank),
        {'img': final_img},
        do_compression=True
    )


def main():
    #
    # Parse the input
    #
    parser = argparse.ArgumentParser(description='Analyze atmosphere')
    parser.add_argument('--mcarats', help='path to reference mcarats results folder')
    parser.add_argument('--ref_images', help='path to reference images')
    parser.add_argument('--sigma', type=float, default=0.0, help='smooth the reference image by sigma')
    args = parser.parse_args()
    
    #
    # Load the reference images
    #
    folder_list = []
    if args.ref_images:
        folder_list = glob.glob(os.path.join(args.ref_images, "*"))
         
    #
    # Load the MISR database.
    #
    particle = getMisrDB()['spherical_nonabsorbing_2.80']

    #
    # Set aerosol parameters
    #
    particle_params = amitibo.attrClass(
        k_RGB=np.array(particle['k']) / np.max(np.array(particle['k'])),
        w_RGB=particle['w'],
        g_RGB=(particle['g'])
        )
    
    if mpi_rank == 0:
        #
        # Set up the solver server.
        #
        master(particle_params, solver='bfgs')
    else:
        if args.mcarats:
            path = os.path.abspath(args.mcarats)
    
            R_ch, G_ch, B_ch = [np.fromfile(os.path.join(path, 'base%d_conf_out' % i), dtype=np.float32) for i in range(3)]
            IMG_SHAPE = (128, 128)
            IMG_SIZE = IMG_SHAPE[0] * IMG_SHAPE[1]
            
            slc = slice((mpi_rank-1)*IMG_SIZE, mpi_rank*IMG_SIZE)
            ref_img = Mcarats.calcMcaratsImg(R_ch, G_ch, B_ch, slc, IMG_SHAPE) / MCARATS_IMG_SCALE
            ref_img = ref_img.astype(np.float)
            
            with open(getResourcePath('CamerasPositions.txt'), 'r') as f:
                lines = f.readlines()
                camera_position = np.array([float(i) for i in lines[mpi_rank-1].strip().split()])*KM_TO_METER

        elif folder_list:
            path = folder_list[mpi_rank-1]
            img_path = os.path.join(path, "RGB_MATRIX.mat")
            data = sio.loadmat(img_path)
            
            ref_img = data['Detector'] / REF_IMG_SCALE
            
            if args.sigma > 0.0:
                for channel in range(ref_img.shape[2]):
                    ref_img[:, :, channel] = \
                        ndimage.filters.gaussian_filter(ref_img[:, :, channel], sigma=args.sigma)
                
            #
            # Parse cameras center file
            #
            with open(os.path.join(path, 'params.txt'), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if parts[0] == 'CameraPosition':
                        camera_position = np.array((float(parts[4])+25000, float(parts[2])+25000, float(parts[3])))/ 1000
                        break
        else:
            camera_position = CAMERA_CENTERS[mpi_rank-1]
            ref_img = None
            
        slave(particle_params, camera_position, ref_img)


if __name__ == '__main__':
    main()
