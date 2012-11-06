"""
Reconstruct a general distribution of aerosols in the atmosphere.
"""
from __future__ import division
import numpy as np
from atmo_utils import calcHG, L_SUN_RGB, RGB_WAVELENGTH
import scipy.io as sio
from camera import Camera
import pickle
import ipopt
import logging
import amitibo
import itertools
import os

#
# Set logging level
#
import logging
logging.basicConfig(filename='run.log',level=logging.DEBUG)
ipopt.setLoggingLevel(logging.DEBUG)

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

MAX_ITERATIONS = 10

#
# Global settings
#
atmosphere_params = amitibo.attrClass(
    cartesian_grids=(
        slice(0, 400, 40), # Y
        slice(0, 400, 40), # X
        slice(0, 80, 20)   # H
        ),
    earth_radius=4000,
    L_SUN_RGB=L_SUN_RGB,
    RGB_WAVELENGTH=RGB_WAVELENGTH,
    air_typical_h=8,
    aerosols_typical_h=1.2
)

camera_params = amitibo.attrClass(
    radius_res=20,
    phi_res=40,
    theta_res=40,
    focal_ratio=0.15,
    image_res=128,
    theta_compensation=False
)

#CAMERA_CENTERS = [(i, 200, 0.2) for i in np.linspace(100, 300, mpi_size-1)]
CAMERA_CENTERS = [(i, j, 0.2) for i, j in itertools.product(np.linspace(100, 300, 6), np.linspace(100, 300, 6))]
SUN_ANGLE = np.pi/4

profile = False


class RadianceProblem(ipopt.problem):
    def __init__(self, A_aerosols, A_air, results_path):

        lb = np.zeros(A_aerosols.size)
        ub = np.ones(A_aerosols.size)
            
        super(RadianceProblem, self).__init__(
                    n=A_aerosols.size,
                    m=0,
                    problem_obj=self,
                    lb=lb,
                    ub=ub,
                    cl=[],
                    cu=[]
                    )
        
        #
        # Send the real atmospheric distribution to all childs so as to create the measurement.
        #
        for i in range(1, mpi_size):
            comm.send([A_aerosols, A_air, results_path], dest=i, tag=IMGTAG)

        self.obj_value = []

    def objective(self, x):
        """Calculate the objective"""

        for i in range(1, mpi_size):
            comm.Send(x, dest=i, tag=OBJTAG)

        sts = MPI.Status()
        images = range(mpi_size-1)
        
        obj = 0
        temp = np.empty(1)
        for i in range(1, mpi_size):
            comm.Recv(temp, source=MPI.ANY_SOURCE, status=sts)
            obj += temp[0]
            
        return obj
    
    def gradient(self, x):
        """The callback for calculating the gradient"""

        for i in range(1, mpi_size):
            comm.Send(x, dest=i, tag=GRADTAG)

        sts = MPI.Status()
        images = range(mpi_size-1)

        temp = np.empty((x.size, 1))
        grad = np.zeros_like(temp)
        for i in range(1, mpi_size):
            comm.Recv(temp, source=MPI.ANY_SOURCE, status=sts)
            grad += temp
            
        return grad

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

        self.obj_value.append(obj_value)
        return True
    
    def constraints(self, x):
        #
        # The callback for calculating the constraints
        #
        return None
    
    
    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian
        #
        return None
    

def master(particle_params):
    #import rpdb2; rpdb2.start_embedded_debugger('pep')
    
    results_path = amitibo.createResultFolder(params=[atmosphere_params, particle_params, camera_params])

    #
    # Create the sky
    #
    Y, X, H = np.mgrid[atmosphere_params.cartesian_grids]
    width = atmosphere_params.cartesian_grids[0].stop
    height = atmosphere_params.cartesian_grids[2].stop
    h = np.sqrt((X-width/2)**2 + (Y-width/2)**2 + (atmosphere_params.earth_radius+H)**2) - atmosphere_params.earth_radius

    #
    # Create the distributions of air & aerosols
    #
    A_aerosols = np.exp(-h/atmosphere_params.aerosols_typical_h)
    A_air = np.exp(-h/atmosphere_params.air_typical_h)
    
    #
    # Define the problem
    #
    problem = RadianceProblem(
        A_aerosols=A_aerosols,
        A_air=A_air,
        results_path=results_path
    )

    #
    # Set solver options
    #
    #problem.addOption('derivative_test', 'first-order')
    #problem.addOption('derivative_test_print_all', 'yes')
    problem.addOption('hessian_approximation', 'limited-memory')
    problem.addOption('mu_strategy', 'adaptive')
    problem.addOption('tol', 1e-9)
    problem.addOption('max_iter', MAX_ITERATIONS)

    #
    # Solve the problem
    #
    #x0 = np.zeros_like(A_aerosols).reshape((-1, 1))
    x0 = A_aerosols.ravel()
    x, info = problem.solve(x0)

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
            'objective': np.array(problem.obj_value)
        },
        do_compression=True
    )


def slave(particle_params):
    #import rpdb2; rpdb2.start_embedded_debugger('pep')

    #
    # Instatiate the camera slave
    #
    cam = Camera(
        SUN_ANGLE,
        atmosphere_params=atmosphere_params,
        camera_params=camera_params,
        camera_position=CAMERA_CENTERS[mpi_rank-1]
    )
    
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
    
    cam.setA_air(A_air)
    
    ref_img = cam.calcImage(
        A_aerosols=A_aerosols,
        particle_params=particle_params
    )

    sio.savemat(
        os.path.join(results_path, 'ref_img%d.mat' % mpi_rank),
        {'img': ref_img},
        do_compression=True
    )

    #sio.savemat(
        #os.path.join(results_path, 'ref_aerosols%d.mat' % mpi_rank),
        #{'A_aerosols': A_aerosols, 'img': ref_img},
        #do_compression=True
    #)

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
            
            #sio.savemat(
                #os.path.join(results_path, 'grad_aerosols%d.mat' % mpi_rank),
                #{'A_aerosols': A_aerosols, 'img': img},
                #do_compression=True
            #)

            gimg = cam.calcImageGradient(
                A_aerosols=A_aerosols,
                particle_params=particle_params
            )

            temp = [-2*(gimg[i]*(ref_img[:, :, i] - img[:, :, i]).reshape((-1, 1))) for i in range(3)]
            
            grad = np.sum(np.hstack(temp), axis=1)

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
    # Load the MISR database.
    #
    with open('misr.pkl', 'rb') as f:
        misr = pickle.load(f)

    #
    # Set aerosol parameters
    #
    particles_list = misr.keys()
    particle = misr[particles_list[0]]
    particle_params = amitibo.attrClass(
        k_RGB=np.array(particle['k']) / np.max(np.array(particle['k'])),#* 10**-12,
        w_RGB=particle['w'],
        g_RGB=(particle['g']),
        visibility=10
        )
    
    if mpi_rank == 0:
        #
        # Set up the solver server.
        #
        master(particle_params)
    else:
        slave(particle_params)


if __name__ == '__main__':
    main()
