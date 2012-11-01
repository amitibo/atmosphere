"""
Reconstruct a general distribution of aerosols in the atmosphere.
"""
from __future__ import division
import numpy as np
from atmo_utils import calcHG, L_SUN_RGB, RGB_WAVELENGTH
from camera import Camera
import pickle
import ipopt
import logging
import amitibo
import os

from mpi4py import MPI

comm = MPI.COMM_WORLD

mpi_size = MPI.COMM_WORLD.Get_size()
mpi_rank = MPI.COMM_WORLD.Get_rank()

OBJTAG = 1
GRADTAG = 2
DIETAG = 3

MAX_ITERATIONS = 100

#
# Global settings
#
atmosphere_params = amitibo.attrClass(
    cartesian_grids=(
        slice(0, 400, 4), # Y
        slice(0, 400, 4), # X
        slice(0, 80, 1)   # H
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

CAMERA_CENTERS = [(i, 200, 0.2) for i in np.linspace(100, 300, mpi_size-1)]
SUN_ANGLE = np.pi/4

profile = False


class radiance(object):
    def __init__(self):

        #
        # Create the sky
        #
        self.Y, self.X, self.H = np.mgrid[atmosphere_params.cartesian_grids]
        width = atmosphere_params.cartesian_grids[0].stop
        height = atmosphere_params.cartesian_grids[2].stop
        h = np.sqrt((self.X-width/2)**2 + (self.Y-width/2)**2 + (atmosphere_params.earth_radius+self.H)**2) - atmosphere_params.earth_radius
    
        #
        # Create the distributions of air & aerosols
        #
        self.ATMO_aerosols = np.exp(-h/atmosphere_params.aerosols_typical_h)
        self.ATMO_air = np.exp(-h/atmosphere_params.air_typical_h)

        #
        # Create the first images
        #
        for i in range(1, mpi_size):
            comm.send([np.ones((5, 1)), self.ATMO_air], dest=i, tag=OBJTAG)

        sts = MPI.Status()
        self.Images = range(mpi_size-1)
        
        for i in range(1, mpi_size):
            img = comm.recv(source=MPI.ANY_SOURCE, status=sts)
            src = sts.Get_source()
            
            self.Images[src-1] = img

        self.obj_value = []

    def getX0(self):
        
        print 'qurying x0'
        
        return np.ones((5, 1))
    
        #
        # Create the initial aerosols distribution
        #
        ATMO_aerosols = np.ones(self.H.shape)
        return ATMO_aerosols.reshape((-1, 1))
    
    def objective(self, x):
        """Calculate the objective"""
        
        print 'objective calculation.'
        
        for i in range(1, mpi_size):
            comm.send([x, self.ATMO_air], dest=i, tag=OBJTAG)

        print 'sent messages'
        
        sts = MPI.Status()
        images = range(mpi_size-1)
        
        for i in range(1, mpi_size):
            img = comm.recv(source=MPI.ANY_SOURCE, status=sts)
            src = sts.Get_source()
            
            images[src-1] = img

        print 'recived messages'
        
        obj = 0
        for ref_img, img in zip(self.Images, images):
            obj += img
            #o = [np.dot(
                #(ref_img[i] - img[i]).T,
                #(ref_img[i] - img[i])
                #) for i in range(3)]
            #obj += np.sum(o)
            
        print 'calculating objective'
        
        return obj
    
    def gradient(self, x):
        """The callback for calculating the gradient"""

        print 'Gradient calculation.'
        
        for i in range(1, mpi_size):
            comm.send([x, self.ATMO_air], dest=i, tag=OBJTAG)

        print 'sent messages.'
            
        sts = MPI.Status()
        images = range(mpi_size-1)
        
        for i in range(1, mpi_size):
            img = comm.recv(source=MPI.ANY_SOURCE, status=sts)
            src = sts.Get_source()
            
            images[src-1] = img
            
        for i in range(1, mpi_size):
            comm.send([x, self.ATMO_air], dest=i, tag=GRADTAG)

        grads = range(mpi_size-1)
        
        for i in range(1, mpi_size):
            gimg = comm.recv(source=MPI.ANY_SOURCE, status=sts)
            src = sts.Get_source()
            
            grads[src-1] = gimg

        grad = None
        for ref_img, img, gimg in zip(self.Images, images, grads):

            #temp = [-2*(gimg[i]*(ref_img[i] - img[i]).reshape((-1, 1))) for i in range(3)]
            
            #g = np.sum(np.hstack(temp), axis=1)

            if grad == None:
                grad = gimg
            else:
                grad += gimg
            
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
    

def master():
    #
    # Define the problem
    #
    
    print 'master called'
    sky = radiance()

    print 'created sky'
    x0 = sky.getX0()

    lb = np.zeros(x0.shape)
    ub = np.ones(x0.shape)
    
    cl = []
    cu = []

    import logging
    logging.basicConfig(filename='run.log',level=logging.DEBUG)
    ipopt.setLoggingLevel(logging.DEBUG)
    
    print 'defining problem'
    
    nlp = ipopt.problem(
                n=len(x0),
                m=len(cl),
                problem_obj=sky,
                lb=lb,
                ub=ub,
                cl=cl,
                cu=cu
                )

    #
    # Set solver options
    #
    #nlp.addOption('derivative_test', 'first-order')
    nlp.addOption('hessian_approximation', 'limited-memory')
    nlp.addOption('mu_strategy', 'adaptive')
    nlp.addOption('tol', 1e-7)
    nlp.addOption('max_iter', MAX_ITERATIONS)

    print 'solving problem'
    
    #
    # Solve the problem
    #
    x, info = nlp.solve(x0)

    print x
    
    #
    # Kill all slaves
    #
    for i in range(1, mpi_size):
        comm.send(0, dest=i, tag=DIETAG)
    
    
def slave(particle_params):
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

    while 1:
        data = comm.recv(source=0, tag=MPI.ANY_TAG, status=sts)

        tag = sts.Get_tag()

        print 'received tag: %d' % tag

        if tag == DIETAG:
            break

        print len(data)
        
        A_air = data[0]
        A_aerosols = data[1]
        
        if tag == OBJTAG:
            img = cam.calcImage(
                A_air=A_air,
                A_aerosols=A_aerosols,
                particle_params=particle_params
            )
            
            comm.send(img, dest=0)
        elif tag == GRADTAG:
            gimg = cam.calcImageGradient(
                A_air=A_air,
                A_aerosols=A_aerosols,
                particle_params=particle_params
            )

            comm.send(gimg, dest=0)
        else:
            raise Exception('Unkown tag')


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
    
    print 'rank: %d' % mpi_rank
    
    if mpi_rank == 0:
        #
        # Set up the solver server.
        #
        print 'calling master'
        master()
    else:
        slave(particle_params)


if __name__ == '__main__':
    main()
