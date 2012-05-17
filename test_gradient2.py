# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import scipy.sparse as sps
from atmo_utils import L_SUN_RGB, RGB_WAVELENGTH
import simulateAtmoGeneral as sa
import pickle
import ipopt
import logging
import matplotlib.pyplot as plt
import amitibo
import os


CAMERA_CENTERS = [(i, 1) for i in range(5, 50, 5)]

SKY_PARAMS = {
    'width': 50,
    'height': 20,
    'dxh': 1,
    'camera_center': (80, 2),
    'camera_dist_res': 100,
    'camera_angle_res': 100,
    'sun_angle': -45/180*np.pi,
    'L_SUN_RGB': L_SUN_RGB,
    'RGB_WAVELENGTH': RGB_WAVELENGTH
}

VISIBILITY = 10

MAX_ITERATIONS = 200


class radiance(object):
    def __init__(self, aerosol_params, sky_params):
        #
        # Set the sky and aerosols params
        #
        self.aerosol_params = aerosol_params
        self.sky_params = sky_params
        
        self.X, self.H = \
          np.meshgrid(
              np.arange(0, self.sky_params['width'], self.sky_params['dxh']),
              np.arange(0, self.sky_params['height'], self.sky_params['dxh'])[::-1]
              )
        
        #
        # Create the distributions of air & aerosols
        #
        self.ATMO_air = np.exp(-self.H/self.aerosol_params["air_typical_h"])
        self.ATMO_aerosols = np.exp(-self.H/self.aerosol_params["aerosols_typical_h"])
        self.ATMO_aerosols[:, :int(self.H.shape[1]/2)] = 0
        
        #
        # Create the first image
        #
        self.Images = []
        for camera_center in CAMERA_CENTERS:
            self.Images.append(
                sa.calcRadianceHelper(
                    self.ATMO_aerosols.reshape((-1, 1)),
                    self.ATMO_air.reshape((-1, 1)),
                    self.X,
                    self.H,
                    self.aerosol_params,
                    self.sky_params,
                    camera_center
                    )
                )

    def getX0(self):
        #
        # Create the initial aerosols distribution
        #
        ATMO_aerosols = np.ones(self.H.shape)
        return ATMO_aerosols.reshape((-1, 1))
    
    def objective(self, x):
        """Calculate the objective"""
        
        obj = 0
        for camera_index, camera_center in enumerate(CAMERA_CENTERS):
            img = sa.calcRadianceHelper(
                x,
                self.ATMO_air.reshape((-1, 1)),
                self.X,
                self.H,
                self.aerosol_params,
                self.sky_params,
                camera_center
                )

            o = [np.dot(
                (self.Images[camera_index][i] - img[i]).T,
                (self.Images[camera_index][i] - img[i])
                ) for i in range(3)]
            obj += np.sum(o)
            
        return obj
    
    def gradient(self, x):
        """The callback for calculating the gradient"""

        grad = None
        for camera_index, camera_center in enumerate(CAMERA_CENTERS):
            img = sa.calcRadianceHelper(
                x,
                self.ATMO_air.reshape((-1, 1)),
                self.X,
                self.H,
                self.aerosol_params,
                self.sky_params,
                camera_center
                )

            gimg = sa.calcRadianceGradientHelper(
                x,
                self.ATMO_air.reshape((-1, 1)),
                self.X,
                self.H,
                self.aerosol_params,
                self.sky_params,
                camera_center
                )

            temp = [-2*(gimg[i]*(self.Images[camera_index][i] - img[i])) for i in range(3)]
            
            g = np.sum(np.hstack(temp), axis=1)

            if grad == None:
                grad = g
            else:
                grad += g
            
        return grad
    
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
    aerosol_params = {
        "k_RGB": np.array(particle['k']) / np.max(np.array(particle['k'])),#* 10**-12,
        "w_RGB": particle['w'],
        "g_RGB": (particle['g']),
        "visibility": VISIBILITY,
        "air_typical_h": 8,
        "aerosols_typical_h": 8,        
    }

    #
    # Create afolder for results
    #
    results_path = amitibo.createResultFolder(params=[aerosol_params, SKY_PARAMS])

    #
    # Define the problem
    #
    sky = radiance(aerosol_params, SKY_PARAMS)
    
    x0 = sky.getX0()

    lb = np.zeros(x0.shape)
    ub = np.ones(x0.shape)
    
    cl = []
    cu = []

    import logging
    logging.basicConfig(filename='run.log',level=logging.DEBUG)
    ipopt.setLoggingLevel(logging.DEBUG)
    
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

    #
    # Solve the problem
    #
    x, info = nlp.solve(x0)

    print "Solution of the primal variables: x=%s\n" % repr(x)
    print "Solution of the dual variables: lambda=%s\n" % repr(info['mult_g'])
    print "Objective=%s\n" % repr(info['obj_val'])

    print info.keys()
    
    #
    # Show the result
    #
    fig1 = plt.figure()
    plt.subplot(221)
    plt.imshow(sky.ATMO_aerosols, interpolation='nearest', cmap='gray')    
    plt.title('target')
    plt.subplot(223)
    plt.imshow(x0.reshape(sky.ATMO_air.shape), interpolation='nearest', cmap='gray')
    plt.title('initial')
    plt.subplot(224)
    plt.imshow(x.reshape(sky.ATMO_air.shape), interpolation='nearest', cmap='gray')
    plt.title('After %d iterations' % MAX_ITERATIONS)

    amitibo.saveFigures(results_path, (fig1,), bbox_inches='tight')

    #
    # Save the results
    #
    np.save(os.path.join(results_path, 'target.npy'), sky.ATMO_aerosols)
    np.save(os.path.join(results_path, 'x_%d_iterations.npy' % MAX_ITERATIONS), x)
    
    #
    # Show some of the images used
    #
    for img in sky.Images:
        IMG = np.transpose(np.array(img, ndmin=3), (1, 2, 0))
        IMG = np.tile(IMG, (1, IMG.shape[0], 1))

        #
        # Account for gamma correction
        #
        IMG **= 0.45
    
        IMG_scaled = IMG / np.max(IMG)
        h = int(IMG_scaled.shape[0] / 2)

        fig1 = plt.figure()
        plt.subplot(211)
        extent = (0, 1, 90, 0)
        plt.imshow(IMG_scaled[h:0:-1, ...], aspect=1/270, extent=extent, interpolation='nearest')
        plt.xticks([0, 0.5, 1.0])
        plt.yticks([0, 30, 60, 90])
        plt.title('Visibility Parameter %d' % aerosol_params["visibility"])

        plt.subplot(212)
        extent = (0, 1, -90, 0)
        plt.imshow(IMG_scaled[h:, ...], aspect=1/270, extent=extent, interpolation='nearest')
        plt.xticks([0, 0.5, 1.0])
        plt.yticks([0, -30, -60, -90])

        amitibo.saveFigures(results_path, (fig1,), bbox_inches='tight')
        
    plt.show()
    
if __name__ == '__main__':
    main()
