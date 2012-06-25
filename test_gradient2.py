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
import cv
import cv2


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

VISIBILITY = 100
ADDED_NOISE = 0.0
MAX_ITERATIONS = 100


class radiance(object):
    def __init__(self, aerosol_params, sky_params, results_path, added_noise=0):
        #
        # Set the sky and aerosols params
        #
        self.aerosol_params = aerosol_params
        self.sky_params = sky_params
        
        self.X, self.H = \
          np.meshgrid(
              np.arange(0, self.sky_params['width'], self.sky_params['dxh']),
              np.arange(0, self.sky_params['height'], self.sky_params['dxh'])
              )
        
        #
        # Create the distributions of air & aerosols
        #
        self.ATMO_air = np.exp(-self.H/self.aerosol_params["air_typical_h"])
        self.ATMO_aerosols = np.exp(-self.H/self.aerosol_params["aerosols_typical_h"])

        aerosols_mask = np.zeros(self.ATMO_aerosols.shape)
        Z1 = (self.X - self.sky_params['width']/3)**2/4 + (self.H - self.sky_params['height']/2)**2/2
        Z2 = (self.X - self.sky_params['width']*2/3)**2/4 + (self.H - self.sky_params['height']/2)**2/2
        aerosols_mask[Z1<25] = 1
        aerosols_mask[Z2<25] = 1
        
        self.ATMO_aerosols = self.ATMO_aerosols * aerosols_mask
        
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
                    camera_center,
                    added_noise=added_noise
                    )
                )

        #
        # Create a video writer for the temporary results
        #
        self.video_writer = cv2.VideoWriter(
            os.path.join(results_path, 'optimization.avi'),
            cv.CV_FOURCC(*"ffds"),
            12,
            (160, 120)
            )

        self.obj_value = []

    def getX0(self):
        #
        # Create the initial aerosols distribution
        #
        ATMO_aerosols = np.ones(self.H.shape)
        return ATMO_aerosols.reshape((-1, 1))
    
    def objective(self, x):
        """Calculate the objective"""

        self.temp_x = x
        
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

        try:
            x = cv2.resize(
                255*self.temp_x.reshape(self.ATMO_aerosols.shape),
                dsize=(160, 120), interpolation=cv.CV_INTER_NN
                )
            self.video_writer.write(x.astype('uint8'))
        except Exception as inst:
            print 'error has occured:\n%s' % repr(inst)

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
    sky = radiance(aerosol_params, SKY_PARAMS, results_path, added_noise=ADDED_NOISE)

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

    #print "Solution of the primal variables: x=%s\n" % repr(x)
    #print "Solution of the dual variables: lambda=%s\n" % repr(info['mult_g'])
    print "Objective=%s\n" % repr(info['obj_val'])

    #
    # Show the optimization result
    #
    pdf = amitibo.figuresPdf(os.path.join(results_path, 'report.pdf'))
    figures = []
    figures.append(plt.figure())
    plt.subplot(211)
    plt.imshow(sky.ATMO_aerosols, interpolation='nearest', cmap='gray')    
    plt.title('target')
    plt.subplot(212)
    plt.imshow(x.reshape(sky.ATMO_air.shape), interpolation='nearest', cmap='gray')
    plt.hold = True
    camera_x, camera_y = zip(*CAMERA_CENTERS)
    plt.plot(camera_x, sky.ATMO_air.shape[0]-1-np.array(camera_y), 'or')
    plt.xlim((0, sky.ATMO_air.shape[1]-1))
    plt.ylim((sky.ATMO_air.shape[0]-1, 0))
    plt.title('Reconstruction Using %d Cameras\nAfter %d iterations' % (len(CAMERA_CENTERS), MAX_ITERATIONS))
    plt.suptitle('Reconstruction Using %d Cameras' % len(CAMERA_CENTERS))

    #
    # Plot the objective
    #
    figures.append(plt.figure())
    plt.plot(np.log10(np.array(sky.obj_value)))
    plt.title('Log of Objective Value per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel(r'$\log_{10}(Objective)$')

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

        figures.append(plt.figure())
        plt.subplot(211)
        extent = (0, 1, 90, 0)
        plt.imshow(IMG_scaled[h:0:-1, ...], aspect=1/270, extent=extent, interpolation='nearest')
        plt.xticks([0, 0.5, 1.0])
        plt.yticks([0, 30, 60, 90])
        plt.title('Visibility Parameter %d, Added Noise (of std) %g' % (aerosol_params["visibility"], ADDED_NOISE))

        plt.subplot(212)
        extent = (0, 1, -90, 0)
        plt.imshow(IMG_scaled[h:, ...], aspect=1/270, extent=extent, interpolation='nearest')
        plt.xticks([0, 0.5, 1.0])
        plt.yticks([0, -30, -60, -90])

    pdf.saveFigures(figures)
        
    plt.show()
    
if __name__ == '__main__':
    main()
