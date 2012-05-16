# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import scipy.sparse as sps
import simulateAtmoGeneral as sa
import pickle
import ipopt


class radiance(object):
    def __init__(self):
        #
        # Load the MISR database.
        #
        with open('misr.pkl', 'rb') as f:
            misr = pickle.load(f)

        particles_list = misr.keys()
        particle = misr[particles_list[0]]
        self.aerosol_params = {
            "k_RGB": np.array(particle['k']) / np.max(np.array(particle['k'])),#* 10**-12,
            "w_RGB": particle['w'],
            "g_RGB": (particle['g']),
            "visibility": sa.VISIBILITY,
            "air_typical_h": 8,
            "aerosols_typical_h": 8,        
        }

        self.sky_params = sa.SKY_PARAMS

        self.sky_params['dxh'] = 2
        
        self.X, self.H = \
          np.meshgrid(
              np.arange(0, self.sky_params['width'], self.sky_params['dxh']),
              np.arange(0, self.sky_params['height'], self.sky_params['dxh'])[::-1]
              )
        
        #
        # Create the distributions of air and aerosols
        #
        self.ATMO_aerosols = np.exp(-self.H/self.aerosol_params["aerosols_typical_h"])
        self.ATMO_aerosols[:, :int(self.H.shape[1]/2)] = 0
        self.ATMO_air = np.exp(-self.H/self.aerosol_params["air_typical_h"])

        self.I = [np.random.rand(self.sky_params['camera_angle_res'], 1) for i in range(3)]

    def getX0(self):
        return self.ATMO_aerosols.reshape((-1, 1))
    
    def objective(self, x):

        img = sa.calcRadianceHelper(
            x,
            self.ATMO_air.reshape((-1, 1)),
            self.X,
            self.H,
            self.aerosol_params,
            self.sky_params
            )

        o = [np.dot(img[i].T, img[i]) for i in range(3)]
        return np.sum(o)
    
    def gradient(self, x):
        #
        # The callback for calculating the gradient
        #
        img = sa.calcRadianceHelper(
            x,
            self.ATMO_air.reshape((-1, 1)),
            self.X,
            self.H,
            self.aerosol_params,
            self.sky_params
            )
        
        gimg = sa.calcRadianceGradientHelper(
            x,
            self.ATMO_air.reshape((-1, 1)),
            self.X,
            self.H,
            self.aerosol_params,
            self.sky_params
            )

        g = [2*(gimg[i] * img[i]) for i in range(3)]
        grad = np.sum(np.hstack(g), axis=1)
        print grad
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
    # Define the problem
    #
    sky = radiance()
    
    x0 = sky.getX0()

    lb = np.zeros(x0.shape)
    ub = np.ones(x0.shape)
    
    cl = []
    cu = []

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
    nlp.addOption('derivative_test', 'first-order')
    nlp.addOption('mu_strategy', 'adaptive')
    nlp.addOption('tol', 1e-7)

    #
    # Solve the problem
    #
    x, info = nlp.solve(x0)
    
    print "Solution of the primal variables: x=%s\n" % repr(x)
    
    print "Solution of the dual variables: lambda=%s\n" % repr(info['mult_g'])
    
    print "Objective=%s\n" % repr(info['obj_val'])


if __name__ == '__main__':
    main()
