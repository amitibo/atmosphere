import unittest

from atmotomo import calcHG, L_SUN_RGB, RGB_WAVELENGTH, getResourcePath, \
     Camera, getMisrDB, single_voxel_atmosphere
import sparse_transforms as spt
import numpy as np
import amitibo
import numdifftools as nd


class Test(unittest.TestCase):
    
    def setUp(self):
        
        self.atmosphere_params = amitibo.attrClass(
            cartesian_grids=spt.Grids(
                np.arange(0, 400, 80.0), # Y
                np.arange(0, 400, 80.0), # X
                np.arange(0, 80, 40.0)   # H
                ),
            L_SUN_RGB=L_SUN_RGB,
            RGB_WAVELENGTH=RGB_WAVELENGTH,
        )
        
        self.camera_params = amitibo.attrClass(
            image_res=16,
            radius_res=16,
            photons_per_pixel=40000
        )

        #
        # Load the MISR database.
        #
        particle = getMisrDB()['spherical_nonabsorbing_2.80']
    
        #
        # Set aerosol parameters
        #
        self.particle_params = amitibo.attrClass(
            k_RGB=np.array(particle['k']) / np.max(np.array(particle['k'])),
            w_RGB=particle['w'],
            g_RGB=(particle['g'])
            )
        
        
    def test01(self):
        """Test the gradient calculation"""
        
        #
        # Create the camera
        #
        CAMERA_CENTERS = (200, 200, 0.2)
        SUN_ANGLE = np.pi/4

        cam = Camera()
        cam.create(
             SUN_ANGLE,
             atmosphere_params=self.atmosphere_params,
             camera_params=self.camera_params,
             camera_position=CAMERA_CENTERS
             )
        
        #
        # Create random atmospheres
        #
        A_air = np.random.rand(*self.atmosphere_params.cartesian_grids.shape)
        A_aerosols = np.random.rand(*self.atmosphere_params.cartesian_grids.shape)
        A_aerosols_ref = np.random.rand(*self.atmosphere_params.cartesian_grids.shape)
         
        cam.setA_air(A_air)
        
        SCALING = 10000
        ref_img = cam.calcImage(
            A_aerosols=A_aerosols_ref,
            particle_params=self.particle_params
        ) * SCALING
        estim_img = cam.calcImage(
            A_aerosols=A_aerosols,
            particle_params=self.particle_params
        )
        
        #
        # Objective function
        #
        def fun(x):
            img = cam.calcImage(
                A_aerosols=x,
                particle_params=self.particle_params
            )
    
            return np.sum((ref_img - SCALING*img)**2)
        
        dfun = nd.Gradient(fun)
        num_grad = dfun(A_aerosols.reshape((-1, 1)))
        
        ana_grad = cam.calcImageGradient(
            img_err=ref_img - SCALING*estim_img,
            A_aerosols=A_aerosols,
            particle_params=self.particle_params
            )*SCALING
        
        err_values = np.max(num_grad - ana_grad) > dfun.error_estimate
        
        self.assertTrue(
            np.all(np.max(num_grad - ana_grad) <= dfun.error_estimate),
            msg='Total errors: %d\n:Numerical: %s\nEstimated: %s\nError Estimate: %s\n' % (err_values.sum(), repr(num_grad[err_values]), repr(ana_grad[err_values]), repr(dfun.error_estimate[err_values]))
        )
        

if __name__ == '__main__':
    unittest.main()
