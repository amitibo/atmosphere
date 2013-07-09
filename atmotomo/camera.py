"""
"""

from __future__ import division
import numpy as np
from . import atmo_utils
import amitibo
import warnings
import sparse_transforms as spt
import numpy as np
import time
import os

__all__ = ["Camera"]

RADIUS_RESOLUTION = 100


class Camera(object):
    """A class that encapsulates the functions of a camera"""
    
    def create(
        self,
        sun_params,
        atmosphere_params,
        camera_params,
        camera_position,
        camera_orientation=None,
        radius_resolution=RADIUS_RESOLUTION,
        save_path=None
        ):
        """"""
        
        assert camera_orientation==None, "orientation not implemented yet"
        
        grids = atmosphere_params.cartesian_grids
        self._shape = grids.shape
        
        #
        # Calculate the distance matrices and scattering angle
        #
        timer = amitibo.timer()
        
        print 'Distances from sun'
        H_distances_from_sun = spt.directionTransform(
            in_grids=grids,
            direction_phi=sun_params.angle_phi,
            direction_theta=sun_params.angle_theta
            )
        timer.tock()
        
        timer.tick()        
        print 'Cartesian to camera projection'
        H_cart2polar = spt.sensorTransform(
            grids,
            camera_position,
            camera_params.resolution,
            radius_resolution,
            samples_num=8000,
            replicate=40
        )
        timer.tock()
        
        sensor_grids = H_cart2polar.out_grids
        
        timer.tick()        
        print 'Distances from voxel'
        H_distances_to_sensor = spt.cumsumTransform(
            sensor_grids,
            axis=0,
            direction=0
        )
        timer.tock()
        
        timer.tick()
        print 'sensor'
        H_sensor = spt.integralTransform(
            sensor_grids,
            axis=0
        )
        print 'finished calculation'
        timer.tock()
        
        #
        # Store the transforms
        #
        self.H_cart2polar = H_cart2polar
        self.H_sensor = H_sensor
        self.calcHDistances(H_distances_to_sensor, H_distances_from_sun)
         
        #
        # Calculated the scattering cosinus angle
        #
        self.mu = atmo_utils.calcScatterMu(H_cart2polar.inv_grids, sun_params.angle_phi, sun_params.angle_theta).reshape((-1, 1))
        
        #
        # Store simulation parameters
        #
        self.camera_params = camera_params
        self.sun_params = sun_params
        
        self.A_air_ = np.empty(1)
        self._air_exts = ()

        #
        # Save to disk
        #
        if save_path != None:
            self._save(save_path, H_distances_to_sensor, H_distances_from_sun)
        
    def calcHDistances(self, H_distances_to_sensor, H_distances_from_sun):
        self.H_distances = H_distances_to_sensor * self.H_cart2polar + self.H_cart2polar * H_distances_from_sun
        
    def _save(self, path, H_distances_to_sensor, H_distances_from_sun):
        import pickle
        
        H_distances_to_sensor.save(os.path.join(path, 'H_distances_to_sensor'))
        H_distances_from_sun.save(os.path.join(path, 'H_distances_from_sun'))
        self.H_cart2polar.save(os.path.join(path, 'H_cart2polar'))
        self.H_sensor.save(os.path.join(path, 'H_sensor'))
        np.save(os.path.join(path, 'mu'), self.mu)
        
        with open(os.path.join(path, 'camera.pkl'), 'w') as f:
            pickle.dump(
                (self.camera_params, self.sun_params),
                f
            )
    
    def load(self, path):
        import pickle
        import gc
        
        #
        # Free up memory
        #
        if hasattr(self, "H_distances"):
            del self.H_distances
            
        del self.H_sensor
        del self.H_cart2polar

        gc.collect()
        
        H_distances_to_sensor = spt.loadTransform(os.path.join(path, 'H_distances_to_sensor'))
        H_distances_from_sun = spt.loadTransform(os.path.join(path, 'H_distances_from_sun'))
        self.H_cart2polar = spt.loadTransform(os.path.join(path, 'H_cart2polar'))
        self.H_sensor = spt.loadTransform(os.path.join(path, 'H_sensor'))
        self.calcHDistances(H_distances_to_sensor, H_distances_from_sun)
        
        self.mu = np.load(os.path.join(path, 'mu.npy'))
        
        with open(os.path.join(path, 'camera.pkl'), 'r') as f:
            self.camera_params, self.sun_params = pickle.load(f)
        
    def setA_air(self, A_air):
        """Store the air distribution"""
        
        self.A_air_ = A_air.reshape((-1, 1))
        air_ext_coef = [1.09e-6 * lambda_**-4.05 * self.A_air_ for lambda_ in self.sun_params.wavelengths]
        self.preCalcAir(air_ext_coef)
        
    def set_air_extinction(self, air_exts):
        """Set the air extinction of the three color channels"""
        
        air_ext_coef = [np.tile(air_ext.reshape((1, 1, -1)), (self._shape[0], self._shape[1], 1)).reshape((-1, 1)) for air_ext in air_exts]
        self.preCalcAir(air_ext_coef)
        
    def preCalcAir(self, air_ext_coefs):
        """Precalculate the air extinction and scattering"""
        
        self._air_exts = [np.exp(-self.H_distances * air_ext_coef) for air_ext_coef in air_ext_coefs]
        self._air_scat = [(3 / (16*np.pi) * (1 + self.mu**2) * (self.H_cart2polar * air_ext_coef)) for air_ext_coef in air_ext_coefs]
        
    def calcImage(self, A_aerosols, particle_params, add_noise=False):
        """Calculate the image for a given aerosols distribution"""
        
        A_aerosols_ = A_aerosols.reshape((-1, 1))
        
        #
        # Precalcuate the aerosols
        #
        exp_aerosols_pre = self.H_distances * A_aerosols_

        img = []
        for L_sun, scatter_air, exp_air, k, w, g in zip(
                self.sun_params.intensities,
                self._air_scat,
                self._air_exts,
                particle_params.k,
                particle_params.w,
                particle_params.g
                ):
            #
            # Calculate scattering and extiniction for aerosols
            #
            scatter_aerosols = w * k * (self.H_cart2polar * A_aerosols_)
            if particle_params.phase == 'isotropic':
                scatter_aerosols *= 1/4/np.pi
            elif particle_params.phase == 'HG':
                scatter_aerosols *= atmo_utils.calcHG_other(self.mu, g)
            else:
                raise Exception('Unsupported phase function %s' % particle_params.phase)
                                
            exp_aerosols = np.exp(-k * exp_aerosols_pre)
            
            #
            # Calculate the radiance
            #
            radiance = (scatter_air + scatter_aerosols) * (exp_air * exp_aerosols)
            
            #
            # Calculate projection on camera
            #
            temp_img = L_sun * self.H_sensor * radiance
            
            #
            # Add noise
            #
            if add_noise:
                max_img = temp_img.max()
                photon_num = np.round(temp_img * self.camera_params.photons_per_pixel / max_img)
                noise = np.sqrt(photon_num) / self.camera_params.photons_per_pixel * max_img * np.random.randn(*temp_img.shape)
                temp_img += noise
                temp_img[temp_img<0] = 0
            
            img.append(temp_img.reshape(*self.camera_params.resolution))
            
        img = np.transpose(np.array(img), (1, 2, 0))
        
        return img
    
    def calcImageGradient(self, img_err, A_aerosols, particle_params):
        """Calculate the image gradient for a given aerosols distribution"""
        
        A_aerosols_ = A_aerosols.reshape((-1, 1))
        
        #
        # Precalcuate the aerosols
        #
        exp_aerosols_pre = self.H_distances * A_aerosols_

        gimg = []
        for i, (L_sun, scatter_air, exp_air, k, w, g) in enumerate(zip(
            self.sun_params.intensities,
            self._air_scat,
            self._air_exts,
            particle_params.k,
            particle_params.w,
            particle_params.g
            )):
            #
            # Calculate scattering and extiniction for aerosols
            #
            if particle_params.phase == 'isotropic':
                P_aerosols = 1/4/np.pi
            elif particle_params.phase == 'HG':
                P_aerosols = atmo_utils.calcHG_other(self.mu, g)
            else:
                raise Exception('Unsupported phase function %s' % particle_params.phase)
            
            scatter_aerosols =  w * k * ((self.H_cart2polar * A_aerosols_) * P_aerosols)
            exp_aerosols = np.exp(-k * exp_aerosols_pre)
            
            ##
            ## Calculate the gradient of the radiance
            ##
            #temp1 =  w * k * atmo_utils.spdiag(P_aerosols)
            #temp2 = k * self.H_distances.T * atmo_utils.spdiag(scatter_air + scatter_aerosols)
            #radiance_gradient = (temp1 - temp2) * atmo_utils.spdiag(exp_air * exp_aerosols)
    
            ##
            ## Calculate projection on camera
            ##
            #gimg.append(-2 * L_sun * radiance_gradient * (self.H_sensor.T * img_err[:, :, i].reshape((-1, 1))))

            #
            # An efficient calculation of the above, just without creating unnecessary huge sparse matrices.
            # The reason for this implementation is to avoid memory problems.
            #
            temp = exp_air * exp_aerosols * (self.H_sensor.T * img_err[:, :, i].reshape((-1, 1)))
            part1 = w * k * (self.H_cart2polar.T * (P_aerosols * temp))
            part2 = k * (self.H_distances.T * ((scatter_air + scatter_aerosols) * temp))

            gimg.append(-2 * L_sun * (part1 - part2))
            
        gimg = np.sum(np.hstack(gimg), axis=1)
        return gimg
    

    def draw_image(self, img):
        """
        """
        
        raise Exception('Note addapted to the new camera model')
    
        img = img**0.45
        img /= np.max(img)
    
        c = img.shape[0]/2
        sun_x = c * (1 + focal_ratio * np.tan(sun_angle))
    
        fig = plt.figure()
        ax = plt.axes([0, 0, 1, 1])
    
        #
        # Draw the sky
        #
        plt.imshow(img)
    
        #
        # Draw the sun
        #
        sun_patch = mpatches.Circle((sun_x, c), 3, ec='r', fc='r')
        ax.add_patch(sun_patch)
    
        #
        # Draw angle arcs
        #
        for arc_angle in range(0, 90, 15)[1:]:
            d = c * focal_ratio * np.tan(arc_angle/180*numpy.pi)
            arc_patch = mpatches.Arc(
                (c, c),
                2*d,
                2*d,
                90,
                15,
                345,
                ec='r'
            )
            ax.add_patch(arc_patch)
            plt.text(c, c+d, str(arc_angle), ha="center", va="center", size=10, color='r')
    
        return fig


def test_scatter_angle():
    """Check that the scatter angle calculation works correctly. The rotation should cause
    the scatter angle to align with the nidar."""
    
    import mayavi.mlab as mlab
    
    atmosphere_params = amitibo.attrClass(
        cartesian_grids=(
            slice(0, 300., 10), # Y
            slice(0, 300., 10), # X
            slice(0, 300., 10)  # H
            ),
    )
    
    camera_position = (150.0, 150.0, .2)
    sun_angle = np.pi/6

    Y, X, Z = np.mgrid[atmosphere_params.cartesian_grids]
    angles = calcScatterAngle(Y, X, Z, camera_position, sun_rotation=(sun_angle, 0, 0))

    H_dist = grids.direction2grids(0, -sun_angle, Y, X, Z)
    x = np.ones(Y.shape)
    y = H_dist * x.reshape((-1, 1))
    
    atmo_utils.viz3D(Y, X, Z, angles, 'Y', 'X', 'Z')
    atmo_utils.viz3D(Y, X, Z, y.reshape(Y.shape), 'Y', 'X', 'Z')
    
    mlab.show()
    

def test_scatter_angle2():
    """Check that the scatter angle calculation works correctly. The rotation should cause
    the scatter angle to align with the nidar."""
    
    import mayavi.mlab as mlab
    
    def calcScatterAngleOld(R, PHI, THETA, sun_rotation):
        """
        Calclculate the scattering angle at each voxel.
        """
    
        H_rot = atmo_utils.calcRotationMatrix(sun_rotation)

        X_ = R * np.sin(THETA) * np.cos(PHI)
        Y_ = R * np.sin(THETA) * np.sin(PHI)
        Z_ = R * np.cos(THETA)
    
        XYZ_dst = np.vstack((X_.ravel(), Y_.ravel(), Z_.ravel(), np.ones(R.size)))
        XYZ_src_ = np.dot(H_rot, XYZ_dst)
    
        Z_rotated = XYZ_src_[2, :]
        R_rotated = np.sqrt(np.sum(XYZ_src_[:3, :]**2, axis=0))
    
        angle = np.arccos(Z_rotated/(R_rotated+amitibo.eps(R_rotated)))
    
        return angle
    
    
    atmosphere_params = amitibo.attrClass(
        cartesian_grids=(
            slice(0, 400, 2), # Y
            slice(0, 400, 2), # X
            slice(0, 20, 0.2)  # H
            ),
    )
    
    camera_params = amitibo.attrClass(
        radius_res=100,
        phi_res=100,
        theta_res=100,
        focal_ratio=0.15,
        image_res=128,
        theta_compensation=False
    )

    camera_position = (200, 200, 0.2)
    sun_angle = 0

    Y, X, Z = np.mgrid[atmosphere_params.cartesian_grids]
    
    angles = calcScatterAngle(Y, X, Z, camera_position, sun_rotation=(sun_angle, 0, 0))

    H_pol, R, PHI, THETA = atmo_utils.sphericalTransformMatrix(
        Y,
        X,
        Z,
        center=camera_position,
        radius_res=camera_params.radius_res,
        phi_res=camera_params.phi_res,
        theta_res=camera_params.theta_res
        )
    
    angles_old = calcScatterAngleOld(R, PHI, THETA, sun_rotation=(sun_angle, 0, 0))

    atmo_utils.viz3D(Y, X, Z, angles, 'Y', 'X', 'Z', title='XYX')
    angles_pol = H_pol * angles.reshape((-1, 1))
    angles_pol.shape = R.shape
    angles_old.shape = R.shape
    atmo_utils.viz3D(R, PHI, THETA, angles_pol, 'R', 'PHI', 'THETA', title='POL')
    atmo_utils.viz3D(R, PHI, THETA, angles_old, 'R', 'PHI', 'THETA', title='OLD')
    
    mlab.show()
    

def main():
    """Main doc """
    
    #test_camera()
    test_scatter_angle()
    
    
if __name__ == '__main__':
    main()

    
    