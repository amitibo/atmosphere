"""
"""

from __future__ import division
import numpy as np
from . import atmo_utils
import amitibo
import warnings
import grids
import time
import os

__all__ = ["Camera"]


def calcScatterAngle(Y, X, Z, camera_position, sun_rotation):
    """
    Calclculate the scattering angle at each voxel.
    """

    H_rot = atmo_utils.calcRotationMatrix(sun_rotation)
    sun_vector = np.dot(H_rot, np.array([[0.], [0.], [1.], [1.]]))
    
    Y_ = Y-camera_position[0]
    X_ = X-camera_position[1]
    Z_ = Z-camera_position[2]
    R = np.sqrt(Y_**2 + X_**2 + Z_**2)

    mu = (Y_ * sun_vector[0] + X_ * sun_vector[1] + Z_ * sun_vector[2])/ (R + amitibo.eps(R))
    
    return mu


class Camera(object):
    """A class that encapsulates the functions of a camera"""
    
    def create(
        self,
        sun_angle,
        atmosphere_params,
        camera_params,
        camera_position,
        camera_orientation=None
        ):
        """"""
        
        assert camera_orientation==None, "orientation not implemented yet"
        
        Y, X, H = np.mgrid[atmosphere_params.cartesian_grids]
        self._shape = H.shape
        
        #
        # Calculate the distance matrices and scattering angle
        #
        timer = amitibo.timer()
        
        print 'Distances1'
        H_distances1 = grids.point2grids(camera_position, Y, X, H)
        timer.tock()
        timer.tick()
        print 'Distances2'
        H_distances2 = grids.direction2grids(0, -sun_angle, Y, X, H)
        timer.tock()
        timer.tick()        
        print 'sensor'
        H_sensor = grids.integrateGrids(
            camera_position, Y, X, H, camera_params.image_res, subgrid_max=camera_params.subgrid_res, subgrid_noise=camera_params.grid_noise
        )
        print 'finished calculation'
        timer.tock()
        
        #
        # Calculate the mu
        #
        warnings.warn('Currently we are using a hack to align the scattering angle with the rotation of the atmosphere')
        mu = calcScatterAngle(Y, X, H, camera_position, sun_rotation=(sun_angle, 0, 0))
        
        #
        # Store the matrices
        #
        self.H_distances = H_distances1 + H_distances2
        self.H_sensor = H_sensor
        self.mu = mu.reshape((-1, 1))
        self.camera_params = camera_params
        self.atmosphere_params = atmosphere_params
        self.A_air_ = np.empty(1)
        self._air_exts = ()
        
    def save(self, path):
        import scipy.io as sio
        import pickle
        sio.savemat(
            os.path.join(path, 'camera.mat'),
            {
                'H_distances': self.H_distances,
                'H_sensor': self.H_sensor,
                'mu': self.mu,
                'A_air_': self.A_air_
            },
            do_compression=True
        )
        
        with open(os.path.join(path, 'camera.pkl'), 'w') as f:
            pickle.dump(
                (self.camera_params, self.atmosphere_params),
                f
            )
    
    def load(self, path):
        import scipy.io as sio
        import pickle
        
        cam = sio.loadmat(os.path.join(path, 'camera.mat'))
        self.H_distances = cam['H_distances']
        self.H_sensor = cam['H_sensor']
        self.mu = cam['mu']
        self.A_air_ = cam['A_air_']
        
        with open(os.path.join(path, 'camera.pkl'), 'r') as f:
            self.camera_params, self.atmosphere_params = pickle.load(f)
        
    def setA_air(self, A_air):
        """Store the air distribution"""
        
        self.A_air_ = A_air.reshape((-1, 1))
        air_ext_coef = [1.09e-3 * lambda_**-4.05 * self.A_air_ for lambda_ in self.atmosphere_params.RGB_WAVELENGTH]
        self.preCalcAir(air_ext_coef)
        
    def set_air_extinction(self, air_exts):
        """Set the air extinction of the three color channels"""
        
        air_ext_coef = [np.tile(air_ext.reshape((1, 1, -1)), (self._shape[0], self._shape[1], 1)).reshape((-1, 1)) for air_ext in air_exts]
        self.preCalcAir(air_ext_coef)
        
    def preCalcAir(self, air_ext_coefs):
        """Precalculate the air extinction and scattering"""
        
        self._air_exts = [np.exp(-self.H_distances * air_ext_coef) for air_ext_coef in air_ext_coefs]
        self._air_scat = [(3 / (16*np.pi) * ((1 + self.mu**2) * air_ext_coef)) for air_ext_coef in air_ext_coefs]
        
    def calcImage(self, A_aerosols, particle_params, add_noise=False):
        """Calculate the image for a given aerosols distribution"""
        
        A_aerosols_ = A_aerosols.reshape((-1, 1))
        
        #
        # Precalcuate the aerosols
        #
        exp_aerosols_pre = self.H_distances * A_aerosols_

        img = []
        for L_sun, scatter_air, exp_air, k, w, g in zip(
                self.atmosphere_params.L_SUN_RGB,
                self._air_scat,
                self._air_exts,
                particle_params.k_RGB,
                particle_params.w_RGB,
                particle_params.g_RGB
                ):
            #
            # Calculate scattering and extiniction for aerosols
            #
            scatter_aerosols = w * k * (A_aerosols_ * atmo_utils.calcHG(self.mu, g))
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
            
            img.append(temp_img.reshape(self.camera_params.image_res, self.camera_params.image_res))
            
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
            self.atmosphere_params.L_SUN_RGB,
            self._air_scat,
            self._air_exts,
            particle_params.k_RGB,
            particle_params.w_RGB,
            particle_params.g_RGB
            )):
            #
            # Calculate scattering and extiniction for aerosols
            #
            P_aerosols = atmo_utils.calcHG(self.mu, g)
            scatter_aerosols =  w * k * (A_aerosols_ * P_aerosols)
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
            part1 = w * k * (P_aerosols * (exp_air * exp_aerosols * (self.H_sensor.T * img_err[:, :, i].reshape((-1, 1)))))
            part2 = k * self.H_distances.T * ((scatter_air + scatter_aerosols) * (exp_air * exp_aerosols * (self.H_sensor.T * img_err[:, :, i].reshape((-1, 1)))))

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


def test_camera():

    import pickle
    from atmo_utils import calcHG, L_SUN_RGB, RGB_WAVELENGTH

    atmosphere_params = amitibo.attrClass(
        cartesian_grids=(
            slice(0, 400, 80), # Y
            slice(0, 400, 80), # X
            slice(0, 80, 40)   # H
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
        image_res=16,
        theta_compensation=False
    )
    
    CAMERA_CENTERS = (200, 200, 0.2)
    SUN_ANGLE = np.pi/4

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
    
    #
    # Create the atmosphere
    #
    Y, X, H = np.mgrid[atmosphere_params.cartesian_grids]
    width = atmosphere_params.cartesian_grids[0].stop
    height = atmosphere_params.cartesian_grids[2].stop
    h = np.sqrt((X-width/2)**2 + (Y-width/2)**2 + (atmosphere_params.earth_radius+H)**2) - atmosphere_params.earth_radius
    A_aerosols = np.exp(-h/atmosphere_params.aerosols_typical_h)
    A_air = np.exp(-h/atmosphere_params.air_typical_h)

    #
    # Create the camera
    #
    cam = Camera(
        SUN_ANGLE,
        atmosphere_params=atmosphere_params,
        camera_params=camera_params,
        camera_position=CAMERA_CENTERS
        )
    
    cam.setA_air(A_air)
    
    ref_img = cam.calcImage(
        A_aerosols=A_aerosols,
        particle_params=particle_params
    )

    #
    # Change a bit the aerosols distribution
    #
    A_aerosols = 2 * A_aerosols
    A_aerosols[A_aerosols>1] = 1.0
    
    #
    # Calculate the gradient
    #
    img = cam.calcImage(
        A_aerosols=A_aerosols,
        particle_params=particle_params
    )
    
    gimg = cam.calcImageGradient(
        A_aerosols=A_aerosols,
        particle_params=particle_params
    )

    temp = [-2*(gimg[i]*(ref_img[:, :, i] - img[:, :, i]).reshape((-1, 1))) for i in range(3)]
    
    grad = np.sum(np.hstack(temp), axis=1)
    
    
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

    
    