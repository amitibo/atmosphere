"""
"""

from __future__ import division
import numpy as np
import atmo_utils
import amitibo
import warnings
import grids
import os


def calcOpticalDistancesMatrix(Y, X, Z, sun_angle, H_pol, R, PHI, THETA):
    """
    Calculate the optical distances from the sun through any voxel in the atmosphere and
    from there to the camera.
    """
    
    #
    # Prepare transformation matrices
    #
    Hrot_forward, rotation, Y_rot, X_rot, Z_rot = atmo_utils.rotation3DTransformMatrix(Y, X, Z, rotation=(0, sun_angle, 0))
    Hrot_backward = atmo_utils.rotation3DTransformMatrix(Y_rot, X_rot, Z_rot, np.linalg.inv(rotation), Y, X, Z)[0]
    
    #
    # Calculate a mask for Hint2
    #
    mask = np.ones_like(X)
    mask_pol = H_pol * mask.reshape((-1, 1))

    Hint1 = atmo_utils.cumsumTransformMatrix((Y_rot, X_rot, Z_rot), axis=2, direction=-1)
    Hint2 = atmo_utils.cumsumTransformMatrix((R, PHI, THETA), axis=0, masked_rows=mask_pol)

    temp1 = H_pol * Hrot_backward * Hint1 * Hrot_forward
    temp2 = Hint2 * H_pol

    #vizTransforms(Y, X, Z, H_pol, Hrot_forward, Hrot_backward, Hint1, Hint2, R, Y_rot, scatter_angle)
    
    return temp2 + temp1


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

        #
        # Calculate the distance matrices and scattering angle
        #
        print 'Distances1'
        H_distances1 = grids.point2grids(camera_position, Y, X, H)
        print 'Distances2'
        H_distances2 = grids.direction2grids(0, -sun_angle, Y, X, H)
        
        print 'sensor'
        H_sensor = grids.integrateGrids(
            camera_position, Y, X, H, camera_params.image_res, camera_params.subgrid_res, noise=0.05
        )
        print 'finished calculation'
        
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
        self.a_air_ = np.empty(1)
        
    def save(self, path):
        import scipy.io as sio
        import pickle
        sio.savemat(
            os.path.join(path, 'camera.mat'),
            {
                'H_distances': self.H_distances,
                'H_sensor': self.H_sensor,
                'mu': self.mu,
                'A_air_': self.a_air_
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
        
    def calcImage(self, A_aerosols, particle_params):
        """Calculate the image for a given aerosols distribution"""
        
        A_aerosols_ = A_aerosols.reshape((-1, 1))
        
        #
        # Precalcuate the air scattering and attenuation
        #
        scatter_air_pre = 3 / (16*np.pi) * ((1 + self.mu**2) * self.A_air_)
        exp_air_pre = self.H_distances * self.A_air_
        
        exp_aerosols_pre = self.H_distances * A_aerosols_

        img = []
        for L_sun, lambda_, k, w, g in zip(
                self.atmosphere_params.L_SUN_RGB,
                self.atmosphere_params.RGB_WAVELENGTH,
                particle_params.k_RGB,
                particle_params.w_RGB,
                particle_params.g_RGB
                ):
            #
            # Calculate scattering and extiniction for aerosols
            #
            extinction_aerosols = k / particle_params.visibility
            scatter_aerosols = w * extinction_aerosols * (A_aerosols_ * atmo_utils.calcHG(self.mu, g))
            exp_aerosols = np.exp(-extinction_aerosols * exp_aerosols_pre)
            
            #
            # Calculate scattering and extiniction for air
            #
            extinction_air = 1.09e-3 * lambda_**-4.05
            scatter_air = extinction_air * scatter_air_pre
            exp_air = np.exp(-extinction_air * exp_air_pre)
            
            #
            # Calculate the radiance
            #
            radiance = (scatter_air + scatter_aerosols) * (exp_air * exp_aerosols)
    
            #
            # Calculate projection on camera
            #
            temp_img = L_sun * self.H_sensor * radiance
            warnings.warn("Noise not implemented yet")
            #temp_img = temp_img + added_noise*temp_img.std()*np.random.randn(*temp_img.shape)
            temp_img[temp_img<0] = 0
            img.append(temp_img.reshape(self.camera_params.image_res, self.camera_params.image_res))
            
        img = np.transpose(np.array(img), (1, 2, 0))
        
        return img
    
    def calcImageGradient(self, A_aerosols, particle_params):
        """Calculate the image gradient for a given aerosols distribution"""
        
        A_aerosols_ = A_aerosols.reshape((-1, 1))
        
        #
        # Precalcuate the air scattering and attenuation
        #
        scatter_air_pre = 3 / (16*np.pi) * ((1 + self.mu**2) * self.A_air_)
        exp_air_pre = self.H_distances * self.A_air_
        
        exp_aerosols_pre = self.H_distances * A_aerosols_

        gimg = []
        for L_sun, lambda_, k, w, g in zip(
                self.atmosphere_params.L_SUN_RGB,
                self.atmosphere_params.RGB_WAVELENGTH,
                particle_params.k_RGB,
                particle_params.w_RGB,
                particle_params.g_RGB
                ):
            #
            # Calculate scattering and extiniction for aerosols
            #
            P_aerosols = atmo_utils.calcHG(self.mu, g)
            extinction_aerosols = k / particle_params.visibility
            scatter_aerosols =  w * extinction_aerosols * (A_aerosols_ * P_aerosols)
            exp_aerosols = np.exp(-extinction_aerosols * exp_aerosols_pre)
            
            #
            # Calculate scattering and extiniction for air
            #
            extinction_air = 1.09e-3 * lambda_**-4.05
            scatter_air = extinction_air * scatter_air_pre
            exp_air = np.exp(-extinction_air * exp_air_pre)
            
            #
            # Calculate the gradient of the radiance
            #
            temp1 =  w * extinction_aerosols * atmo_utils.spdiag(P_aerosols) * self.H_pol.T
            temp2 = extinction_aerosols * self.H_distances.T * atmo_utils.spdiag(self.H_pol * (scatter_air + scatter_aerosols))
            radiance_gradient = (temp1 - temp2) * atmo_utils.spdiag(exp_air * exp_aerosols)
    
            #
            # Calculate projection on camera
            #
            gimg.append(L_sun * radiance_gradient * self.H_sensor.T)

        return gimg
    

    def draw_image(self, img, focal_ratio, sun_angle):
        """
        """
        
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

    
    