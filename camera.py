"""
"""

from __future__ import division
import numpy as np
import atmo_utils
import amitibo
import warnings


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

    scatter_angle = calcScatterAngle(R, PHI, THETA, rotation)
    
    #vizTransforms(Y, X, Z, H_pol, Hrot_forward, Hrot_backward, Hint1, Hint2, R, Y_rot, scatter_angle)
    
    return temp2 + temp1, scatter_angle


def calcScatterAngle(R, PHI, THETA, rotation):
    """
    Calclculate the scattering angle at each voxel.
    """

    X_ = R * np.sin(THETA) * np.cos(PHI)
    Y_ = R * np.sin(THETA) * np.sin(PHI)
    Z_ = R * np.cos(THETA)

    XYZ_dst = np.vstack((X_.ravel(), Y_.ravel(), Z_.ravel(), np.ones(R.size)))
    XYZ_src_ = np.dot(rotation, XYZ_dst)

    Z_rotated = XYZ_src_[2, :]
    R_rotated = np.sqrt(np.sum(XYZ_src_[:3, :]**2, axis=0))

    angle = np.arccos(Z_rotated/(R_rotated+amitibo.eps(R_rotated)))

    return angle


class Camera(object):
    """A class that encapsulates the functions of a camera"""
    
    def __init__(
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

        H_pol, R, PHI, THETA = atmo_utils.sphericalTransformMatrix(
            Y,
            X,
            H,
            center=camera_position,
            radius_res=camera_params.radius_res,
            phi_res=camera_params.phi_res,
            theta_res=camera_params.theta_res
            )
        
        #
        # Calculate the distance matrices and scattering angle
        #
        H_distances, scatter_angle = \
            calcOpticalDistancesMatrix(
                Y,
                X,
                H,
                sun_angle,
                H_pol,
                R,
                PHI,
                THETA
                )

        H_int = atmo_utils.integralTransformMatrix((R, PHI, THETA))
        H_camera = atmo_utils.cameraTransformMatrix(
            PHI[0, :, :],
            THETA[0, :, :],
            focal_ratio=camera_params.focal_ratio,
            image_res=camera_params.image_res,
            theta_compensation=camera_params.theta_compensation
        )

        #
        #
        #
        self.H_distances = H_distances
        self.H_pol = H_pol
        self.H_sensor = H_camera * H_int
        self.mu = np.cos(scatter_angle).reshape((-1, 1))
        self.camera_params = camera_params
        self.atmosphere_params = atmosphere_params

    def calcImage(self, A_air, A_aerosols, particle_params):
        """Calculate the image for a given aerosols distribution"""
        
        A_aerosols_ = A_aerosols.reshape((-1, 1))
        A_air_ = A_air.reshape((-1, 1))
        
        #
        # Precalcuate the air scattering and attenuation
        #
        scatter_air_pre = (1 + self.mu**2) * 3 / (16*np.pi) * (self.H_pol * A_air_)
        exp_air_pre = self.H_distances * A_air_
        scatter_aerosols_pre = self.H_pol * A_aerosols_
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
            scatter_aerosols = w * extinction_aerosols * atmo_utils.calcHG(self.mu, g) * scatter_aerosols_pre
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
            radiance = (scatter_air + scatter_aerosols) * exp_air * exp_aerosols
    
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
    
    def calcImageGradient(self, A_air, A_aerosols, particle_params):
        """Calculate the image gradient for a given aerosols distribution"""
        
        A_aerosols_ = A_aerosols.reshape((-1, 1))
        A_air_ = A_air.reshape((-1, 1))
        
        #
        # Precalcuate the air scattering and attenuation
        #
        scatter_air_pre = (1 + self.mu**2) * 3 / (16*np.pi) * (self.H_pol * A_air_)
        exp_air_pre = self.H_distances * A_air_
        scatter_aerosols_pre = self.H_pol * A_aerosols_
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
            extinction_aerosols = k / particle_params.visibility
            scatter_aerosols_partial = w * extinction_aerosols * atmo_utils.calcHG(self.mu, g)
            scatter_aerosols =  atmo_utils.spdiag(scatter_aerosols_partial * scatter_aerosols_pre)
            scatter_aerosols_grad = self.H_pol.T * atmo_utils.spdiag(scatter_aerosols_partial)
            exp_aerosols = atmo_utils.spdiag(np.exp(-extinction_aerosols * exp_aerosols_pre))
            exp_aerosols_grad = -extinction_aerosols * self.H_distances.T * exp_aerosols
            
            #
            # Calculate scattering and extiniction for air
            #
            extinction_air = 1.09e-3 * lambda_**-4.05
            scatter_air = atmo_utils.spdiag(extinction_air * scatter_air_pre)
            exp_air = atmo_utils.spdiag(np.exp(-extinction_air * exp_air_pre))
            
            #
            # Calculate the gradient of the radiance
            #
            temp1 = scatter_aerosols_grad * exp_aerosols + exp_aerosols_grad * scatter_aerosols
            temp2 = exp_aerosols_grad * scatter_air
            radiance_gradient = (temp1 + temp2) * exp_air
    
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
    
    cam = Camera(
        SUN_ANGLE,
        atmosphere_params=atmosphere_params,
        camera_params=camera_params,
        camera_position=CAMERA_CENTERS
        )
    
    Y, X, H = np.mgrid[atmosphere_params.cartesian_grids]
    width = atmosphere_params.cartesian_grids[0].stop
    height = atmosphere_params.cartesian_grids[2].stop
    h = np.sqrt((X-width/2)**2 + (Y-width/2)**2 + (atmosphere_params.earth_radius+H)**2) - atmosphere_params.earth_radius
    A_aerosols = np.exp(-h/atmosphere_params.aerosols_typical_h)
    A_air = np.exp(-h/atmosphere_params.air_typical_h)

    gimg = cam.calcImageGradient(
         A_air,
         A_aerosols,
         particle_params
    )
    
    

def main():
    """Main doc """
    
    test_camera()

    
if __name__ == '__main__':
    main()

    
    