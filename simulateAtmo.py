# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 18:21:25 2011

@author: amitibo

The calulations are based on Ch.12 of "Applied Optics" by Leo Levi.

The sun iradiance is based on the following data:
1) http://en.wikipedia.org/wiki/Sunlight#Composition
Compares the sunlight to blackbody at 5250C

2) http://www.vendian.org/mncharity/dir3/blackbody/UnstableURLs/bbr_color.html
Gives the RGB values for black body according to the following fields
#  K     temperature (K)
#  CMF   {" 2deg","10deg"}
#          either CIE 1931  2 degree CMFs with Judd Vos corrections
#              or CIE 1964 10 degree CMFs
#  x y   chromaticity coordinates
#  P     power in semi-arbitrary units
#  R G B {0-1}, normalized, mapped to gamut, logrithmic
#        (sRGB primaries and gamma correction)
#  r g b {0-255}
#  #rgb  {00-ff} 

  5500 K   2deg  0.3346 0.3451  1.363e+15    1.0000 0.8541 0.7277  255 238 222  #ffeede
  5500 K  10deg  0.3334 0.3413  1.479e+15    1.0000 0.8403 0.7437  255 236 224  #ffece0

Note: 5250 C = 5520 K

The RGB wavelengths are taken from
http://en.wikipedia.org/wiki/Color
"""


from __future__ import division
from enthought.traits.api import HasTraits, Enum, Range, Float, on_trait_change
from enthought.traits.ui.api import View, Item, VGroup
from enthought.chaco.api import Plot, ArrayPlotData, VPlotContainer
from enthought.enable.component_editor import ComponentEditor
import numpy
import scipy.interpolate
import argparse
import math
import matplotlib.pyplot as plt
import pickle
import os
from IPython.parallel import Client, require

#
# Global parameters
#
SUN_ANGLES = numpy.linspace(-numpy.pi/2, numpy.pi/2, 10)


class skyAnalayzer(HasTraits):
    tr_scaling = Range(0.0, 30.0, 0.0, desc='Radiance scaling logarithmic')
    tr_sun_angle = Range(float(SUN_ANGLES[0]), float(SUN_ANGLES[-1]), 0.0, desc='Zenith of the sun')
    tr_sky_max = Float( 0.0, desc='Maximal value of raw sky image (before scaling)' )
    
    traits_view  = View(
                        VGroup(
                            Item('plot_sky', editor=ComponentEditor(), show_label=False),
                            Item('tr_sky_max', label='Maximal value', style='readonly'),
                            Item('tr_scaling', label='Radiance Scaling'),
                            Item('tr_sun_angle', label='Index of sun angle')
                            ),
                        resizable = True
                    )
                        
    def __init__(self, sky_list):
        super( skyAnalayzer, self ).__init__()
        
        self.sky_list = sky_list
        self.tr_sky_max = numpy.max(self.sky_list[0])
            
        #
        # Prepare all the plots.
        # ArrayPlotData - A class that holds a list of numpy arrays.
        # Plot - Represents a correlated set of data, renderers, and axes in a single screen region.
        # VPlotContainer - A plot container that stacks plot components vertically.
        #
        self.plotdata = ArrayPlotData( sky_img=self.scaleImg() )
        plot_img = Plot(self.plotdata)
        plot_img.img_plot("sky_img")
    
        self.plot_sky = VPlotContainer( plot_img )

    def scaleImg(self):
        sky_list_index = numpy.argmin(numpy.abs(SUN_ANGLES - self.tr_sun_angle))
        
        tmpimg = self.sky_list[sky_list_index]*10**self.tr_scaling
        tmpimg[tmpimg > 255] = 255
        self.tr_sky_max = numpy.max(self.sky_list[sky_list_index])
        return tmpimg.astype(numpy.uint8)
                   
    @on_trait_change('tr_scaling, tr_sun_angle')
    def _updateImgScale(self):
        self.plotdata.set_data( 'sky_img', self.scaleImg() )
    


@require(numpy, scipy.interpolate)
def calcCamIR(params):
    """Calculate the Iradiance at the camera"""
    
    L_sun_RGB=(255, 236, 224)
    RGB_wavelength = (700e-3, 530e-3, 470e-3)

    def calc_H_Phi_LS(sky_params):
        """Create the sky matrices: Height matrice, LS (line sight) angles and distances (from the camera) """
        
        h, w = numpy.round(numpy.array((sky_params['height'], sky_params['width'])) / sky_params['dxh'])
        X, H = numpy.meshgrid(numpy.arange(w)*sky_params['dxh'], numpy.arange(h)[::-1]*sky_params['dxh'])
    
        Phi_LS = numpy.arctan2(X - sky_params['camera_x'], H)
        Distances = H / numpy.cos(Phi_LS) + numpy.finfo(float).eps
    
        return H, Phi_LS, Distances
    
    
    def calc_attenuation(H, Distances, sun_angle, Phi_LS, lambda_):
        """Calc the attenuation of the Sun's radiance per sky voxel.
    The calculation takes into account the path from the top of the sky to the voxel
    and the path from the voxel to the camera."""
        
        h_star = 8000
        e_H = numpy.exp(-H/h_star)
        alpha = 1.09e-3 * lambda_**-4.05
        temp = -alpha * ((1 - e_H) / numpy.cos(Phi_LS) + e_H / numpy.cos(sun_angle))
        Phi_scatter = sun_angle + numpy.pi - Phi_LS
        attenuation = numpy.exp(temp) * alpha / h_star * e_H * (1 + numpy.cos(Phi_scatter)**2)
        attenuation = attenuation / (Distances + 1)
    
        print lambda_, numpy.max(attenuation)
        return attenuation
    
    
    def cameraProject(Iradiance, Distances, Angles, dist_res, angle_res, interp_method):
        """Interpolate a uniform polar grid of a nonuniform polar data"""
        
        max_R = numpy.max(Distances)
        
        grid_phi, grid_R = \
            numpy.mgrid[-numpy.pi/2:numpy.pi/2:numpy.complex(0, angle_res), 0:max_R:numpy.complex(0, dist_res)]
        
        points = numpy.vstack((Distances.flatten(), Angles.flatten())).T
        polar_attenuation = scipy.interpolate.griddata(points, Iradiance.flatten(), (grid_R, grid_phi), method=interp_method, fill_value=0)
        polar_attenuation[polar_attenuation<0] = 0
        
        jac = numpy.linspace(0, max_R, dist_res)
        camera_projection = numpy.sum(polar_attenuation * jac, axis = 1)
        
        return camera_projection

    sky_params, sun_angle = params
    H, Phi_LS, Distances = calc_H_Phi_LS(sky_params)

    cam_radiance = numpy.zeros((sky_params['angle_res'], 3))
   
    #
    # Caluclate the iradiance separately for each color channel.
    #
    for ch_ind, (L_sun, lambda_) in enumerate(zip(L_sun_RGB, RGB_wavelength)):
        Iradiance = L_sun * calc_attenuation(H, Distances, sun_angle, Phi_LS, lambda_)
        
        cam_radiance[:, ch_ind] = cameraProject(
                                    Iradiance,
                                    Distances,
                                    Phi_LS,
                                    sky_params['dist_res'],
                                    sky_params['angle_res'],
                                    sky_params['interp_method']
                                    )

    #
    # Tile the 2D camera to 3D camera.
    #
    cam_radiance = numpy.tile(cam_radiance.reshape(1, sky_params['angle_res'], 3), (sky_params['angle_res'], 1, 1))  
    
    return cam_radiance


def main():
    #
    # Parse the command line
    #
    parser = argparse.ArgumentParser(description='Simulate the sky.')
    parser.add_argument('--folder', type=str, default='', help='Load previously calculated sky optical paths.')
    args = parser.parse_args()


    results_folder = 'results'

    if args.folder:
        file_name = os.path.join(args.folder, 'blue_sky.pkl')
        with open(file_name, 'rb') as f:
            cam_radiances = pickle.load(f)
    else:
        #
        # Set the params of the run
        #
        sky_params = {
                    "width": 1e5,
                    "height": 1e4,
                    "dxh": 50,
                    "camera_x": 1e5/2,
                    "angle_res": 360,
                    "dist_res": 100,
                    "interp_method": 'cubic'
        }
    
        #
        # Handle the parallelism
        #
        rc = Client()
        lview = rc.load_balanced_view()
        lview.block = True
        
        #
        # Run the simulation
        #
        cam_radiances = lview.map(calcCamIR, [(sky_params, sun_angle) for sun_angle in SUN_ANGLES])
        
        file_path = os.path.join(results_folder, 'blue_sky.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(cam_radiances, f)
            
    #
    # Show the results in a GUI.
    #
    skyAnalayzer(cam_radiances).configure_traits()


if __name__ == '__main__':
    main()
    
