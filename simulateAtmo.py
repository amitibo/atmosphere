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
from utils import createResultFolder, attrClass
from enthought.traits.api import HasTraits, Enum, Range, Float, on_trait_change
from enthought.traits.ui.api import View, Item, VGroup
from enthought.chaco.api import Plot, ArrayPlotData, VPlotContainer
from enthought.enable.component_editor import ComponentEditor
import numpy as np
from scipy.interpolate import griddata
import argparse
import math
import matplotlib.pyplot as plt
import pickle
import os


#
# Global parameters
#
L_sun_RGB=(255, 236, 224)
RGB_wavelength = (700e-3, 530e-3, 470e-3)
h_star = 8000
eps = np.finfo(float).eps


class skyAnalayzer(HasTraits):
    tr_scaling = Range(0.0, 30.0, 0.0, desc='Radiance scaling logarithmic')
    tr_sun_angle = Range(-np.pi, np.pi, 0.0, desc='Zenith of the sun')
    tr_sky_max = Float( 0.0, desc='Maximal value of raw sky image (before scaling)' )
    
    traits_view  = View(
                        VGroup(
                            Item('plot_sky', editor=ComponentEditor(), show_label=False),
                            Item('tr_sky_max', label='Maximal value', style='readonly'),
                            Item('tr_scaling', label='Radiance Scaling')
                            ),
                        resizable = True
                    )
                        
    def __init__(self, sky):
        super( skyAnalayzer, self ).__init__()

        self.sky_img = sky
        self.sky_img[self.sky_img<0] = 0
        self.tr_sky_max = np.max(self.sky_img)
            
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
        tmpimg = self.sky_img*10**self.tr_scaling
        tmpimg[tmpimg > 255] = 255
        return tmpimg.astype(np.uint8)
                   
    @on_trait_change('tr_scaling')
    def _updateImgScale(self):
        self.plotdata.set_data( 'sky_img', self.scaleImg() )
    

def calc_H_Phi_LS(sky_params):
    """Create the sky matrices: Height matrice, LS (line sight) angles and distances (from the camera) """
    
    h, w = np.round(np.array((sky_params.height, sky_params.width)) / sky_params.dxh)
    X, H = np.meshgrid(np.arange(w)*sky_params.dxh, np.arange(h)[::-1]*sky_params.dxh)

    Phi_LS = np.arctan2(X - sky_params.camera_x, H)
    Distances = H / np.cos(Phi_LS) + eps

    return H, Phi_LS, Distances


def calc_attenuation(H, Distances, sun_angle, Phi_LS, lambda_):
    """Calc the attenuation of the Sun's radiance per sky voxel.
The calculation takes into account the path from the top of the sky to the voxel
and the path from the voxel to the camera."""
    
    e_H = np.exp(-H/h_star)
    alpha = 1.09e-3 * lambda_**-4.05
    temp = -alpha * ((1 - e_H) / np.cos(Phi_LS) + e_H / np.cos(sun_angle))
    Phi_scatter = sun_angle + np.pi - Phi_LS
    attenuation = np.exp(temp) * alpha / h_star * e_H * (1 + np.cos(Phi_scatter)**2)
    attenuation = attenuation / (Distances + 1)

    print lambda_, np.max(attenuation)
    return attenuation


def cameraProject(Iradiance, Distances, Angles, dist_res, angle_res):
    """Interpolate a uniform polar grid of a nonuniform polar data"""
    
    max_R = np.max(Distances)
    
    grid_phi, grid_R = \
        np.mgrid[np.pi/2:-np.pi/2:np.complex(0, angle_res), 0:max_R:np.complex(0, dist_res)]
    
    points = np.vstack((Distances.flatten(), Angles.flatten())).T
    polar_attenuation = griddata(points, Iradiance.flatten(), (grid_R, grid_phi), method='linear', fill_value=0)
    
    jac = np.linspace(0, max_R, dist_res)
    
    camera_projection = np.sum(polar_attenuation * jac, axis = 1)
    
    return camera_projection


def calcCamIR(sky_params, autoscale=False):
    """Calculate the Iradiance at the camera"""
    
    H, Phi_LS, Distances = calc_H_Phi_LS(sky_params)

    cam_radiance = np.zeros((sky_params.angle_res, 3))
   
    #
    # Caluclate the iradiance separately for each color channel.
    #
    for ch_ind, (L_sun, lambda_) in enumerate(zip(L_sun_RGB, RGB_wavelength)):
        Iradiance = L_sun * calc_attenuation(H, Distances, sky_params.sun_angle, Phi_LS, lambda_)
        
        cam_radiance[:, ch_ind] = cameraProject(
                                    Iradiance,
                                    Distances,
                                    Phi_LS,
                                    sky_params.dist_res,
                                    sky_params.angle_res
                                    )

    #
    # Tile the 2D camera to 3D camera.
    #
    cam_radiance = np.tile(cam_radiance.reshape(1, sky_params.angle_res, 3), (sky_params.angle_res, 1, 1))  
    
    #
    # Stretch the values.
    #
    if autoscale:
        cam_radiance = (cam_radiance/np.max(cam_radiance)*255).astype(np.uint8)
    
    return cam_radiance


def main():
    #
    # Set the params of the run
    #
    sky_params = attrClass(
                width=1e5,
                height=1e4,
                dxh=50,
                sun_angle=-np.pi/3,
                camera_x=1e5/2,
                angle_res=180,
                dist_res=100
                )

    #
    # Run the simulation
    #
    cam_radiance = calcCamIR(sky_params)
    
    #
    # Show the results in a GUI.
    #
    skyAnalayzer(cam_radiance).configure_traits()


if __name__ == '__main__':
    main()
    
