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
    

def show_img(data):

    plt.figure()
    res = plt.imshow(data)
    plt.colorbar(res)


def calc_H_Phi_LS(sky_params):
    """Calc phi"""
    
    h, w = np.round(np.array((sky_params.height, sky_params.width)) / sky_params.dxh)
    X, H = np.meshgrid(np.arange(w)*sky_params.dxh, np.arange(h)[::-1]*sky_params.dxh)

    Phi_LS = np.arctan2(X - sky_params.camera_x, H)
    Distances = H / np.cos(Phi_LS) + eps

    return H, Phi_LS, Distances


def calc_attenuation(H, Distances, sun_angle, Phi_LS, lambda_):

    e_H = np.exp(-H/h_star)
    alpha = 1.09e-3 * lambda_**-4.05
    temp = -alpha * ((1 - e_H) / np.cos(Phi_LS) + e_H / np.cos(sun_angle))
    Phi_scatter = sun_angle + np.pi - Phi_LS
    attenuation = np.exp(temp) * alpha / h_star * e_H * (1 + np.cos(Phi_scatter)**2)
    attenuation = attenuation / (Distances + 1)

    print lambda_, np.max(attenuation)
    return attenuation

    
def calcCamIR(sky_params, autoscale=False):

    H, Phi_LS, Distances = calc_H_Phi_LS(sky_params)

    Phi_scatter = sky_params.sun_angle + np.pi - Phi_LS 
    angle_indices = np.round((Phi_LS/np.pi + 0.5) * sky_params.cam_resolution)    
    cam_radiance = np.zeros((1, sky_params.cam_resolution+1, 3))
   
    for ch_ind, (L_sun, lambda_) in enumerate(zip(L_sun_RGB, RGB_wavelength)):
        Iradiance = L_sun * calc_attenuation(H, Distances, sky_params.sun_angle, Phi_LS, lambda_)
        for i, v, d in zip(angle_indices.flatten(), Iradiance.flatten(), Distances.flatten()):
            if d > sky_params.dist_treshold:
                cam_radiance[0, i, ch_ind] += v

    #
    # Tile the 2D camera to 3D camera.
    #
    cam_radiance = np.tile(cam_radiance, (sky_params.cam_resolution, 1, 1))  
    
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
                dxh=10,
                sun_angle=np.pi/6,
                camera_x=1e5/2,
                cam_resolution=360,
                dist_treshold=1000
                )

    #
    # Run the simulation
    #
    cam_radiance = calcCamIR(sky_params)
    
    skyAnalayzer(cam_radiance).configure_traits()


if __name__ == '__main__':
    main()
    
