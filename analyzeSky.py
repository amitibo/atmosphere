# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 23:52:19 2011

@author: amitibo
"""
from __future__ import division
import numpy as np
from enthought.traits.api import HasTraits, Enum, Range, Float, on_trait_change
from enthought.traits.ui.api import View, Item, VGroup
from enthought.chaco.api import Plot, ArrayPlotData, VPlotContainer
from enthought.enable.component_editor import ComponentEditor
from utils import attrClass
from simulateSky import calcCamIR
import argparse
import pickle
import os


def main(folder):
    #
    # Load the misr data base
    #
    with open('misr.pkl', 'rb') as f:
        misr = pickle.load(f)
    
    particles_list = misr.keys()
    
    #
    # Load the sky data
    #
    file_name = 'sky.pkl'
    if folder:
        file_name = os.path.join(args.folder, file_name)
        
    with open(file_name, 'rb') as f:
        sky = pickle.load(f)
    
    class skyAnalayzer(HasTraits):
        tr_particles = Enum(particles_list, desc='Name of particle')
        tr_scaling = Range(0.0, 30.0, 0.0, desc='Radiance scaling logarithmic')
        tr_sky_max = Float( 0.0, desc='Maximal value of raw sky image (before scaling)' )
    
        traits_view  = View(
                            VGroup(
                                Item('plot_sky', editor=ComponentEditor(), show_label=False),
                                Item('tr_particles', label='Particle Name'),
                                Item('tr_sky_max', label='Maximal value', style='readonly'),
                                Item('tr_scaling', label='Radiance Scaling')
                                ),
                            resizable = True
                        )
                        
        def __init__(self, sky, misr):
            super( skyAnalayzer, self ).__init__()

            self.sky = sky
            self.misr = misr
            
            self.updateImg()
            
            #
            # Prepare all the plots.
            #
            self.plotdata = ArrayPlotData( sky_img=self.scaleImg() )
            plot_img = Plot(self.plotdata)
            plot_img.img_plot("sky_img")
    
            self.plot_sky = VPlotContainer( plot_img )

        def scaleImg(self):
            tmpimg = self.sky_img*10**self.tr_scaling
            tmpimg[tmpimg > 255] = 255
            return (tmpimg).astype(np.uint8)
            
        def updateImg(self):
            print self.tr_particles
            particle = self.misr[self.tr_particles]
            
            aerosol_params = attrClass(
                        ANGLE_ACCURACY=2,
                        k_RGB=np.array(particle['k']) * 10**-12,
                        w_RGB=particle['w'],
                        g_RGB=(particle['g']),
                        F_sol_RGB=(255, 236, 224)
                        )
                        
            self.sky_img = calcCamIR( self.sky, aerosol_params, autoscale=False )
            self.tr_sky_max = np.max(self.sky_img)
        
        @on_trait_change('tr_particles')
        def _updateImg(self):
            self.updateImg()
            self.plotdata.set_data( 'sky_img', self.scaleImg() )
    
        @on_trait_change('tr_scaling')
        def _updateImgScale(self):
            self.plotdata.set_data( 'sky_img', self.scaleImg() )
    
    
    skyAnalayzer(sky, misr).configure_traits()
    
        
if __name__ == '__main__':
    #
    # Parse the command line
    #
    parser = argparse.ArgumentParser(description='Analyze the sky.')
    parser.add_argument('folder', type=str, default='', help='Path to calculated sky optical paths.')
    args = parser.parse_args()
    
    main(args.folder)
