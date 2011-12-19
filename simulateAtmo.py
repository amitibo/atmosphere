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
import argparse
import math
import time
import matplotlib.pyplot as plt
import pickle
import pp
import os
import scipy.interpolate

import pycuda.driver as cuda
import pycuda.gpuarray as ga
import pycuda.cumath as cumath
import pycuda.autoinit
from pycuda.elementwise import ElementwiseKernel
from pycuda.compiler import SourceModule


#
# Global parameters
#
SUN_ANGLES = numpy.linspace(-numpy.pi/2, numpy.pi/2, 5)[1:-1]
AEROSOL_VISIBILITY = numpy.linspace(10, 200, 2)

SKY_PARAMS = {
    'width': 100,
    'height': 10,
    'dxh': 0.1,
    'camera_x': 50,
    'angle_res': 180,
    'dist_res': 50,
    'interp_method': 'cubic'
}
RESULTS_FOLDER = 'results'

SERVER_LIST = ()#['gpu%d.ef.technion.ac.il' % i for i in (1, 2, 3, 5)]
NCPUS = 1

def calc_H_Phi_LS(sky_params):
    """Create the sky matrices: Height matrice, LS (line sight) angles and distances (from the camera) """
    
    h, w = numpy.round(numpy.array((sky_params['height'], sky_params['width'])) / sky_params['dxh'])
    X, H = numpy.meshgrid(numpy.arange(w)*sky_params['dxh'], numpy.arange(h)[::-1]*sky_params['dxh'])

    Phi_LS = numpy.arctan2(X - sky_params['camera_x'], H)
    Distances = H / numpy.cos(Phi_LS) + numpy.finfo(numpy.float32).eps

    return ga.to_gpu(H.astype(numpy.float32)), ga.to_gpu(Phi_LS.astype(numpy.float32)), ga.to_gpu(Distances.astype(numpy.float32))


def calcHG(sun_angle, Phi_LS, g):
    """Calculate the Henyey-Greenstein function for each voxel.
    The HG function is taken from: http://www.astro.umd.edu/~jph/HG_note.pdf
    """
    
    henyey_green = ElementwiseKernel(
        "float g, float sun_angle, float *Phi_LS, float *HG",
        "HG[i] = (1 - g*g) / pow((1 + g*g - 2*g*cosf(sun_angle + CUDART_PI_F - Phi_LS[i])), 1.5f) / (4*CUDART_PI_F)",
        "henyey_green_kernel",
        preamble="""#include <math_constants.h>"""
        )

    HG = ga.empty_like(Phi_LS)
    henyey_green(g, sun_angle, Phi_LS, HG)

    return HG


def calc_attenuation(H, Distances, sun_angle, Phi_LS, lambda_, aerosol_visibility, k, w, g):
    """Calc the attenuation of the Sun's radiance per sky voxel.
The calculation takes into account the path from the top of the sky to the voxel
and the path from the voxel to the camera."""
    
    h_star_air = numpy.float32(8.0)
    h_star_aerosol = numpy.float32(1.2)
    alpha_air = numpy.float32(1.09e-3 * lambda_**-4.05)
    alpha_aerosol = numpy.float32(1 / aerosol_visibility * k) #/ 0.0005*10**-12
    
    p = calcHG(sun_angle, Phi_LS, g)
    
    atten_mod = SourceModule("""
        #include <math_constants.h> 

        __global__ void attenuation_kernel(
                            float h_star_air,
                            float h_star_aerosol,
                            float alpha_air,
                            float alpha_aerosol,
                            float sun_angle,
                            float w,
                            float *p,
                            float *H,
                            float *Phi_LS,
                            float *Distances,
                            float *attenuation,
                            int n
                            )
        {
          unsigned tid = threadIdx.x;
          unsigned total_threads = gridDim.x*blockDim.x;
          unsigned cta_start = blockDim.x*blockIdx.x;
          unsigned i;
          float e_H_air;
          float e_H_aerosol;
          float temp = 0;

          for (i = cta_start + tid; i < n; i += total_threads)
          {
            e_H_air = expf(-H[i]/h_star_air);
            e_H_aerosol = expf(-H[i]/h_star_aerosol);
    
            temp = -alpha_air * h_star_air * ((1 - e_H_air) / cosf(Phi_LS[i]) + e_H_air / cosf(sun_angle));
            temp += -alpha_aerosol * h_star_aerosol * ((1 - e_H_aerosol) / cosf(Phi_LS[i]) + e_H_aerosol / cosf(sun_angle));
            attenuation[i] = expf(temp) * (alpha_air * e_H_air * (1 + pow(cosf(sun_angle + CUDART_PI_F - Phi_LS[i]), 2.0f)) +  alpha_aerosol * e_H_aerosol * w * p[i]);
            attenuation[i] = attenuation[i] / (Distances[i] + 0.001);
          };
        }
"""
                       )

    atten_func = atten_mod.get_function("attenuation_kernel")

    attenuation = ga.empty_like(Phi_LS)
    atten_func(
        h_star_air,
        h_star_aerosol,
        alpha_air,
        alpha_aerosol,
        numpy.float32(sun_angle),
        numpy.float32(w),
        p,
        H,
        Phi_LS,
        Distances,
        attenuation,
        numpy.int32(Phi_LS.size),
        block=(256,1,1)                            
        )

    return attenuation.get()


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


def calcCamIR(sky_params, aerosol_params, sun_angle):
    """Calculate the Iradiance at the camera"""
    
    L_sun_RGB=(255, 236, 224)
    RGB_wavelength = (700e-3, 530e-3, 470e-3)

    tic = time.time()
    H, Phi_LS, Distances = calc_H_Phi_LS(sky_params)
    cam_radiance = numpy.zeros((sky_params['angle_res'], 3))
    print 'preparation - %f' % (time.time() - tic)

    #
    # Caluclate the iradiance separately for each color channel.
    #
    for ch_ind, (L_sun, lambda_, k, w, g) in enumerate(zip(L_sun_RGB, RGB_wavelength, aerosol_params["k_RGB"], aerosol_params["w_RGB"], aerosol_params["g_RGB"])):
        tic = time.time()
        Iradiance = L_sun * calc_attenuation(
                                H,
                                Distances,
                                sun_angle,
                                Phi_LS,
                                lambda_,
                                aerosol_params["visibility"],
                                k,
                                w,
                                g
                                )
        tac = time.time()
        cam_radiance[:, ch_ind] = cameraProject(
                                    Iradiance,
                                    Distances.get(),
                                    Phi_LS.get(),
                                    sky_params['dist_res'],
                                    sky_params['angle_res'],
                                    sky_params['interp_method']
                                    )
        toc = time.time()
        print "attenuation - %f, projection - %f" % (tac-tic, toc-tic)

    return cam_radiance


def main():
    #
    # Parse the command line
    #
    parser = argparse.ArgumentParser(description='Simulate the sky.')
    parser.add_argument('--folder', type=str, default='', help='Load previously calculated sky optical paths.')
    args = parser.parse_args()

    results_folder = 'results'

    #
    # Load the misr data base
    #
    with open('misr.pkl', 'rb') as f:
        misr = pickle.load(f)
    
    particles_list = misr.keys()
    
    class skyAnalayzer(HasTraits):
        tr_scaling = Range(0.0, 30.0, 0.0, desc='Radiance scaling logarithmic')
        tr_sun_angle = Range(float(SUN_ANGLES[0]), float(SUN_ANGLES[-1]), 0.0, desc='Zenith of the sun')
        tr_aeros_viz = Range(float(AEROSOL_VISIBILITY[0]), float(AEROSOL_VISIBILITY[-1]), desc='Visibility due to aerosols [km]')
        tr_sky_max = Float( 0.0, desc='Maximal value of raw sky image (before scaling)' )
        tr_particles = Enum(particles_list, desc='Name of particle')
        
        traits_view  = View(
                            VGroup(
                                Item('plot_sky', editor=ComponentEditor(), show_label=False),
                                Item('tr_sky_max', label='Maximal value', style='readonly'),
                                Item('tr_particles', label='Particle Name'),                                
                                Item('tr_scaling', label='Radiance Scaling'),
                                Item('tr_sun_angle', label='Sun Angle'),
                                Item('tr_aeros_viz', label='Aerosol Visibility [km]')
                                ),
                            resizable = True
                        )
                            
        def __init__(self, folder, misr):
            super( skyAnalayzer, self ).__init__()

            self.misr = misr

            if folder:
                file_name = os.path.join(folder, 'blue_sky.pkl')
                with open(file_name, 'rb') as f:
                    self.sky_list = pickle.load(f)
            else:
                self.calcSkyList()
            
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
    
        def calcSkyList(self):

            particle = self.misr[self.tr_particles]
            
            aerosol_params = {
                "k_RGB": numpy.array(particle['k']) / numpy.max(numpy.array(particle['k'])),#* 10**-12,
                "w_RGB": particle['w'],
                "g_RGB": (particle['g']),
                "visibility": 1
            }
                        
            #
            # Run the simulation
            #
            tic = time.time()
            self.sky_list = []
            for aerosol_viz in AEROSOL_VISIBILITY:
                aerosol_params["visibility"] = aerosol_viz
                temp = []
                for sun_angle in SUN_ANGLES:
                    temp.append(calcCamIR(SKY_PARAMS, aerosol_params, sun_angle))
                self.sky_list.append(temp)

            print 'Time used %d' % (time.time()- tic)
            
            if not os.path.exists(RESULTS_FOLDER):
                os.mkdir(RESULTS_FOLDER)
                
            file_path = os.path.join(RESULTS_FOLDER, 'blue_sky.pkl')
            with open(file_path, 'wb') as f:
                pickle.dump(self.sky_list, f)
            
        def scaleImg(self):
            """Scale and tile the image to valid values"""
            
            angle_index = numpy.argmin(numpy.abs(SUN_ANGLES - self.tr_sun_angle))
            aerosol_index = numpy.argmin(numpy.abs(AEROSOL_VISIBILITY - self.tr_aeros_viz))
            
            tmpimg = self.sky_list[aerosol_index][angle_index]*10**self.tr_scaling
            tmpimg[tmpimg > 255] = 255
            self.tr_sky_max = numpy.max(self.sky_list[aerosol_index][angle_index])
            
            #
            # Tile the 2D camera to 3D camera.
            #
            pixel_num = tmpimg.shape[0]
            tmpimg = numpy.tile(tmpimg.reshape(1, pixel_num, 3), (pixel_num, 1, 1))  
    
            return tmpimg.astype(numpy.uint8)
                       
        @on_trait_change('tr_particles')
        def _updateImg(self):
            self.calcSkyList()
            self.plotdata.set_data( 'sky_img', self.scaleImg() )
    
        @on_trait_change('tr_scaling, tr_sun_angle, tr_aeros_viz')
        def _updateImgScale(self):
            self.plotdata.set_data( 'sky_img', self.scaleImg() )

    #
    # Show the results in a GUI.
    #
    skyAnalayzer(args.folder, misr).configure_traits()


if __name__ == '__main__':
    main()
    
