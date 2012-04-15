from __future__ import division

from enthought.traits.api import HasTraits, Range, on_trait_change, Float, Enum
from enthought.traits.ui.api import View, Item, Handler, Action, VGroup
from enthought.chaco.api import Plot, ArrayPlotData, PlotAxis, VPlotContainer
from enthought.enable.component_editor import ComponentEditor
from atmo_utils import L_SUN_RGB, RGB_WAVELENGTH
from simulateAtmoGeneral import calcRadiance
import matplotlib.pyplot as plt
import numpy as np
import pickle
import amitibo


SKY_PARAMS = {
    'width': 200,
    'height': 50,
    'dxh': 1,
    'camera_center': (80, 2),
    'sun_angle': 0,
    'L_SUN_RGB': L_SUN_RGB,
    'RGB_WAVELENGTH': RGB_WAVELENGTH
}


class TC_Handler(Handler):

    def do_savefig(self, info):

        sky_object = info.object

        h = int(sky_object.sky_img.shape[0]/2)
        
        plt.subplot(211)
        extent = (0, 1, 90, 0)
        plt.imshow(sky_object.sky_img[h:0:-1, ...], aspect=1/270, extent=extent, interpolation='nearest')
        plt.xticks([0, 0.5, 1.0])
        plt.yticks([0, 30, 60, 90])
        plt.title(sky_object.makeTitle())

        plt.subplot(212)
        extent = (0, 1, -90, 0)
        plt.imshow(sky_object.sky_img[h:, ...], aspect=1/270, extent=extent, interpolation='nearest')
        plt.xticks([0, 0.5, 1.0])
        plt.yticks([0, -30, -60, -90])

        amitibo.saveFigures(sky_object.results_path, bbox_inches='tight')


class skyAnalayzer(HasTraits):
    """Gui Application"""
    
    tr_scaling = Range(-5.0, 5.0, 0.0, desc='Radiance scaling logarithmic')
    tr_sun_angle = Range(-np.pi/2, np.pi/2, 0.0, desc='Zenith of the sun [radians]')
    tr_camera_center = Range(10.0, SKY_PARAMS['width']-10.0, 80, desc='X axis of camera')
    tr_aeros_viz = Range(1, 100, desc='Visibility due to aerosols [km]')
    tr_sky_max = Float( 0.0, desc='Maximal value of raw sky image (before scaling)' )
    tr_gamma = Range(0.4, 1.0, 0.45, desc='Gamma encoding value')
    save_button = Action(name = "Save Fig", action = "do_savefig")
    tr_aeros_scale = Range(1, 20, desc='Exponential scale of the aerosols [km]')

    traits_view  = View(
        VGroup(
            Item('img_container', editor=ComponentEditor(), show_label=False),
            Item('tr_sky_max', label='Maximal value', style='readonly'),
            Item('tr_particles', label='Particle Name'),                                
            Item('tr_scaling', label='Radiance Scaling'),
            Item('tr_sun_angle', label='Sun Angle'),
            Item('tr_camera_center', label='Camera Center'),
            Item('tr_aeros_viz', label='Aerosol Visibility [km]'),
            Item('tr_aeros_scale', label='Aerosol Scale [km]'),
            Item('tr_gamma', label='Gamma Encoding')
            ),
            resizable = True,
            handler=TC_Handler(),
            buttons = [save_button]
        )

    def __init__(self, results_path):
        super(skyAnalayzer, self).__init__()

        self.results_path = results_path
        
        #
        # Load the misr data base
        #
        with open('misr.pkl', 'rb') as f:
            self.misr = pickle.load(f)

        self.add_trait('tr_particles',  Enum(self.misr.keys(), desc='Name of particle'))
    
        #
        # Prepare all the plots.
        # ArrayPlotData - A class that holds a list of numpy arrays.
        # Plot - Represents a correlated set of data, renderers, and
        # axes in a single screen region.
        #
        self.plotdata = ArrayPlotData()
        self.img_container = VPlotContainer(
            padding=-20,
            fill_padding=True,
            spacing=-40,
            stack_order='top_to_bottom',
            bgcolor='cyan',
            use_backbuffer=True
            )

        self._updateImg()
        for i in range(2):
            p = Plot(self.plotdata)
            p.img_plot("sky_img%d" % i, xbounds=(-0.5, 0.5), ybounds=(0, 90))
            p.overlays.append(
                PlotAxis(
                    orientation='bottom',
                    title= "View Angle [rad]",
                    component=p
                    )
                )
            self.img_container.add(p)
            
    
    def _scaleImg(self):
        """Scale and tile the image to valid values"""

        tmpimg = self.base_img * 10**self.tr_scaling
        tmpimg = tmpimg ** self.tr_gamma
        tmpimg[tmpimg > 255] = 255

        self.sky_img = tmpimg.astype(np.uint8)
        h = int(self.sky_img.shape[0]/2)
        self.plotdata.set_data('sky_img0', self.sky_img[-1:h:-1, ...])
        self.plotdata.set_data('sky_img1', self.sky_img[:h, ...])

    def _calcImg(self):
        particle = self.misr[self.tr_particles]

        aerosol_params = {
            "k_RGB": np.array(particle['k']) / np.max(np.array(particle['k'])),
            "w_RGB": particle['w'],
            "g_RGB": particle['g'],
            "visibility": self.tr_aeros_viz,
            "air_typical_h": 8,
            "aerosols_typical_h": self.tr_aeros_scale
            }

        SKY_PARAMS['sun_angle'] = self.tr_sun_angle
        SKY_PARAMS['camera_center'] = (self.tr_camera_center, 2)

        tmp_img = calcRadiance(aerosol_params, SKY_PARAMS)
        tmp_img = np.transpose(np.array(tmp_img, ndmin=3), (2, 0, 1))
        
        self.base_img = np.tile(tmp_img, (1, tmp_img.shape[0], 1))
        self.tr_sky_max = np.max(self.base_img)
        
    @on_trait_change('tr_scaling, tr_gamma')
    def _updateImgScale(self):
        self.img_container.title = self.makeTitle()
        self._scaleImg()

    @on_trait_change('tr_aeros_scale, tr_particles, tr_sun_angle, tr_aeros_viz, tr_camera_center')
    def _updateImg(self):
        self.img_container.title = self.makeTitle()
        self._calcImg()
        self._scaleImg()

    def makeTitle(self):
        return "%s\nVisibility: %d[km], Sun Angle: %.1f[rad]" % (self.tr_particles, self.tr_aeros_viz, self.tr_sun_angle)
        

def main():
    """Main function"""

    results_path = amitibo.createResultFolder()
    
    app = skyAnalayzer(results_path)
    app.configure_traits()
    
    
if __name__ == '__main__':
    main()
