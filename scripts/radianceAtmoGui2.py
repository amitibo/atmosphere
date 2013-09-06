"""
Visualize side by side the results of MC simulations and Single Scattering model.
This GUI is useful when the MC results were used as reference to the SS scattering analysis.
Just drag and drop one of the images into the GUI.
The GUI enables browsing and comparing the results of different cameras and scaling separately
the MC and single scattering results.
"""

from __future__ import division

import numpy as np
import scipy.io as sio
from traits.api import HasTraits, Range, List, Instance, on_trait_change, Enum, Float
from traitsui.api import View, Item, HGroup, VGroup, DropEditor, EnumEditor, spring, Action, Handler
from tvtk.pyface.scene_editor import SceneEditor
from enthought.chaco.api import Plot, ArrayPlotData, PlotAxis, VPlotContainer, Legend, PlotLabel
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene
from enthought.enable.component_editor import ComponentEditor
from enthought.io.api import File
from tvtk.api import tvtk
from mayavi import mlab
import atmotomo
import amitibo
import os



def mse(estim, orig):
    #old_err_state = np.seterr(divide='ignore')
    ans = np.abs(estim-orig).sum()/np.abs(orig).sum()
    #np.seterr(**old_err_state)
    #ans = ans.mean()
    return ans


def zeroBorders(s, margin=1):
    if margin < 1:
        return s
    
    s = s.copy()
    s[0:margin, :, :] = 0
    s[:, 0:margin, :] = 0
    s[-margin:, :, :] = 0
    s[:, -margin:, :] = 0
    
    return s


class TC_Handler(Handler):
    
    def do_savefig(self, info):

        gui_object = info.object
        
        gui_object.scene1.anti_aliasing_frames = 16
        gui_object.scene2.anti_aliasing_frames = 16
        mlab.savefig(os.path.join(gui_object.base_path, 'snapshot1.png'), magnification=2, figure=gui_object.scene1.mayavi_scene)
        mlab.savefig(os.path.join(gui_object.base_path, 'snapshot2.png'), magnification=2, figure=gui_object.scene2.mayavi_scene)

    def do_savemovie(self, info):
        
        gui_object = info.object
        
        scene1 = gui_object.scene1
        scene2 = gui_object.scene2
        scene1.anti_aliasing_frames = 16
        scene2.anti_aliasing_frames = 16
        
        ANGLE_STEP = 10
        for i in range(int(360/ANGLE_STEP)):
            scene1.camera.azimuth(ANGLE_STEP)
            scene1.scene.render()
            mlab.savefig(
                os.path.join(
                    gui_object.base_path,
                    'snapshot1_%d.png' % i
                ),
                magnification=2,
                figure=scene1.mayavi_scene
            )
            scene2.scene.render()
            mlab.savefig(
                os.path.join(
                    gui_object.base_path,
                    'snapshot2_%d.png' % i
                ),
                magnification=2,
                figure=scene2.mayavi_scene
            )

        
class Visualization(HasTraits):
    scene1 = Instance(MlabSceneModel, ())
    scene2 = Instance(MlabSceneModel, ())
    scene3 = Instance(MlabSceneModel, ())
    scene4 = Instance(MlabSceneModel, ())
    scene5 = Instance(MlabSceneModel, ())
    visualization_mode = Enum('iso-surfaces', 'cross-planes')
    objective_plot = Instance(Plot)
    tr_DND = List(Instance(File))
    mse_margin = Range(0, 20, 0)
    crop_TOA = Range(0, 20, 0)
    save_button = Action(name = "Save Fig", action = "do_savefig")
    movie_button = Action(name = "Save Movie", action = "do_savemovie")
    mse = Float( 0.0, desc='MSE of reconstruction (after removing margin from the sides)' )
    max_range_set = Float( 0.0, desc='Max range of colorbar manually set.' )
    max_range_calc = Float( 0.0, desc='Max range of colorbar automatically calculated.' )
    mass1 = Float( 0.0, desc='Mass of original density (sum of density over grid)' )
    mass2 = Float( 0.0, desc='Mass of reconstructed density (sum of density over grid)' )
    mass_ratio = Float( 0.0, desc='Ratio between reconstructed and original mass' )
    
    # the layout of the dialog created
    view = View(
        VGroup(
            HGroup(
                Item('scene1',
                     editor=SceneEditor(scene_class=MayaviScene),
                     height=250,
                     width=300,
                     show_label=False
                     ),
                Item('scene2',
                     editor=SceneEditor(),
                     height=250,
                     width=300,
                     show_label=False
                     ),
                Item('objective_plot', editor=ComponentEditor(), show_label=False),
                ),
            HGroup(
                Item('scene3',
                     editor=SceneEditor(),
                     height=250,
                     width=300,
                     show_label=False
                     ),
                Item('scene4',
                     editor=SceneEditor(),
                     height=250,
                     width=300,
                     show_label=False
                     ),
                Item('scene5',
                     editor=SceneEditor(),
                     height=250,
                     width=300,
                     show_label=False
                     ),
                ),
            '_',
            Item('tr_DND', label='Drag radiance mat here', editor=DropEditor()),
            Item(
                name='visualization_mode',
                style='custom',
                editor=EnumEditor(
                    values={
                        'iso-surfaces' : '1:iso-surfaces',
                        'cross-planes' : '2:cross-planes'
                    }
                )
                ),
            HGroup(
                Item('mse_margin', width=400, label='Error Margin'),
                Item('mse', label='MSE', style='readonly'),
                spring,
                Item('crop_TOA', width=400, label='Crop excess TOA rows'),
                ),
            HGroup(
                Item('max_range_set', label='MSE'),
                Item('max_range_calc', label='MSE', style='readonly'),
                spring,
                ),
            HGroup(
                Item('mass1', label='Original Mass', width=300, style='readonly'),
                Item('mass2', label='Reconstructed Mass', width=300,style='readonly'),
                Item('mass_ratio', label='Mass ratio', width=300,style='readonly'),                
                spring,
                ),
        ),
        handler=TC_Handler(),
        buttons = [save_button, movie_button],
        resizable = True
    )

    def __init__(self):
        super(Visualization, self).__init__()
        
        self.plotdata = ArrayPlotData(x=np.arange(100), y=np.zeros(100))
        self.objective_plot = Plot(self.plotdata, resizable="h")
        plot = self.objective_plot.plot(("x", "y"))
        self.objective_plot.overlays.append(PlotLabel("Log of Objective",
                                      component=self.objective_plot,
                                      font = "swiss 16",
                                      overlay_position="top"))        
        
    @on_trait_change('visualization_mode, crop_TOA, max_range_set')
    def _updatePlot(self):
        self.plotdata.set_data('x', np.arange(self.objective.size))
        self.plotdata.set_data('y', np.log(self.objective))
        
        abs_err = np.abs(self.radiance1 - self.radiance2)
        rel_err = abs_err / (self.radiance1 + amitibo.eps(self.radiance1))
        laplacian = np.abs(atmotomo.weighted_laplace(self.radiance2, weights=(1, 1, 0.1)))
        color_bars = []
        
        if self.crop_TOA > 0:
            Y = self.Y[:, :, :-self.crop_TOA]
            X = self.X[:, :, :-self.crop_TOA]
            Z = self.Z[:, :, :-self.crop_TOA]
        else:
            Y = self.Y
            X = self.X
            Z = self.Z
        
        #
        # Positions of the cameras
        #
        camera_grid_size = (5, 5)
        width = 50000
        cameras_X, cameras_Y = np.meshgrid(
            np.linspace(0, width, camera_grid_size[0]+2)[1:-1],
            np.linspace(0, width, camera_grid_size[1]+2)[1:-1]
            )
        cameras_Z = np.zeros_like(cameras_X)
        
        for radiance, scene in zip(
            (self.radiance1, self.radiance2, abs_err, rel_err, laplacian),
            (self.scene1, self.scene2, self.scene3, self.scene4, self.scene5)
            ):
            if self.crop_TOA > 0:
                radiance = radiance[:, :, :-self.crop_TOA]
            
            mlab.clf(figure=scene.mayavi_scene)

            #
            # Add the cameras
            # Note:
            # Incase you don't see the cameras, then it most probable that the coordinates or scale factor is wrong
            #
            mlab.points3d(cameras_X.ravel(), cameras_Y.ravel(), cameras_Z.ravel(), colormap="copper", scale_factor=1000.)
            
            src = scene.mlab.pipeline.scalar_field(
                Y,
                X,
                Z,
                radiance/1e6,
                figure=scene.mayavi_scene
            )
            
            if self.visualization_mode == 'iso-surfaces':
                iso = scene.mlab.pipeline.iso_surface(src, figure=scene.mayavi_scene)
                
                iso.contour.number_of_contours = 10
                iso.actor.property.opacity = 0.2
                
            else:
                ipw_x = scene.mlab.pipeline.image_plane_widget(src, plane_orientation='x_axes', figure=scene.mayavi_scene)
                ipw_x.ipw.reslice_interpolate = 'linear'
                ipw_x.ipw.texture_interpolate = False
                ipw_y = scene.mlab.pipeline.image_plane_widget(src, plane_orientation='y_axes', figure=scene.mayavi_scene)
                ipw_y.ipw.reslice_interpolate = 'linear'
                ipw_y.ipw.texture_interpolate = False
                ipw_z = scene.mlab.pipeline.image_plane_widget(src, plane_orientation='z_axes', figure=scene.mayavi_scene)
                ipw_z.ipw.reslice_interpolate = 'linear'
                ipw_z.ipw.texture_interpolate = False
                
            scene.scene.background = (1.0, 1.0, 1.0)
            scene.scene.foreground = (0.0, 0.0, 0.0)
            color_bar = scene.mlab.colorbar(title='Density', label_fmt='%.1f', orientation='vertical', nb_labels=5)
            color_bar.scalar_bar_representation.position = np.array([0.88, 0.12])
            color_bar.scalar_bar_representation.position2 = np.array([0.1, 0.72])
            color_bars.append(color_bar)
            scene.mlab.outline(color=(0, 0, 0), line_width=2.0, extent=(0, 50000, 0, 50000, 0, 10000))
            #scene.mlab.outline(color=(0, 0, 0), line_width=2.0, extent=self.limits)
            #scene.mlab.axes(ranges=(0, 50, 0, 50, 0, 10), extent=self.limits)
 
            #
            # Camera used for opex2013. Images were saved when the guid was enlarged to the
            # screen at home.
            #
            scene.scene.camera.position = [-17803.911400310248, 117225.67286360709, 51554.720463531274]
            scene.scene.camera.focal_point = [23859.570823474398, 22883.951370260442, 2934.7398505589249]
            scene.scene.camera.view_angle = 30.0
            scene.scene.camera.view_up = [0.19117384324776399, -0.38162881313193109, 0.9043295918222477]
            scene.scene.camera.clipping_range = [47098.475474809886, 193522.77246003647]

            scene.scene.camera.compute_view_plane_normal()

        #
        # Set all color bars to the same value
        #
        self.max_range_calc = np.ceil(max((color_bars[0].lut.range[1], color_bars[2].lut.range[1])))
        for color_bar, scene in zip(
            color_bars[:3],
            (self.scene1, self.scene2, self.scene3)
            ):
            color_bar.lut.range = np.array((0.0, self.max_range_set))
            mlab.draw(figure=scene.mayavi_scene)
        
        mlab.sync_camera(reference_figure=self.scene1.mayavi_scene, target_figure=self.scene2.mayavi_scene)
        
        if self.crop_TOA > 0:
            self.mass1 = self.radiance1[:, :, :-self.crop_TOA].sum()
            self.mass2 = self.radiance2[:, :, :-self.crop_TOA].sum()
        else:
            self.mass1 = self.radiance1.sum()
            self.mass2 = self.radiance2.sum()

        if self.mass1 == 0.0:
            self.mass_ratio = 0.0
        else:
            self.mass_ratio = self.mass2 / self.mass1
            
    @on_trait_change('mse_margin, crop_TOA')
    def _updateMSE(self):
        orig1 = zeroBorders(self.radiance1, margin=self.mse_margin)
        estim1 = zeroBorders(self.radiance2, margin=self.mse_margin)
        
        if self.crop_TOA > 0:
            orig1 = orig1[:, :, :-self.crop_TOA]
            estim1 = estim1[:, :, :-self.crop_TOA]

        self.mse = mse(orig1, estim1)        
        
    @on_trait_change('tr_DND')
    def _updateDragNDrop(self):
        path = self.tr_DND[0].absolute_path
        self.base_path = os.path.split(path)[0]
        
        data = sio.loadmat(path)
        data_keys = [key for key in data.keys() if not key.startswith('__')]
    
        if len(data_keys) == 0:
            raise Exception('No matrix found in data. Available keys: %d', data.keys())
        
        #
        # The ratio is calculated as 1 / visibility / k_aerosols = 1 / 50[km] / 0.00072 [um**2]
        # This comes from out use of A:
        # exp(-A / visibility * length) = exp(-k * N * length)
        #
        self.radiance1 = atmotomo.fixmat(data['true'])
        self.radiance2 = atmotomo.fixmat(data['estimated'])
        self.objective = atmotomo.fixmat(data['objective']).ravel()
        
        if data.has_key('limits'):
            self.Y = atmotomo.fixmat(data['Y'])
            self.X = atmotomo.fixmat(data['X'])
            self.Z = atmotomo.fixmat(data['Z'])
            self.limits = data['limits'].ravel()
        else:
            shape = self.radiance1.shape
            self.Y, self.X, self.Z = np.mgrid[0:50000:complex(shape[0]), 0:50000:complex(shape[1]), 0:10000:complex(shape[2])]
            self.limits = (0, 50000, 0, 50000, 0, 10000)
        
        #
        # Center the girds
        #
        self.Y += (self.Y[1, 0, 0] - self.Y[0, 0, 0]) / 2
        self.X += (self.X[0, 1, 0] - self.X[0, 0, 0]) / 2
        self.Z += (self.Y[0, 0, 1] - self.Y[0, 0, 0]) / 2
        
        self._updatePlot()
        self._updateMSE()

def main():
    """Main function"""

    app = Visualization()
    app.configure_traits()


if __name__ == '__main__':
    main()
