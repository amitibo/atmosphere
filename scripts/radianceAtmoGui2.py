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
from traits.api import HasTraits, Range, List, Instance, on_trait_change, Enum
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


def zeroBorders(s, margin=1):
    if margin == 0:
        return s
    
    t = s[margin:, :, :]
    t = t[:, margin:, :]
    t = t[:-margin, :, :]
    t = t[:, :-margin, :]
    
    return t


class TC_Handler(Handler):
    
    def do_savefig(self, info):

        gui_object = info.object
        
        path = 'C:/Users/amitibo/Desktop'
        
        gui_object.scene1.anti_aliasing_frames = 16
        gui_object.scene2.anti_aliasing_frames = 16
        mlab.sync_camera(reference_figure=gui_object.scene1.mayavi_scene, target_figure=gui_object.scene2.mayavi_scene)
        mlab.savefig(os.path.join(path, 'snapshot1.png'), magnification=2, figure=gui_object.scene1.mayavi_scene)
        mlab.savefig(os.path.join(path, 'snapshot2.png'), magnification=2, figure=gui_object.scene2.mayavi_scene)


class Visualization(HasTraits):
    scene1 = Instance(MlabSceneModel, ())
    scene2 = Instance(MlabSceneModel, ())
    scene3 = Instance(MlabSceneModel, ())
    scene4 = Instance(MlabSceneModel, ())
    visualization_mode = Enum('iso-surfaces', 'cross-planes')
    objective_plot = Instance(Plot)
    tr_DND = List(Instance(File))
    margin = Range(0, 20, 0)
    save_button = Action(name = "Save Fig", action = "do_savefig")

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
                spring,
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
            Item('margin', label='Error Margin'),
        ),
        handler=TC_Handler(),
        buttons = [save_button],
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
        
    @on_trait_change('visualization_mode, margin')
    def _updatePlot(self):
        self.plotdata.set_data('x', np.arange(self.objective.size))
        self.plotdata.set_data('y', np.log(self.objective))
        
        abs_err = np.abs(self.radiance1 - self.radiance2)
        rel_err = abs_err / (self.radiance1 + amitibo.eps(self.radiance1))
        for radiance, scene in zip(
            (self.radiance1, self.radiance2, abs_err, rel_err),
            (self.scene1, self.scene2, self.scene3, self.scene4)
            ):
            mlab.clf(figure=scene.mayavi_scene)
            #scene.mayavi_scene.on_mouse_pick(self._updateCameras)
            src = scene.mlab.pipeline.scalar_field(
                zeroBorders(self.Y, self.margin),
                zeroBorders(self.X, self.margin),
                zeroBorders(self.Z, self.margin),
                zeroBorders(radiance/1e6, self.margin),
                figure=scene.mayavi_scene
            )
            src.update_image_data = True    

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
            color_bar.scalar_bar_representation.position = np.array([0.88, 0.16])            
            scene.mlab.outline(color=(0, 0, 0), extent=self.limits)
            #scene.mlab.axes(ranges=(0, 50, 0, 50, 0, 10), extent=self.limits)

    @on_trait_change('scene1.camera')
    def _updateCameras(self, picker):
        print picker
        
    @on_trait_change('tr_DND')
    def _updateDragNDrop(self):
        path = self.tr_DND[0].absolute_path
         
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
        
        if data.has_key('limit'):
            self.Y = atmotomo.fixmat(data['Y'])
            self.X = atmotomo.fixmat(data['X'])
            self.Z = atmotomo.fixmat(data['Z'])
            self.limits = data['limits'].ravel()
        else:
            shape = self.radiance1.shape
            self.Y, self.X, self.Z = np.mgrid[0:50000:complex(shape[0]), 0:50000:complex(shape[1]), 0:10000:complex(shape[2])]
            self.limits = (0, 50000, 0, 50000, 0, 10000)
            
        
        self._updatePlot()
        

def main():
    """Main function"""

    app = Visualization()
    app.configure_traits()


if __name__ == '__main__':
    main()
