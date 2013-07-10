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
from traits.api import HasTraits, Range, List, Instance, on_trait_change
from traitsui.api import View, Item, HGroup, VGroup, DropEditor
from tvtk.pyface.scene_editor import SceneEditor
from enthought.chaco.api import Plot, ArrayPlotData, PlotAxis, VPlotContainer, Legend, PlotLabel
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene
from enthought.enable.component_editor import ComponentEditor
from enthought.io.api import File
from mayavi import mlab


class Visualization(HasTraits):
    scene1 = Instance(MlabSceneModel, ())
    scene2 = Instance(MlabSceneModel, ())
    objective_plot = Instance(Plot)
    tr_DND = List(Instance(File))

    # the layout of the dialog created
    view = View(
        VGroup(
            HGroup(
                Item('scene1',
                     editor=SceneEditor(),
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
            '_',
            Item('tr_DND', label='Drag radiance mat here', editor=DropEditor()),
            ),
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
        
    def _updatePlot(self):
        self.plotdata.set_data('x', np.arange(self.objective.size))
        self.plotdata.set_data('y', np.log(self.objective))
        
        for radiance, scene, trans_flag in zip((self.radiance1, self.radiance2), (self.scene1, self.scene2), (False, False)):
            mlab.clf(figure=scene.mayavi_scene)

            if trans_flag:
                radiance = np.transpose(radiance)
                
            src = scene.mlab.pipeline.scalar_field(self.Y, self.X, self.Z, radiance, figure=scene.mayavi_scene)
            src.update_image_data = True    
            ipw_x = scene.mlab.pipeline.image_plane_widget(src, plane_orientation='x_axes', figure=scene.mayavi_scene)
            ipw_x.ipw.reslice_interpolate = 'linear'
            ipw_x.ipw.texture_interpolate = False
            ipw_y = scene.mlab.pipeline.image_plane_widget(src, plane_orientation='y_axes', figure=scene.mayavi_scene)
            ipw_y.ipw.reslice_interpolate = 'linear'
            ipw_y.ipw.texture_interpolate = False
            ipw_z = scene.mlab.pipeline.image_plane_widget(src, plane_orientation='z_axes', figure=scene.mayavi_scene)
            ipw_z.ipw.reslice_interpolate = 'linear'
            ipw_z.ipw.texture_interpolate = False
            scene.mlab.colorbar()
            scene.mlab.outline()
    
            limits = []
            for grid in (self.Y, self.X, self.Z):
                limits += [grid.min()]
                limits += [grid.max()]
            scene.mlab.axes(ranges=limits)

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
        self.Y = data['Y']
        self.X = data['X']
        self.Z = data['Z']
        self.radiance1 = data['true']
        self.radiance2 = data['estimated']
        self.objective = data['objective'].ravel()
        
        self._updatePlot()
        

def main():
    """Main function"""

    app = Visualization()
    app.configure_traits()


if __name__ == '__main__':
    main()
