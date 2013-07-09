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
from traitsui.api import View, Item, HGroup, DropEditor
from tvtk.pyface.scene_editor import SceneEditor
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene
from enthought.io.api import File
from mayavi import mlab


class Visualization(HasTraits):
    scene = Instance(MlabSceneModel, ())
    tr_DND = List(Instance(File))

    # the layout of the dialog created
    view = View(
        Item('scene',
             editor=SceneEditor(scene_class=MayaviScene),
             height=250,
             width=300,
             show_label=False
             ),
        HGroup(
            '_',
            Item('tr_DND', label='Drag radiance mat here', editor=DropEditor()),
            ),
    )

    def __init__(self):
        super(Visualization, self).__init__()
        
        #x, y, z, t = curve(self.meridional, self.transverse)
        #self.plot = self.scene.mlab.plot3d(x, y, z, t, colormap='Spectral')

    def _updatePlot(self):
        ratio = 1e15 / 50 / 0.00072 / 1e15 
        radiance = self.radiance * ratio
        shape= radiance.shape
        X, Y, Z = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
        
        mlab.clf(figure=self.scene.mayavi_scene)
        src = self.scene.mlab.pipeline.scalar_field(X, Y, Z, radiance)
        src.spacing = [1, 1, 1]
        src.update_image_data = True    
        ipw_x = self.scene.mlab.pipeline.image_plane_widget(src, plane_orientation='x_axes')
        ipw_x.ipw.reslice_interpolate = 'linear'
        ipw_x.ipw.texture_interpolate = False
        ipw_y = self.scene.mlab.pipeline.image_plane_widget(src, plane_orientation='y_axes')
        ipw_y.ipw.reslice_interpolate = 'linear'
        ipw_y.ipw.texture_interpolate = False
        ipw_z = self.scene.mlab.pipeline.image_plane_widget(src, plane_orientation='z_axes')
        ipw_z.ipw.reslice_interpolate = 'linear'
        ipw_z.ipw.texture_interpolate = False
        self.scene.mlab.colorbar()
        self.scene.mlab.outline()

        limits = []
        for grid in (X, Y, Z):
            limits += [grid.min()]
            limits += [grid.max()]
        self.scene.mlab.axes(ranges=limits)

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
        self.radiance = data['estimated']

        self._updatePlot()
        

def main():
    """Main function"""

    app = Visualization()
    app.configure_traits()


if __name__ == '__main__':
    main()
