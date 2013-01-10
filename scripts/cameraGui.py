"""
Analyze a result of the atmospheric simulater.
Its input is a numpy array.
Enables scaling and applying gamma correction.
"""

from __future__ import division

from enthought.traits.api import HasTraits, Range, on_trait_change, Float, \
     List, Directory, Str, Bool, Instance, Tuple, Array
from enthought.traits.ui.api import View, Item, Handler, DropEditor, HSplit, \
     VGroup, EnumEditor, DirectoryEditor
from enthought.chaco.api import Plot, ArrayPlotData, PlotAxis, VPlotContainer
from enthought.chaco.tools.api import PanTool, ZoomTool
from enthought.enable.component_editor import ComponentEditor
from enthought.io.api import File
from mayavi import mlab
from mayavi.core.ui.api import MlabSceneModel, SceneEditor
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import amitibo
from atmotomo import Camera
import glob
import os


class resultAnalayzer(HasTraits):
    """Gui Application"""
    
    tr_scaling = Range(-5.0, 5.0, 0.0, desc='Radiance scaling logarithmic')
    tr_img_list = List()
    tr_gamma_correction = Bool()
    tr_DND = List(Instance(File))
    
    scene = Instance(MlabSceneModel, ())
    x = Range(0, 100., 50, desc='pixel coord x', enter_set=True,
              auto_set=False)
    y = Range(0, 100., 50, desc='pixel coord y', enter_set=True,
              auto_set=False)
    z = Range(0, 10., 5, desc='pixel coord z', enter_set=True,
              auto_set=False)
    r = Range(1, 5., 1, desc='radius of ball', enter_set=True, auto_set=False)
    
    # Tuple of x, y, z arrays where the field is sampled.
    points = Tuple(Array, Array, Array)

    traits_view  = View(
        HSplit(
            VGroup(
                Item('scene',
                     editor=SceneEditor(), height=250,
                     width=300),
                'x',
                'y',
                'z',
                'r',
                ),
            VGroup(
                Item('img_container', editor=ComponentEditor(), show_label=False),
                Item('tr_scaling', label='Radiance Scaling'),
                Item('tr_gamma_correction', label='Apply Gamma Correction'),
                Item('tr_DND', label='Drag Here', editor=DropEditor())
                ),
            ),
        resizable = True,
    )

    def __init__(self):
        super(resultAnalayzer, self).__init__()

        self.cam = Camera()
        self.loadParticles()
        
        #
        # Prepare all the plots.
        # ArrayPlotData - A class that holds a list of numpy arrays.
        # Plot - Represents a correlated set of data, renderers, and
        # axes in a single screen region.
        #
        self._img = np.zeros((256, 256, 3), dtype=np.uint8)
        
        self.plotdata = ArrayPlotData()
        self._updateImg()
        
        self.img_container = Plot(self.plotdata)
        self.img_container.img_plot('result_img')
        self.img_container.tools.append(PanTool(self.img_container))
        self.img_container.tools.append(ZoomTool(self.img_container))
                
    @on_trait_change('scene.activated')
    def create_scene(self):
        mlab.clf(figure=self.scene.mayavi_scene)
        Y, X, Z = self.points
        r = (Y-self.y)**2 + (X-self.x)**2 + (Z-self.z)**2
        b = np.ones_like(Y)
        b[r>self.r**2] = 0
        
        self.src = mlab.pipeline.scalar_field(Y, X, Z, b, figure=self.scene.mayavi_scene)
        ipw_x = mlab.pipeline.image_plane_widget(self.src, plane_orientation='x_axes')
        ipw_y = mlab.pipeline.image_plane_widget(self.src, plane_orientation='y_axes')
        ipw_z = mlab.pipeline.image_plane_widget(self.src, plane_orientation='z_axes')
        mlab.colorbar()
        mlab.axes()

    @on_trait_change('tr_scaling, tr_gamma_correction')
    def _updateImg(self):
        img = self._img * 10**self.tr_scaling
        if self.tr_gamma_correction:
            img**=0.4
        
        img[img<0] = 0
        img[img>255] = 255
        
        self.plotdata.set_data('result_img', img.astype(np.uint8))

    @on_trait_change('tr_DND')
    def _updateDragNDrop(self):
        path = os.path.split(self.tr_DND[0].absolute_path)[0]
        self.cam.load(path)
        Y, X, Z = np.mgrid[self.cam.atmosphere_params.cartesian_grids]
        self.points = (Y, X, Z)
        self.create_scene()
        self.update_volume()
        
    @on_trait_change('x, y, z, r')
    def update_volume(self):
        Y, X, Z = self.points
        r = (Y-self.y)**2 + (X-self.x)**2 + (Z-self.z)**2
        b = np.ones_like(Y)
        b[r>self.r**2] = 0
        self.src.mlab_source.scalars = b
        
        self._img = self.cam.calcImage(b, self.particle_params)
        self._updateImg()
        
    def _points_default(self):
        Y, X, Z = np.mgrid[0:10:0.1, 0:10:0.1, 0:10:0.1]
        return Y, X, Z

    def loadParticles(self):
        import pickle
        
        with open('misr.pkl', 'rb') as f:
            misr = pickle.load(f)
        
        particles_list = misr.keys()
        particle = misr[particles_list[0]]
        self.particle_params = amitibo.attrClass(
            k_RGB=np.array(particle['k']) / np.max(np.array(particle['k'])),#* 10**-12,
            w_RGB=particle['w'],
            g_RGB=(particle['g']),
            visibility=50
            )


def main():
    """Main function"""

    app = resultAnalayzer()
    app.configure_traits()
    
    
if __name__ == '__main__':
    main()
