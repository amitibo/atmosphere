"""
Analyze a result of the atmospheric simulater.
Its input is a numpy array.
Enables scaling and applying gamma correction.
"""

from __future__ import division

from enthought.traits.api import HasTraits, Range, on_trait_change, Float, List, Directory, Str, Bool, Instance
from enthought.traits.ui.api import View, Item, Handler, DropEditor, VGroup, EnumEditor, DirectoryEditor
from enthought.chaco.api import Plot, ArrayPlotData, PlotAxis, VPlotContainer
from enthought.chaco.tools.api import PanTool, ZoomTool
from enthought.enable.component_editor import ComponentEditor
from enthought.io.api import File
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import amitibo
import glob
import os


class resultAnalayzer(HasTraits):
    """Gui Application"""
    
    tr_scaling = Range(-5.0, 5.0, 0.0, desc='Radiance scaling logarithmic')
    tr_folder = Directory()
    tr_img_list = List()
    tr_img_name = Str()
    tr_gamma_correction = Bool()
    tr_DND = List(Instance(File))
    
    traits_view  = View(
        VGroup(
            Item('img_container', editor=ComponentEditor(), show_label=False),
            Item('tr_scaling', label='Radiance Scaling'),
            Item('tr_gamma_correction', label='Apply Gamma Correction'),
            Item('tr_folder', label='Result Folder', editor=DirectoryEditor()),
            Item('tr_img_name', label='Images List', editor=EnumEditor(name="tr_img_list")),
            Item('tr_DND', label='Drag Here', editor=DropEditor())
            ),
            resizable = True,
        )

    def __init__(self):
        super(resultAnalayzer, self).__init__()

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
                
    @on_trait_change('tr_scaling, tr_gamma_correction')
    def _updateImg(self):
        img = self._img * 10**self.tr_scaling
        if self.tr_gamma_correction:
            img**=0.4
        
        img[img<0] = 0
        img[img>255] = 255
        
        self.plotdata.set_data('result_img', img.astype(np.uint8))

    @on_trait_change('tr_img_name')
    def _updateImgName(self):
        if os.path.exists(self.tr_img_name):
            data = sio.loadmat(self.tr_img_name)
            self._img = data['img']
        else:
            self._img = np.zeros((256, 256, 3), dtype=np.uint8)
            
        self._updateImg()
    
    @on_trait_change('tr_folder')
    def _updateFolder(self):
        
        self.tr_img_list = glob.glob(os.path.join(self.tr_folder, "*.mat"))
                
        self.tr_img_name = ''
        
    @on_trait_change('tr_DND')
    def _updateDragNDrop(self):
        self.tr_img_name = self.tr_DND[0].absolute_path
 
        
def main():
    """Main function"""

    app = resultAnalayzer()
    app.configure_traits()
    
    
if __name__ == '__main__':
    main()
