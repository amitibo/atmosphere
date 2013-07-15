"""
Visualize side by side the results of MC simulations and Single Scattering model.
This GUI is useful when the MC results were used as reference to the SS scattering analysis.
Just drag and drop one of the images into the GUI.
The GUI enables browsing and comparing the results of different cameras and scaling separately
the MC and single scattering results.
"""

from __future__ import division

from traits.api import HasTraits, Range, on_trait_change, Float, List, Directory, Str, \
     Bool, Instance, DelegatesTo, Enum, Int, DelegatesTo, Array
from traitsui.api import View, Item, Handler, DropEditor, VGroup, HGroup, EnumEditor, \
     DirectoryEditor, Action, spring, UItem, Group
from chaco.api import Plot, ArrayPlotData, PlotAxis, VPlotContainer, HPlotContainer, \
     Legend, PlotLabel, ColorBar
from chaco.tools.api import LineInspector, PanTool, ZoomTool, LegendTool
from chaco.tools.cursor_tool import CursorTool, BaseCursorTool
from enable.component_editor import ComponentEditor
from pyface.api import warning
from enthought.io.api import File
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.io as sio
import numpy as np
import amitibo
import glob
import os
import re

IMG_SIZE = 32


def calcRatio(ref_img, single_img, erode=False):
    #
    # Calc a joint mask
    #
    mask = (ref_img > 0) * (single_img > 0)
    if erode:
        for i in range(3):
            mask[:, :, i] = morph.greyscale_erode(mask[:, :, i].astype(np.uint8) , morph.disk(1))
        mask = mask>0
    
    ratio = ref_img[mask].mean() / single_img[mask].mean()

    return ratio


class CrossPlots(HasTraits):

    # container for all plots
    container = Instance(HPlotContainer)
    
    # plot data
    pd = Instance(ArrayPlotData)

    cross_plot_X = Instance(Plot)
    cross_plot_Y = Instance(Plot)

    cursor = Instance(BaseCursorTool)
    
    traits_view = View(
        Group(
            UItem('container', editor=ComponentEditor(size=(800,600)))
            ),
        resizable=True
    )
    
    img0 = Array()
    img1 = Array()
    
    def __init__(self, *args, **kwargs):
        super(CrossPlots, self).__init__(*args, **kwargs)
        self.createCrossPlots()
    
    def createCrossPlots(self):

        self.pd = ArrayPlotData(
            basex=np.array([]),
            img0_x=np.array([]),
            img1_x=np.array([]),
            basey=np.array([]),
            img0_y=np.array([]),
            img1_y=np.array([]),
        )

        #
        # Create the cross section plots
        #
        self.cross_plot_X = Plot(self.pd, resizable="h")
        self.cross_plot_X.height = 30
        
        plots = self.cross_plot_X.plot(("basex", "img0_x", "img1_x"))
        plots[1].line_style = 'dot'
        
        legend = Legend(component=self.cross_plot_X, padding=5, align="ur")
        legend.tools.append(LegendTool(legend, drag_button="right"))
        legend.plots = dict(zip(('MC', 'Sim'), plots))
        
        self.cross_plot_X.overlays.append(legend)
        self.cross_plot_X.overlays.append(
            PlotLabel(
                "X section",
                component=self.cross_plot_X,
                font = "swiss 16",
                overlay_position="top"
            )
        )
        
        self.cross_plot_Y = Plot(self.pd, resizable="h")
        self.cross_plot_Y.height = 30
        
        plots = self.cross_plot_Y.plot(("basey", "img0_y", "img1_y"))
        plots[1].line_style = 'dot'
        
        self.cross_plot_Y.overlays.append(
            PlotLabel(
                "Y section",
                component=self.cross_plot_Y,
                font = "swiss 16",
                overlay_position="top"
            )
        )

    @on_trait_change('cursor, img0, img1')
    def update_plots(self):
        self.pd.set_data('basex', np.arange(self.img0.shape[1]))
        self.pd.set_data('basey', np.arange(self.img0.shape[0]))
        
        for i, img in enumerate((self.img0, self.img1)):
            self.pd.set_data('img%d_x' % i, img[self.cursor.current_index[1], :])
            self.pd.set_data('img%d_y' % i, img[:, self.cursor.current_index[0]])
    
        
class resultAnalayzer(HasTraits):
    """Gui Application"""
    
    tr_scaling = Range(-14.0, 14.0, 0.0, desc='Radiance scaling logarithmic')
    tr_relative_scaling = Range(-10.0, 10.0, 0.0, desc='Relative radiance scaling logarithmic')
    tr_err=Float(0.0, desc='Error between the MC and single images')
    tr_sun_angle = Range(-0.5, 0.5, 0.0, desc='Sun angle in parts of radian')
    tr_folder = Directory()
    tr_gamma_correction = Bool()
    tr_DND = List(Instance(File))
    tr_min = Int(0)
    tr_len = Int(0)
    tr_index = Range('tr_min', 'tr_len', 0, desc='Index of image in amit list')
    save_button = Action(name = "Save Fig", action = "do_savefig")
    movie_button = Action(name = "Make Movie", action = "do_makemovie")
    cross_plots = Instance('cross_plots')
    cursor = DelegatesTo(BaseCursorTool)
    tr_channel = Enum(0, 1, 2)
    
    traits_view  = View(
        VGroup(
            HGroup(
                Item('img_container1', editor=ComponentEditor(), show_label=False),
                Item('img_container2', editor=ComponentEditor(), show_label=False),
                Item('img_container3', editor=ComponentEditor(), show_label=False),
                ),
            UItem('cross_plots'),
            Item('tr_scaling', label='Radiance Scaling'),
            HGroup(
                Item('tr_relative_scaling', width=600, label='Relative Radiance Scaling'),
                spring,
                Item('tr_err', label='Error', style='readonly')
                ),
            Item('tr_sun_angle', label='Sun angle'),
            Item('tr_gamma_correction', label='Apply Gamma Correction'),
            Item('tr_index', label='Image Index'),
            Item('tr_DND', label='Drag Image here', editor=DropEditor()),
            Item('tr_channel', label='Plot Channel', editor=EnumEditor(values={0: 'R', 1: 'G', 2: 'B'}), style='custom')
            ),
        resizable = True
    )

    def __init__(self):
        super(resultAnalayzer, self).__init__()

        self.cross_plots = CrossPlots()
        
        #
        # Prepare all the plots.
        # ArrayPlotData - A class that holds a list of numpy arrays.
        # Plot - Represents a correlated set of data, renderers, and
        # axes in a single screen region.
        #
        self._ref_images = [np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)]
        self._sim_images = [np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)]
        
        self.plotdata = ArrayPlotData(
            result_img0=self._ref_images[0],
            result_img1=self._sim_images[0],
            result_img2=np.zeros((IMG_SIZE, IMG_SIZE))
        )
        
        #
        # Create image containers
        #
        self.img_container1 = Plot(self.plotdata)
        img_plot = self.img_container1.img_plot('result_img0')[0]
        self.img_container1.overlays.append(
            PlotLabel(
                "Monte-Carlo",
                component=self.img_container1,
                font = "swiss 16",
                overlay_position="top"
            )
        )

        self.img_container2 = Plot(self.plotdata)
        self.img_container2.img_plot('result_img1')
        self.img_container2.overlays.append(
            PlotLabel(
                "Single-Scattering Simulation",
                component=self.img_container2,
                font = "swiss 16",
                overlay_position="top"
            )
        )
        
        self.img_container3 = Plot(self.plotdata)
        self.img_container3.img_plot('result_img2')
        self.img_container3.overlays.append(
            PlotLabel(
                "Single-Scattering Reconstruction",
                component=self.img_container3,
                font = "swiss 16",
                overlay_position="top"
            )
        )        
        
        #
        # Create the cursor
        #
        self.cursor = CursorTool(
            component=img_plot,
            drag_button='left',
            color='white',
            line_width=1.0
        )                
        img_plot.overlays.append(self.cursor)
        self.cursor.current_position = 1, 1

        #
        # Initialize the images
        #
        self._updateImg()

    @on_trait_change('tr_scaling, tr_relative_scaling, tr_gamma_correction, tr_channel, cursor.current_index, tr_index')
    def _updateImg(self):
        
        h, w, d = self._ref_images[self.tr_index].shape
        
        self.cross_plots.img0 = self._ref_images[self.tr_index][:, :, self.tr_channel]
        self.cross_plots.img1 = self._sim_images[self.tr_index][:, :, self.tr_channel]
        
        err = 0
        err_img = []
        for i, (ref_img, sim_img) in enumerate(zip(self._ref_images, self._sim_images)):
            ref_img = self._scaleImg(ref_img)
            sim_img = self._scaleImg(sim_img, self.tr_relative_scaling)
            
            if i == self.tr_index:
                self._showImgGraph(0, ref_img)                
                self._showImgGraph(1, sim_img)

            err += calcRatio(ref_img, sim_img)
            err_img.append(ref_img[:, :, self.tr_channel] - sim_img[:, :, self.tr_channel])
            
        self.tr_err = err/len(self._ref_images)
        std_img = np.dstack(err_img).std(axis=2)
        self.plotdata.set_data('result_img2', std_img)
        
    def _showImgGraph(self, i, img):
        img_croped = cropImg(img)        
        self.plotdata.set_data('result_img%d' % i, img_croped)            

    def _scaleImg(self, img, relative_scale=0.0):
        img = img * 10**(self.tr_scaling+relative_scale)
        if self.tr_gamma_correction:
            img**=0.4
            
        return img
            
    @on_trait_change('tr_DND')
    def _updateDragNDrop(self):
        path = self.tr_DND[0].absolute_path
         
        path, file_name =  os.path.split(path)
        ref_list = glob.glob(os.path.join(path, "ref_img*.mat"))
        if not ref_list:
            warning(info.ui.control, "No img found in the folder", "Warning")
            return
        sim_list = glob.glob(os.path.join(path, "sim_img*.mat"))
        
        self._ref_images = []
        self._sim_images = []
        
        for ref_path, sim_path in zip(sorted(ref_list), sorted(sim_list)):
            data = sio.loadmat(ref_path)
            self._ref_images.append(data['img'])
            data = sio.loadmat(sim_path)
            self._sim_images.append(data['img'])

        self.tr_len = len(self._ref_images) - 1

        self._updateImg()

def cropImg(img):
    img_croped = img.copy()
    img_croped[img<0] = 0
    img_croped[img>255] = 255
    return img_croped.astype(np.uint8)
        
        
def main():
    """Main function"""

    app = resultAnalayzer()
    app.configure_traits()
    
    
if __name__ == '__main__':
    main()
