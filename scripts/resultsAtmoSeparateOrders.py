"""
Visualize side by side the results of MC simulations and Single Scattering model.
This GUI is useful when the MC and SS results are in separate folders.
In the case of the MC, drag and drop one of the results folder, in the case of the single scattering
drag and drop one of the images (image matrix).
The GUI enables browsing and comparing the results of different cameras and scaling separately
the MC and single scattering results.
"""

from __future__ import division

from enthought.traits.api import HasTraits, Range, on_trait_change, Float, List, Directory, Str, Bool, Instance, DelegatesTo, Enum, Int
from enthought.traits.ui.api import View, Item, Handler, DropEditor, VGroup, HGroup, EnumEditor, DirectoryEditor, Action
from enthought.chaco.api import Plot, ArrayPlotData, PlotAxis, VPlotContainer, Legend, PlotLabel
from enthought.chaco.tools.api import LineInspector, PanTool, ZoomTool, LegendTool
from chaco.tools.cursor_tool import CursorTool, BaseCursorTool
from enthought.enable.component_editor import ComponentEditor
from enthought.pyface.api import warning
from enthought.io.api import File
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.io as sio
from atmotomo import loadVadimData
import numpy as np
import amitibo
import glob
import os
import re

IMG_SIZE = 128


class TC_Handler(Handler):

    def do_savefig(self, info):

        results_object = info.object
        
        path = 'C:/Users/amitibo/Desktop'
        
        for i in (1, 2):
            figure_name = '%d.svg' % i
            
            img = results_object.plotdata.get_data('result_img%d' % i)
        
            #
            # Draw the image
            #
            fig = plt.figure()
            ax = plt.axes([0, 0, 1, 1]) 
            plt.imshow(img.astype(np.uint8))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
            img_center = img.shape[0]/2
    
            amitibo.saveFigures(path, bbox_inches='tight', figures_names=(figure_name, ))

            #
            # Draw cross sections
            # "basex", "img1_x", "img2_x", "img3_x"
            for i, axis in enumerate(('x', 'y')):
                fig = plt.figure()
                base = results_object.plotdata.get_data('base%s' % axis)
                img1 = results_object.plotdata.get_data('img1_%s' % axis)
                img2 = results_object.plotdata.get_data('img2_%s' % axis)
                    
                ax = plt.axes([0, 0, 1, 1])
                plt.plot(
                    base, img1, 'k',
                    base, img2, 'k:',
                    linewidth=2.0
                )
                
                plt.legend(
                    ('MC', 'Single'),
                    'lower right',
                    shadow=True
                )
                plt.grid(False)
                plt.xlabel('%s Axis' % axis.upper())
                plt.ylabel('Intensity')
                plt.title('%s Cross Section' % axis.upper())
                plt.xlim(0, IMG_SIZE)
                
                fig.savefig(
                    os.path.join(
                        path,
                        'cross_section_%s.svg' % axis
                        ), 
                    format='svg'
                )

    def do_makemovie(self, info):

        results_object = info.object

        path, file_name =  os.path.split(results_object.tr_img_name)
        file_pattern = re.search(r'(.*?)\d+.mat', file_name).groups()[0]
        image_list = glob.glob(os.path.join(path, "%s*.mat" % file_pattern))
        if not image_list:
            warning(info.ui.control, "No ref_img's found in the folder", "Warning")
            return

        import matplotlib.animation as manim
        
        FFMpegWriter = manim.writers['ffmpeg']
        metadata = dict(title='Results Movie', artist='Matplotlib', comment='Movie support!')
        writer = FFMpegWriter(fps=2, bitrate=-1, metadata=metadata)

        fig = plt.figure()
        with writer.saving(fig, os.path.join(path, "%s.mp4" % file_pattern), 100):
            for i in range(0, len(image_list)+1):
                img_path = os.path.join(path, "%s%d.mat" % (file_pattern, i))
                try:
                    data = sio.loadmat(img_path)
                except:
                    continue
                
                if 'img' in data.keys():
                    img = data['img']
                else:
                    img = data['rgb']
                img = img * 10**results_object.tr_scaling
                if results_object.tr_gamma_correction:
                    img**=0.4
                
                img[img<0] = 0
                img[img>255] = 255
                
                plt.imshow(img.astype(np.uint8))
                plt.title('image %d' % i)
                writer.grab_frame()            
                fig.clear()
                

class resultAnalayzer(HasTraits):
    """Gui Application"""
    
    tr_scaling = Range(-10.0, 10.0, 0.0, desc='Radiance scaling logarithmic')
    tr_folder = Directory()
    tr_gamma_correction = Bool()
    tr_DND_first_set = List(Instance(File))
    tr_DND_second_set = List(Instance(File))
    tr_min = Int(1)
    tr_max = Int(1)
    tr_base_name = Enum(
        'L1aerosol_MATRIX.mat',
        'L1air_MATRIX.mat',
        'L1both_MATRIX.mat',
        'L2both_MATRIX.mat',
        'L3both_MATRIX.mat',
        'L4both_MATRIX.mat',
        'L5both_MATRIX.mat',
        'LNS_MATRIX.mat'
    )
    tr_index = Range(low='tr_min', high='tr_max', value=1, desc='Index of image')
    save_button = Action(name = "Save Fig", action = "do_savefig")
    movie_button = Action(name = "Make Movie", action = "do_makemovie")
    tr_cross_plot1 = Instance(Plot)
    tr_cross_plot2 = Instance(Plot)
    tr_cursor1 = Instance(BaseCursorTool)
    tr_cursor2 = Instance(BaseCursorTool)
    tr_channel = Enum(0, 1, 2)
    
    traits_view  = View(
        VGroup(
            HGroup(
                Item('img_container1', editor=ComponentEditor(), show_label=False),
                Item('img_container2', editor=ComponentEditor(), show_label=False),
                ),
            HGroup(
                Item('tr_cross_plot1', editor=ComponentEditor(), show_label=False),
                Item('tr_cross_plot2', editor=ComponentEditor(), show_label=False),
                ),
            Item('tr_scaling', label='Radiance Scaling'),
            Item('tr_gamma_correction', label='Apply Gamma Correction'),
            Item('tr_index', label='Index of images'),
            Item('tr_DND_first_set', label='Drag first set', editor=DropEditor()),
            Item('tr_DND_second_set', label='Drag second set', editor=DropEditor()),
            HGroup(
                Item('tr_channel', label='Plot Channel', editor=EnumEditor(values={0: 'R', 1: 'G', 2: 'B'}), style='custom'),
                Item('tr_base_name', label='Order')
            )
            ),
        handler=TC_Handler(),
        buttons = [save_button, movie_button],
        resizable = True
    )

    def __init__(self):
        super(resultAnalayzer, self).__init__()

        #
        # Prepare all the plots.
        # ArrayPlotData - A class that holds a list of numpy arrays.
        # Plot - Represents a correlated set of data, renderers, and
        # axes in a single screen region.
        #
        self._images_first_set = [np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)]
        self._images_second_set = [np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)]
        
        self.plotdata = ArrayPlotData(result_img1=np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8), results_img2=np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
        
        self.img_container1 = Plot(self.plotdata)
        img_plot1 = self.img_container1.img_plot('result_img1')[0]
        self.tr_cursor1 = CursorTool(
            component=img_plot1,
            drag_button='left',
            color='white',
            line_width=1.0
        )                
        img_plot1.overlays.append(self.tr_cursor1)
        self.img_container1.overlays.append(PlotLabel("First Set",
                                      component=self.img_container1,
                                      font = "swiss 16",
                                      overlay_position="top"))        
        

        self.img_container2 = Plot(self.plotdata)
        img_plot2 = self.img_container2.img_plot('result_img2')[0]
        self.tr_cursor2 = CursorTool(
            component=img_plot2,
            drag_button='left',
            color='white',
            line_width=1.0
        )                
        img_plot2.overlays.append(self.tr_cursor2)
        self.tr_cursor2.current_position = 1, 1
        self.img_container2.overlays.append(PlotLabel("Second Set",
                                      component=self.img_container2,
                                      font = "swiss 16",
                                      overlay_position="top"))        
        
        self._updateImg()

        self.tr_cross_plot1 = Plot(self.plotdata, resizable="h")
        self.tr_cross_plot1.height = 30
        plots = self.tr_cross_plot1.plot(("basex", "img1_x", "img2_x"))
        plots[1].line_style = 'dot'
        legend = Legend(component=self.tr_cross_plot1, padding=5, align="ur")
        legend.tools.append(LegendTool(legend, drag_button="right"))
        legend.plots = dict(zip(('MC', 'Single'), plots))
        self.tr_cross_plot1.overlays.append(legend)
        self.tr_cross_plot1.overlays.append(PlotLabel("X section",
                                      component=self.tr_cross_plot1,
                                      font = "swiss 16",
                                      overlay_position="top"))        
        self.tr_cross_plot2 = Plot(self.plotdata, resizable="h")
        self.tr_cross_plot2.height = 30
        plots = self.tr_cross_plot2.plot(("basey", "img1_y", "img2_y"))
        plots[1].line_style = 'dot'
        self.tr_cross_plot2.overlays.append(PlotLabel("Y section",
                                      component=self.tr_cross_plot2,
                                      font = "swiss 16",
                                      overlay_position="top"))        
        
        
    @on_trait_change('tr_scaling, tr_gamma_correction, tr_channel, tr_cursor1.current_index, tr_index')
    def _updateImg(self):
        
        if self.tr_cursor2:
            self.tr_cursor2.current_index = self.tr_cursor1.current_index
        
        h, w, d = self._images_first_set[0].shape
        
        self.plotdata.set_data('basex', np.arange(w))
        self.plotdata.set_data('basey', np.arange(h))

        index1 = min(self.tr_index, len(self._images_first_set))
        index2 = min(self.tr_index, len(self._images_second_set))
        for i, img in enumerate((self._images_first_set[index1-1], self._images_second_set[index2-1])):
            img = img * 10**self.tr_scaling * 10
            if self.tr_gamma_correction:
                img**=0.4
            
            img_croped = img.copy()
            img_croped[img<0] = 0
            img_croped[img>255] = 255
            
            self.plotdata.set_data('result_img%d' % (i+1), img_croped.astype(np.uint8))
            self.plotdata.set_data('img%d_x' % (i+1), img[self.tr_cursor1.current_index[1], :, self.tr_channel])
            self.plotdata.set_data('img%d_y' % (i+1), img[:, self.tr_cursor1.current_index[0], self.tr_channel])
            
    @on_trait_change('tr_cursor2.current_index')
    def _updateCursor(self):
        
        self.tr_cursor1.current_index = self.tr_cursor2.current_index
        
    @on_trait_change('tr_DND_first_set, tr_base_name')
    def _updateDragNDrop1(self):
        path = self.tr_DND_first_set[0].absolute_path
         
        path, folder_name =  os.path.split(path)
        self._images_first_set, dumb = loadVadimData(path, base_name=self.tr_base_name)        
        self.tr_max = min(len(self._images_first_set), len(self._images_second_set))
        
        self._updateImg()

    @on_trait_change('tr_DND_second_set, tr_base_name')
    def _updateDragNDrop2(self):
        path = self.tr_DND_second_set[0].absolute_path
         
        path, folder_name =  os.path.split(path)
        self._images_second_set, dumb = loadVadimData(path, base_name=self.tr_base_name)        
        self.tr_max = max(len(self._images_first_set), len(self._images_second_set))

        self._updateImg()
        
        
def main():
    """Main function"""

    app = resultAnalayzer()
    app.configure_traits()
    
    
if __name__ == '__main__':
    main()
