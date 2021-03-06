"""
Visualize side by side the results of MC simulations and Single Scattering model.
This GUI is useful when the MC and SS results are in separate folders.
In the case of the MC, drag and drop one of the results folder, in the case of the single scattering
drag and drop one of the images (image matrix).
The GUI enables browsing and comparing the results of different cameras and scaling separately
the MC and single scattering results.
"""

from __future__ import division

from traits.api import HasTraits, Range, on_trait_change, Float, List, Directory, Str, Bool, Instance, DelegatesTo, Enum, Int
from traitsui.api import View, Item, Handler, DropEditor, VGroup, HGroup, EnumEditor, DirectoryEditor, Action
from chaco.api import Plot, ArrayPlotData, PlotAxis, VPlotContainer, Legend, PlotLabel
from chaco.tools.api import LineInspector, PanTool, ZoomTool, LegendTool
from chaco.tools.cursor_tool import CursorTool, BaseCursorTool
from enable.component_editor import ComponentEditor
from pyface.api import warning
from io import FileIO
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.io as sio
import scipy.ndimage as ndimage
from atmotomo import loadVadimData
import numpy as np
import amitibo
import glob
import os
import re

IMG_SIZE = 64


class TC_Handler(Handler):

    def do_savefig(self, info):

        results_object = info.object
        
        sun_angle = results_object.tr_sun_angle
        
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
    
            #
            # Draw sun
            #
            sun_x = img_center * (1 + sun_angle * 2)
            sun_patch = mpatches.Circle((sun_x, img_center), 2, ec='y', fc='y')
            ax.add_patch(sun_patch)
    
            #
            # Draw angle arcs
            #
            for arc_angle in range(0, 90, 30)[1:]:
                d = img_center * arc_angle / 90
                arc_patch = mpatches.Arc(
                    (img_center, img_center),
                    2*d,
                    2*d,
                    90,
                    25,
                    335,
                    ec='w',
                    ls='dashed',
                    lw=4
                )
                ax.add_patch(arc_patch)
                plt.text(
                    img_center,
                    img_center+d,
                    "$%s^{\circ}$" % str(arc_angle),
                    ha="center",
                    va="center",
                    size=30,
                    color='w'
                )
        
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
    tr_relative_scaling = Range(-15.0, 15.0, 0.0, desc='Relative radiance scaling logarithmic')
    tr_sun_angle = Range(-0.5, 0.5, 0.0, desc='Sun angle in parts of radian')
    tr_folder = Directory()
    tr_gamma_correction = Bool()
    tr_DND_vadim = List(Instance(FileIO))
    tr_DND_amit = List(Instance(FileIO))
    tr_min = Int(0)
    tr_vadim_len = Int(0)
    tr_amit_len = Int(0)
    tr_vadim_index = Range('tr_min', 'tr_vadim_len', 0, desc='Index of image in vadim list')
    tr_amit_index = Range('tr_min', 'tr_amit_len', 0, desc='Index of image in amit list')
    tr_sigma = Range(0.0, 3.0, 0.0, desc='Blur sigma of MC image')
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
            Item('tr_relative_scaling', label='Relative Radiance Scaling'),
            Item('tr_sun_angle', label='Sun angle'),
            Item('tr_gamma_correction', label='Apply Gamma Correction'),
            Item('tr_vadim_index', label='Index of Vadim'),
            Item('tr_amit_index', label='Index of Amit'),
            Item('tr_sigma', label='Blur Sigma'),
            Item('tr_DND_vadim', label='Drag Vadim', editor=DropEditor()),
            Item('tr_DND_amit', label='Drag Amit', editor=DropEditor()),
            Item('tr_channel', label='Plot Channel', editor=EnumEditor(values={0: 'R', 1: 'G', 2: 'B'}), style='custom')
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
        self._images_amit = [np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)]
        self._images_vadim = [np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)]
        
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
        self.img_container1.overlays.append(PlotLabel("Monte-Carlo Simulation",
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
        self.img_container2.overlays.append(PlotLabel("Single-Scattering",
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
        
        
    @on_trait_change('tr_sigma, tr_scaling, tr_relative_scaling, tr_gamma_correction, tr_channel, tr_cursor1.current_index, tr_vadim_index, tr_amit_index')
    def _updateImg(self):
        
        if self.tr_cursor2:
            self.tr_cursor2.current_index = self.tr_cursor1.current_index
        
        relative_scaling = [0, self.tr_relative_scaling]
        h, w, d = self._images_amit[self.tr_amit_index].shape
        
        self.plotdata.set_data('basex', np.arange(w))
        self.plotdata.set_data('basey', np.arange(h))

        for i, img in enumerate((self._images_vadim[self.tr_vadim_index], self._images_amit[self.tr_amit_index])):
            img = img * 10**self.tr_scaling * 10**relative_scaling[i]
            if self.tr_gamma_correction:
                img**=0.4
            
            if i == 0 and self.tr_sigma > 0:
                for j in range(3):
                    img[:, :, j] = ndimage.filters.gaussian_filter(img[:, :, j], sigma=self.tr_sigma)
                
            img_croped = img.copy()
            img_croped[img<0] = 0
            img_croped[img>255] = 255
            
            self.plotdata.set_data('result_img%d' % (i+1), img_croped.astype(np.uint8))
            self.plotdata.set_data('img%d_x' % (i+1), img[self.tr_cursor1.current_index[1], :, self.tr_channel])
            self.plotdata.set_data('img%d_y' % (i+1), img[:, self.tr_cursor1.current_index[0], self.tr_channel])
            
    @on_trait_change('tr_cursor2.current_index')
    def _updateCursor(self):
        
        self.tr_cursor1.current_index = self.tr_cursor2.current_index
        
    @on_trait_change('tr_DND_vadim')
    def _updateDragNDrop1(self):
        path = self.tr_DND_vadim[0].absolute_path
         
        path, folder_name =  os.path.split(path)
        self._images_vadim, dumb = loadVadimData(path)        
        self.tr_vadim_len = len(self._images_vadim) - 1
        
        self._updateImg()

    @on_trait_change('tr_DND_amit')
    def _updateDragNDrop2(self):
        path = self.tr_DND_amit[0].absolute_path
         
        path, file_name =  os.path.split(path)
        file_pattern = re.search(r'(.*?)\d+.mat', file_name).groups()[0]
        image_list = glob.glob(os.path.join(path, "%s*.mat" % file_pattern))
        if not image_list:
            warning(info.ui.control, "No img found in the folder", "Warning")
            return
        
        self._images_amit = []
        for img_path in sorted(image_list):
            try:
                data = sio.loadmat(img_path)
            except:
                continue
            
            self._images_amit.append(data['img'])

        self.tr_amit_len = len(self._images_amit) - 1

        self._updateImg()
        
        
def main():
    """Main function"""

    app = resultAnalayzer()
    app.configure_traits()
    
    
if __name__ == '__main__':
    main()
