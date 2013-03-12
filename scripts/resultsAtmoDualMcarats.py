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
import scipy.ndimage as ndimage
import scipy.stats as stats
import numpy as np
import amitibo
import glob
import os
import re


class TC_Handler(Handler):

    def do_savefig(self, info):

        results_object = info.object
        
        sun_angle = results_object.tr_sun_angle
        
        path, file_name =  os.path.split(results_object.tr_img_name)
        figure_name, dump = os.path.splitext(file_name)
        figure_name += '.svg'
        
        img = results_object._img * 10**results_object.tr_scaling
        if results_object.tr_gamma_correction:
            img**=0.4
        
        img[img<0] = 0
        img[img>255] = 255
    
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
    tr_relative_scaling = Range(-10.0, 10.0, 0.0, desc='Relative radiance scaling logarithmic')
    tr_sun_angle = Range(-0.5, 0.5, 0.0, desc='Sun angle in parts of radian')
    tr_folder = Directory()
    tr_gamma_correction = Bool()
    tr_DND_mcarats = List(Instance(File))
    tr_DND_amit = List(Instance(File))
    tr_min = Int(0)
    tr_mcarats_len = Int(0)
    tr_amit_len = Int(0)
    tr_mcarats_index = Range('tr_min', 'tr_mcarats_len', 0, desc='Index of image in Mcarats list')
    tr_amit_index = Range('tr_min', 'tr_amit_len', 0, desc='Index of image in amit list')
    tr_sigma = Range(0.0, 3.0, 0.0, desc='Blur sigma of MC image')
    save_button = Action(name = "Save Fig", action = "do_savefig")
    movie_button = Action(name = "Make Movie", action = "do_makemovie")
    tr_cross_plot1 = Instance(Plot)
    tr_cross_plot2 = Instance(Plot)
    tr_cursor1 = Instance(BaseCursorTool)
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
            Item('tr_mcarats_index', label='Index of mcarats'),
            Item('tr_amit_index', label='Index of Amit'),
            Item('tr_sigma', label='Blur Sigma'),
            Item('tr_DND_mcarats', label='Drag mcarats', editor=DropEditor()),
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
        self._images_amit = [np.zeros((128, 128, 3), dtype=np.uint8)]
        self._images_mcarats = [np.zeros((128, 128, 3), dtype=np.uint8)]
        
        self.plotdata = ArrayPlotData(result_img1=np.zeros((128, 128, 3), dtype=np.uint8), results_img2=np.zeros((128, 128, 3), dtype=np.uint8))
        
        self.img_container1 = Plot(self.plotdata)
        img_plot = self.img_container1.img_plot('result_img1')[0]
        self.tr_cursor1 = CursorTool(
            component=img_plot,
            drag_button='left',
            color='white',
            line_width=1.0
        )                
        img_plot.overlays.append(self.tr_cursor1)
        self.tr_cursor1.current_position = 64, 64
        self.img_container1.overlays.append(PlotLabel("Monte-Carlo Simulation",
                                      component=self.img_container1,
                                      font = "swiss 16",
                                      overlay_position="top"))        
        

        self.img_container2 = Plot(self.plotdata)
        self.img_container2.img_plot('result_img2')
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
        
        
    @on_trait_change('tr_sigma, tr_scaling, tr_relative_scaling, tr_gamma_correction, tr_channel, tr_cursor1.current_index, tr_mcarats_index, tr_amit_index')
    def _updateImg(self):
        
        relative_scaling = [0, self.tr_relative_scaling]
        h, w, d = self._images_amit[self.tr_amit_index].shape
        
        self.plotdata.set_data('basex', np.arange(w))
        self.plotdata.set_data('basey', np.arange(h))

        for i, img in enumerate((self._images_mcarats[self.tr_mcarats_index], self._images_amit[self.tr_amit_index])):
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
    
    def calcRmax(self, x, fmax=1.2, FACMIN=1.3):
        xmn = x.mean()
        xst = x.std()
        
        return max(xmn*FACMIN, xmn + xst*fmax)
    
    def removeSunSpot(self, ch, ys, xs, MARGIN=2):
        ymin = ys.min()-MARGIN
        ymax = ys.max()+MARGIN
        xmin = xs.min()-MARGIN
        xmax = xs.max()+MARGIN
        
        ch_part = ch[ymin:ymax, xmin:xmax].copy()
        ch_part[ys-ymin, xs-xmin] = np.nan

        ch[ymin:ymax, xmin:xmax] = np.mean(stats.nanmean(ch_part))
        
        return ch
    
    def calcMcaratsImg(self, R_ch, G_ch, B_ch, slc, IMG_SHAPE):
        R, G, B = [ch[slc].reshape(IMG_SHAPE) for ch in (R_ch, G_ch, B_ch)]
        Rmax = self.calcRmax(R)
        ys, xs = np.nonzero(R>Rmax)
        
        R, G, B = [self.removeSunSpot(ch, ys, xs) for ch in (R, G, B)]
                   
        img = np.dstack((R, G, B))
        
        return img
    
    @on_trait_change('tr_DND_mcarats')
    def _updateDragNDrop1(self):
        path = self.tr_DND_mcarats[0].absolute_path

        R_ch, G_ch, B_ch = [np.fromfile(os.path.join(path, 'base%d_conf_out' % i), dtype=np.float32) for i in range(3)]
        IMG_SHAPE = (512, 512)
        IMG_SIZE = IMG_SHAPE[0] * IMG_SHAPE[1]
        
        self._images_mcarats = []
        for i in range(int(R_ch.size / IMG_SIZE)):
            slc = slice(i*IMG_SIZE, (i+1)*IMG_SIZE)
            self._images_mcarats.append(self.calcMcaratsImg(R_ch, G_ch, B_ch, slc, IMG_SHAPE))

        self.tr_mcarats_len = len(self._images_mcarats) - 1
        
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
        for i in range(0, len(image_list)+1):
            img_path = os.path.join(path, "%s%d.mat" % (file_pattern, i))
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
