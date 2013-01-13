"""
Analyze a result of the atmospheric simulater.
Its input is a numpy array.
Enables scaling and applying gamma correction.
"""

from __future__ import division

from enthought.traits.api import HasTraits, Range, on_trait_change, Float, List, Directory, Str, Bool, Instance
from enthought.traits.ui.api import View, Item, Handler, DropEditor, VGroup, EnumEditor, DirectoryEditor, Action
from enthought.chaco.api import Plot, ArrayPlotData, PlotAxis, VPlotContainer
from enthought.chaco.tools.api import PanTool, ZoomTool
from enthought.enable.component_editor import ComponentEditor
from enthought.pyface.api import warning
from enthought.io.api import File
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.io as sio
import numpy as np
import amitibo
import glob
import os


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
        image_list = glob.glob(os.path.join(path, "ref_img*.mat"))
        if not image_list:
            warning(info.ui.control, "No ref_img's found in the folder", "Warning")
            return

        import matplotlib.animation as manim
        
        FFMpegWriter = manim.writers['ffmpeg']
        metadata = dict(title='Results Movie', artist='Matplotlib', comment='Movie support!')
        writer = FFMpegWriter(fps=5, bitrate=-1, metadata=metadata)

        fig = plt.figure()
        with writer.saving(fig, os.path.join(path, "ref_images.mp4"), 100):
            for i in range(1, len(image_list)+1):
                img_path = os.path.join(path, "ref_img%d.mat" % i)
                data = sio.loadmat(img_path)
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
    tr_sun_angle = Range(-0.5, 0.5, 0.0, desc='Sun angle in parts of radian')
    tr_folder = Directory()
    tr_img_list = List()
    tr_img_name = Str()
    tr_gamma_correction = Bool()
    tr_DND = List(Instance(File))
    save_button = Action(name = "Save Fig", action = "do_savefig")
    movie_button = Action(name = "Make Movie", action = "do_makemovie")
    
    traits_view  = View(
        VGroup(
            Item('img_container', editor=ComponentEditor(), show_label=False),
            Item('tr_scaling', label='Radiance Scaling'),
            Item('tr_sun_angle', label='Sun angle'),
            Item('tr_gamma_correction', label='Apply Gamma Correction'),
            Item('tr_folder', label='Result Folder', editor=DirectoryEditor()),
            Item('tr_img_name', label='Images List', editor=EnumEditor(name="tr_img_list")),
            Item('tr_DND', label='Drag Here', editor=DropEditor())
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
            if 'img' in data.keys():
                self._img = data['img']
            else:
                self._img = data['rgb']
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
