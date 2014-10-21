from __future__ import division
from scipy.misc import lena
import enaml
from enaml.qt.qt_application import QtApplication
from atom.api import Atom, Int, List, Value, Float, Typed
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np


def calc_snr(f0, f):
    
    f0 = f0.astype(np.float)
    f0 /= f0.max()
    f = f.astype(np.float)
    f /= f.max()
    
    snr = -20*np.log10(np.linalg.norm(f-f0)/np.linalg.norm(f0))    
    return snr


class NoiseModel(Atom):
    """The data model of the client."""

    QE = Float(default=0.3)
    F = Int(default=20000)
    B = Int(default=8)
    alpha = Float()
    DN_mean = Float(default=6)
    DN_sigma = Float(default=2)
    t = Int(default=10)
    
    figure1 = Value()
    figure2 = Value()

    snr = Float()
    
    def _default_figure1(self):
        
        figure = Figure()
        ax = figure.add_subplot(111)
        ax.imshow(lena(), cmap='gray', interpolation='nearest')
        ax.set_axis_off()
        
        return figure
        
    def _default_figure2(self):
        
        figure = Figure()
        ax = figure.add_subplot(111)
        ax.imshow(lena(), cmap='gray', interpolation='nearest')
        ax.set_axis_off()
        
        return figure
        
    def process(self):
        
        self.alpha = 2**self.B/self.F
        
        #
        # Calculate the number of photons
        #
        photons = np.random.poisson(lena().astype(np.uint) * self.t)
        photons[photons>self.F] = self.F
        
        #
        # Convert to electrons and add the dark noise
        # based on Baer, Richard L. "A model for dark current characterization and simulation." Electronic Imaging 2006. International Society for Optics and Photonics, 2006.
        #
        electrons = (self.QE*photons.astype(np.float)).astype(np.uint)
        DN_noise = np.random.lognormal(mean=np.log(self.DN_mean), sigma=np.log(self.DN_sigma), size=electrons.shape).astype(np.uint)
        electrons += DN_noise
        
        #
        # Convert to gray level
        #
        g = electrons.astype(np.float) * self.alpha
        
        #
        # Quantisize
        #
        g_q = g.astype(np.uint)
        g_q[g_q>2**self.B] = 2**self.B

        #
        # Visualize
        #
        figure = Figure()
        ax = figure.add_subplot(111)
        ax.imshow(g_q, cmap='gray', interpolation='nearest')
        ax.set_axis_off()
                
        self.figure2 = figure

        self.snr = calc_snr(lena(), g_q)

#
# Import the enaml view.
#
with enaml.imports():
    from noise_ui import Main

class Controller(Atom):
    
    model = Typed(NoiseModel)
    view = Typed(Main)


#
# Main of application
#
def main():
    """Main doc"""

    import enaml
    from enaml.qt.qt_application import QtApplication

    #
    # Instansiate the data model
    #
    noise_model = NoiseModel()
    
    #
    # Start the Qt application
    #
    app = QtApplication()

    view = Main(model=noise_model)

    controller = Controller(model=noise_model, view=view)

    view.show()

    app.start()


if __name__ == '__main__':
    main()

