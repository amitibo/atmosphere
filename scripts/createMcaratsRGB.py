"""
Convert mcarsts radiance binary files to reference matrices for the analysis script.
"""
from __future__ import division
import numpy as np
from atmotomo import Mcarats
import scipy.io as sio
import os
import argparse
import matplotlib.pyplot as plt
import amitibo


def main():
    #
    # Parse the input
    #
    parser = argparse.ArgumentParser(description='Create mcarats RGB images')
    parser.add_argument('mcarats', help='path to mcarats images')
    args = parser.parse_args()
    
    path = os.path.abspath(args.mcarats)

    R_ch, G_ch, B_ch = [os.path.join(path, 'base%d_conf_out' % i) for i in range(3)]

    #
    # Process the results
    #
    imgs = Mcarats.calcRGBImg(R_ch, G_ch, B_ch)
    
    #
    # Show the results
    #
    figures = []
    for img in imgs:
        figures.append(plt.figure())
        plt.imshow(img)
    
    amitibo.saveFigures(path, bbox_inches='tight', figures=figures)
    
    plt.show()

if __name__ == '__main__':
    main()
