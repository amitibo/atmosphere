"""
Convert mcarsts radiance binary files to reference matrices for the analysis script.
"""
from __future__ import division
import numpy as np
from atmotomo import Mcarats
import scipy.io as sio
import os
import argparse


def main():
    #
    # Parse the input
    #
    parser = argparse.ArgumentParser(description='Create mcarats images')
    parser.add_argument('--remove_sun', action='store_true', help='remove the sun spots from the images.')
    parser.add_argument('mcarats', help='path to mcarats images')
    args = parser.parse_args()
    
    path = os.path.abspath(args.mcarats)

    R_ch, G_ch, B_ch = [np.fromfile(os.path.join(path, 'base%d_conf_out' % i), dtype=np.float32) for i in range(3)]
    IMG_SHAPE = (128, 128)
    IMG_SIZE = IMG_SHAPE[0] * IMG_SHAPE[1]
    
    for i in range(int(R_ch.size/IMG_SIZE)):
        slc = slice(i*IMG_SIZE, (i+1)*IMG_SIZE)
        ref_img = Mcarats.calcMcaratsImg(R_ch, G_ch, B_ch, slc, IMG_SHAPE, remove_sun=args.remove_sun)
        ref_img = ref_img.astype(np.float)
        
        sio.savemat(
            os.path.join(path, 'ref_img%d.mat' % i),
            {'img': ref_img},
            do_compression=True
        )



if __name__ == '__main__':
    main()
