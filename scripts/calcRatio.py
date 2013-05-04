"""
Calculate the ratio between two sets of images.
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
import scipy.io as sio
import re
import os


def getMcMatList(path):
    
    folder_list = glob.glob(os.path.join(path, "*"))
    if not folder_list:
        warning(info.ui.control, "No img found in the folder", "Warning")
        return
    
    img_list = []
    for folder in folder_list:
        img_path = os.path.join(folder, "RGB_MATRIX.mat")
        try:
            data = sio.loadmat(img_path)
        except:
            continue
        
        img_list.append(data['Detector'])
    
    return img_list


def getSingleMatList(path):
    
    path = os.path.abspath(path)
    files_list = glob.glob(os.path.join(path, "*.mat"))
    
    img_list = []
    for img_path in files_list:
        try:
            data = sio.loadmat(img_path)
        except:
            continue
        
        img_list.append(data['img'])
        
    return img_list


def main(mc_path, single_path):
    """Main doc """
    
    mc_imgs = getMcMatList(mc_path)
    single_imgs = getSingleMatList(single_path)

    means = []
    stds = []
    for mc_img, single_img in zip(mc_imgs, single_imgs):
        mask = (mc_img > 0) * (single_img > 0)
        
        ratios = single_img[mask] / mc_img[mask]
        means.append(ratios.mean())
        stds.append(ratios.std())
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar(np.arange(len(means)), means, color='r', yerr=stds)
    plt.show()
    
        
if __name__ == '__main__':
    #
    # Parse the input
    #
    parser = argparse.ArgumentParser(description='Calculate the ratio between two sets of images')
    parser.add_argument('mc', type=str, help='The monte carlo set of images.')
    parser.add_argument('single', type=str, help='The single scatter set of images.')
    args = parser.parse_args()

    main(args.mc, args.single)