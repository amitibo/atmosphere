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
import skimage.morphology as morph


def getMcMatList(path):
    
    folder_list = glob.glob(os.path.join(path, "*"))
    if not folder_list:
        raise Exception("No img found in the folder")
    
    img_list = []
    for folder in folder_list:
        print folder
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
        print img_path
        try:
            data = sio.loadmat(img_path)
        except:
            continue
        
        img_list.append(data['img'])
        
    return img_list


def findMatched(ref_img, single_imgs):

    mask_results = []
    masks = []
    
    if (ref_img>0).sum() > (ref_img.size*3/4):
        #
        # The mcarats ref images have values all over the sky
        # so I supress all values below the median
        #
        ref_img = ref_img.copy()
        mean_val = ref_img.mean()
        ref_img[ref_img < mean_val] = 0
        
    for single_img in single_imgs:
        #
        # Calc a joint mask
        #
        mask = (ref_img > 0) * (single_img > 0)
        for i in range(3):
            mask[:, :, i] = morph.greyscale_erode(mask[:, :, i].astype(np.uint8) , morph.disk(1))
        mask = mask>0
        
        temp = single_img[mask]
        mask_results.append(temp.sum())
        masks.append(mask)
    
    ind = np.argmax(mask_results)
    print ind
    return single_imgs[ind], masks[ind]


def main(ref_path, single_path, vadim_ref):
    """Main doc """
    
    if vadim_ref:
        ref_imgs = getMcMatList(ref_path)
    else:
        ref_imgs = getSingleMatList(ref_path)
        
    single_imgs = getSingleMatList(single_path)

    matched_single_imgs = []
    matched_masks = []
    for ref_img in ref_imgs:
        single_img, mask = findMatched(ref_img, single_imgs)
        matched_single_imgs.append(single_img)
        matched_masks.append(mask)
        
    means = []
    for ref_img, single_img, mask in zip(ref_imgs, matched_single_imgs, matched_masks):
        ratio = ref_img[mask].mean() / single_img[mask].mean()
        
        means.append(ratio)
        
        showImages(ref_img, single_img, mask, ratio)
        
    print 'Mean: %g (=10^%g)' % (np.mean(means[7:]), np.log10(np.mean(means[7:])))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar(np.arange(len(means)), means, color='r')
    plt.show()
    

def showImages(ref_img, single_img, mask, ratio):
    plt.figure()
    plt.subplot(131)
    plt.imshow(ref_img/ref_img.max())
    plt.title('Reference')
    plt.subplot(132)
    plt.imshow(single_img/single_img.max())
    plt.title('Single')
    plt.subplot(133)
    plt.imshow(mask)
    plt.title('Mask')


if __name__ == '__main__':
    #
    # Parse the input
    #
    parser = argparse.ArgumentParser(description='Calculate the ratio between two sets of images')
    parser.add_argument('--vadim_ref', action='store_true', help='The reference images are stored in Vadim\'s format (folder of folders).')
    parser.add_argument('reference', type=str, help='The reference set of images.')
    parser.add_argument('single', type=str, help='The single scatter set of images.')
    args = parser.parse_args()

    main(args.reference, args.single, args.vadim_ref)