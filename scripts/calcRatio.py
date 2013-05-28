"""
Calculate the ratio between two sets of images.
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import atmotomo
import argparse
import glob
import scipy.io as sio
import re
import os
import skimage.morphology as morph


def getMcMatList(path):

    img_list, cameras = atmotomo.loadVadimData(path, remove_sunspot=False)
    
    return img_list


def getSingleMatList(path):
    
    path = os.path.abspath(path)
    files_list = glob.glob(os.path.join(path, "*.mat"))
    
    file_pattern = re.search(r'(.*?)\d+.mat', files_list[0]).groups()[0]
    
    img_list = []
    for i in range(0, len(files_list)+1):
        img_path = os.path.join(path, "%s%d.mat" % (file_pattern, i))
        try:
            data = sio.loadmat(img_path)
        except:
            continue
        
        img_list.append(data['img'])
        
    return img_list


def findMatched(ref_img, single_imgs):

    mask_results = []

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
    
    ind = np.argmax(mask_results)
        
    print ind
    return single_imgs[ind]


def calcRatio(ref_img, single_img, erode):
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


def main(ref_path, single_path, vadim_ref):
    """Main doc """
    
    #
    # Load the reference images and single scattering images
    #
    if vadim_ref:
        ref_imgs = getMcMatList(ref_path)
    else:
        ref_imgs = getSingleMatList(ref_path)
        
    single_imgs = getSingleMatList(single_path)

    #
    # Match the images.
    #
    #matched_single_imgs = []
    #for ref_img in ref_imgs:
        #single_img = findMatched(ref_img, single_imgs)
        #matched_single_imgs.append(single_img)
    if vadim_ref:
        matched_single_imgs = single_imgs[:]
    else:
        matched_single_imgs = single_imgs[:]
    
    means = []
    for ref_img, single_img in zip(ref_imgs, matched_single_imgs):
        means.append(calcRatio(ref_img, single_img, not vadim_ref))
        showImages(ref_img, single_img)
        
    print 'Mean: %g (=10^%g)' % (np.mean(means[7:]), np.log10(np.mean(means[7:])))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar(np.arange(len(means)), means, color='r')
    if vadim_ref:
        plt.title('Ratio Between Single Scattering and Vadim MC (isotropic phase)')
    else:
        plt.title('Ratio Between Single Scattering and MCARATS')
        
    plt.show()
    

def showImages(ref_img, single_img):
    plt.figure()
    plt.subplot(121)
    plt.imshow(ref_img/ref_img.max())
    plt.title('Reference')
    plt.subplot(122)
    plt.imshow(single_img/single_img.max())
    plt.title('Single')


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