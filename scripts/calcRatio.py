import scipy.io as sio
import atmotomo
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid


def getSingleMatList(path):
    
    path = os.path.abspath(path)
    files_list = glob.glob(os.path.join(path, "sim_img*.mat"))
    files_list.sort()
    
    img_list = []
    for img_path in files_list:
        try:
            data = sio.loadmat(img_path)
        except:
            continue
        
        img_list.append(data['img'])
        
    return img_list


def calcRatio(ref_img, single_img, sun_mask):
    #
    # Calc a joint mask
    #
    mask = (ref_img > 0) * (single_img > 0)

    ref_img = ref_img * sun_mask
    single_img = single_img * sun_mask
    
    ratio = ref_img[mask].mean() / single_img[mask].mean()

    return ratio


def compareImgs(mc_imgs, single_imgs):
    #
    # Calculate perliminary ratio
    # 
    means = []
    for i, (mc_img, single_img) in enumerate(zip(mc_imgs, single_imgs)):
        means.append(calcRatio(mc_img, single_img, sun_mask=1))
    
    ratio = np.mean(means)
    
    #
    # Calculate a sun mask
    #
    err_imgs = []
    for mc_img, single_img in zip(mc_imgs, single_imgs):
        err_imgs.append(mc_img/ratio - single_img)
    
    std = np.dstack(err_imgs).std(axis=2)
    sun_mask = np.exp(-std)
    sun_mask = np.tile(sun_mask[:, :, np.newaxis], (1, 1, 3))
    
    #
    # Recalculate the ratio
    #
    means = []
    for i, (mc_img, single_img) in enumerate(zip(mc_imgs, single_imgs)):
        means.append(calcRatio(mc_img, single_img, sun_mask))
        
    return means


def processSimulations():
    
    mc_base_path = '../data/monte_carlo_simulations'
    single_base_path = '../data/single_scatter_simulations'
    
    mc_folders = (
        ('front_high_density_medium_resolution',
        'front_low_density_medium_resolution',),
        ('high_cloud_high_density_medium_resolution',
        'high_cloud_low_density_medium_resolution',),
        ('low_cloud_high_density_medium_resolution',
        'low_cloud_low_density_medium_resolution',),
        ('two_clouds_high_density_medium_resolution',
        'two_clouds_low_density_medium_resolution',),
        ('two_clouds_high_density_high_resolution',
        'two_clouds_low_density_high_resolution',),
    )
    
    single_folders = (
        ('front_high_density_medium_resolution',
        'front_low_density_medium_resolution',),
        ('high_cloud_high_density_medium_resolution',
        'high_cloud_low_density_medium_resolution',),
        ('low_cloud_high_density_medium_resolution',
        'low_cloud_low_density_medium_resolution',),
        ('two_clouds_high_density_medium_resolution',
        'two_clouds_low_density_medium_resolution',),
        ('two_clouds_high_density_high_resolution',
        'two_clouds_low_density_high_resolution',),
    )
    
    for mc_folder, single_folder in zip(mc_folders, single_folders):
        fig = plt.figure(figsize=(12, 6))
        
        for i in range(2):
            try:
                mc_path = os.path.abspath(os.path.join(mc_base_path, mc_folder[i]))
                single_path = os.path.abspath(os.path.join(single_base_path, single_folder[i]))
            
                mc_imgs, cameras = atmotomo.loadVadimData(mc_path, remove_sunspot=False)

                single_imgs = getSingleMatList(single_path)
                
                means = compareImgs(mc_imgs, single_imgs)
                
                ax = plt.subplot(121+i)
                ax.bar(np.arange(len(means)), means, color='r')
                title(mc_folder[i])
            except:
                pass
            

if __name__ == '__main__':
    processSimulations()
    plt.show()