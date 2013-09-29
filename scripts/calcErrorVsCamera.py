"""
Scan a folder containing atmosphere analysis results and
calculate the reconstruction error per camera num.
Before running you need to check the CROP_TOA and CAMERA_NUM parameters.
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
import os
import atmotomo
import scipy.io as sio

CROP_TOA = 10
CAMERA_NUM = 36
FINAL_ERROR = 0.65

def mse(estim, orig):
    #old_err_state = np.seterr(divide='ignore')
    ans = np.abs(estim-orig).sum()/np.abs(orig).sum()
    #np.seterr(**old_err_state)
    #ans = ans.mean()
    return ans


def main(path):
    """Main doc """
    
    folders = [f for f in glob.glob(os.path.join(path, '*')) if os.path.isdir(f)]
    
    errors = [[] for i in range(CAMERA_NUM)]
    for folder in folders:
        #
        # Get number of cameras
        #
        camera_num = CAMERA_NUM-1
        with open(os.path.join(folder, 'params.txt')) as f:
            for line in f:
                line = line.strip()
                if line.startswith('camera_num:'):
                    temp = int(line[12:])
                    if temp > 0:
                        camera_num = temp-1

        #
        # Calc error
        #
        data = sio.loadmat(os.path.join(folder, 'radiance.mat'))
        density_true = atmotomo.fixmat(data['true'])
        density_estim = atmotomo.fixmat(data['estimated'])
        
        if CROP_TOA > 0:
            density_estim = density_estim[:, :, :-CROP_TOA],
            density_true = density_true[:, :, :-CROP_TOA]
            
        error = mse(
            density_estim,
            density_true
        )

        errors[camera_num].append(error)
    
    index = []
    mean = []
    std = []
    for i in range(CAMERA_NUM):
        if errors[i] == []:
            continue
        index.append(i)
        mean.append(np.mean(errors[i]))
        std.append(np.std(errors[i]))
    
    plt.figure()
    plt.bar(
        index,
        mean,
        alpha=0.4,
        yerr=std
    )
    plt.title('Reconstruction Error vs. Cameras Quantity')
    plt.xlabel('Cameras Quantity')
    plt.ylabel('Error')
    
    plt.axhline(y=FINAL_ERROR, color='k', ls='dashed')
    
    plt.show()
    
if __name__ == '__main__':
    main(sys.argv[1])
    
    