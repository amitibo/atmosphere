"""
Visualize 3D radiance reconstruction results.
"""

from __future__ import division
import numpy as np
import scipy.io as sio
import mayavi.mlab as mlab
import matplotlib.pyplot as plt
import argparse
import amitibo


def main():
    """Main doc """

    parser = argparse.ArgumentParser(description='Visualize a 3D matlab matrix')
    parser.add_argument('path', help='Path to matlab matrix')
    
    args = parser.parse_args()
    
    data = sio.loadmat(args.path)
    data_keys = [key for key in data.keys() if not key.startswith('__')]

    if len(data_keys) == 0:
        raise Exception('No matrix found in data. Available keys: %d', data.keys())
    
    #
    # The ratio is calculated as 1 / visibility / k_aerosols = 1 / 50[km] / 0.00072 [um**2]
    # This comes from out use of A:
    # exp(-A / visibility * length) = exp(-k * N * length)
    #
    ratio = 1e15 / 50 / 0.00072
    radiance_true = data['true'] * ratio
    radiance_estim = data['estimated'] * ratio
    
    #
    # Remove negative values from the estimation
    #
    radiance_estim[radiance_estim<0] = 0
    
    X, Y, Z = np.mgrid[0:50:1.0, 0:50:1.0, 0:10.0:0.1]
    
    radiance_estim[radiance_estim/1e15 > 16] = 16
    
    amitibo.viz3D(X, Y, Z, radiance_true/1e15, title='Original')
    amitibo.viz3D(X, Y, Z, radiance_estim/1e15, title='Estimated',)
    
    error_per_particle = np.sum(np.abs(radiance_estim - radiance_true)) / np.sum(radiance_true)
    print 'Absolute error normalized to num of particles:', error_per_particle
    error_per_voxel = np.sum(np.abs(radiance_estim - radiance_true)) / radiance_true.size
    print 'Absolute error normalized to num of particles:', error_per_voxel
    
    #mlab.show()
    
    plt.figure(figsize=(8, 2))
    plt.plot(np.log(data['objective']))
    plt.title('Objective vs Iteration')
    plt.ylabel('log of objective')
    plt.xlabel('iteration')
    plt.show()

if __name__ == '__main__':
    main()

    
    