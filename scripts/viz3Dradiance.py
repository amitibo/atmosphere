"""
Visualize 3D radiance reconstruction results.
"""

from __future__ import division
import numpy as np
import scipy.io as sio
import mayavi.mlab as mlab
import matplotlib.pyplot as plt
from atmotomo import single_cloud_vadim
import sparse_transforms as spt
import argparse
import amitibo


atmosphere_params = amitibo.attrClass(
    cartesian_grids=spt.Grids(
        np.arange(0, 50000, 1000.0), # Y
        np.arange(0, 50000, 1000.0), # X
        np.arange(0, 10000, 100.0)   # H
        ),
    earth_radius=4000000,
    air_typical_h=8000,
    aerosols_typical_h=2000
)

def main():
    """Main doc """

    parser = argparse.ArgumentParser(description='Visualize a 3D matlab matrix')
    parser.add_argument('path', help='Path to radiance matrix')
    
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
    #ratio = 1 / 10000 / 0.00072
    radiance_estim = data['estimated'] 
    radiance_true = data['true']
    
    Y, X, H = atmosphere_params.cartesian_grids.expanded
    
    amitibo.viz3D(Y, X, H, radiance_true, title='Monte Carlo')
    amitibo.viz3D(Y, X, H, radiance_estim, title='Estimated',)
    
    error_per_particle = np.sum(np.abs(radiance_estim - radiance_true)) / np.sum(radiance_true)
    print 'Absolute error normalized to num of particles:', error_per_particle
    error_per_voxel = np.sum(np.abs(radiance_estim - radiance_true)) / radiance_true.size
    print 'Absolute error normalized to num of particles:', error_per_voxel
    
    mlab.show()
    
    #plt.figure()
    #plt.imshow(radiance_true[:, 24, ::-1].T, interpolation='nearest', cmap='gray')
    #plt.colorbar()
    #plt.title('True Density')
    
    #plt.figure()
    #plt.imshow(radiance_estim[:, 24, ::-1].T, interpolation='nearest', cmap='gray')
    #plt.colorbar()
    #plt.title('Estimated Density')
    
    #plt.show()
    
    #plt.figure(figsize=(8, 2))
    #plt.plot(np.log(data['objective']))
    #plt.title('Objective vs Iteration')
    #plt.ylabel('log of objective')
    #plt.xlabel('iteration')
    #plt.show()

if __name__ == '__main__':
    main()

    
    