"""
Visualize a 3D matlab matrix useful for seeing one of Vadim radiance results
"""

from __future__ import division
import numpy as np
import scipy.io as sio
import mayavi.mlab as mlab
import argparse


def viz3D(X, Y, Z, V, X_label='X', Y_label='Y', Z_label='Z'):

    mlab.figure()

    src = mlab.pipeline.scalar_field(X, Y, Z, V)
    src.spacing = [1, 1, 1]
    src.update_image_data = True    
    ipw_x = mlab.pipeline.image_plane_widget(src, plane_orientation='x_axes')
    ipw_y = mlab.pipeline.image_plane_widget(src, plane_orientation='y_axes')
    ipw_z = mlab.pipeline.image_plane_widget(src, plane_orientation='z_axes')
    mlab.colorbar(orientation='vertical')
    mlab.outline()
    mlab.xlabel(X_label)
    mlab.ylabel(Y_label)
    mlab.zlabel(Z_label)

    limits = []
    for grid in (X, Y, Z):
        limits += [grid.min()]
        limits += [grid.max()]
    mlab.axes(ranges=limits)


def main():
    """Main doc """

    parser = argparse.ArgumentParser(description='Visualize a 3D matlab matrix')
    parser.add_argument('path', help='Path to matlab matrix')
    
    args = parser.parse_args()
    
    data = sio.loadmat(args.path)
    data_keys = [key for key in data.keys() if not key.startswith('__')]

    if len(data_keys) == 0:
        raise Exception('No matrix found in data. Available keys: %d', data.keys())
    
    mat = data[data_keys[0]]
    
    X, Y, Z = np.mgrid[0:mat.shape[0], 0:mat.shape[1], 0:mat.shape[2]]
    
    viz3D(X, Y, Z, mat)
    viz3D(X, Y, Z, np.log(mat+1))
    
    print mat.min(), mat.max()
    
    mlab.show()
    

if __name__ == '__main__':
    main()

    
    