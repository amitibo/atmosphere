#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import subprocess as sbp
import os.path
import time
import copy

def assert_array_approx_equal(actual,desired,significant=7,err_msg='',verbose=True):
    """Compare two arrays up to some siginificant digits"""
    
    if np.all(desired==actual):
        return
    
    eps = np.power(10., -(significant+1))
    
    error = np.abs(actual-desired)
    scale = np.abs(desired)
    scale[scale==0.0] = 1.0
    error = error/scale

    msg = 'Items are not equal to %d significant digits:\nactual=\n%s\ndesired=\n%s\nerror=\n%s\n' % (significant, repr(actual), repr(desired), repr(error))
    
    if np.max(error) > eps:
        raise AssertionError(msg)


def plotLabeledResults(X, Y, clf2D=None, title=''):
    """Plot labeled results {X, Y}.
If clf2D is specified, the function will plot the zero level set of the
classifier. In this case clf2D must be a function of the form
    z = clf2D(x, y)
that implements the classifier for two dimenstions {x, y} and returns the
classification result {z}. z should be an array of the same form of x, y.
"""

    #
    # Plot the results
    #
    plt.figure()
    plt.hold = True
    ind_neg = Y == -1
    ind_pos = Y == 1
    plt.plot(X[0, ind_neg], X[1, ind_neg], 'ro')
    plt.plot(X[0, ind_pos], X[1, ind_pos], 'bo')

    #
    # Plot the zero level set of the classifier
    #
    if clf2D:
        x1, x2, y1, y2 = plt.gca().axis()
        x = np.linspace(x1, x2)
        y = np.linspace(y1, y2)
        x, y = np.meshgrid(x, y)
        z = clf2D(x, y)
        plt.contour(x, y, z, [0])
        plt.title(title)
    

def createResultFolder(base_path, params=[], diff_file=True):
    """Create folder for results.
Name of the folder uses hg version and time stamp. optional: The folder will contain diff of the working copy stored in file and parameters file."""

    #
    # Create the folder for results:
    #
    p = sbp.Popen('hg identify -n', stdout=sbp.PIPE)
    ver = p.stdout.read().strip()
    ts = time.strftime('%y_%m_%d__%H_%M_%S')
    results_folder = os.path.join(base_path, ver, ts)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    if diff_file:
        f = open(os.path.join(results_folder, 'hg_diff.txt'), 'w')
        p = sbp.Popen('hg diff', stdout=f)
        f.close()

    if params:
        f = open(os.path.join(results_folder, 'params.txt'), 'w')
        for item in params.__dict__.items():
            f.write('%s: %s\n' % (item[0], repr(item[1])))
        f.close()

    return results_folder


class attrClass(object):
    """Class used for storing attributes"""
    def __init__(self, **attrs):
        self.__dict__.update(attrs)


