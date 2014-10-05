# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 20:12:32 2011

@author: amitibo
"""
from __future__ import division
from pkg_resources import resource_filename
from atmotomo import ColoredParam
import pickle
import os


def main():
    """Main function"""
    
    misr = {}
    
    txt_path1 = resource_filename('atmotomo', 'data/misr.txt')
    txt_path2 = resource_filename('atmotomo', 'data/misr_physical.txt')
    pkl_path = resource_filename('atmotomo', 'data/misr.pkl')
    
    with open(txt_path1, 'rb') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 4):
            #
            # The parameters are read in the order blue, green red
            #
            color_params = [line.strip().split() for line in lines[i:i+3]][::-1]
            
            misr[color_params[2][6]] = {
                'description': ''.join(color_params[2][6:]),
                'refractive index': ColoredParam(*[complex(float(params[1]), float(params[2])) for params in color_params]),
                'k': ColoredParam(*[float(params[3]) for params in color_params]),
                'w': ColoredParam(*[float(params[4]) for params in color_params]),
                'g': ColoredParam(*[float(params[5]) for params in color_params]),
            }
    
    with open(txt_path2, 'rb') as f:
        for line in f:
            parts = line.strip().split()
            misr[parts[8]].update(
                dict(
                    zip(
                        ('min radius', 'max radius', 'char radius', 'char width', 'density', 'bottom', 'top', 'scale'),
                        [float(p) for p in parts[0:8]]
                    )
                )
            )
    
    with open(pkl_path, 'wb') as f:
        pickle.dump(misr, f)


if __name__ == '__main__':
    main()
    
