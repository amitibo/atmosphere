# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 20:12:32 2011

@author: amitibo
"""
from __future__ import division
from pkg_resources import resource_filename
import pickle
import os


def main():
    """Main function"""
    
    misr = {}
    
    txt_path1 = resource_filename('shdom', 'data/misr.txt')
    txt_path2 = resource_filename('shdom', 'data/misr_physical.txt')
    pkl_path = resource_filename('shdom', 'data/misr.pkl')
    
    with open(txt_path1, 'rb') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 4):
            blue = lines[i].strip().split()
            green = lines[i+1].strip().split()
            red = lines[i+2].strip().split()
            
            misr[blue[6]] = {
                'description': ''.join(blue[6:]),
                'k': (float(red[3]), float(green[3]), float(blue[3])),
                'w': (float(red[4]), float(green[4]), float(blue[4])),
                'g': (float(red[5]), float(green[5]), float(blue[5]))
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
    
