# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 20:12:32 2011

@author: amitibo
"""
from __future__ import division
import pickle
import os

def main():
    """Main function"""
    
    misr = {}
    
    with open('misr.txt', 'rb') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 4):
            blue = lines[i].strip().split()
            green = lines[i+1].strip().split()
            red = lines[i+2].strip().split()
            
            misr[blue[6]] = {
                'description': blue[7],
                'k': (float(red[3]), float(green[3]), float(blue[3])),
                'w': (float(red[4]), float(green[4]), float(blue[4])),
                'g': (float(red[5]), float(green[5]), float(blue[5]))
    
            }
    
    with open('misr.pkl', 'wb') as f:
        pickle.dump(misr, f)


if __name__ == '__main__':
    main()
    
