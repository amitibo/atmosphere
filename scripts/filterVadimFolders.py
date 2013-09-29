"""
Visit the results folder of Vadim's simulations with separated scattering
order and leave only a specific order (or sum of orders).
"""

import os
import sys
import shutil


def main(path):
    
    base = os.path.abspath(path)
    
    for root, dirs, files in os.walk(base):
        if 'Matrixs' in dirs:
            try:
                os.rename(os.path.join(root, 'Matrixs', 'RGB_MATRIX.mat'), os.path.join(root, 'RGB_MATRIX.mat'))
            except:
                pass
            
            shutil.rmtree(os.path.join(root, 'Matrixs'))
            dirs.remove('Matrixs')
            
        if 'Images' in dirs:
            shutil.rmtree(os.path.join(root, 'Images'))
            dirs.remove('Images')
        
        for f in files:            
            if f == 'LNS_MATRIX.mat':
                os.rename(os.path.join(root, f), os.path.join(root, 'RGB_MATRIX.mat'))
                continue
            
            os.remove(os.path.join(root, f))
        
    
if __name__ == '__main__':
    main(sys.argv[1])
    