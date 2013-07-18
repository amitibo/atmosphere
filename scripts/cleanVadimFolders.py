import os
import sys
import shutil


def main(path):
    
    base = os.path.abspath(path)
    
    for root, dirs, files in os.walk(base):
        if 'Images' in dirs:
            shutil.rmtree(os.path.join(root, 'Images'))
            dirs.remove('Images')
        
        for f in files:
            if os.path.splitext(f)[1] == '.mat':
                continue
            
            if f == 'hg_diff.txt':
                os.remove(os.path.join(root, f))
        
    
if __name__ == '__main__':
    main(sys.argv[1])
    