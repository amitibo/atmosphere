"""
Convert the names of Vadim's result folders to consequtive names. This
is useful when joining results that were created on different folders
therefore the order of the results (corresponding to camera index) are
not necessarilly chronological. Converting the folder names to indexes
enables comparing them to my results in the GUI (or in the calcRatio
script).
"""

import glob
import argparse
import os
import shutil


def main(input_path, output_path, start_index, prefix):
    
    folder_list = glob.glob(os.path.join(input_path, '*'))
    folder_list.sort()
        
    for i, src_path in enumerate(folder_list):
        #
        # Check that the folder is not empty
        #
        if not os.path.exists(os.path.join(src_path, "RGB_MATRIX.mat")):
            continue
        
        dst_path = os.path.join(output_path, '%s%s' % (prefix, ('00%d' % (i+start_index))[-3:]))
        shutil.copytree(src_path, dst_path)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Change names of folders')
    parser.add_argument('--output_path', default='.', help='path to output folder')
    parser.add_argument('--start_index', type=int, default=0, help='start index')
    parser.add_argument('--prefix', default='img', help='prefix to the destination output images')
    parser.add_argument('input_path', help='path to input folder')
    args = parser.parse_args()

    main(
        os.path.abspath(args.input_path),
        os.path.abspath(args.output_path),
        args.start_index,
        args.prefix
        )
