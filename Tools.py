from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
from PIL import Image

def main(args):
    dir = args.input_dir
    save_dir = args.output_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    degrees_to_rotate = args.degrees_to_rotate
    for subdir in os.listdir(dir):
        path = dir + subdir + '/'
        save_subdir = save_dir + subdir
        if not os.path.exists(save_subdir):
            os.mkdir(save_subdir)
        for filename in os.listdir(path):
            img_path = path + filename
            save_path = save_subdir + '/' + filename
            image = Image.open(img_path)
            rotate_image = image.rotate(degrees_to_rotate)
            rotate_image.save(save_path)
    print('Done!!!')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir',type=str,help='Directory contains unrotated image.')
    parser.add_argument('output_dir',type=str,help='Directory contains rotated image.')
    parser.add_argument('--degrees_to_rotate',type=int,help='Degrees to rotate image: -90 0 90',default=-90)
    return parser.parse_args(argv)

if __name__ == '__main__':
    print('Loading...')
    main(parse_arguments(sys.argv[1:]))