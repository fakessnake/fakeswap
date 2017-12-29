import sys
sys.path.append('./face_swap')
from PIL import Image,ImageDraw
import imageio
import cv2

import argparse
import cv2
import dlib
import json
import numpy
import skimage
from pathlib import Path
from tqdm import tqdm
from umeyama import umeyama

def transform( image, mat, size, padding=0 ):
    mat = mat * size
    mat[:,2] += padding
    new_size = int( size + padding * 2 )
    return cv2.warpAffine( image, mat, ( new_size, new_size ) )

def main( args ):
    output_dir = Path( args.output_dir ) 
    output_dir.mkdir( parents=True, exist_ok=True )
    input_json =  str(output_dir / "faces.json")

    reader = imageio.get_reader(args.input_file,  'ffmpeg')
    fps = reader.get_meta_data()['fps']
    print(fps)
    output_dir = Path( args.output_dir ) / "old_faces" 
    output_dir.mkdir( parents=True, exist_ok=True )

    with open(input_json,'r') as f:
        alignments = json.load(f)
    
    last_count = -1
    cv2_image = None 
    for frame_count_str, face_file, mat in tqdm( alignments ):
        frame_count = (int)(frame_count_str)
        if (last_count != frame_count):
            im = reader.get_data((int)(frame_count))
            pil_image = Image.fromarray(im)
            draw = ImageDraw.Draw(pil_image)
            array = numpy.array(pil_image)    
            cv2_image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        mat = numpy.array(mat).reshape(2,3)
        aligned_image = transform( cv2_image, mat, 160, 48 )
        cv2.imwrite(str(output_dir /face_file),aligned_image)
        last_count = frame_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( "input_file"  , type=str )
    parser.add_argument( "output_dir", type=str)

    parser.set_defaults( only_one_face=False )
    parser.add_argument('--one-face' , dest='only_one_face', action='store_true'  )
    parser.add_argument('--all-faces', dest='only_one_face', action='store_false' )

    main( parser.parse_args() )
