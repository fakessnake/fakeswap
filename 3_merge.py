import sys
sys.path.append('./face_swap')
import argparse
import cv2
import json
import numpy
from pathlib import Path
from tqdm import tqdm
from PIL import Image,ImageDraw
import imageio

def merge_one_face(  image, face, mat ):
    assert face.shape == (256,256,3)
    crop = slice(48,208)
    face = face[crop,crop]
    size = 160
    image_size = image.shape[1], image.shape[0]
    cv2.warpAffine( face, mat * size, image_size, image, cv2.WARP_INVERSE_MAP, cv2.BORDER_TRANSPARENT )
    return image

def main( args ):
    output_dir = Path( args.output_dir ) 
    output_dir.mkdir( parents=True, exist_ok=True )
    input_json =  str(output_dir / "faces.json")

    input_dir = output_dir / "new_faces"
    assert input_dir.is_dir()
    output_dir = output_dir / "merged" 
    output_dir.mkdir( parents=True, exist_ok=True )

    reader = imageio.get_reader(args.input_file,  'ffmpeg')
    fps = reader.get_meta_data()['fps']
    print(fps)

    with open(input_json,'r') as f:
        alignments = json.load(f)

    last_count = -1
    last_count_str = ""
    image = None 
    for frame_count_str, face_file, mat in tqdm( alignments ):
        #print(frame_count_str, face_file)
        frame_count = (int)(frame_count_str)
        if (last_count != frame_count):
            if (last_count != -1):
                #print(args.output_dir + "/" + last_count_str + ".png")
                file_name = last_count_str + ".png"
                cv2.imwrite(str( output_dir / file_name ),  image)
            im = reader.get_data((int)(frame_count))
            pil_image = Image.fromarray(im)
            draw = ImageDraw.Draw(pil_image)
            array = numpy.array(pil_image)    
            image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        last_count = frame_count
        last_count_str = frame_count_str
        face  = cv2.imread(str( input_dir / face_file ))#( args.input_dir + "/" + face_file )
        mat = numpy.array(mat).reshape(2,3)
        if image is None: continue
        if face  is None: continue
        image = merge_one_face( image, face, mat )
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( "input_file", type=str )
    parser.add_argument( "output_dir", type=str )
    main( parser.parse_args() )

