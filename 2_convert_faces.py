import sys
sys.path.append('./face_swap')
import argparse
import cv2
import numpy
from pathlib import Path

from utils import get_image_paths

#from model import autoencoder_A, decoder_A
from model import autoencoder_B, decoder_B
from model import encoder
from script import convert_one_image

def main( args ):
    output_dir = Path( args.output_dir ) 
    encoder  .load_weights( str(output_dir/ "models" / "encoder.h5"))
    #decoder_A.load_weights( str(output_dir/ "models" / "decoder_A.h5" ))
    decoder_B.load_weights( str(output_dir/ "models" / "decoder_B.h5" ))
    input_dir = output_dir / "old_faces"
    output_dir = output_dir / "new_faces" 
    output_dir.mkdir( parents=True, exist_ok=True )

    images_A = get_image_paths( str(input_dir) )

    for fn in images_A:
        print(Path(fn).name)
        image = cv2.imread(fn)
        new_image = convert_one_image( autoencoder_B, image )
        output_file = output_dir / Path(fn).name
        cv2.imwrite( str(output_file), new_image )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( "output_dir", type=str )
    main( parser.parse_args() )