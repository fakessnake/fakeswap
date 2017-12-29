import cv2
import numpy
from pathlib import Path

from utils import get_image_paths

from model import autoencoder_A
from model import autoencoder_B
from model import encoder, decoder_A, decoder_B


def adjust_avg_color(img_old,img_new):
    w,h,c = img_new.shape
    for i in range(img_new.shape[-1]):
        old_avg = img_old[:, :, i].mean()
        new_avg = img_new[:, :, i].mean()
        diff_int = (int)(old_avg - new_avg)
        for m in range(img_new.shape[0]):
            for n in range(img_new.shape[1]):
                temp = (img_new[m,n,i] + diff_int)
                if temp < 0:
                    img_new[m,n,i] = 0
                elif temp > 255:
                    img_new[m,n,i] = 255
                else:
                    img_new[m,n,i] = temp

def smooth_mask(img_old,img_new):
    w,h,c = img_new.shape
    crop = slice(0,w)
    mask = numpy.zeros_like(img_new)
    mask[h//15:-h//15,w//15:-w//15,:] = 255
    mask = cv2.GaussianBlur(mask,(15,15),10)
    img_new[crop,crop] = mask/255*img_new + (1-mask/255)*img_old

def convert_one_image( autoencoder, image ):
    use_smoothed_mask = True
            
    assert image.shape == (256,256,3)
    crop = slice(48,208)
    face = image[crop,crop]

    old_face = face.copy()

    face = cv2.resize( face, (64,64) )
    
    face = numpy.expand_dims( face, 0 )
    new_face = autoencoder.predict( face / 255.0 )[0]
    new_face = numpy.clip( new_face * 255, 0, 255 ).astype( image.dtype )
    new_face = cv2.resize( new_face, (160,160) )

    adjust_avg_color(old_face,new_face)
    smooth_mask(old_face,new_face)

    new_image = image.copy()
    
    new_image[crop,crop] = new_face

    return new_image



if __name__ == '__main__':
    output_dir = Path( 'output' )
    output_dir.mkdir( parents=True, exist_ok=True )
    encoder  .load_weights( "models/encoder.h5"   )
    decoder_A.load_weights( "models/decoder_A.h5" )
    decoder_B.load_weights( "models/decoder_B.h5" )

    images_A = get_image_paths( "data/trump" )
    images_B = get_image_paths( "data/cage" )

    for fn in images_A:
        image = cv2.imread(fn)
        new_image = convert_one_image( autoencoder_B, image )
        output_file = output_dir / Path(fn).name
        cv2.imwrite( str(output_file), new_image )