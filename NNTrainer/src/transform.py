from keras.layers import Input, merge
from keras.models import Model,Sequential
from keras.optimizers import Adam, SGD,Nadam,Adadelta
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from scipy.misc import imsave
import time
import numpy as np 
import argparse
import h5py
import tensorflow as tf

from skimage import color, exposure, transform
from scipy import ndimage
from scipy.ndimage.filters import median_filter

import sys
sys.path.insert(0, 'src')
import nets
from layers import VGGNormalize,ReflectionPadding2D,Denormalize,conv_bn_relu,res_conv,dconv_bn_nolinear
from loss import dummy_loss,StyleReconstructionRegularizer,FeatureReconstructionRegularizer,TVRegularizer
from img_util import preprocess_image, preprocess_image_for_generating, preprocess_reflect_image, crop_image

# from 6o6o's fork. https://github.com/6o6o/chainer-fast-neuralstyle/blob/master/generate.py
def original_colors(original, stylized,original_color):
    # Histogram normalization in v channel
    ratio=1. - original_color 

    hsv = color.rgb2hsv(original/255)
    hsv_s = color.rgb2hsv(stylized/255)

    hsv_s[:,:,2] = (ratio* hsv_s[:,:,2]) + (1-ratio)*hsv [:,:,2]
    img = color.hsv2rgb(hsv_s)    
    return img

def blend(original, stylized, alpha):
    return alpha * original + (1 - alpha) * stylized



def median_filter_all_colours(im_small, window_size):
    """
    Applies a median filer to all colour channels
    """
    ims = []
    for d in range(3):
        im_conv_d = median_filter(im_small[:,:,d], size=(window_size,window_size))
        ims.append(im_conv_d)

    im_conv = np.stack(ims, axis=2).astype("uint8")
    
    return im_conv

def load_weights(model,file_path):
    f = h5py.File(file_path)

    layer_names = [name for name in f.attrs['layer_names']]

    for i, layer in enumerate(model.layers[:31]):
        g = f[layer_names[i]]
        weights = [g[name] for name in g.attrs['weight_names']]
        layer.set_weigh
        ts(weights)

    f.close()
    
    print('Pretrained Model weights loaded.')

def main(args):
    style= args.style
    #img_width = img_height =  args.image_size
    output_file =args.output
    input_file = args.input
    original_color = args.original_color
    blend_alpha = args.blend
    media_filter = args.media_filter

    aspect_ratio, x = preprocess_reflect_image(input_file, size_multiple=4)
    print(x.shape)
    imsave('%s_input.png' % output_file, x[0])

    img_width= img_height = x.shape[1]
    net = nets.image_transform_net_simple(img_width,img_height)
    #model = nets.loss_net(net.output,net.input,img_width,img_height,"",0,0)

    #model.summary()

    net.compile(Adam(),  dummy_loss)  # Dummy loss since we are learning from regularizes

    net.load_weights("model/"+style+'_weight.h5',by_name=False)

    
    t1 = time.time()
    y = net.predict(x)[0] 
    y = crop_image(y, aspect_ratio)

    print("process: %s" % (time.time() -t1))

    ox = crop_image(x[0], aspect_ratio)

    y =  median_filter_all_colours(y, media_filter)

    if blend_alpha > 0:
        y = blend(ox,y,blend_alpha)


    if original_color > 0:
        y = original_colors(ox,y,original_color )

    imsave('%s_output.png' % output_file, y)
        
 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time style transfer')

    parser.add_argument('--style', '-s', type=str, required=True,
                        help='style image file name without extension')

    parser.add_argument('--input', '-i', default=None, required=True,type=str,
                        help='input file name')

    parser.add_argument('--output', '-o', default=None, required=True,type=str,
                        help='output file name without extension')

    parser.add_argument('--original_color', '-c', default=0, type=float,
                        help='0~1 for original color')

    parser.add_argument('--blend', '-b', default=0, type=float,
                        help='0~1 for blend with original image')

    parser.add_argument('--media_filter', '-f', default=3, type=int,
                        help='media_filter size')
    parser.add_argument('--image_size', default=256, type=int)

    args = parser.parse_args()
    main(args)
