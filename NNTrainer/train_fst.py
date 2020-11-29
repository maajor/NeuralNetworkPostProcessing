from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.optimizers import Adam, SGD,RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import imageio
import time
import numpy as np
import argparse

from tensorflow.keras.callbacks import TensorBoard

import sys
sys.path.insert(0, 'src')
from trainer import Trainer



def display_img(i,x,style,is_val=False):
    # save current generated image
    img = x #deprocess_image(x)
    if is_val:
        #img = ndimage.median_filter(img, 3)

        fname = 'datasets/output/%s_%d_val.png' % (style,i)
    else:
        fname = 'datasets/output/%s_%d.png' % (style,i)
    imageio.imwrite(fname, img)
    print('Image saved as', fname)

def get_style_img_path(style):
    return "datasets/style/"+style+".jpg"


def main(args):
    style_weight= args.style_weight
    content_weight= args.content_weight
    tv_weight= args.tv_weight
    style= args.style
    img_width = img_height =  args.image_size

    style_image_path = get_style_img_path(style)

    trainer = Trainer(style_image_path, "datasets/coco2017/val", content_weight=content_weight, style_weight=style_weight, tv_weight=tv_weight)

    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time style transfer')
        
    parser.add_argument('--style', '-s', type=str, required=True,
                        help='style image file name without extension')
          
    parser.add_argument('--output', '-o', default=None, type=str,
                        help='output model file path without extension')
    parser.add_argument('--tv_weight', default=1e-6, type=float,
                        help='weight of total variation regularization according to the paper to be set between 10e-4 and 10e-6.')
    parser.add_argument('--content_weight', default=1.0, type=float)
    parser.add_argument('--style_weight', default=5.0, type=float)
    parser.add_argument('--image_size', default=512, type=int)

    args = parser.parse_args()
    main(args)
