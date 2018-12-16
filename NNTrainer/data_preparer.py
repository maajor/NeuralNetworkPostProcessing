import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import datetime
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='terrain', help='name of the dataset')
parser.add_argument('--datanum', dest='datanum', default=100, help='data count')
args = parser.parse_args()

class DataPrepare():
    def __init__(self, dataset_name, img_res=(512, 512), datanum = 100):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.datanum = datanum
        self.combine_data()

    def combine_data(self):
        folder = "./datasets/" + self.dataset_name + "/source/"
        targetfolder = "./datasets/" + self.dataset_name + "/target/"
        for i in range(0, self.datanum):
            colorpath = folder + 'image_color_%04d' % i + ".png"
            depth = folder + 'image_depth_%04d' % i + ".png"
            target = targetfolder + 'image_color_%04d' % i + ".png"
            images = map(Image.open, [colorpath, depth, target])
            new_im = Image.new('RGB', (self.img_res[0] * 3, self.img_res[1]))
            x_offset = 0
            for im in images:
                new_im.paste(im, (x_offset,0))
                x_offset += self.img_res[0]
            outpath = "./datasets/" + self.dataset_name + "/train/" + 'image_%04d' % i + ".png"
            new_im.save(outpath)
            print("save to " + outpath)

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)

if __name__ == '__main__':
    DataPrepare(args.dataset_name, (512,512), int(args.datanum))