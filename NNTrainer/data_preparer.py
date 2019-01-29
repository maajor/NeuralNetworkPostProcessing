import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import datetime
from PIL import Image
import argparse
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='terrain', help='name of the dataset')
args = parser.parse_args()

class DataPrepare():
    def __init__(self, dataset_name, img_res=(512, 512)):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.combine_data()

    def combine_data(self):
        sourcefolder = "./datasets/" + self.dataset_name + "/source/"
        targetfolder = "./datasets/" + self.dataset_name + "/target/"
        sourcefiles = os. listdir(sourcefolder)
        targetfiles = os.listdir(targetfolder)
        assert(len(sourcefiles) == len(targetfiles))
        for i in range(0, len(sourcefiles)):
            colorpath = sourcefolder + sourcefiles[i];
            target = targetfolder + targetfiles[i]
            images = map(Image.open, [colorpath, target])
            new_im = Image.new('RGB', (self.img_res[0] * 2, self.img_res[1]))
            x_offset = 0
            for im in images:
                new_im.paste(im, (x_offset,0))
                x_offset += self.img_res[0]
            if i % 5 == 0:
                outpath = "./datasets/" + self.dataset_name + "/test/" + 'image_%04d' % i + ".png"
            else:
                outpath = "./datasets/" + self.dataset_name + "/train/" + 'image_%04d' % i + ".png"
            new_im.save(outpath)
            print("save to " + outpath)

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)

if __name__ == '__main__':
    DataPrepare(args.dataset_name, (512,512))