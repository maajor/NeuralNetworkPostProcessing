import scipy.misc
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import datetime
import imageio

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False, is_debug = False):
        data_type = "train" if not is_testing else "test"
        data_type = data_type if not is_debug else "debug"
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        np.random.seed(datetime.datetime.now().microsecond)
        batch_images = np.random.choice(path, size=batch_size)

        imgs_A = []
        imgs_B = []
        for img_path in batch_images:
            img = self.imread(img_path)

            h, w, _ = img.shape
            _w = int(w/3)
            img_color, img_dep, img_target = img[:, :_w, :], img[:, _w:2*_w, :],  img[:, 2*_w:, :]
            img_src = np.zeros([h,_w,4])
            img_dst = np.zeros([h,_w,4])
            img_src[:,:,:3] = img_color
            img_src[:,:,3]  = img_dep[:,:,1]
            img_dst[:,:,:3] = img_target
            img_dst[:,:,3]  = img_dep[:,:,1]
            img_B, img_A = img_src, img_dst

            img_A = scipy.misc.imresize(img_A, self.img_res)
            img_B = scipy.misc.imresize(img_B, self.img_res)

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.

        return imgs_A, imgs_B

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        self.n_batches = int(len(path) / batch_size)

        for i in range(self.n_batches-1):
            batch = path[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img in batch:
                img = self.imread(img)
                h, w, _ = img.shape

                _w = int(w/3)
                img_color, img_dep, img_target = img[:, :_w, :], img[:, _w:2*_w, :],  img[:, 2*_w:, :]
                img_src = np.zeros([h,_w,4])
                img_dst = np.zeros([h,_w,4])
                img_src[:,:,:3] = img_color
                img_src[:,:,3]  = img_dep[:,:,1]
                img_dst[:,:,:3] = img_target
                img_dst[:,:,3]  = img_dep[:,:,1]
                img_B, img_A = img_src, img_dst

                img_A = scipy.misc.imresize(img_A, self.img_res)
                img_B = scipy.misc.imresize(img_B, self.img_res)

                if not is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B


    def imread(self, path):
        return imageio.imread(path).astype(np.float)


if __name__ == '__main__':
    dataloader = DataLoader("terrain", (512,512))
    iga, igb = dataloader.load_data()
    print(igb.shape)

    fig, axs = plt.subplots(1,1, figsize=(15,15))
    axs.imshow(igb[0,:,:,3])
    fig.savefig("test.png")