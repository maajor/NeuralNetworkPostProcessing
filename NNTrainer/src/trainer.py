from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg19 import preprocess_input
from collections import namedtuple
from glob import glob
import os
import datetime
import numpy as np
import tensorflow as tf
import imageio

from nets import SimpleTransformNet
from vgg import VGGModel
from img_util import get_image, resize_img

Loss = namedtuple('Loss', 'total_loss style_loss content_loss tv_loss')


class Trainer:
    def __init__(self,
                 style_path,
                 content_file_path,
                 epochs=10,
                 batch_size=4,
                 content_weight=1e0,
                 style_weight=4e1,
                 tv_weight=2e2,
                 learning_rate=1e-3,
                 log_period=100,
                 save_period=1000,
                 content_layers=["conv4_2"],
                 style_layers=["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"],
                 content_layer_weights=[1],
                 style_layer_weights=[0.2, 0.2, 0.2, 0.2, 0.2],
                 img_res=512):
        self.style_path = style_path
        self.style_name = style_path.split("/")[-1].split(".")[0]
        self.content_file_path = content_file_path
        assert (len(content_layers) == len(content_layer_weights))
        self.content_layers = content_layers
        self.content_layer_weights = content_layer_weights
        assert (len(style_layers) == len(style_layer_weights))
        self.style_layers = style_layers
        self.style_layer_weights = style_layer_weights
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_period = log_period
        self.save_period = save_period
        self.saved_model_path = "model/{0}_weight.h5".format(self.style_name)

        self.style_weight = style_weight
        self.content_weight = content_weight
        self.tv_weight = tv_weight
        self.img_res = img_res

        self.transform = SimpleTransformNet()
        self.vgg = VGGModel(content_layers, style_layers)
        self.learing_rate = learning_rate
        self.train_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learing_rate)

    def run(self):
        self.S_outputs = self._get_S_outputs()
        self.S_style_grams = [self._gram_matrix(tf.convert_to_tensor(m, tf.float32)) for m in
                              self.S_outputs[self.vgg.partition_idx:]]

        content_images = glob(os.path.join(self.content_file_path, "*.jpg"))
        num_images = len(content_images) - (len(content_images) % self.batch_size)
        print("Training on %d images" % num_images)

        self.iteration = 0

        for e in range(self.epochs):
            for e_i, batch in enumerate(
                    [content_images[i:i + self.batch_size] for i in range(0, num_images, self.batch_size)]):

                content_imgs = [get_image(img_path, (self.img_res, self.img_res, 3)) for img_path in batch]
                content_imgs = np.array(content_imgs)
                content_tensors = tf.convert_to_tensor(content_imgs)

                loss = self._train_step(content_tensors)

                if (self.iteration % self.log_period == 0):
                    self._log_protocol(loss)
                if (self.iteration % self.save_period == 0):
                    self._save_protocol()

                self.iteration += 1

                if self.iteration % 100 == 0:
                    self._display_img(self.iteration)

            self._log_protocol(loss)
            self._save_protocol()
            print("Epoch complete.")
        print("Training finished.")

    def _get_S_outputs(self):
        img = tf.convert_to_tensor(get_image(self.style_path), tf.float32)
        img = tf.expand_dims(img, 0)
        img = self.vgg.preprocess(img)
        return self.vgg.model(img)

    @tf.function
    def _train_step(self, content_tensors):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.transform.get_variables())
            C = self.transform.preprocess(content_tensors)
            X = self.transform.model(C)
            X = self.transform.postprocess(X)

            X_vgg = self.vgg.preprocess(X)
            Y_hat = self.vgg.forward(X_vgg)
            Y_hat_content = Y_hat.content_output
            Y_hat_style = Y_hat.style_output

            C_vgg = self.vgg.preprocess(content_tensors)
            Y = self.vgg.forward(C_vgg)
            Y_content = Y.content_output

            L = self._get_loss(Y_hat_content, Y_hat_style, Y_content, X)
        grads = tape.gradient(L.total_loss, self.transform.get_variables())
        self.train_optimizer.apply_gradients(zip(grads, self.transform.get_variables()))
        return L

    def _get_loss(self, transformed_content, transformed_style, content, transformed_img):

        content_loss = self._get_content_loss(transformed_outputs=transformed_content, content_outputs=content)
        style_loss = self._get_style_loss(transformed_style)
        tv_loss = self._get_total_variation_loss(transformed_img)

        L_style = style_loss * self.style_weight
        L_content = content_loss * self.content_weight
        L_tv = tv_loss * self.tv_weight

        total_loss = L_style + L_content + L_tv

        return Loss(total_loss=total_loss,
                    style_loss=L_style,
                    content_loss=L_content,
                    tv_loss=L_tv)

    def _get_content_loss(self, transformed_outputs, content_outputs):
        content_loss = 0
        assert (len(transformed_outputs) == len(content_outputs))
        for i, output in enumerate(transformed_outputs):
            weight = self.content_layer_weights[i]
            B, H, W, CH = output.get_shape()
            HW = H * W
            loss_i = weight * 2 * tf.nn.l2_loss(output - content_outputs[i]) / (B * HW * CH)
            content_loss += loss_i
        return content_loss

    def _get_style_loss(self, transformed_outputs):
        style_loss = 0
        assert (len(transformed_outputs) == len(self.S_style_grams))
        for i, output in enumerate(transformed_outputs):
            weight = self.style_layer_weights[i]
            B, H, W, CH = output.get_shape()
            G = self._gram_matrix(output)
            A = self.S_style_grams[i]
            style_loss += weight * 2 * tf.nn.l2_loss(G - A) / (B * (CH ** 2))
        return style_loss

    def _gram_matrix(self, input_tensor, shape=None):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = input_tensor.get_shape()
        num_locations = input_shape[1] * input_shape[2] * input_shape[3]
        num_locations = tf.cast(num_locations, tf.float32)
        return result / num_locations

    def _get_total_variation_loss(self, img):
        B, W, H, CH = img.get_shape()
        return tf.reduce_sum(tf.image.total_variation(img)) / (W * H)

    def _log_protocol(self, L):
        tf.print("iteration: %d, total_loss: %f, style_loss: %f, content_loss: %f, tv_loss: %f" \
                 % (self.iteration, L.total_loss, L.style_loss, L.content_loss, L.tv_loss))

    def _save_protocol(self):
        self.transform.model.save_weights(self.saved_model_path)

    def _display_img(self, i):
        img = get_image("datasets/chicago.jpg")
        img_tensor = tf.convert_to_tensor(img)
        img_tensor = tf.expand_dims(img_tensor, 0)
        res = self.transform.model.predict(img_tensor)
        res = tf.clip_by_value(res, 0, 255)
        res = res.numpy()
        res = tf.squeeze(res)
        res = res.numpy()
        res = res.astype(int)
        fname = 'datasets/output/train_%d_val.png' % (i)
        imageio.imwrite(fname, res)
