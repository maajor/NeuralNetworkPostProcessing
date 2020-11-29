import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, Add, Layer, Conv2DTranspose, Activation
from layers import InputNormalize, Denormalize, conv_bn_relu, res_conv, dconv_bn_nolinear

class SimpleTransformNet:
    def __init__(self):
        self.model = self._get_model()

    def _get_model(self):
        x = tf.keras.Input(shape=(None,None,3))
        a = InputNormalize()(x)
        #a = ReflectionPadding2D(padding=(40,40),input_shape=(img_width,img_height,3))(a)
        a = conv_bn_relu(8, 9, 9, stride=(1,1))(a)
        a = conv_bn_relu(16, 3, 3, stride=(2,2))(a)
        a = conv_bn_relu(32, 3, 3, stride=(2,2))(a)
        for i in range(2):
            a = res_conv(32,3,3)(a)
        a = dconv_bn_nolinear(16,3,3)(a)
        a = dconv_bn_nolinear(8,3,3)(a)
        a = dconv_bn_nolinear(3,9,9,stride=(1,1),activation="tanh")(a)
        # Scale output to range [0, 255] via custom Denormalize layer
        y = Denormalize(name='transform_output')(a)
        return tf.keras.Model(x, y, name="transformnet")

    def get_variables(self):
        return self.model.trainable_variables

    def preprocess(self, img):
        return img

    def postprocess(self, img):
        return tf.clip_by_value(img, 0.0, 255.0)