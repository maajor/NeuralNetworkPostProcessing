from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Conv2DTranspose, Layer, add, UpSampling2D

class InputNormalize(Layer):
    def __init__(self, **kwargs):
        super(InputNormalize, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def compute_output_shape(self,input_shape):
        return input_shape

    def call(self, x, mask=None):
        x = (x - 127.5)/ 127.5
        return x



def conv_bn_relu(nb_filter, nb_row, nb_col,stride):   
    def conv_func(x):
        x = Conv2D(nb_filter, (nb_row, nb_col), strides=stride,padding='same')(x)
        x = BatchNormalization(momentum = 0.8)(x)
        #x = LeakyReLU(0.2)(x)
        x = Activation("relu")(x)
        return x
    return conv_func



#https://keunwoochoi.wordpress.com/2016/03/09/residual-networks-implementation-on-keras/
def res_conv(nb_filter, nb_row, nb_col,stride=(1,1)):
    def _res_func(x):
        #identity = Cropping2D(cropping=((2,2),(2,2)))(x)

        a = Conv2D(nb_filter, (nb_row, nb_col), strides=stride, padding='same')(x)
        a = BatchNormalization(momentum = 0.8)(a)
        #a = LeakyReLU(0.2)(a)
        a = Activation("relu")(a)
        a = Conv2D(nb_filter, (nb_row, nb_col), strides=stride, padding='same')(a)
        y = BatchNormalization(momentum = 0.8)(a)

        return  add([x, y])

    return _res_func


def dconv_bn_nolinear(nb_filter, nb_row, nb_col,stride=(2,2),activation="relu"):
    def _dconv_bn(x):
        #TODO: Deconvolution2D
        #x = Deconvolution2D(nb_filter,nb_row, nb_col, output_shape=output_shape, subsample=stride, border_mode='same')(x)
        #x = UpSampling2D(size=stride)(x)
        x = UpSampling2D(size=stride)(x)
        #x = ReflectionPadding2D(padding=stride)(x)
        x = Conv2D(nb_filter, (nb_row, nb_col), padding='same')(x)
        x = BatchNormalization(momentum = 0.8)(x)
        x = Activation(activation)(x)
        return x
    return _dconv_bn

class Denormalize(Layer):
    '''
    Custom layer to denormalize the final Convolution layer activations (tanh)
    Since tanh scales the output to the range (-1, 1), we add 1 to bring it to the
    range (0, 2). We then multiply it by 127.5 to scale the values to the range (0, 255)
    '''

    def __init__(self, **kwargs):
        super(Denormalize, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        '''
        Scales the tanh output activations from previous layer (-1, 1) to the
        range (0, 255)
        '''

        return (x + 1) * 127.5

    def compute_output_shape(self,input_shape):
        return input_shape
