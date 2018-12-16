from keras.layers import Input
from keras.layers.merge import concatenate
from keras.models import Model,Sequential
from layers import InputNormalize,VGGNormalize,ReflectionPadding2D,Denormalize,conv_bn_relu,res_conv,dconv_bn_nolinear
from loss import StyleReconstructionRegularizer,FeatureReconstructionRegularizer,TVRegularizer
from keras import backend as K
from VGG16 import VGG16
import img_util


def image_transform_net_simple(img_width,img_height,tv_weight=1):
    x = Input(shape=(img_width,img_height,3))
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
    
    model = Model(inputs=x, outputs=y)
    
    if tv_weight > 0:
        add_total_variation_loss(model.layers[-1],tv_weight)
        
    return model 


def image_transform_net(img_width,img_height,tv_weight=1):
    x = Input(shape=(img_width,img_height,3))
    a = InputNormalize()(x)
    #a = ReflectionPadding2D(padding=(40,40),input_shape=(img_width,img_height,3))(a)
    a = conv_bn_relu(32, 9, 9, stride=(1,1))(a)
    a = conv_bn_relu(64, 9, 9, stride=(2,2))(a)
    a = conv_bn_relu(128, 3, 3, stride=(2,2))(a)
    for i in range(5):
        a = res_conv(128,3,3)(a)
    a = dconv_bn_nolinear(64,3,3)(a)
    a = dconv_bn_nolinear(32,3,3)(a)
    a = dconv_bn_nolinear(3,9,9,stride=(1,1),activation="tanh")(a)
    # Scale output to range [0, 255] via custom Denormalize layer
    y = Denormalize(name='transform_output')(a)
    
    model = Model(inputs=x, outputs=y)
    
    if tv_weight > 0:
        add_total_variation_loss(model.layers[-1],tv_weight)
        
    return model 




def loss_net(x_in, trux_x_in,width, height,style_image_path,content_weight,style_weight):
    # Append the initial input to the FastNet input to the VGG inputs
    x = concatenate([x_in, trux_x_in], axis=0)
    
    # Normalize the inputs via custom VGG Normalization layer
    x = VGGNormalize(name="vgg_normalize")(x)

    vgg = VGG16(include_top=False,input_tensor=x)

    vgg_output_dict = dict([(layer.name, layer.output) for layer in vgg.layers[-18:]])
    vgg_layers = dict([(layer.name, layer) for layer in vgg.layers[-18:]])

    if style_weight > 0:
        add_style_loss(vgg,style_image_path , vgg_layers, vgg_output_dict, width, height,style_weight)   

    if content_weight > 0:
        add_content_loss(vgg_layers,vgg_output_dict,content_weight)

    # Freeze all VGG layers
    for layer in vgg.layers[-19:]:
        layer.trainable = False

    return vgg

def add_style_loss(vgg,style_image_path,vgg_layers,vgg_output_dict,img_width, img_height,weight):
    style_img = img_util.preprocess_image(style_image_path, img_width, img_height)
    print('Getting style features from VGG network.')

    style_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']

    style_layer_outputs = []

    for layer in style_layers:
        style_layer_outputs.append(vgg_output_dict[layer])

    vgg_style_func = K.function([vgg.layers[-19].input], style_layer_outputs)

    style_features = vgg_style_func([style_img])

    # Style Reconstruction Loss
    for i, layer_name in enumerate(style_layers):
        layer = vgg_layers[layer_name]

        feature_var = K.variable(value=style_features[i][0])
        style_loss = StyleReconstructionRegularizer(
                            style_feature_target=feature_var,
                            weight=weight)(layer)

        layer.add_loss(style_loss)

def add_content_loss(vgg_layers,vgg_output_dict,weight):
    # Feature Reconstruction Loss
    content_layer = 'block3_conv3'
    content_layer_output = vgg_output_dict[content_layer]

    layer = vgg_layers[content_layer]
    content_regularizer = FeatureReconstructionRegularizer(weight)(layer)
    layer.add_loss(content_regularizer)


def add_total_variation_loss(transform_output_layer,weight):
    # Total Variation Regularization
    layer = transform_output_layer  # Output layer
    tv_regularizer = TVRegularizer(weight)(layer)
    layer.add_loss(tv_regularizer)

