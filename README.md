Neural Network Post Processing
========

Post Processing for Unity using Convolution Neural Network. CNN Model trained with pix2pix/GAN.  
You can create your style offline and train the network with your own data, making your own NNPP!

![model](Imgs/99_0.png)  
[![NNPP](http://img.youtube.com/vi/qYcST5reOzY/0.jpg)](http://www.youtube.com/watch?v=qYcST5reOzY "NNPP")

Run ~60fps on GTX980 at 480p

## How to Run:

Open ***HirezScene*** scene and run!

### Requirement
* Unity 2018.2+
* Compute Shader support (DX11+, Vulkan, Metal)

## For Your Project:

### How to Train your model:

1. Prepare your training data(source):
	* Add a ***TrailRecorder*** to your Character Controller, run the game. Your trail will be saved.
	* Add a ***TrailPlayer*** to you Character Controller, link the trail record, disable ***TrailRecorder***, play the game. Use UnityRecorder to save color frames, naming: image_color_XXXX.png
	* With ***TrailPlayer*** on, add a ***RenderDepth*** component, play the game. Use UnityRecorder to save depth frames, naming: image_depth_XXXX.png
2. Prepare your training data(target):
	* Make your stylish action in Photoshop, batch on all your screenshots and saves, naming: image_out_XXXX.png
	* Copy all color/depth/out file to "NNTrainer/datasets/(yourdatasetname)/source"
	* run ```python data_prepare.py --dataset_name <yourdatasetname> --datanum <yourdatanumber>``` to generate training datasets
The training data should like this:
![data](Imgs/image_0009.png =500x)
2. Train
	* run ```python train.py --dataset_name <yourdatasetname>``` to train your model
	* Currently the model is:
![model](Imgs/model_architecture.png =300x)  
	* During training, model will export predicted pictures in "NNTrainer/images/(yourdatasetname)"
3. Export
	* run ```python exporter.py --dataset_name <yourdatasetname>``` to export your model to Unity
4. Run
	* In Unity, Open ***HirezScene*** scene and run!

### Requirement
* Python 3.6
* Tensorflow 1.10
* Keras 2.2.4




Reference
========

* https://github.com/eriklindernoren/Keras-GAN  
* https://github.com/keijiro/Pix2Pix
