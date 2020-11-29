Neural Network Post Processing
========

## 1. data preparation

I use coco2017 val for training datasets, download at [coco2017](http://images.cocodataset.org/zips/val2017.zip), unzip into "datasets" folder, pictures should be in "datasets/coco2017/val/"

add your style image to "datasets/style", remember that the image name is `<your-style-name>`  

## 2. modify the network

You can decrease layers and filters it if it runs too slow, increase if it underfit.

in "src/nets.py", method `SimpleTransformNet()`

## 3. train the network

run `python train_fst.py --style <your-style-name>`

## 4. export to unity

run `python exporter.py --dataset_name <your-style-name>`

## 5. parse model in unity

-  In Unity Project Window selected your recent imported model, should be `Assets/Script/RawModel/<your-dataset-name>.json`, right click it, from the pop-up menu select "ParseFromRawModel"  
-  Add `<your-dataset-name>` to script enum "NNStyle"  

Now you should be able to select the model from NNPostProcessingEffect inspector window  

## Requirements
python>=3.8
tensorflow>=2.3.0
scipy>1.5.4
imageio>=2.9.0
scikit-image>=0.17.2