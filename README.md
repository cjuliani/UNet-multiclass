# UNet for multiclass semantic segmentation
![Tensorflow](https://img.shields.io/badge/Implemented%20in-Tensorflow-green.svg) <br>

The demo was developed for segmenting topographic features. Input data are *256x256* pixels patch-based samples of elevation data and RGB terrain attributes transformed by principal component analysis. Annotations correspond to mounds (class-2) and mound summits (class-1). The project includes a sample generator with built-in random augmentation (position and colors) and batching systems, prediction and testing modules, and a data processing pipeline.

## Notes
- Images and annotations have specific names, e.g. annotations_1_34, indicating the annotation #34 of class 1. The symbol underscore is required.
- Validation data can be either taken randomly from the training set with a percentage defined in config.json, and/or manually pre-defined in the datasets/validation folder.
- Classes considered in the datasets must match the number of classes defined in in config.json.
- Metrics are saved when a training epoch starts or ends.
- Averaged metrics for each epoch are saved separately in text files in /train.
- A pre-trained model is saved in /train/selected_checkpoint if a restoration is needed.
- Summaries for training and validation are separated in 2 folders. Call /train/summary in TensorBoard for displaying both trends.

| ![alt text](https://raw.githubusercontent.com/cjuliani/tf-unet-multiclass/master/unet-multiclass.png) |
|:--:|
| *Standard U-Net architecture for multi-class object segmentation in conv10_1 (class 1) and conv10_2 (class 2). Inputs consist of  RGB images and depth data (1+3+3 channels). The number of filters is indicated next to convolutional layers. Convolution operations (conv) use  kernels with ReLU as activation function, excepted for segmentation and the bottleneck. The total number of trainable parameters is 5,881,828.*|

| ![alt text](https://raw.githubusercontent.com/cjuliani/tf-unet-multiclass/master/bathymetry.PNG) |
|:--:|
| *Map-view example of processed digital surface data (2-m resolution) showing complex interplays between an eruptive center, individual and overlapping seafloor mounds (black dots and white contours, respectively), fault scarps and fractures, and sediment or lava flows (black arrows).*|

| ![alt text](https://raw.githubusercontent.com/cjuliani/tf-unet-multiclass/master/segmentation.PNG) |
|:--:|
| *Weighted predictive map of class 1 (multiplied by 0.3) and class 2 (0.7) superimposed on digital surface map.*|


| ![alt text](https://raw.githubusercontent.com/cjuliani/tf-unet-multiclass/master/curvature.PNG) |
|:--:|
| *Map-view examples of RGB curvature resulting from PCA-based data transformation. Annotated mounds consist of 2 sub-units: the convex base (class 2) and the summit (class 1)*|
