# UNet for multiclass semantic segmentation

The demo was initially developed for segmenting terrain features. Input data are 256x256 pixels patch-based samples of elevation data and RGB terrain attributes transformed by principal component analysis. Annotations correspond to mounds (class-2) and mound summits (class-1). The project includes a sample generator which applies random data augmentation (position and colors).

**NOTES**
- Images and annotations have specific names, e.g. annotations_1_34, indicating the annotation #34 of class 1. The symbol underscore is required.
- Validation data can be either taken randomly from the training set with a percentage defined in config.json, and/or manually pre-defined in the datasets/validation folder.
- Classes considered in the datasets must match the number of classes defined in in config.json.
- Metrics are saved when a training epoch starts or ends.
- Averaged metrics for each epoch are saved separately in text files in /train.
- A pre-trained model is saved in /train/selected_checkpoint if a restoration is needed.
- Summaries for training and validation are separated in 2 folders. Call /train/summary in TensorBoard for displaying both trends.

*Make sure you use Python 3.x and that related libraries have correct versions for compatibility (see requirements). Especially, some modules of TensorFlow may be depreciated and do not work properly.*
