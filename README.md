# Image classification for car defect detection

## Abstract

### Samples of original pictures for each class
|Class|Images|Num of pics|
|:---:|:---:|:---:|
|`Normal`|![Normal Images](/imgs/nor_merged.png)|42|
|`Error`|![Error Images](/imgs/err_merged.png)|47|

## Purpose

### `1.Automate preprocessing`
* Denoising - Gamma correction, Morphology
* Selecting and croping the regions of interest(ROI)
* Resizing and picklizing

### `2.Use both ML and neural network`
* Find compact model(least compute but use gpu on nerual model)

### `3.Select best classification model`
* Best precision model

## License
This repo follows The MIT License (MIT)
