# Automation for car inspection using Mach-band image

## Abstract

### Samples of original pictures for each class
|Class|Num of pics|Images|
|:---:|:---:|:---:|
|`Normal`|42|![Normal Images](/imgs/nor_merged.png)|
|`Error`|47|![Error Images](/imgs/err_merged.png)|

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
