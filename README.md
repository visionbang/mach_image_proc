# Automation of car dent detection using Mach-band image


## 1.Abstract
In terms of car dent inspection process, 

* Workers are suffering from extremly high illumination(about 2000 Lux)

### Samples of original pictures for each class
|Class|Num of pics|Mach-band images|
|:---:|:---:|:---:|
|`Normal`|65|![Normal Images](/imgs/nor_merged.png)|
|`Error`|77|![Error Images](/imgs/err_merged.png)|

## 2.Purpose

### `1.Automate preprocessing`
* Denoising - Gamma correction, Morphology, Edge detect
* Selecting and croping the regions of interest(ROI)
* Resizing and picklizing

### `2.Select best classification model`
* Best score model
* Compact model(least compute)
* Least time counsuming

## 3.License
This repo follows GNU General Public License v3.0
