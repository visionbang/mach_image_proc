# Automation of car defect detection using Mach-band image

## Abstract


### Samples of original pictures for each class
|Class|Num of pics|Mach-band images|
|:---:|:---:|:---:|
|`Normal`|42|![Normal Images](/imgs/nor_merged.png)|
|`Error`|47|![Error Images](/imgs/err_merged.png)|

## Purpose

### `1.Automate preprocessing`
* Denoising - Gamma correction, Morphology, Edge detect
* Selecting and croping the regions of interest(ROI)
* Resizing and picklizing

### `2.Select best classification model`
* Best accuracy model
* Compact model(least compute)

### `3.PS.`
* Use GPU on neural model
* Try Ensenble on ML models


## License
This repo follows GNU General Public License v3.0
