# Automation of car dent detection using Mach-band image

List of files and sepcifications

#### Ipython Notebook files
* `represent_0415.ipynb` : practical analysis, visualized
* `represent_preproc_0415.ipynb` : demonstration of preprocessing 
* `project_image_sandbox.ipynb` : experimental preprocessing
* `theano_sandbox.ipynb` : experimental modeling


#### python files
* `modules.py` : all of modules are defined in here
* `pipe_preproc.py` : all of preprocessing can be done in here 
* `modeling.py` : practical analysis 


## 1.Abstract

![current inspection](/imgs/inspection.png)

In terms of car dent inspection process, 
* Workers are suffering from extremely high illumination(about 2000 Lux)
* Inspecting with their eyes
* A defect rate of newly producted car(within 3 months) is about 11% for surface defects

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
