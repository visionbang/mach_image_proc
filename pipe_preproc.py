#-*- coding:utf-8 -*-

from os import path , mkdir
import modules as ip
import PIL
import numpy as np

# Set Parameter for GAMMA correction
GAMMA = 8

# Set list of dir Path of input pictures 
#[0] - for normal
#[1] - for error
PATHS = ['d://jupyter_notebooks/imgs/nor_new/',
        'd://jupyter_notebooks/imgs/err_new/' ]

# Set list of dir Path of resized pictures 
#[0] - for normal
#[1] - for error
OUT_PATHS = ['D://nor_new3/','d://err_new3/']


# Set list of path for exporting images and labels to pickle, should have '.p' in the end of string
#[0] - for imgages
#[1] - for labels
EXP_PATHS = ['D://stack_images.p','D://stack_lables.p']

# Set margin and size of output file
OUT_MARGIN = 200
OUT_SIZE = (50,50)

# For labeling append a character 
OUT_PATHS[0] += 'n'
OUT_PATHS[1] += 'e'

# Check output folder exist
for out_path in OUT_PATHS :
    if not path.isdir(out_path[:-2]):
        mkdir(out_path[:-2])
        print('not existing output folder, create folder : ', out_path[:-2])

for path,opath in zip(PATHS,OUT_PATHS):
    get_files = ip.get_pure_imagef(dir_path = path,formats=['jpg','.JPG'])
    
    print('List of imgs in ' + path  + ' are : ' , get_files)
    for imgs in get_files:
        img = PIL.Image.open(path+imgs)
        # Adjust gamma and boost pixel to get clear edge
        # @TODO: histogram control is needed , median filter may be  needed
        img = img.point(lambda x : x+60 if x > 110 else x).point(lambda x : 255*(x/255)**GAMMA)
        img_c = img.convert('L')    # Convert to Greyscale
        img_ic = ip.grey2ero_dil(img_c,iterat=1, n_ero=30,n_dil= 35)
        img_bic = ip.make_bw(img_ic)
        img_bic = img_bic.filter(PIL.ImageFilter.FIND_EDGES)    # Find edges 
        img_bicr = ip.pil_inv(img_bic)     # Invert images 

#         it's remained further use for getting n objects of feature
#         lab_arr,num_lab = ndimage.measurements.label(img_bic,structure=[[1,1,1],[1,0,1] ,[1,1,1]])     # To confirm clear images, label objects
#         print("num of objects :  " ,imgs, num_lab)
#         print("array of objects :  " ,lab_arr)
        print('for image : ', imgs)
        idx_crop = ip.find_rect(np.array(img_bicr),margin=OUT_MARGIN)    # Get index rectangular position
        print(idx_crop)
        im_final = img_c.crop(idx_crop)     # Crop from gamma corrected image with pre-get position
        im_final = im_final.resize(OUT_SIZE,resample=PIL.Image.BICUBIC)     # Equalize and resize images
        im_final.save(opath + imgs)
print('Preprocessing and saving completed')

print('Start picklizing')
lst_imgs, lst_labs = ip.stack_imgs(OUT_PATHS[0][:-1], OUT_PATHS[1][:-1])
ip.mat2pickle(EXP_PATHS[0], EXP_PATHS[1], lst_imgs, lst_labs)
print('picklinzing completed')

