from scipy import ndimage
from os import path , mkdir
from image_Proc import get_pure_imagef,make_bw,grey2ero_dil, find_rect2,mat2pickle,stack_imgs
import PIL.ImageOps
import numpy as np

# Set Parameter for GAMMA correction
GAMMA = 10

# Set list of dir Path of input pictures 
#[0] - for normal
#[1] - for error

PATHS = ['d://jupyter_notebooks/imgs/nor_new/',
        'd://jupyter_notebooks/imgs/err_new/' ]

# Set list of dir Path of output pictures 
OUT_PATHS = ['D://nor_new2/','d://err_new2/']


# Set list of path for exporting images and labels to pickle
EXP_PATHS = ['D://stack_images.p','D://stack_lables.p']

# Set margin and size of output file
OUT_MARGIN = 200
OUT_SIZE = (50,50)

OUT_PATHS[0] += 'n'
OUT_PATHS[1] += 'e'

# Check output folder exist
for out_path in OUT_PATHS :
    if not path.isdir(out_path[:-2]):
        mkdir(out_path[:-2])

for path,opath in zip(PATHS,OUT_PATHS):
    get_files = get_pure_imagef(path = path,format=['jpg','.JPG'])
    for imgs in get_files:
        img = PIL.Image.open(path+imgs)
        # Adjust gamma and boost pixel to get clear edge
        # @TODO: histogram control is needed , median filter is needed
        img = img.point(lambda x : x+60 if x > 110 else x).point(lambda x : 255*(x/255)**GAMMA)
        img_c = img.convert('L')    # Convert to Grayscale
#         img_ic = PIL.ImageOps.invert(img_c)
        img_ic = grey2ero_dil(img_c,iterat=1, n_ero=30,n_dil= 30)
        img_bic = make_bw(img_ic)
        img_bic = img_bic.filter(PIL.ImageFilter.FIND_EDGES)    # Find edges 
        img_bicr = PIL.ImageOps.invert(img_bic)     # Invert images 

#         it's remained further use for getting n pieces of feature
#         lab_arr,num_lab = ndimage.measurements.label(img_bic,structure=[[1,1,1],[1,0,1] ,[1,1,1]])     # To confirm clear images, label objects
#         print("num of objects :  " ,imgs, num_lab)
#         print("array of objects :  " ,lab_arr)

        idx_crop = find_rect2(np.array(img_bicr),margin=OUT_MARGIN)    # Get index rectangular position
        print(idx_crop)
        im_final = img_c.crop(idx_crop)     # Crop with pre-get position
        im_final = im_final.resize(OUT_SIZE,resample=PIL.Image.BICUBIC)     # Equalize and resize images
        im_final.save(opath + imgs)

