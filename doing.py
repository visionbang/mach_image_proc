from scipy import ndimage
from os import path , mkdir
from image_proc import make_bw, grey2ero_dil, find_rect2
import PIL.ImageOps
import image_proc as ip

# Set Parameter
GAMMA = 8

# Set list of dir Path of input pictures 
#[0] - for normal
#[1] - for error

PATHS = ['d://jupyter_notebooks/imgs/nor_ori/',
        'd://jupyter_notebooks/imgs/error_ori/' ]

# Set list of dir Path of output pictures 
# use 
OUT_PATHS = ['D://nor/n','d://err/e']

# Set margin and size of output file
OUT_MARGIN = 200
OUT_SIZE = (50,50)

# Check output folder exist
for out_path in OUT_PATHS :
    if not path.isdir(out_path[:-2]):
        mkdir(out_path)
        
for path,opath in zip(PATHS,OUT_PATHS):
    get_files = ip.get_pure_imagef(path = path,format='jpg')
    for imgs in get_files:
        img = PIL.Image.open(path+imgs)
        # Adjust gamma 
        # @TODO: histogram control is need , median filter is needed
        img = img.point(lambda x : x+60 if x > 110 else x).point(lambda x : 255*(x/255)**GAMMA)
        img_c = img.convert('L')    # Convert to Grayscale
        img_ic = PIL.ImageOps.invert(img_c)
        img_ic = grey2ero_dil(img_ic,iterat=10, n_ero=20,n_dil= 30)
        img_bic = make_bw(img_ic)
        img_bic = img_bic.filter(PIL.ImageFilter.FIND_EDGES)    # Find edges 
        img_bicr = PIL.ImageOps.invert(img_bic)     # Invert images 
        lab_arr,num_lab = ndimage.measurements.label(img_bicr,structure=[[1,1,1]
                                                                         ,[1,0,1]
                                                                         ,[1,1,1]])     # To confirm clear images, label objects
        print("num of object :  " ,imgs, num_lab)
        print("array of object :  " ,lab_arr)
        idx_crop = find_rect2(lab_arr,margin=OUT_MARGIN)    # Get index rectangular position
        print(idx_crop)
        im_final = img_c.crop(idx_crop)     # Crop with pre-get position
        im_final = im_final.resize(OUT_SIZE,resample=PIL.Image.BICUBIC)     # Equalize and resize images
        im_final.save(opath + imgs)
