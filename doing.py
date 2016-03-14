from skimage import morphology
from scipy import ndimage
from os import listdir
from image_Proc import make_bw, grey2ero_dil, find_rect2
from nt import mkdir
import PIL
import skimage
import PIL.ImageOps
import numpy
import image_Proc as ip


GAMMA = 8
OUT_PATH = ['D://nor/n','d://err/e']

paths = ['d://jupyter_notebooks/imgs/nor_ori/',
        'd://jupyter_notebooks/imgs/error_ori/' ]

max_size=[0,0]

# for opath in OUT_PATH:
#     mkdir(opath[:-2])
#     
for path,opath in zip(paths,OUT_PATH):
    get_files = ip.get_pure_image(path = path,format='jpg')
    for imgs in get_files:
        img = PIL.Image.open(path+imgs)
        img = img.point(lambda x : x+60 if x > 110 else x).point(lambda x : 255*(x/255)**GAMMA)
        img_c = img.convert('L')
        img_ic = PIL.ImageOps.invert(img_c)
        img_ic = grey2ero_dil(img_ic,iterat=10, n_ero=20,n_dil= 30)
        img_bic = make_bw(img_ic)
        img_bic = img_bic.filter(PIL.ImageFilter.FIND_EDGES)
        img_bicr = PIL.ImageOps.invert(img_bic)
        lab_arr,num_lab = ndimage.measurements.label(img_bicr,structure=[[1,1,1],[1,0,1],[1,1,1]])
        print("num of object :  " ,imgs, num_lab)
        print("num of object :  " ,lab_arr)

        idx_crop = find_rect2(lab_arr,margin=200)
        print(idx_crop)
        im_final = img_c.crop(idx_crop)
        
#         if im_final.size[0] > max_size[0]:
#             max_size[0] = im_final.size[0]
#         if im_final.size[1] > max_size[1]:
#             max_size[1] = im_final.size[1]
        
        im_final = im_final.resize((50,50),resample=PIL.Image.BICUBIC)
        im_final.save(opath + imgs)

# print(max size)
# 
# 
# # TODO : Streching Imgs to max size 
# for imgs in listdir(path+'/test'):
#     img = PIL.open(path+'/test/'+imgs)
#     img
