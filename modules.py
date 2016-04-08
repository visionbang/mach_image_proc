#-*- coding:utf-8 -*-

from scipy import ndimage
from os import listdir, path
import matplotlib.pylab as plt
import PIL.ImageOps
import pickle
import PIL


def get_pure_imagef(dir_path,formats = None):    
    '''
    Get list of only format files
    
    #Args 
    
    dir_path : Str, path for directory that is files exist
    formats : list, bunch of extension
    
    e.g)get_pure_imagef('d://', formats = ['.jpg','.JPG'])    

    '''
    if not path.isdir(dir_path):
        print('Please enter valid directory path')
        return

    
    get_files = listdir(dir_path)
    get_files.sort()
    real_imgs = []
    
    # Assign jpg type as default
    if formats is None:
        formats = ['.jpg','.JPG'] 
    if type(formats) != list :
        print('Please enter valid list to "formats"')
        return
    
    
    for i,file in enumerate(get_files):
        for format in formats:
            if file.find(format) != -1:
                print(i,get_files[i])
                real_imgs.append(get_files[i])
    return real_imgs

def make_bw (grey,threshold= 0.5):
    '''
    Convert gray image to binary image
    
    #Args

    gray :  1D PIL.Image, greyscale image
    threshold : float, threshold would be 255 * threshold for uint8 pixel values, 0< x < 1 
    '''
    
    bw = grey.point(lambda x : 255 if x > int( 255 * threshold) else 0)
    
    return bw

def grey2ero_dil (grey,iterat = 3, n_ero = 30, n_dil=20):
    '''
    Function for morphological opening 
    
    #Args

    grey :  PIL.Image, greyscaled
    iterat :int, number of iteration
    n_ero : ints or tuple, size of erosion
    n_dil : ints or tuple, size of dilation
    
    
    '''
    for i in xrange(iterat):
        
        ero = ndimage.grey_erosion(grey, n_ero)
        ero_dil= ndimage.grey_dilation(ero, n_dil)

    return PIL.Image.fromarray(ero_dil)

def pil_inv(img):
    '''
    Invert image 
    
    #Args:
    
    img : PIL.Image
    '''
    
    img = PIL.ImageOps.invert(img)
    return img

def find_rect(x,margin = 100):
    '''
    Find feature's rectangular coordinate by finding none zero pixel value
    
    #Args:
    
    x : numpy.ndarray, array of image
    margin : int, size of margin
    
    #Returns list of position - [left, upper, right, bottom]
    '''
    rect = []
    x_sum_1=x.sum(axis=1)
    x_sum_0=x.sum(axis=0)
    
    for i,j in  enumerate(x_sum_0):
        if j != x_sum_0[1] & j != 0 :
            print("left is ",i)
            rect.append(i-margin)
            break
        
    for i,j in  enumerate(x_sum_1):
        if j != x_sum_1[1] & j != 0:
            print("upper is ",i)
            rect.append(i-margin)
            break

    for i,j in  enumerate(reversed(x_sum_0)):
        if j != x_sum_0[-2] & j != 0:
            print("right is ",x.shape[1]-i)
            rect.append(x.shape[1]-i+margin)
            break

    for i,j in  enumerate(reversed(x_sum_1)):
        if j != x_sum_1[-2] & j != 0:
            print("bottom is ",x.shape[0]-i)
            rect.append(x.shape[0]-i+margin)
            break
    return rect

def stack_imgs (nor_dir,err_dir):
    '''
    Make images vectorize and stack to matrix
    
    #Args:
    nor_dir : Str , directory existing normal images
    err_dir : Str , directory existing error images
    
    
    #Returns list of vectorized images, labels
    '''
    
    PATHS = [err_dir,  nor_dir]
    matrix = []
    label = []
    
    # Stack all pictures as vector 
    for path in PATHS:
        imgs = listdir(path)
        for img in imgs:
            matrix.append(plt.imread(path + img).flatten())
            if img[0] != 'n':   # If file name got 'e' at first append 1
                label.append(1)
            else:
                label.append(0)

    print('num of pics: ' , len(matrix))
    print('num of labs: ' , len(label))
    return matrix , label
                
def mat2pickle(img_out_path ,lab_out_path,list_img,list_lab):
    '''
    Make list to pickle
    
    #Args:
    
    img_out_path : Str, path that you want to export picklized image list, should be formated as '.p'   
    lab_out_path : Str, path that you want to export picklized label list, should be formated as '.p'
    list_img : list, list of vectorized images
    list_lab : list, list of vectorized labels
    
    #Returns None
    '''
    
    # Save as pickle for convenience and further use
    with  open(img_out_path ,mode='wb+') as sm:
        pickle.dump(list_img,sm)
        sm.close()
    with  open(lab_out_path,mode='wb+') as sl:
        pickle.dump(list_lab,sl)
        sl.close()





