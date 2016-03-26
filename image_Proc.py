#-*- coding:utf-8 -*-

from scipy import ndimage
from os import listdir
import PIL
import PIL.ImageOps

    
def get_pure_imagef(path,format = '.jpg'):    
    
    '''
    Get list of only format files
    
    #Args 
    
    path : path for directory that is files exist
    format : only extension 
    '''
    
    get_files = listdir(path)
    get_files.sort()
    for i,file in enumerate(get_files):
        if file.find(format) == -1:
            print(i,get_files.pop(i))
    print(get_files)
    
    return get_files

def make_bw (gray,threshold= 0.5):
    
    '''
    Convert gray image to binary image
    
    #Args

    gray :  greyscaled or 1D PIL.Image 
    threshold : final threshold would be 255 * threshold for uint8 pixel values, 0< x < 1 float only 
    '''
    
    bw = gray.point(lambda c : 255 if c > int( 255 * threshold) else 0)
    
    return bw

def grey2ero_dil (gray,iterat = 3, n_ero = 30, n_dil=20):
    '''
    Function for image opening 
    
    #Args

    gray :  greyscaled PIL.Image 
    iterat : number of iteration , int
    n_ero : size of erosion, ints or tuple
    n_dil : size of dilation, ints or tuple
    
    
    '''
    for i in xrange(iterat):
        
        ero = ndimage.grey_erosion(gray, n_ero)
        ero_dil= ndimage.grey_dilation(ero, n_dil)

    return PIL.Image.fromarray(ero_dil)

def pil_inv(img):
    img = PIL.ImageOps.invert(img)
    return img

def find_rect2(x,margin = 0):
    
    '''
    Find feature's rectangular coordinate by finding none zero pixel value
    
    #Args:
    x : PIL.Image
    margin : size of margin
    
    #Returns list of position - [left, upper, right, bottom]
    '''
    
    rect = []
    x_sum_1=x.sum(axis=1)
    x_sum_0=x.sum(axis=0)
    
    for j,i in  enumerate(x_sum_0):
        if i != x_sum_0[1] & i != 0 :
            print("left is ",j)
            rect.append(j-margin)
            break
    for j,i in  enumerate(x_sum_1):
        if i != x_sum_1[1] & i != 0:
            print("upper is ",j)
            rect.append(j-margin)
            break

    for j,i in  enumerate(reversed(x_sum_0)):
        if i != x_sum_0[-2] & i != 0:
            print("right is ",x.shape[1]-j)
            rect.append(x.shape[1]-j+margin)
            break

    for j,i in  enumerate(reversed(x_sum_1)):
        if i != x_sum_1[-2] & i != 0:
            print("bottom is ",x.shape[0]-j)
            rect.append(x.shape[0]-j+margin)
            break
    return rect

    

