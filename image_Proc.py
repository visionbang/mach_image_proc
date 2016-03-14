from skimage import morphology
from scipy import ndimage
from os import listdir
import PIL
import skimage
import PIL.ImageOps
import numpy


path = 'D:\project_tensorflow_class\Data (1)\original/err/'

%matplotlib

    
    
def get_pure_image():    
    j = 0
    get_files = listdir(path)
    get_files.sort()
    for i,file in enumerate(get_files):
        if file.find('.jpg') == -1:
            print(i,get_files.pop(i-j))
    print(get_files)
    return get_files

def make_bw (rgb):
    bw = rgb.point(lambda c : 255 if c >127 else 0)
    return bw

def grey2ero_dil (gray,iterat = 3, n_ero = 30, n_dil=20):
    for i in range(iterat):
        
        ero = ndimage.grey_erosion(gray, n_ero)
        #,structure=[[0,1,2],[0,2,0],[1,2,0]]

        ero_dil= ndimage.grey_dilation(e, n_dil)

    return PIL.Image.fromarray(ero_dil)

def pil_inv(img):
    img = PIL.ImageOps.invert(img)
    return img


def find_rect(x,margin = 0):
    '''return left upper right bottom'''
    max_size = [0,0]
    rect = []
    for j,i in  enumerate(x.sum(axis=0)):
        if i != x.shape[0]:
            print("left is ",j)
            rect.append(j-margin)
            break
        
    for j,i in  enumerate(x.sum(axis=1)):
        if i != x.shape[1]:
            print("upper is ",j)
            rect.append(j-margin)
            break

    for j,i in  enumerate(reversed(x.sum(axis=0))):
        if i != x.shape[0] :
            print("right is ",x.shape[1]-j)
            rect.append(x.shape[1]-j+margin)
            break

    for j,i in  enumerate(reversed(x.sum(axis=1))):
        if i != x.shape[1]:
            print("bottom is ",x.shape[0]-j)
            rect.append(x.shape[0]-j+margin)
            break
    return rect

def 





def imsave(img):
    img.save()
    
