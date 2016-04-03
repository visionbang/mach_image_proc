from os import listdir
import pickle
import matplotlib.pylab as plt

PATHS = ['d://err_new/', 'd://nor_new/']

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

# Save as pickle for convenience and further use
with  open('d://images_new.p',mode='wb+') as sm:
    pickle.dump(matrix,sm)
    sm.close()
with  open('d://labels_new.p',mode='wb+') as sl:
    pickle.dump(label,sl)
    sl.close()