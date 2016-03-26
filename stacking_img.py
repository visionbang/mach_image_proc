from os import listdir
import pickle
import matplotlib.pylab as plt

PATHS = ['d://err/', 'd//nor/']

matrix = []
label = []

# Stack all pictures as vector 
for path in PATHS:
    imgs = listdir(path)
    for img in imgs:
        matrix.append(plt.imread(path + img).flatteb())
        if img[0] != 'n':   # If file name got 'n' at first append 1
            label.append(1)
        else:
            label.append(0)
            
print('num of pics: ' + len(matrix))
print('num of labs: ' + len(label))

# Save as pickle for convenience and further use
with  open('d://images.p',mode='wb+') as sm:
    pickle.dump(matrix,sm)
    sm.close()
with  open('d://labels.p',mode='wb+') as sl:
    pickle.dump(label,sl)
    sl.close()