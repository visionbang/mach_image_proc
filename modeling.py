from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D ,MaxPooling2D,Flatten
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from keras.callbacks import History
from keras.models import Sequential
from keras.utils import np_utils
from keras.optimizers import SGD
import matplotlib.pylab as plt
import numpy as np
import pickle

# Load files and regularize 
a = open('d://labels.p')
b = open('d://images.p')
labels = pickle.load(a)
imgs = np.array(pickle.load(b))
imgs /= 255
labels = np_utils.to_categorical(labels,nb_classes=2)

# Categorical to Label for further use in classification_report
def cat2lab (x):
    '''only for binary category'''
    return [0 if s[0] else 1 for s in x]

#imgs for CNN - it's shape (1,50,50) 
imgs2d= []
for img in imgs:
    imgs2d.append(np.reshape(img,(1,50,50)))

x_tr,x_te,y_tr,y_te = train_test_split(imgs,labels,test_size= 0.2,random_state= 123)    # Flattened imgs
x_tr2,x_te2,y_tr2,y_te2 = train_test_split(imgs2d,labels,test_size= 0.2,random_state= 123)    #  (1,50,50) shaped imgs for CNN

# Simple neural for trial - sigmoid
model1 = Sequential()
model1.add(Dense(30, input_dim=2500, activation="sigmoid", init='normal'))
model1.add(Dropout(0.25))
model1.add(Dense(1, activation="softmax", init='normal'))
model1.compile(loss='categorical_crossentropy', optimizer=SGD())

hist1 = model1.fit(np.array(x_tr2), y_tr2, nb_epoch=500,validation_split=0.3, batch_size=30,show_accuracy=True,shuffle=True)

print(model1.summary())
plt.plot(hist1.history['acc'],label='acc')
plt.plot(hist1.history['loss'],label='loss')
plt.plot(hist1.history['val_loss'],label='val_loss')
plt.plot(hist1.history['val_acc'],label='val_acc')
plt.legend()
plt.grid('off')
plt.show()
 
# CNN modeling some layers are hashed to make compact model

model2 = Sequential()
history2 = History() # Remained for callback
model2.add(Convolution2D(10,10,10, border_mode='same', input_shape=(1, 50, 50)))
model2.add(Activation('relu'))
# model2.add(Convolution2D(10,10,10,init='uniform'))
# model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))

model2.add(Convolution2D(10,10,10, border_mode='same'))
model2.add(Activation('relu'))
# model2.add(Convolution2D(10,10,10,init='uniform'))
# model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))

model2.add(Flatten())
model2.add(Dense(1250,init='uniform'))
model2.add(Activation('relu'))
model2.add(Dense(2))
model2.add(Activation('softmax'))
model2.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.6, nesterov=True))

hist2 = model2.fit(np.array(x_tr2), y_tr2, nb_epoch=200,validation_split=0.2,
                    batch_size=30,show_accuracy=True,shuffle=True,callbacks=[history2])

print(model2.summary())
plt.figure(figsize=(15,15))
plt.plot(history2.history['acc'],label='acc')
plt.plot(history2.history['loss'],label='loss')
plt.plot(history2.history['val_loss'],label='val_loss')
plt.plot(history2.history['val_acc'],label='val_acc')
plt.legend()
plt.grid('off')
plt.show()

plt.figure(figsize=(15,15))

def plot_wegh (model):
    '''
    Plot weights of convolution layer
    only for first layer
    
    #Args
    model : fitted model
    '''
    wegh_arr = model.get_weights()
    num = len(wegh_arr[0])
    if type(np.sqrt(num)) is int:
        col = np.sqrt(num)
        row = np.sqrt(num) 
    else:
        col = int(num/2)
        row = int(num/col)
        
    fig ,axes = plt.subplots(row,col, subplot_kw={'xticks': [], 'yticks': []})
    plt.subplots_adjust(hspace=0.02,wspace = 0.05)
    
    for i, ax in zip(xrange(num),axes.flat):
        
        ax.imshow(wegh_arr[0][i][0])
        ax.grid('off')
    plt.show()

y_pred2 = model2.predict_classes(np.array(x_te2))
y_ten2 = cat2lab(y_te2)

print(classification_report(y_ten2,y_pred2))

### Remained for experimental use
# model3 = Sequential()
# history3 = History()
# model3.add(Convolution2D(50,5, 5, border_mode='same',activation='tanh' ,input_shape=(1, 50, 50)))
# # model3.add(Convolution2D(50, 5, 5,,activation='logistic' ,init='uniform'))
# model3.add(MaxPooling2D(pool_size=(2, 2)))
# model3.add(Dropout(0.25))
# 
# # model3.add(Convolution2D(100, 5, 5, border_mode='same'))
# # model3.add(Activation('LSTM'))
# # model3.add(Convolution2D(100, 5, 5,init='uniform'))
# # model3.add(Activation('LSTM'))
# # model3.add(MaxPooling2D(pool_size=(2, 2)))
# # model3.add(Dropout(0.25))
# 
# model3.add(Flatten())
# model3.add(Dense(1250,init='uniform'))
# model3.add(Activation('tanh'))
# model3.add(Dense(2,activation='softmax'))
# model3.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.6, nesterov=True))

