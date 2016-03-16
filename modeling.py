from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense, Dropout, Activation
from keras import backend
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
import numpy as np

import pickle
from keras.layers import Convolution2D ,MaxPooling2D,Flatten
a = open('d://labels.p')
b = open('d://images.p')
labels = pickle.load(a)
imgs = np.array(pickle.load(b))
imgs /= 255
labels = np_utils.to_categorical(labels,nb_classes=2)

#imgs for CNN

imgs2d= []
for img in imgs:
    imgs2d.append(np.reshape(img,(1,50,50)))
x_tr2,x_te2,y_tr2,y_te2 = train_test_split(imgs2d,labels,test_size= 0.2,random_state= 123)
# x_tr,x_te,y_tr,y_te = train_test_split(imgs,labels,test_size= 0.2,random_state= 123)
# model1 = Sequential()
# model1.add(Dense(30, input_dim=2500, activation="sigmoid", init='normal'))
# model1.add(Dense(1, activation="softmax", init='normal'))
# model1.compile(loss='categorical_crossentropy', optimizer=SGD())

model2 = Sequential()
model2.add(Convolution2D(50,3, 3, border_mode='same', input_shape=(1, 50, 50)))
model2.add(Activation('relu'))
model2.add(Convolution2D(50, 3, 3))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))

model2.add(Convolution2D(100, 3, 3, border_mode='same'))
model2.add(Activation('relu'))
model2.add(Convolution2D(100, 3, 3))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))

model2.add(Flatten())
model2.add(Dense(1250))
model2.add(Activation('relu'))
model2.add(Dense(2))
model2.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
model2.compile(loss='mse', optimizer=SGD(lr=0.09, decay=1e-6, momentum=0.9, nesterov=True))
# let's train the model using SGD + momentum (how original)
hist2 = model2.fit(np.array(x_tr2), y_tr2, nb_epoch=40,validation_split=0.1, batch_size=36,show_accuracy=True,shuffle=True)
model2.summary()