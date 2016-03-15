from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense, Dropout, Activation
from keras import backend
from sklearn.cross_validation import train_test_split
import pickle
a = open('d://labels.p')
b = open('d://images.p')
labels = pickle.load(a)
imgs = pickle.load(b)
x_tr,x_te,y_tr,y_te = train_test_split(imgs,labels,test_size= 0.2,random_state= 123)
model1 = Sequential()
model1.add(Dense(30, input_dim=2500, activation="sigmoid", init='normal'))
model1.add(Dense(1, activation="softmax", init='normal'))
model1.compile(loss='categorical_crossentropy', optimizer=SGD())

