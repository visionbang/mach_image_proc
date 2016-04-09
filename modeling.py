#-*- coding:utf-8 -*-
from __future__ import division
from keras.layers import Convolution2D ,MaxPooling2D,Flatten
from keras.layers.core import Dense, Dropout, Activation
from keras.regularizers import l2,activity_l2
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split, StratifiedKFold ,cross_val_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import matplotlib.pylab as plt
import numpy as np
import pickle


# Categorical to Label for further use in classification_report
def cat2lab (cat): 
    '''only for binary category
    
    #Args:
    
    cat : binary categorical variable
    '''
    return np.array([0 if s[0] else 1 for s in cat])

def plot_wegh (model):
    '''
    Plot weights of convolution layer
    
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


# Load files and regularize 
img_pickle = open('d://labels_new.p')
lab_pickle = open('d://images_new.p')
labels = np.array(pickle.load(img_pickle))
imgs = np.array(pickle.load(lab_pickle))
reg_imgs = imgs /255
cat_labels = np_utils.to_categorical(labels,nb_classes=2) 

#reshape to shape (50,50)
reg_imgs_2d =[]
for img in reg_imgs:
    reg_imgs_2d.append(np.reshape(img,(50,50))) 
reg_imgs_2d = np.array(reg_imgs_2d)

#reshape to shape (1,50,50) for CNN
reg_imgs_3d =[]
for img in reg_imgs:
    reg_imgs_3d.append(np.reshape(img,(1,50,50)))
reg_imgs_3d = np.array(reg_imgs_3d)

x_tr1,x_te1,y_tr1,y_te1 = train_test_split(reg_imgs,cat_labels,test_size= 0.2,random_state= 123)
x_tr2,x_te2,y_tr2,y_te2 = train_test_split(reg_imgs_2d,cat_labels,test_size= 0.2,random_state= 123)
x_tr3,x_te3,y_tr3,y_te3 = train_test_split(reg_imgs_3d,cat_labels,test_size= 0.2,random_state= 123)
x_trn1,x_ten1,y_trn1,y_ten1 = train_test_split(reg_imgs,labels,test_size= 0.2,random_state= 123)

#Simple neuron 
model1 = Sequential()
model1.add(Dense(2500, input_dim=2500,init='uniform',activation='relu'))
model1.add(Dropout(0.25))
#                  ,activity_regularizer=activity_l2(0.01)))
model1.add(Dense(2, activation="softmax"))
model1.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.005,decay= 1e-6,momentum=0.7,nesterov=True))

hist1 = model1.fit(x_tr1, y_tr1, nb_epoch=500,validation_split=0.2 ,batch_size=50,show_accuracy=True,verbose=1)

print(model1.summary())
print(model1.evaluate(x_te1,y_te1,batch_size=50,show_accuracy=True))
y_pred1 = model1.predict_classes(x_te1,20)
y_ten1 = cat2lab(y_te1)
print(classification_report(y_ten1,y_pred1))

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
model2.add(Convolution2D(10,10, 10, border_mode='same', input_shape=(1, 50, 50)))
model2.add(Activation('relu'))
# model2.add(Convolution2D(50, 5, 5,init='uniform'))
# model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Convolution2D(10, 10, 10,init='uniform' ,border_mode='same'))
model2.add(Activation('relu'))
# model2.add(Convolution2D(100, 5, 5,init='uniform'))
# model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.35))

model2.add(Flatten())
model2.add(Dense(1250,init='uniform'))
model2.add(Activation('relu'))
model2.add(Dense(2,activation='softmax'))
model2.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01,decay=1e-6,
                                                              momentum=0.6,
                                                              nesterov=True))

hist2 = model2.fit(x_tr3, y_tr3, nb_epoch=300 , batch_size=50 ,validation_split=0.2, show_accuracy=True ,shuffle=True,verbose =1)

print(model2.summary())
print(model2.evaluate(x_te3,y_te3,batch_size=50,show_accuracy=True,verbose=1))
y_pred2 = model2.predict_classy_ten2 = cat2lab(y_te3)
y_ten2 = cat2lab(y_te3)
print(classification_report(y_ten2,y_pred2))

plt.figure(figsize=(25,15))
plt.plot(hist2.history['acc'],label='acc')
plt.plot(hist2.history['loss'],label='loss')
plt.plot(hist2.history['val_acc'],'--',label='val_acc')
plt.plot(hist2.history['val_loss'],'--',label='val_loss')
plt.legend()
plt.ylim(0,max(hist2.history['acc'])+0.05)
plt.grid('off')
plt.show()

plot_wegh(model2)

y_pred2 = model2.predict_classes(x_te3)
y_ten2 = cat2lab(y_te3)
print('score for CNN is :')
print(model2.evaluate(x_te3,y_te3, batch_size=50, show_accuracy=True, verbose=1))
print(classification_report(y_ten2,y_pred2))


cv = StratifiedKFold(labels,n_folds=10,shuffle=True)
params = {'C' : [1e1, 1e2, 1e3,1e4,1e5],
           'gamma' : [0.0001,0.0005,0.001,0.005,0.01]}

clf_grid = GridSearchCV(SVC(kernel='rbf'),params,cv=cv)

model3 = clf_grid.fit(reg_imgs,labels)
print('score for grid-searched SVM is :')
print(model3.best_score_,model3.best_score_)

#demonstration of upper GridSearchCV method

svc_rslt = []
for x,y in cv: 
    clf = SVC(kernel='rbf',C=10.0,gamma = 0.005,)
    clf.fit(reg_imgs[x],labels[x])
    svc_rslt.append(clf.score(reg_imgs[y], labels[y]))
svc_rslt = np.array(svc_rslt)

print('cross validated SVC score is ' , svc_rslt.mean())


# tried ensenble with various algorithm
ens1 = RandomForestClassifier(n_estimators =  250 , max_depth= None,verbose=1)

ens2 = AdaBoostClassifier(SVC(kernel='rbf',gamma=0.005,C = 10.0),
                          algorithm="SAMME",
                          n_estimators=100,
                          learning_rate=0.01)

ens3  = AdaBoostClassifier(DecisionTreeClassifier(max_depth=None),
                         algorithm="SAMME",
                         n_estimators=100,
                          learning_rate=0.01)
ens1.fit(x_trn1, y_trn1)
ens2.fit(x_trn1, y_trn1)
ens3.fit(x_trn1, y_trn1)

print('RandomForest score : ',ens1.score(x_ten1,y_ten1))
print('Adaboost-SVM score : ',ens2.score(x_ten1,y_ten1))
print('Adaboost-Decision score : ',ens3.score(x_ten1,y_ten1))

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







