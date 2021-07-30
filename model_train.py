
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import cv2
import os
from keras.utils import np_utils
from sklearn.utils import shuffle
from tensorflow.python.keras.engine import sequential
from keras import regularizers
from keras.regularizers import l1
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import *

# Path to the dataset Train and Test folders

img_folder = '../DevanagariHandwrittenCharacterDataset/Train'
test_folder = '../DevanagariHandwrittenCharacterDataset/Test'

# Initializing some variables

img_data_array  =[]
class_name = []

test_data_array=[]
test_name=[]

# Preparing Data

for dir1 in os.listdir(img_folder):
     for file in os.listdir(os.path.join(img_folder,dir1)):
          image_path = os.path.join(img_folder,dir1,file)
          image = cv2.imread(image_path)
          image= cv2.resize(image,(28,28))
          image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
          image = np.array(image)
          image = image.astype('float32')
          image = image/255
          img_data_array.append(image)
          class_name.append(dir1)

for dir2 in os.listdir(test_folder):
     for file in os.listdir(os.path.join(test_folder,dir2)):
          test_path=os.path.join(test_folder,dir2,file)
          test = cv2.imread(test_path)
          test = cv2.resize(test,(28,28))
          test = cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)
          test=np.array(test)
          test=test.astype('float32')
          test=test/255
          test_data_array.append(test)
          test_name.append(dir2)

class_name = list(map(int,class_name))
img_data_array = np.array(img_data_array)
class_name = np.array(class_name)
class_name = class_name - 1
img_data_array = img_data_array.reshape(-1,28,28,1)
class_name = np_utils.to_categorical(class_name,36)

test_name=list(map(int,test_name))
test_data_array=np.array(test_data_array)
test_name=np.array(test_name)
test_name=test_name-1
test_data_array=test_data_array.reshape(-1,28,28,1)
test_name=np_utils.to_categorical(test_name,36)

img_data_array,class_name=shuffle(img_data_array,class_name)

# Building model 

model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape = (28,28,1),data_format='channels_last'))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dense(36,activation='softmax'))

sgd=SGD(learning_rate=0.01,decay=1e-6,momentum=0.9,nesterov=True)

model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(img_data_array,class_name,batch_size=32 ,epochs=30,validation_split=0.2,shuffle=True)

loss, accuracy= model.evaluate(test_data_array,test_name,batch_size=32,verbose=1)

print(accuracy)
print(loss)

model.save('model_final.h5')