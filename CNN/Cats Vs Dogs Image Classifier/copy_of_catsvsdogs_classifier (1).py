# -*- coding: utf-8 -*-
"""Copy of CatsVsDogs Classifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14PVs1vO6ZxSkC1t2paVAt7pfIJ4omZu3
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 17:48:57 2018

@author: harsha
"""


from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
#import numpy as np


model =Sequential()
#model.add(Sequential)
model.add(Convolution2D(32,(3,3), input_shape =(64, 64,3,), activation='relu'))
model.add(MaxPooling2D(pool_size =(2,2)))
model.add(Convolution2D(32,(3,3),  activation='relu'))
model.add(MaxPooling2D(pool_size =(2,2)))

model.add(Flatten())

model.add(Dense(units =128, activation ='relu'))
model.add(Dense(units=1, activation ='sigmoid'))


model.compile(optimizer='adam' , loss='binary_crossentropy' , metrics =['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        '/datasets/train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        '/datasets/test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        training_set,
        steps_per_epoch=2000,
        epochs=10,
        validation_data=test_set,
        validation_steps=800)




import cv2
img = cv2.imread('/datasets/testset/copy_of_test/20.jpg')
img = cv2.resize(img, (64,64))
img = img.reshape(1, 64,64,3)
print(model.predict(img))