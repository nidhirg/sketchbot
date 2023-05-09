# -*- coding: utf-8 -*-

import sys
import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
import tensorflow as tf
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils
from matplotlib import image
from skimage.color import rgb2gray
from skimage import io

#mount drive



#Load data
apple = np.load('data/apple.npy')
banana = np.load('data/banana.npy')



#add a column for labels
apple = np.c_[apple, np.zeros(len(apple))]
banana = np.c_[banana, np.ones(len(banana))]


# Merging arrays and splitting the features and labels
y = np.concatenate((apple[:1000,-1], banana[:1000,-1]), axis=0).astype('float32') # the last column
X = np.concatenate((apple[:1000,:-1], banana[:1000,:-1]), axis=0).astype('float32') # all columns but the last

# Split data between train and test (80 - 20 ratio). Normalizing the value between 0 and 1
X_train, X_test, y_train, y_test = train_test_split(X/255.,y,test_size=0.2,random_state=0)

# Convert grayscale data to RGB format
X_train_rgb = np.stack((X_train,)*3, axis=-1)
X_test_rgb = np.stack((X_test,)*3, axis=-1)

# # reshape to be [samples][pixels][width][height]
X_train_cnn = X_train_rgb.reshape(X_train_rgb.shape[0], 28, 28, 3).astype('float32')
X_test_cnn = X_test_rgb.reshape(X_test_rgb.shape[0], 28, 28, 3).astype('float32')

# one hot encode outputs
y_train_cnn = np_utils.to_categorical(y_train)
y_test_cnn = np_utils.to_categorical(y_test)
num_classes = y_test_cnn.shape[1]



#create CNN model
def cnn_model():
    # create model
    model = Sequential()
    model.call = tf.function(model.call)
    model.add(Conv2D(30, (3, 3), input_shape=(28, 28,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam',  metrics=['accuracy'])
    return model


np.random.seed(0)
# Build the model
model_cnn = cnn_model()
# Fit the model
model_cnn.fit(X_train_cnn, y_train_cnn, validation_data=(X_test_cnn, y_test_cnn), epochs=3, batch_size=200)

model_cnn.save('model_cnn.h5')

# def confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
#     df_cm = pd.DataFrame(
#         confusion_matrix, index=class_names, columns=class_names, 
#     )
#     fig = plt.figure(figsize=figsize)
#     try:
#         heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
#     except ValueError:
#         raise ValueError("Confusion matrix values must be integers.")
#     heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
#     heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')

# class_names = ['apple', 'banana']
# confusion_matrix(c_matrix, class_names, figsize = (10,7), fontsize=14)

