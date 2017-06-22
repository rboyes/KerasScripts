import os   
import datetime
import sys
import time 
import string 
import random
import pandas as pd
import numpy as np

if(len(sys.argv) < 2):
    print('Usage: DirectoryClassifier.py training-folder validation-folder model.h5 model.log')
    sys.exit(1)

sys.setrecursionlimit(10000)

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import *
import keras.preprocessing.image as image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger

from keras.layers import Input, merge, Dropout, Dense, Flatten, Activation
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras.models import Model, load_model
from keras import regularizers

from keras import backend as K
from keras.utils.data_utils import get_file


from keras.applications.resnet50 import ResNet50

K.set_image_dim_ordering('th')

trainingDir = sys.argv[1]
validationDir = sys.argv[2]
modelName = sys.argv[3]
logName = sys.argv[4]
nchannel = 3
nx = 224
ny = 224
batchSize = 25

resnetModel = ResNet50(include_top=False, weights='imagenet', input_shape=(nchannel, nx, ny))
        
print('The number of layers in the resnet model = %d' % (len(resnetModel.layers)))

numberOfBottleneckFeatures = 100
bottleneckTrainingDataGenerator = ImageDataGenerator(rescale = 1.0/255.0)
bottleneckValidationDataGenerator = ImageDataGenerator(rescale = 1.0/255.0)
bottleneckTrainingGenerator = bottleneckTrainingDataGenerator.flow_from_directory(trainingDir, target_size = (nx, ny), batch_size = batchSize, shuffle = False)
bottleneckValidationGenerator = bottleneckValidationDataGenerator.flow_from_directory(validationDir, target_size = (nx, ny), batch_size = batchSize, shuffle = False)
bottleneckTrainingFeatures = resnetModel.predict_generator(bottleneckTrainingGenerator, bottleneckTrainingGenerator.classes.shape[0])
bottleneckValidationFeatures = resnetModel.predict_generator(bottleneckValidationGenerator, bottleneckValidationGenerator.classes.shape[0])
bottleneckTrainingCategories = np_utils.to_categorical(bottleneckTrainingGenerator.classes)
bottleneckValidationCategories = np_utils.to_categorical(bottleneckValidationGenerator.classes)

newTop = Sequential()
newTop.add(Flatten(input_shape = bottleneckTrainingFeatures.shape[1:]))
newTop.add(Dense(512, activation='relu'))
newTop.add(Dropout(0.5))
newTop.add(Dense(bottleneckTrainingGenerator.nb_class, activation='softmax', name='predictions'))
newTop.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1.0E-3), metrics=['accuracy']) 

newTop.fit(bottleneckTrainingFeatures, bottleneckTrainingCategories, batch_size = batchSize, nb_epoch = 4, 
           validation_data = (bottleneckValidationFeatures, bottleneckValidationCategories), verbose = 1)

finalModel = Model(input = resnetModel.input, output = newTop(resnetModel.output))
print('The number of layers in the final model = %d' % (len(finalModel.layers)))

for layer in finalModel.layers[:(len(resnetModel.layers) - 21)]:
    layer.trainable = False

finalModel.compile(loss='categorical_crossentropy',optimizer=SGD(lr=1e-4, momentum=0.9),metrics=['accuracy'])
print(finalModel.summary())

trainingDataGenerator = ImageDataGenerator(rescale=1./255, rotation_range = 40, zoom_range = 0.15, horizontal_flip = True,                                       
                                           width_shift_range = 0.1, height_shift_range = 0.1, shear_range = 0.1)
validationDataGenerator = ImageDataGenerator(rescale=1./255)
trainingGenerator = trainingDataGenerator.flow_from_directory(trainingDir, target_size = (nx, ny), batch_size = batchSize, shuffle = True)
validationGenerator = validationDataGenerator.flow_from_directory(validationDir, target_size = (nx, ny), batch_size = batchSize, shuffle = False)

csvLogger = CSVLogger(logName, append=True)
checkPointer = ModelCheckpoint(filepath=modelName, verbose = 1, save_best_only = True)
finalModel.fit_generator(trainingGenerator, trainingGenerator.classes.shape[0], 100, validation_data = validationGenerator, 
                         nb_val_samples = validationGenerator.classes.shape[0], callbacks = [checkPointer, csvLogger])