import os   
import datetime
import sys
import time 
import string 
import random
import pandas as pd
import numpy as np
import gc

if(len(sys.argv) < 2):
    print('Usage: CSVTrainer.py train.csv validation.csv model.h5 log.txt')
    sys.exit(1)

trainingName = sys.argv[1]
validationName = sys.argv[2]
modelName = sys.argv[3]
logName = sys.argv[4]

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

from sklearn.metrics import accuracy_score

from keras.applications import resnet50

def readCSV(fileList):

    namesDataFrame = pd.read_csv(fileList)

    flatten = lambda l: [item for sublist in l for item in sublist]
    labels = sorted(list(set(flatten([l.split(' ') for l in namesDataFrame['tags'].values]))))

    labelMap = {l: i for i, l in enumerate(labels)}

    numberOfLabels = len(labels)
    numberOfImages = len(namesDataFrame)

    fileNames = []
    y = np.zeros((numberOfImages, numberOfLabels), np.float32)
    for index in range(0, numberOfImages):
        inputImage = image.img_to_array(image.load_img(namesDataFrame.iloc[index][0]))
        fileNames.append(namesDataFrame.iloc[index][0])
        tags = namesDataFrame.iloc[index][1]
        for t in tags.split(' '):
            y[index, labelMap[t]] = 1.0 
    return (fileNames, y, labelMap)

print('Loading images..........', end = '',flush = True)

(trainingFileNames, trainY, trainingLabelMap) = readCSV(trainingName)
(validationFileNames, validationY, validationLabelMap) = readCSV(validationName)

print('done.', flush = True)

if len(trainingLabelMap) != len(validationLabelMap):
    print("Label maps for training and validation are not equal")
    sys.exit(1)

numberOfTrainingImages = len(trainingFileNames)
numberOfValidationImages = len(validationFileNames)
numberOfChannels = 3
nx = 256
ny = 256
batchSize = 25
lossName = 'binary_crossentropy'
activationName = 'sigmoid'

resnetModel = resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(numberOfChannels, nx, ny))
        
print('The number of layers in the resnet model = %d' % (len(resnetModel.layers)))

bottleneckTrainingDataGenerator = ImageDataGenerator(rescale = 1.0/255.0)
bottleneckValidationDataGenerator = ImageDataGenerator(rescale = 1.0/255.0)
bottleneckTrainingGenerator = bottleneckTrainingDataGenerator.flow_from_filenames(trainingFileNames, target_size = (nx, ny), batch_size = batchSize, shuffle = False)
bottleneckValidationGenerator = bottleneckTrainingDataGenerator.flow_from_filenames(validationFileNames, target_size = (nx, ny), batch_size = batchSize, shuffle = False)
bottleneckTrainingFeatures = resnetModel.predict_generator(bottleneckTrainingGenerator, numberOfTrainingImages)
bottleneckValidationFeatures = resnetModel.predict_generator(bottleneckValidationGenerator, numberOfValidationImages)

newTop = Sequential()
newTop.add(Flatten(input_shape = bottleneckTrainingFeatures.shape[1:]))
newTop.add(Dense(512, activation='relu'))
newTop.add(Dropout(0.5))
newTop.add(Dense(len(trainingLabelMap), activation=activationName, name='predictions'))
newTop.compile(loss=lossName, optimizer=Adam(lr=1.0E-3)) 

print('Fitting predicted features...', flush = True)

newTop.fit(bottleneckTrainingFeatures, trainY, validation_data = (bottleneckValidationFeatures, validationY), verbose = 1, batch_size = batchSize, nb_epoch = 25)

print('Done.', flush = True)

finalModel = Model(input = resnetModel.input, output = newTop(resnetModel.output))
print('The number of layers in the final model = %d' % (len(finalModel.layers)))

for layer in finalModel.layers[:(len(resnetModel.layers) - 21)]:
    layer.trainable = False

finalModel.compile(loss=lossName,optimizer=SGD(lr=1e-4, momentum=0.9))
print(finalModel.summary())

# Could add vertical_flip = True
trainingDataGenerator = ImageDataGenerator(rescale = 1.0/255.0, rotation_range = 40, zoom_range = 0.15, horizontal_flip = True,                                       
                                           width_shift_range = 0.1, height_shift_range = 0.1, shear_range = 0.1)
validationDataGenerator = ImageDataGenerator(rescale = 1.0/255.0)
trainingGenerator = trainingDataGenerator.flow_from_filenames(trainingFileNames, trainY, batch_size = batchSize, target_size = (nx, ny))
validationGenerator = validationDataGenerator.flow_from_filenames(validationFileNames, validationY, batch_size = batchSize, target_size = (nx, ny))

csvLogger = CSVLogger(logName, append=True)
checkPointer = ModelCheckpoint(filepath=modelName, verbose = 1, save_best_only = True)
finalModel.fit_generator(trainingGenerator, numberOfTrainingImages, 50, validation_data = validationGenerator, 
                         nb_val_samples = numberOfValidationImages, callbacks = [checkPointer, csvLogger])
