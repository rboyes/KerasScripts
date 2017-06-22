import os   
import datetime
import sys
import time 
import string 
import random
import pandas as pd
import numpy as np

if(len(sys.argv) < 2):
    print('Usage: DirectoryTester.py test-folder model.h5 output.txt')
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

from sklearn.metrics import accuracy_score

from keras.applications import vgg16, vgg19
from keras.applications import resnet50
from keras.applications import inception_v3

def writeSubmission(testingDir, modelName, outputName):

    model = load_model(modelName)

    testingDataGenerator = ImageDataGenerator(rescale=1./255, rotation_range = 40, zoom_range = 0.15, horizontal_flip = True,                                       
                                              width_shift_range = 0.1, height_shift_range = 0.1, shear_range = 0.1)

    numberOfSamples = testingDataGenerator.flow_from_directory(testingDir).classes.shape[0]
    numberOfAugmentations = 9
    for index in range(numberOfAugmentations):
        print('Predicting outcome from augmentation %d of %d' % (index + 1, numberOfAugmentations))
        random_seed = np.random.randint(0, 100000 + 1)
        testingGenerator = testingDataGenerator.flow_from_directory(testingDir, target_size = (224, 224), batch_size = 25, shuffle = False,                                                                
                                                                    class_mode = None, classes = None, seed = random_seed)
        if index == 0:
            predictions = model.predict_generator(testingGenerator, numberOfSamples)
        else:
            predictions = predictions + model.predict_generator(testingGenerator, numberOfSamples)

    predictions = predictions/numberOfAugmentations
    
    baseNames = []
    for index in range(0, predictions.shape[0]):
        baseNames.append(os.path.basename(testingGenerator.filenames[index]))

    outputData = np.column_stack([baseNames, predictions])

    np.savetxt(outputName, outputData, delimiter=',', fmt='%s')

K.set_image_dim_ordering('th')

writeSubmission(sys.argv[1], sys.argv[2], sys.argv[3])
