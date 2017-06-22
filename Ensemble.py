import os   
import datetime
import sys
import time 
import string 
import random
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss

def ensemblePrediction(predictionFiles, outputName):

    namesDataFrame = pd.read_csv(predictionFiles[0], header=None)
    numberOfImages = len(namesDataFrame)     
    names = []
    for index in range(0, numberOfImages):
        names.append(namesDataFrame.iloc[index][0])

    for fileIndex, predictionFile in enumerate(predictionFiles):
        prediction = pd.read_csv(predictionFile, header=None) 
        prediction.drop(prediction.columns[[0]], axis = 1, inplace = True)
        
        predictionMatrix = prediction.as_matrix().astype(float)

        if fileIndex == 0:
            averagePrediction = predictionMatrix
        else:
            averagePrediction += predictionMatrix

    averagePrediction /= len(predictionFiles)
    outputData = np.column_stack([names, averagePrediction])
    np.savetxt(outputName, outputData, delimiter=',', fmt='%s') 

if(len(sys.argv) < 2):
    print('Usage: Ensemble.py a.csv b.csv c.csv output.csv')
    sys.exit(1)

ensemblePrediction(sys.argv[1:len(sys.argv)-1], sys.argv[len(sys.argv) - 1])
