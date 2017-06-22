import os   
import datetime
import sys
import time 
import string 
import random
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss

def readCSV(csvName):
    dataFrame = pd.read_csv(csvName, header=None, skiprows = 1)
    dataFrame.sort_values(0, axis = 0, inplace = True)
    dataFrame.drop(dataFrame.columns[[0]], axis = 1, inplace = True)
    return dataFrame.as_matrix()

if(len(sys.argv) < 2):
    print('Usage: Logloss.py gold-standard.csv test.csv')
    sys.exit(1)

goldMatrix = readCSV(sys.argv[1])
goldLabels = np.argmax(goldMatrix, axis = 1)
testMatrix = readCSV(sys.argv[2])
print(log_loss(goldLabels, testMatrix))
