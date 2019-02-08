import os.path
import sys, getopt

import pandas as pd
import numpy as np
from mlPipeline.pipeline import Pipeline

# output my results
import time

def main(argv):
   inputfile = ''
   outputfile = ''
   messageLog = ''
   datasetLog = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:m:d:",["ifile=","ofile=","mfile=","dfile="])
   except getopt.GetoptError:
      print 'test.py -i <inputfile> -o <outputfile>'
      sys.exit(2)

   for opt, arg in opts:
      if opt == '-h':
         print 'test.py -i <inputfile> -o <outputfile> -m <message>'
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
      elif opt in ("-m", "--mfile"):
         messageLog = arg
      elif opt in ("-d", "--dfile"):
          datasetLog = arg

   print 'Input file is "', inputfile
   print 'Output directory is "', outputfile
   print 'Message is', messageLog
   print 'Dataset is', messageLog

   return [inputfile,outputfile,messageLog, datasetLog]

if __name__ == "__main__":
   filenames = main(sys.argv[1:])
   inputFile = filenames[0]
   outputDir = filenames[1]
   messageLog = filenames[2]
   datasetLog = filenames[3]

   print filenames[0]
   SEP = ","
   features = pd.read_csv(filenames[0], skiprows=0, sep=SEP)
   print "Dataframe length is:", features.iloc[:,0].size

   results = pd.DataFrame([])

   pipe = Pipeline(features,outputDir,["free-viewing","intentions"], datasetLog)

   # kolikaty je label column odzadu

   # featuresStart = 0
   # featuresEnd = 12
   # labelColumn = 15
   # userColumn = 14

   featuresStart = 0
   featuresEnd = 89
   labelColumn = 92
   userColumn = 91

   #pipe.LoadFeatures(featuresStart, featuresEnd, labelColumn)
   pipe.LoadFeatures(featuresStart, featuresEnd, labelColumn, userColumn)

   #Test 1. all features
   pipe.writeLog(pd.DataFrame([messageLog]))

   X = pipe.x
   Y = pipe.y
   USERS = pipe.users
   allUsers = np.unique(USERS)

   for user in range(0,len(allUsers)):
      print("Individual: ", user)

      idx = np.where(USERS == user)
      pipe.x = X.iloc[idx[0]]
      pipe.y = Y[idx[0]]
      pipe.users = USERS[idx[0]]

      # Transformations
      # note for better practice: tohle by se melo stat jenom na train.set a az po feature selection
      pipe.SelectFeatures(0.35)
      pipe.UpsampleMinority()
      pipe.Normalize()

      print("Split into training and testing set")

   # nastaveni cross-validation
      SPLIT_RATIO = 0.4 #split na train and test
      PARAM_C = 750
      PARAM_Gamma = 0.0001
      NUM_TRIALS = 1  # kolikrat se ma cross-validace random zopakovat - vnejsi loop
      KSPLITS = 5 # vnitrni loop

      pipe.SplitTrainTest(SPLIT_RATIO)

      params = pipe.GridSearchRandom()
      PARAM_C = params['C']
      PARAM_Gamma = params['gamma']

      pipe.CrossValidationKFold(PARAM_C, PARAM_Gamma, NUM_TRIALS, KSPLITS)

      pipe.TrainTestSaveFinalModel(PARAM_C, PARAM_Gamma)

print "All good, folks!"
exit(0)
