import os.path
import sys, getopt

import pandas as pd
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

   featuresStart = 0
   featuresEnd = 12
   labelColumn = 15
   userColumn = 14

   #pipe.LoadFeatures(featuresStart, featuresEnd, labelColumn)
   pipe.LoadFeatures(featuresStart, featuresEnd, labelColumn, userColumn)


   #Test 1. all features
   pipe.writeLog(pd.DataFrame([messageLog]))

# note for better practice: tohle by se melo stat jenom na train.set
   pipe.Normalize()
   pipe.SelectFeatures()
   #pipe.AddPolynomialFeatures()

   #pipe.SimpleTrain()
   print("Split into training and testing set")

# nastaveni cross-validation
   SPLIT_RATIO = 0.4 #split na train and test
   PARAM_C = 750
   PARAM_Gamma = 0.0001
   NUM_TRIALS = 2  # kolikrat se ma cross-validace random zopakovat - vnejsi loop
   KSPLITS = 5 # vnitrni loop


   # naivni zpusob, kfold
   # pipe.SplitTrainTest(SPLIT_RATIO)
   # pipe.UpsampleMinorityTrain()     #17k negative and 17k positive
   # pipe.CrossValidationKFold(PARAM_C, PARAM_Gamma, NUM_TRIALS, KSPLITS)

   #lepsi zpusob: pres lidi
   pipe.SplitTrainTestUser(SPLIT_RATIO)
   # pipe.UpsampleMinorityPerson()
   pipe.CrossValidationPerson(PARAM_C, PARAM_Gamma, NUM_TRIALS, KSPLITS)


#    pipe.CrossvalidationUsers(outputDir + "Logo.Smote.35.gamma.auto.class.weights")

print "All good, folks!"
exit(0)
