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
   print 'Dataset is', datasetLog

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
   pipe.SplitTrainTest(SPLIT_RATIO)
   #pipe.SelectFeatures()
   pipe.UpsampleMinorityTrain()     #17k negative and 17k positive

   params = pipe.GridSearchRandom()
   PARAM_C = params['C']
   PARAM_Gamma = params['gamma']

   #TODO: UNCOMMENT HERE!!!!!
   pipe.CrossValidationKFold(PARAM_C, PARAM_Gamma, NUM_TRIALS, KSPLITS)

   pipe.TrainTestSaveFinalModel(PARAM_C, PARAM_Gamma)

# But the purpose of cross-validation is not to come up with our final model. We don't use these 5 instances of our trained model to do any real prediction. For that we want to use all the data we have to come up with the best model possible. The purpose of cross-validation is model checking, not model building.

# Now, say we have two models, say a linear regression model and a neural network. How can we say which model is better? We can do K-fold cross-validation and see which one proves better at predicting the test set points. But once we have used cross-validation to select the better performing model, we train that model (whether it be the linear regression or the neural network) on all the data. We don't use the actual model instances we trained during cross-validation for our final predictive model.

# Note that there is a technique called bootstrap aggregation (usually shortened to 'bagging') that does in a way use model instances produced in a way similar to cross-validation to build up an ensemble model, but that is an advanced technique beyond the scope of your question here.

# Overfitting has to do with model complexity, it has nothing to do with the amount of data used to train the model. Model complexity has to do with the method the model uses, not the values its parameters take.

# I rather say that overfitting has to do with having too few training cases for too complex a model. So it (also) has to do with numbers of training cases. But having more training cases will reduce the risk of overfitting (for constant model complexity)

# Basically we use CV (e.g. 80/20 split, k-fold, etc) to estimate how well your whole procedure (including the data engineering, choice of model (i.e. algorithm) and hyper-parameters, etc.) will perform on future unseen data. And once you've chosen the winning "procedure", the fitted models from CV have served their purpose and can now be discarded. You then use the same winning "procedure" and train your final model using the whole data set.

# https://machinelearningmastery.com/train-final-machine-learning-model/

#    pipe.CrossvalidationUsers(outputDir + "Logo.Smote.35.gamma.auto.class.weights")

print "All good, folks!"
exit(0)
