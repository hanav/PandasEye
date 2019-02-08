import os.path
import sys, getopt

import pandas as pd
from mlPipeline.pipeline import Pipeline

from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score, auc, roc_auc_score, f1_score,matthews_corrcoef, precision_score, recall_score

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
   pipe.UpsampleMinorityTrain()     #17k negative and 17k positive

   params = pipe.GridSearchRandom()
   PARAM_C = params['C']
   PARAM_Gamma = params['gamma']

# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
   from keras.models import Sequential
   from keras.layers import Dense
   import numpy
   import keras_metrics

   # fix random seed for reproducibility
   numpy.random.seed(7)

   X = pipe.x_train
   Y = pipe.y_train

   # create model
   model = Sequential()
   model.add(Dense(16, input_dim=12, activation='relu'))
   model.add(Dense(12, activation='relu'))
   model.add(Dense(1, activation='sigmoid'))

   print("Compile model")
   model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',keras_metrics.precision(), keras_metrics.recall()])

   print("Fit the model")
   model.fit(X, Y, epochs=150, batch_size=10)

   # evaluate the model
   scores = model.evaluate(X, Y)
   print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

   # make predictions on training set
   predictions = model.predict(X)
   rounded = [round(x[0]) for x in predictions]

   print("Binary: predict")

   target_names = ['intent', 'non-intent']  # todo: zkontrolovat, jestli intent je 1

   train_accuracy = accuracy_score(Y, rounded)
   train_f1 = f1_score(Y, rounded, target_names)
   train_precision = precision_score(Y, rounded, target_names)
   train_recall = recall_score(Y, rounded, target_names)

   # make predictions on testing set
   predictions = model.predict(pipe.x_unseen)
   rounded = [round(x[0]) for x in predictions]

   test_accuracy = accuracy_score(pipe.y_unseen, rounded)
   test_f1 = f1_score(pipe.y_unseen, rounded, target_names)
   test_precision = precision_score(pipe.y_unseen, rounded, target_names)
   test_recall = recall_score(pipe.y_unseen, rounded, target_names)
   print("Accuracy: ", train_accuracy, "-", test_accuracy)
   print("F1: ", train_f1,"-",test_f1)
   print("Precision: ",train_precision,"-",test_precision)
   print("Recall: ",train_recall,"-",test_recall)


   # #TODO: UNCOMMENT HERE!!!!!
   # pipe.CrossValidationKFold(PARAM_C, PARAM_Gamma, NUM_TRIALS, KSPLITS)
   #
   # pipe.TrainTestSaveFinalModel(PARAM_C, PARAM_Gamma)

print "All good, folks!"
exit(0)
