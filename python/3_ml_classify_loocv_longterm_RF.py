import os.path
import sys, getopt
import pandas as pd
from time import gmtime, strftime
from imblearn.combine import SMOTETomek

from mlPipeline.pipeline import Pipeline

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

   # Read source data
   features = pd.read_csv(filenames[0], skiprows=0, sep=SEP)
   print "Data.frame samples is:", features.shape[0],"x", features.shape[1]
   features = pd.read_csv(inputFile, skiprows=0, sep=",")


   # Test 1: All features, prediction of valence
   datasetLog1 = datasetLog + '_ALL_Valence'
   pipe = Pipeline(features,outputDir,["neutral","emotional"], datasetLog1)

   featuresStart = 0
   featuresEnd =  54

   userColumn = 55
   labelColumn = 58 # binary classification

   pipe.LoadFeaturesHot(featuresStart,featuresEnd,labelColumn,userColumn)
   pipe.ImputeMissingData() #tohle je jenom zachrana brzda
   pipe.ScaleFeatures()

   pipe.SplitTrainTestShuffleStratified(0.30)

   # bestParams = None
   # # bestParams = pipe.RandomGridSearchRandomForest() # - for the best classifier - SVC
   # print("Best selected parameters are: ", bestParams)
   #
   print("Kfold Dummy Crossvalidation - how does it perform with random classifier?")
   pipe.CrossvalidationKFoldLongtermDummy("ALL")
   #
   # # http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html#sphx-glr-auto-examples-ensemble-plot-adaboost-twoclass-py
   #
   # print("KFold CrossValidation - how does it perform on kFold?")
   # # pipe.CrossvalidationKFoldLongterm(bestParams,"ALL")

   # print("LOPO CrossValidation - how does it perform on New User?")
   # pipe.CrossvalidationUsersLongterm(bestParams,"ALL")

##################################################################################################################
   # Test 1a: Gaze features
   featuresStart = 0
   featuresEnd = 32

   pipe.LoadFeaturesHot(featuresStart,featuresEnd,labelColumn,userColumn)
   pipe.ImputeMissingData()
   pipe.ScaleFeatures()

   pipe.SplitTrainTestShuffleStratified(0.30)

   # bestParams = None
   # bestParams = pipe.RandomGridSearchRandomForest() # - for the best classifier - SVC
   # print("Best selected parameters are: ", bestParams)

   # http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html#sphx-glr-auto-examples-ensemble-plot-adaboost-twoclass-py

   #
   # print("KFold CrossValidation - how does it perform on kFold?")
   # pipe.CrossvalidationKFoldLongterm(bestParams,"GAZE")
   #
   # print("LOPO CrossValidation - how does it perform on New User?")
   # pipe.CrossvalidationUsersLongterm(bestParams,"GAZE")

   ##################################################################################################################
   # Test 1b: GSR features
   featuresStart = 33
   featuresEnd = 44

   pipe.LoadFeaturesHot(featuresStart, featuresEnd, labelColumn, userColumn)
   pipe.ImputeMissingData()
   pipe.ScaleFeatures()

   pipe.SplitTrainTestShuffleStratified(0.30)

   # bestParams = None
   # bestParams = pipe.RandomGridSearchRandomForest()  # - for the best classifier - SVC
   # print("Best selected parameters are: ", bestParams)

   # http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html#sphx-glr-auto-examples-ensemble-plot-adaboost-twoclass-py


   # print("KFold CrossValidation - how does it perform on kFold?")
   # pipe.CrossvalidationKFoldLongterm(bestParams, "GSR")
   #
   # print("LOPO CrossValidation - how does it perform on New User?")
   # pipe.CrossvalidationUsersLongterm(bestParams, "GSR")


   ##################################################################################################################
   # Test 1c: TM features
   featuresStart = 45
   featuresEnd = 54

   pipe.LoadFeaturesHot(featuresStart, featuresEnd, labelColumn, userColumn)
   pipe.ImputeMissingData()
   pipe.ScaleFeatures()

   pipe.SplitTrainTestShuffleStratified(0.30)

   # bestParams = None
   # bestParams = pipe.RandomGridSearchRandomForest()  # - for the best classifier - SVC
   # print("Best selected parameters are: ", bestParams)

   # http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html#sphx-glr-auto-examples-ensemble-plot-adaboost-twoclass-py


   # print("KFold CrossValidation - how does it perform on kFold?")
   # pipe.CrossvalidationKFoldLongterm(bestParams, "TM")
   #
   # print("LOPO CrossValidation - how does it perform on New User?")
   # pipe.CrossvalidationUsersLongterm(bestParams, "TM")

   ##################################################################################################################


   # print("Final Train-test")
   # pipe.FinalTrainTestRandomForest2(outputDir, bestParams) #todo: continue here - make full pipeline - with and without SMOTE in CV and unseen, not in randomGridSearch
   #
   # print("Dummy Train-test")
   # pipe.DummyTrainTest()

   # Test 1: All features, prediction of valence
   datasetLog2 = datasetLog + '_ALL_Arousal'
   pipe = Pipeline(features, outputDir, ["neutral", "emotional"], datasetLog2)

   featuresStart = 0
   featuresEnd = 54

   userColumn = 55
   labelColumn = 60 # binary classification

   pipe.LoadFeaturesHot(featuresStart, featuresEnd, labelColumn, userColumn)
   pipe.ImputeMissingData()
   pipe.ScaleFeatures()

   pipe.SplitTrainTestShuffleStratified(0.30)

   # bestParams = None
   # bestParams = pipe.RandomGridSearchRandomForest() # - for the best classifier - SVC
   # # print("Best selected parameters are: ", bestParams)

   # http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html#sphx-glr-auto-examples-ensemble-plot-adaboost-twoclass-py

   print("Kfold Dummy Crossvalidation - how does it perform with random classifier?")
   pipe.CrossvalidationKFoldLongtermDummy("ALL")

   # print("KFold CrossValidation - how does it perform on kFold?")
   # pipe.CrossvalidationKFoldLongterm(bestParams,"ALL")
   #
   # print("LOPO CrossValidation - how does it perform on New User?")
   # pipe.CrossvalidationUsersLongterm(bestParams,"ALL")

##################################################################################################################
   # Test 2a: Gaze features
   featuresStart = 0
   featuresEnd = 32

   pipe.LoadFeaturesHot(featuresStart,featuresEnd,labelColumn,userColumn)
   pipe.ImputeMissingData()
   pipe.ScaleFeatures()

   pipe.SplitTrainTestShuffleStratified(0.30)

   # bestParams = None
   # bestParams = pipe.RandomGridSearchRandomForest() # - for the best classifier - SVC
   # print("Best selected parameters are: ", bestParams)

   # http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html#sphx-glr-auto-examples-ensemble-plot-adaboost-twoclass-py

   # print("KFold CrossValidation - how does it perform on kFold?")
   # pipe.CrossvalidationKFoldLongterm(bestParams,"GAZE")
   #
   # print("LOPO CrossValidation - how does it perform on New User?")
   # pipe.CrossvalidationUsersLongterm(bestParams,"GAZE")

   ##################################################################################################################
   # Test 2b: GSR features
   featuresStart = 33
   featuresEnd = 44

   pipe.LoadFeaturesHot(featuresStart, featuresEnd, labelColumn, userColumn)
   pipe.ImputeMissingData()
   pipe.ScaleFeatures()

   pipe.SplitTrainTestShuffleStratified(0.30)

   # bestParams = None
   # bestParams = pipe.RandomGridSearchRandomForest()  # - for the best classifier - SVC
   # print("Best selected parameters are: ", bestParams)

   # http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html#sphx-glr-auto-examples-ensemble-plot-adaboost-twoclass-py

   # print("KFold CrossValidation - how does it perform on kFold?")
   # pipe.CrossvalidationKFoldLongterm(bestParams, "GSR")
   #
   # print("LOPO CrossValidation - how does it perform on New User?")
   # pipe.CrossvalidationUsersLongterm(bestParams, "GSR")


   ##################################################################################################################
   # Test 2c: TM features
   featuresStart = 45
   featuresEnd = 54

   pipe.LoadFeaturesHot(featuresStart, featuresEnd, labelColumn, userColumn)
   pipe.ImputeMissingData()
   pipe.ScaleFeatures()

   pipe.SplitTrainTestShuffleStratified(0.30)

   # bestParams = None
   # bestParams = pipe.RandomGridSearchRandomForest()  # - for the best classifier - SVC
   # print("Best selected parameters are: ", bestParams)

   # http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html#sphx-glr-auto-examples-ensemble-plot-adaboost-twoclass-py

   # print("KFold CrossValidation - how does it perform on kFold?")
   # pipe.CrossvalidationKFoldLongterm(bestParams, "TM")
   #
   # print("LOPO CrossValidation - how does it perform on New User?")
   # pipe.CrossvalidationUsersLongterm(bestParams, "TM")

   ##################################################################################################################


print("***********************************\n All good\n")
exit(0)