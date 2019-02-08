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

   features = pd.read_csv(filenames[0], skiprows=0, sep=SEP)
   print "Data.frame samples is:", features.shape[0],"x", features.shape[1]

   features = pd.read_csv(inputFile, skiprows=0, sep=",")
   pipe = Pipeline(features,outputDir,["neutral","emotional"], datasetLog)

   featuresStart = 4
   featuresEnd =  34
   labelColumn = 3
   userColumn = 0

   pipe.LoadFeatures(featuresStart,featuresEnd,labelColumn,userColumn)
   #todo: Rko - prejmenovat sloupecky, at vime, ktere sensors tam vlastne mame

   # pipe.UpsampleMinorityPersonWhole() - tady to spadne

   pipe.ImputeMissingData()
   pipe.ScaleFeatures()

   bestParams = pipe.RandomGridSearch() # - for the best classifier - SVC
   print("Best selected parameters are: ", bestParams)

   # http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html#sphx-glr-auto-examples-ensemble-plot-adaboost-twoclass-py
   pipe.CrossvalidationUsers(outputDir)

   # results = pd.DataFrame([])
   # results = results.append(pipe.outputResults, ignore_index=True)
   #
   #
   # resultsOut = os.path.join(outputDir,'hot_all_results'+strftime("%Y-%m-%d %H:%M:%S", gmtime())+'.csv')
   #  #results.to_csv(resultsOut, mode='a', header=False, index=False)
   # with open(resultsOut, 'a') as f:
   #      results.to_csv(f, header=False, index=False)

print("***********************************\n All good\n")
exit(0)