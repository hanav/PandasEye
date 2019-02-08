#!/usr/bin/python
import numpy as np
import pandas as pd
import sklearn as sk


import sys, getopt


def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print 'test.py -i <inputfile> -o <outputfile>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print 'test.py -i <inputfile> -o <outputfile>'
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
   print 'Input file is "', inputfile
   print 'Output file is "', outputfile
   return [inputfile,outputfile]

if __name__ == "__main__":
   filenames = main(sys.argv[1:])
   print filenames[0]
   df = pd.read_csv(filenames[0], skiprows=0, sep=",")
   print "Dataframe length is:", df.iloc[:,0].size

# print 'Number of arguments:', len(sys.argv), 'arguments.'
# print 'Argument List:', str(sys.argv)

print "All good, folks!"
exit(0)