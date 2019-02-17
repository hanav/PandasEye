# date: 04/30/18
# author: Hana Vrzakova
# description: A main script to start the data segmentation and feature
# extraction.
# Inputs:
# - prefix: number of fixation prior to the mouse click
# - suffix: number of fixation after the mouse click
# Outputs:
# - Results/features_prefix_suffix.csv

import os.path
import pandas as pd

import os
from record import Record

def extractParticipantPrefix(x):
    return x.split('.')[0]

def mergePath(fileDir,x):
    return os.path.join(fileDir, x)

userhome = os.path.expanduser('~')

# Input directory
folderPath = os.path.dirname(os.path.realpath(__file__))
aoiFile = os.path.join(folderPath,"ExampleData","AOI_codes.csv")
outputPath = os.path.join(folderPath,"ExampleData")

# Create results directory
folderPath = os.path.dirname(os.path.realpath(__file__))
outputDir = os.path.join(folderPath, "Results")
if not os.path.exists(outputDir):
    os.mkdir(outputDir)

fileArray  = os.listdir(outputPath)
folderArray = filter (lambda x:x.endswith(".txt") , fileArray)
eventArray = filter (lambda x:x.endswith("_event.csv") , fileArray)

if len(eventArray)==0:
    for i in range(0, (len(folderArray))):
        gazefile = folderArray[i]
        gazeFilePath = mergePath(outputPath, gazefile)
        gazeDF = pd.read_csv(gazeFilePath, skiprows=18, sep="\t")

        userID = extractParticipantPrefix(gazefile)
        print "Event processing: ", userID

        record = Record()
        eventDir = mergePath(outputPath,userID+"_event.csv") #relict from producing event fixations
        record.exportEvents8P(gazeFilePath,aoiFile, eventDir) #relict from producing event fixations

eventArray = filter (lambda x:x.endswith("_event.csv") , fileArray)
folderArray.sort()
eventArray.sort()

prefix = -4
suffix = -1

outputDataset = pd.DataFrame()

for i in range(0,(len(folderArray))):
    print "Processing participant: "

    gazefile = folderArray[i]
    gazeFilePath= mergePath(outputPath,gazefile)
    gazeDF = pd.read_csv(gazeFilePath, skiprows=18, sep="\t")
    eventFilePath = mergePath(outputPath, eventArray[i])

    userID = extractParticipantPrefix(gazefile)
    print userID

    record = Record()
    record.loadPrefixSuffix(prefix,suffix)
    record.loadPath(gazeFilePath)

    record.loadData8P(gazeFilePath,aoiFile, eventFilePath)

    outputUser =record.cutFixations8P()
    outputUser['participant'] = userID

    outputDataset = outputDataset.append(outputUser)

outputDataset['action_binary'] = 0
outputDataset['action_binary'].loc[outputDataset['action']==1] = 1

outputFileDir = mergePath(outputDir,"features_"+str(prefix)+"_"+str(suffix)+".csv")
outputDataset.to_csv(path_or_buf =outputFileDir, sep=",", index=False)

print("All good, folks!")
exit(0)