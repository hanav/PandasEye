import os.path
import pandas as pd

import os
from record import Record


def extractParticipantPrefix(x):
    return x.split('.')[0]

def mergePath(fileDir,x):
    return os.path.join(fileDir, x)


userhome = os.path.expanduser('~')

aoiFile = "/Users/icce/Dropbox/_thesis_framework/_dataset_8Puzzles/properly_anonymized_data_GazeAugmented/AOI_codes.csv"
outputPath = '/Users/icce/Dropbox/_thesis_framework/_dataset_8Puzzles/properly_anonymized_data_GazeAugmented/'

fileArray  = os.listdir(outputPath)
folderArray = filter (lambda x:x.endswith(".txt") , fileArray)
eventArray = filter (lambda x:x.endswith("_event.csv") , fileArray)

if len(eventArray)==0:
    for i in range(0, (len(folderArray))):
        gazefile = folderArray[i]
        gazeFilePath = mergePath(outputPath, gazefile)
        gazeDF = pd.read_csv(gazeFilePath, skiprows=18, sep="\t")

        userID = extractParticipantPrefix(gazefile)
        print userID

        record = Record()
        eventDir = mergePath(outputPath,userID+"_event.csv") #relict from producing event fixations
        record.exportEvents8P(gazeFilePath,aoiFile, eventDir) #relict from producing event fixations

eventArray = filter (lambda x:x.endswith("_event.csv") , fileArray)

folderArray.sort()
eventArray.sort()


# todo: replace this with pandas Series, dataFrames
features = []
pupils = []
diff1 = []
diff2 = []
spectrum = []
cepstrum = []
all = []

prefix = -4
suffix = -1

outputDataset = pd.DataFrame()

# todo: jeden dedicated for na vytvoreni patricnych _event.csv


for i in range(0,(len(folderArray))):
#for i in range(0,1):
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

outputFileDir = mergePath(outputPath,"output.csv")
outputDataset.to_csv(path_or_buf =outputFileDir, sep=",", index=False)

print("All good, folks!")
exit(0)