# Author: Hana Vrzakova
# Date: 23.1.2011
# Date: 10.8.2012
# Date: 29.8.2012
# Date: 16.4.2014
# Date: 24.1.2015
# Date: 4.4. 2017
# Date: 9.4.2018

import os
import numpy as np
import pandas as pd

def mergePath(fileDir,x):
    return os.path.join(fileDir, x)

def extractParticipantPrefix(x):
    return x.split('.')[0]


############################################################

# opravit pupily
# ukladani po featurech - quick analysis
# ukladani feature fixations
# ukladat basic stats (abs. + rel. hodnoty) + spocitat vahy, ulozit
# vykreslit timeline pro featury - ukladat timestamps - zacatek a konec sekvence (pozdeji)


# C:\Python27\python D:\Dropbox\dizertacka\python\parseFeatures\main.py ga -5 -3
# prefix = -6 #pocet fixaci pred eventem, cislujeme od nuly => [0,1,2,event,4,5] = 3 + event + 2 = 6fixaci na analyzu
# suffix = -2 #pocet fixaci po eventu - pada to na suffixu

inputDir = "/Users/icce/Dropbox/_thesis_framework/_dataset_8Puzzles/_properly_anonymized_data_Mouse"
outputDir = "/Users/icce/Dropbox/_thesis_framework/_dataset_8Puzzles/processed"

prefix = 5
suffix = 0

fileArray  = os.listdir(inputDir)
paths = filter (lambda x:x.endswith("_aoi.csv") , fileArray)

i = 0
for gazeFile in paths:
    gazeFilePath= mergePath(inputDir,gazeFile)
    gazeDF = pd.read_csv(gazeFilePath, sep=",")
    userID = extractParticipantPrefix(gazeFile)

    idx = gazeDF[gazeDF['Event']=='LMouseButton'].index
    clickCount = len(idx)

    #Q1: how often the click happend during the single fixation
    withinFixation = 0
    outsideFixation = 0


    emptyPreFixation = 0
    emptyPostFixation = 0
    emptyBoth = 0
    emptyFirst = 0
    emptySecond = 0

    for id in idx:
        preEventFixation = gazeDF['Fixation'].iloc[id -1]
        postEventFixation = gazeDF['Fixation'].loc[id+1]

        # if preEventFixation == " ":
        #     emptyPreFixation += 1
        #
        # if postEventFixation == " ":
        #     emptyPostFixation +=1

        if preEventFixation == " " and postEventFixation == " ":
            emptyBoth +=1

        if preEventFixation == " " and postEventFixation != " ":
            emptyFirst +=1

        if preEventFixation != " " and postEventFixation == " ":
            emptySecond +=1


        # if preEventFixation == postEventFixation:
        #     withinFixation +=1
        # else:
        #     outsideFixation +=1
    # print "Ratio: ", withinFixation / float(len(idx))

    print userID,"-",len(idx)
    print "Empty both fixations: ", emptyBoth, "(", emptyBoth / float(len(idx)),"%)"
    print "Empty pre fixation: ", emptyFirst, "(", emptyFirst / float(len(idx)),"%)"
    print "Empty post fixation: ", emptySecond, "(", emptySecond / float(len(idx)),"%)"

    #record = Record()
    #record.loadPrefixSuffix(prefix, suffix)
    #record.loadPath(gazeFilePath)
    #record.loadData()
    #record.prepareRawData()
    #record.cutFixations()
    #record.extractFeatures()

    i +=1


print("All good, folks!")
exit(0)

