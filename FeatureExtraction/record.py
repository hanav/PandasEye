# date: 06/30/18
# author: Hana Vrzakova
# description: Class representing recorded data of each participant.

import numpy as np
import pandas as pd

from recordPath import RecordPath
from rawData import RawData
from mist.aoi import Aoi
from pupildilations.pupillaryResponse import PupillaryResponse
from mist.saveOutput import SaveOutput
from features.features import Features
from pupildilations.pupilSequences import PupilSequences
from mist.elanEvents import ElanEvents

class Record(SaveOutput):
    def __init__(self):
        self.rawData = RawData()
        self.fixations = []
        self.events = []
        self.aoiBorders = Aoi()

        self.recordPath = RecordPath()
        self.outputDir = []
        self.pupilResp = PupillaryResponse()

        self.prefix = 0
        self.suffix = 0
        

        self.eventFixation = []
        self.fixationSequencesNumbers = []
        self.fixationSequence = []
        self.fixationArray = []
        
        self.eventSequence = []
        self.eventArray = []
        
        self.nonSequence = []
        self.nonArray = []
        
        self.allSamples = []

        self.ndType = [('timestamp',int), 
                       ('pupilL', np.float32),  ('validityL',int),
                       ('pupilR', np.float32), ('validityR',int),
                       ('fixationNumber', int),
                       ('gazePointX',int), ('gazePointY',int),
                       ('event','S15'), 
                        ('rawX', int), ('rawY', int)]

    def loadPrefixSuffix(self,prefix, suffix):
        
        if prefix >= suffix:
            quit()
        self.prefix = prefix
        self.suffix = suffix

    def loadPath(self, xmlPaths):
        self.recordPath = xmlPaths

    def loadData(self):
        self.rawData.load(self.recordPath.cmdPath)

        if(not self.recordPath.elanPath):
            elanEvents = ElanEvents(self.recordPath.elanPath)
            self.events = elanEvents.separateElanEvents()
        else:
            self.events = self.rawData.separateEvents()
            
        self.fixations = self.rawData.separateFixations()
        self.aoiBorders.load(self.recordPath.aoiPath)

    def loadData8P(self, cmdPath, aoiPath, eventFilePath):
        self.rawData.loadDataFrame8P(cmdPath)
        self.cmdData = self.rawData.data
        self.events = pd.read_csv(eventFilePath, sep=",")
        self.fixations = self.rawData.separateFixations8P()
        self.aoiBorders = pd.read_csv(aoiPath, skiprows=0, sep="\t")

    def exportEvents8P(self, cmdPath, aoiPath, eventFilePath):
        self.rawData.loadDataFrame8PEvents(cmdPath)
        self.events = self.rawData.separateEvents8P()
        self.events.to_csv(path_or_buf = eventFilePath , sep=",", index=False)

    def loadDataWTP(self, cmdPath, aoiPath, eventFilePath):
        self.rawData.loadDataFrameWTP(cmdPath)
        self.cmdData = self.rawData.data
        self.events = self.rawData.separateEventsWTP()
        self.fixations = self.rawData.separateFixations8P()
        self.aoiBorders = pd.read_csv(aoiPath, skiprows=0, sep="\t")

    def prepareRawData(self):
        self.rawData.eraseEvents()
        self.rawData.countBaseline()

    def cutFixations(self):
        self.cutFixationSequences()
        self.extractEventFixationNumbers()
        self.saveEventFixations()

    def cutFixations8P(self):
        fixationNumbers = pd.Series(self.fixations['fixationNumber'].unique())
        fixationNumbers = fixationNumbers.dropna()
        fixationNumbers = fixationNumbers.astype(int)

        features = Features()
        outputDf = pd.DataFrame()

        # generate sequences & check they are in range and continuous
        for fixationNumber in fixationNumbers:
            idxSeq = self.returnSeqFixationNumbers(fixationNumber, self.prefix, self.suffix)

            if np.unique(np.diff(idxSeq)) != 1:
                # print "Skips in sequence: ", idxSeq
                continue

            results = [any(fixationNumbers == x) for x in idxSeq]

            if (False in results):
                # print "Not existing in the dataset:",idxSeq
                continue

            seqDf = self.cutFixationSequencePandas(idxSeq)
            dfSequenceRaw = self.cutFixationSequenceRawPandas(idxSeq)

            if len(seqDf['fixationNumber'].unique()) != (self.suffix-self.prefix+1):
                # print "Sequence lenght is wrong:", len(seqDf['fixationNumber'].unique())
                continue

            features.load(seqDf,self.aoiBorders,  self.prefix,  self.suffix)
            features.loadRawPandas(dfSequenceRaw)

            row = features.extractFeaturesPandas()
            rowPupils = features.extractFeaturesPupilPandas()
            row = pd.concat([row, rowPupils], axis=1)

            # is event embedded in the fixation sequence?
            actionTag = 0

            eventsInBetween = self.events['eventFixation'].between(idxSeq[0], idxSeq[-1])

            if eventsInBetween.any():

                eventIdx = eventsInBetween[eventsInBetween==True].index[0]
                eventFixation = self.events['eventFixation'].loc[eventIdx]
                # print "Sequence: ", idxSeq, "- fixation before event: ", eventFixation
                if idxSeq[self.suffix] == eventFixation:
                    # print "- right spot"
                    actionTag = 1
                else:
                    # print "- not exactly right spot but in the sequence: "
                    actionTag = 0.5

            row['action'] = actionTag

            outputDf = outputDf.append(row)

        self.events['eventFixation'].count() #369
        outputDf['action'].loc[outputDf['action'] == 1].count() #113

        return outputDf

    def cutFixationSequences(self):
        fixationNumbers = np.array(self.returnFixationNumbers())

        for fixationNumber in fixationNumbers:
            idxSeq = self.returnSeqFixationNumbers(fixationNumber,  self.prefix,  self.suffix)

            if self.isExistingSequence(idxSeq)==False:
                continue

            partSequence = self.cutFixationSequence(idxSeq)
            partArray = self.cutFixationArray(idxSeq)
            
            self.fixationSequence.append(partSequence)
            self.fixationArray.append(partArray)
            self.fixationSequencesNumbers.append( idxSeq)
        

    def extractEventFixationNumbers(self):
        for event in self.events:
            index = np.where(self.fixations['timestamp'] < event['timestamp'])
            idx = index[0][-1]
            fixationNumber = self.fixations[idx]['fixationNumber']
            self.eventFixation.append(fixationNumber)



    def saveEventFixations(self):
        fixationSequences = np.array(self.fixationSequencesNumbers)
        
        for i in range(0, len(fixationSequences)):

            if(self.checkOverlay(self.fixationArray[i]) == 1):
                continue

            eventFixation = fixationSequences[i, 0]
  
            if self.prefix <0 and self.suffix < 0:
                eventFixation =  fixationSequences[i, 0] - self.prefix
                
            elif self.prefix >0 and self.suffix >0 :
                 eventFixation =  fixationSequences[i, 0] - self.prefix

            if self.prefix <0 and self.suffix >0:
                eventFixation = fixationSequences[i, 0-(self.prefix)]

            if ( eventFixation in self.eventFixation)==True:
                self.eventArray.append(self.fixationArray[i])
                self.eventSequence.append(self.fixationSequence[i])
                continue
            
            else:
                self.nonArray.append(self.fixationArray[i])
                self.nonSequence.append(self.fixationSequence[i])
                
        print len(self.eventArray), "-",   len(self.nonArray)

    def returnSeqFixationNumbers(self, fixationNumber,  prefix,  suffix):
            fixBegin = fixationNumber + prefix + 1
            fixEnd = fixationNumber + suffix + 1
            idxSeq = range(fixBegin,fixEnd+1)
            return idxSeq

    def isExistingSequence(self,idxSeq):
        results = [False for number in idxSeq if (number in self.fixations['fixationNumber'])==False]
        if (False in results):
            return False
        return True

    def cutFixationSequence(self, idxSequence):
        sequence = []
        idxBegin = np.where(self.rawData.data['fixationNumber'] == idxSequence[0])
        idxEnd = np.where(self.rawData.data['fixationNumber'] == idxSequence[-1])
        sequence = self.rawData.data[idxBegin[0][0]:(idxEnd[0][-1]+1)]
        return sequence

    def cutFixationSequencePandas(self, idxSequence):
        startFixation = idxSequence[0]
        endFixation = idxSequence[-1]
        sequence = self.cmdData.loc[ (self.cmdData['fixationNumber'] >=startFixation) & (self.cmdData['fixationNumber'] <= endFixation) ]
        return sequence

    def cutFixationSequenceRawPandas(self, idxSequence):
        startFixation = idxSequence[0]
        endFixation = idxSequence[-1]

        startTimestamp = self.cmdData.loc[ self.cmdData['fixationNumber'] == startFixation,'timestamp']
        endTimestamp = self.cmdData.loc[ self.cmdData['fixationNumber'] == endFixation,'timestamp']

        sequence = self.cmdData.loc[ (self.cmdData['timestamp'] >=startTimestamp.iloc[0]) & (self.cmdData['timestamp'] <= endTimestamp.iloc[-1]) ]
        return sequence

    def cutFixationArray(self, idxSequence):
        fixationArray = []
        for i in range(idxSequence[0],(idxSequence[-1]+1)): 
            idx = np.where(self.fixations['fixationNumber'] == i)
            fixationArray.append(self.fixations[idx])
        return fixationArray

    def returnFixationNumbers(self):    
        fixNumbers = []
        
        for row in self.fixations:
            fixNumber = row['fixationNumber']
            if ((fixNumber in fixNumbers) == False):
                fixNumbers.append(fixNumber)
        return fixNumbers

    def labelFixations(self,inputList, label):
        outputList = []
        for i in range(0,len(inputList)):
            seq = np.append(label,inputList[i])
            outputList.append(seq)
        return outputList      

    def returnFeatures(self):
        
        intFeatures = self.featuresIntent.returnAllFeatures()   
        nonFeatures = self.featuresNon.returnAllFeatures()
        
        labelIntent = self.labelFixations(intFeatures,"intent")
        labelNon = self.labelFixations(nonFeatures,"non")
        features = labelIntent + labelNon
        return features

# ----------------------------------------------------------------------------------------------
    def returnDiff1(self):
        diff1 = self.pupilsIntent.returnDiff1() + self.pupilsNon.returnDiff1()
        return diff1

    def returnDiff2(self):
        diff2 = self.pupilsIntent.returnDiff2() + self.pupilsNon.returnDiff2()
        return diff2

    def returnSpectrum(self):
        spectrum = self.pupilsIntent.returnSpectrum() + self.pupilsNon.returnSpectrum()
        return spectrum

    def returnCepstrum(self):
        cepstrum = self.pupilsIntent.returnCepstrum() + self.pupilsNon.returnCepstrum()
        return cepstrum

    def returnAllSamples(self):
        return self.allSamples

    def checkOverlay(self,  seq):
        
        for i in range(0,len(seq)-1):
            
            vY = seq[i+1][0]['gazePointY'] - seq[i][0]['gazePointY']
            vX = seq[i+1][0]['gazePointX'] - seq[i][0]['gazePointX']
                
            if vY == 0 and vX == 0:
                return 1
        return 0    
