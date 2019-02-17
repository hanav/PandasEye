# Pupillary Response
import numpy as np


class PupillaryResponse:
    def __init__(self):
        self.data = []
        self.direction = []
        self.tileNumber = []

        self.tileCodebook = {}  
        self.tileMeanbook = {}

        self.dirCodebook = {}
        self.dirMeanbook = {}

        self.dirMean = []
        self.tileMean = []
        
    def sortSequences(self):
        for i in range(0, len(self.data)):
            #tiles
            key = str(self.tileNumber[i]) + "_" + self.direction[i]['direction']
            meanPupil = self.countMeanPupil(self.data[i])
            self.tileMeanbook.setdefault(key,[]).append(meanPupil)

            #direction
            key = self.direction[i]['direction']
            self.dirMeanbook.setdefault(key,[]).append(meanPupil)

    def countMeanPupil(self,sequence):
        meanPupil = (sequence['pupilL'] + sequence['pupilR'])/2
        return meanPupil

    def createTileCodebook(self):
        for key in self.tileMeanbook.iterkeys():
            items = self.tileMeanbook[key]
            meanVector = self.countMean(items)
            self.tileCodebook[key] = meanVector
        self.createTileMean()

    def createDirCodebook(self):
        for key in self.dirMeanbook.iterkeys():
            items = self.dirMeanbook[key]
            meanVector = self.countMean(items)
            self.dirCodebook[key] = meanVector
        self.createDirMean()

    def countMean(self,sequence):
        meanSequence = np.zeros(shape=0, dtype=np.float32)

        for i in range(0, len(sequence[0])):
            pupil = np.zeros(shape = len(sequence))
            for j in range(0, len(sequence)):
                pupil[j] =  sequence[j][i]
            
            meanPupil= pupil.mean()
            meanSequence = np.append(meanSequence, meanPupil)
        return meanSequence

    def createDirMean(self):
        sequence = []
        for key in self.dirCodebook.iterkeys():
            sequence.append(self.dirCodebook[key])
        self.dirMean = self.countMean(sequence)

    def createTileMean(self):
        sequence = []
        for key in self.tileCodebook.iterkeys():
            sequence.append(self.tileCodebook[key])
        self.tileMean = self.countMean(sequence)
 
