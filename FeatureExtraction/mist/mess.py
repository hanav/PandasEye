# Class mess
# jednoho krasneho dne uklidit

import numpy as np
import random
from saveOutput import SaveOutput

class Mess:
    def __init__(self):
        pass

    def saveTrainTest(self, inputArray, dir, ratio):    
        output = np.array(inputArray) 
        index = self.shuffleNorm(output)
        border = int(len(output) * ratio)
        #print index,  "-",  border

        idx = np.array(index[0:border])
        trainSet = output[idx]
        idx = index[border:-1]
        testSet = output[idx]

# trainSize - desetinne cislo
    def binaryTrainTest(self, inputArray, dir,  trainSize):

        input = np.array(inputArray)
        
        positive = np.where(input[:, 0] == 'intent')
        negative = np.where(input[:, 0] == 'non')
        mid = np.where(input[:, 0] == 'mid')
        
        intIdx = self.splitTrainTestIndex(positive, trainSize)
        nonIdx = self.splitTrainTestIndex(negative, trainSize)
        midIdx = self.splitTrainTestIndex(mid, trainSize)
        
        trainIdx = np.concatenate([intIdx[0], midIdx[0],  nonIdx[0]]) #ty dalsi zavorky tam fakt musi byt
        testIdx = np.concatenate([intIdx[1], midIdx[1],  nonIdx[1]]) 
        
        trainSet = input[trainIdx]
        testSet = input[testIdx]
        
        obj = SaveOutput()
        trainName = obj.createPathName(dir, "features_train")
        testName = obj.createPathName(dir, "features_test")
        obj.saveArray(trainName, trainSet)
        obj.saveArray(testName, testSet)


# general: zamichat a rozdelit na train a test
    def splitTrainTestIndex(self,  indexArray,  trainSize):
        indeces = indexArray[0]
        random.shuffle(indeces)
        border = np.floor(len(indeces) * trainSize)
        
        trainIdx = np.array(indeces[0:border])
        testIdx = np.array(indeces[border:len(indeces)])

        return (trainIdx,  testIdx)
        
    #nejaka zbloudila funkce    
    def createHistogram(self, array):
        unnest = reduce(lambda x,y: x+y, array)
        unnest = np.array(unnest)
        flatted = unnest.flatten()
        flatted.sort()
        quantile_2 = (len(flatted) / 100) * 2
        histMin = flatted[quantile_2 - 1]
        histMax = flatted[len(flatted) - quantile_2]
        binCounts = 16 + 1
        step = (histMax - histMin) / binCounts
        histBins = np.arange(histMin, histMax, step)

        histograms = []
        for sample in unnest:
            prob,bins = np.histogram(sample, bins = histBins, normed=False)
            length = np.linalg.norm(prob)
            normProb = prob / length
            histograms.append(normProb)
        return histograms
