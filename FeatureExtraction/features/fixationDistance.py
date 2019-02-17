# date: 04/30/18
# author: Hana Vrzakova
# description: Features derived from the distance between fixations.

import numpy as np
import pandas as pd
##Todo:
## polygonal area covered by fixations

class FixationDistance():
    def __init__(self):

        self.Distances = []
        
        self.Mean = []
        self.Median = []
        self.Stddev = []
        self.Var = []
        self.Max = []
        self.Min = []
        self.Sum = []
        
        self.Prior = []
        self.Last = []

        self.RatioPriorLast = []
        
    def countStatistics(self, seq,  prefix,  suffix):
        distances = []
        for i in range(0,len(seq)-1):
            distance = self.countEuclidean(seq[i],seq[i+1])
            distances.append(distance)
        
        self.Distances.append(distances)
        
        self.Mean.append(np.mean(distances))
        self.Median.append(np.median(distances))
        self.Stddev.append(np.std(distances))
        self.Var.append(np.var(distances))
        self.Max.append(np.max(distances))
        self.Min.append(np.min(distances))
        self.Sum.append(np.sum(distances))
        
        self.Prior.append(distances[-2])
        self.Last.append(distances[-1]) 
        
        ratio = -1
        if distances[-1] > 0:
            ratio = distances[-2]/distances[-1]
        self.RatioPriorLast.append(ratio)


    def subtract_fixation_coords(self, df):
        return pd.Series(data= [df['gazePointX'].iloc[0], df['gazePointY'].iloc[0]])

    def countStatisticsPandas(self,df):
        fixationCoords = df.groupby(['fixationNumber']).apply(self.subtract_fixation_coords) # index je fixation id, hodnota je fixation duration

        self.fixationCoords = fixationCoords

        diffCoords = fixationCoords.diff()
        distances = (diffCoords **2).sum(axis=1).pow(1./2)
        distances = distances.iloc[1:].reset_index(drop=True) #the first row is 0 due to diff

        #for velocity between fixations
        self.distances = distances

        distMean = distances.mean()
        distMedian = distances.median()
        distVariance = distances.var()
        distSTD = distances.std()
        distMin = distances.min()
        distMax = distances.max()
        distSum = distances.sum()

        distFirst = distances.iloc[0]
        distLast = distances.iloc[-1]
        distPriorLast = distances.iloc[-2]

        distRatioFirstLast= distFirst / distLast
        distRatioFirstPrior = distFirst / distPriorLast
        distRatioPriorLast = distPriorLast / distLast

        dfOutput = pd.DataFrame([distMean,distMedian,distVariance,distSTD,
                               distMin,distMax,distSum,
                               distFirst,distLast,distPriorLast,
                               distRatioFirstLast, distRatioFirstPrior, distRatioPriorLast]).transpose()

        dfOutput.columns = ["fixDistMean","fixDistMedian","fixDistVariance","fixDistSTD",
                               "fixDistMin","fixDistMax","fixDistSum",
                               "fixDistFirst","fixDistLast","fixDistPriorLast",
                               "fixDistRatioFirstLast", "fixDistRatioFirstPrior", "fixDistRatioPriorLast"]

        return dfOutput

    def countEuclidean(self,fixation1,fixation2):
        point1 = np.array([fixation1[0]['gazePointX'],fixation1[0]['gazePointY']])
        point2 = np.array([fixation2[0]['gazePointX'],fixation2[0]['gazePointY']])
        distance = np.linalg.norm(point1 - point2)    
        return distance
    
    def allToString(self):
        self.strMean = np.array(self.Mean).astype('|S10')
        self.strMedian= np.array(self.Median).astype('|S10')
        self.strStddev = np.array(self.Stddev).astype('|S10')
        self.strVar = np.array(self.Var).astype('|S10')
        self.strMax = np.array(self.Max).astype('|S10')
        self.strMin = np.array(self.Min).astype('|S10')
        self.strSum = np.array(self.Sum).astype('|S10')
        
        self.strPrior = np.array(self.Prior).astype('|S10')
        self.strLast = np.array(self.Last).astype('|S10')
        
        self.strRatioPriorLast = np.array(self.RatioPriorLast).astype('|S10')     
#-------------------------------------------------------------------------------
## Fixation polygonal area
#-------------------------------------------------------------------------------
    def countFixationPolygonalArea(self, seq):
        pass


