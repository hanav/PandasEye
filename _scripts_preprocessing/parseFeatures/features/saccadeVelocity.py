#-------------------------------------------------------------------------------
## Saccadic velocity
#-------------------------------------------------------------------------------
from __future__ import division
import numpy as np
import pandas as pd

class SaccadeVelocity:
    def __init__(self):
        self.Velocities = []
        
        self.Mean = []
        self.Median = []
        self.Max = []
        self.Min = []
        self.Sum = []
        
        self.Prior = []
        self.Last = []
        
        self.RatioPriorLast = []

        
    def countStatistics(self,  fixationDistance,  saccadeDuration,  prefix,  suffix):
        velocities = np.divide(fixationDistance ,  saccadeDuration) 
        
        self.Velocities.append(velocities)
        
        self.Mean.append( np.sum(fixationDistance) / np.sum(saccadeDuration))
        self.Median.append( np.median(velocities) )
        #outSTD se pocita jinak?
        #outVar repete
        self.Max.append( np.max(velocities)) 
        self.Min.append( np.min(velocities)) # a na jake pozici - abs. hodnoty rikaji malo,
        self.Sum.append(np.sum(velocities))
        
        self.Prior.append(velocities[-2])
        self.Last.append( velocities[-1])
        
        ratio = -1
        if(velocities[-1] > 0):
            ratio = velocities[-2]/velocities[-1]
        self.RatioPriorLast.append(ratio)

    def countStatisticsPandas(self, distances, durations):
        velocities = distances/durations

        self.velocities = velocities

        distMean = velocities.mean()
        distMedian = velocities.median()
        distVariance = velocities.var()
        distSTD = velocities.std()
        distMin = velocities.min()
        distMax = velocities.max()
        distSum = velocities.sum()

        distFirst = velocities.iloc[0]
        distLast = velocities.iloc[-1]
        distPriorLast = velocities.iloc[-2]

        distRatioFirstLast = distFirst / distLast
        distRatioFirstPrior = distFirst / distPriorLast
        distRatioPriorLast = distPriorLast / distLast

        dfOutput = pd.DataFrame([distMean, distMedian, distVariance, distSTD,
                                 distMin, distMax, distSum,
                                 distFirst, distLast, distPriorLast,
                                 distRatioFirstLast, distRatioFirstPrior, distRatioPriorLast]).transpose()

        dfOutput.columns = ["saccVelMean", "saccVelMedian", "saccVelVariance", "saccVelSTD",
                            "saccVelMin", "saccVelMax", "saccVelSum",
                            "saccVelFirst", "saccVelLast", "saccVelPriorLast",
                            "saccVelRatioFirstLast", "saccVelRatioFirstPrior", "saccVelRatioPriorLast"]

        return dfOutput

    def allToString(self):
        self.strMean = np.array(self.Mean).astype('|S10')
        self.strMedian= np.array(self.Median).astype('|S10')
        #self.strStddev = np.array(self.Stddev).astype('|S10')
        #self.strVar = np.array(self.Var).astype('|S10')
        self.strMax = np.array(self.Max).astype('|S10')
        self.strMin = np.array(self.Min).astype('|S10')
        self.strSum = np.array(self.Sum).astype('|S10')
        
        self.strPrior = np.array(self.Prior).astype('|S10')
        self.strLast = np.array(self.Last).astype('|S10')

        self.strRatioPriorLast = np.array(self.RatioPriorLast).astype('|S10')       

