#-------------------------------------------------------------------------------
## Saccadic direction (absolute)
#-------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import pandas as pd

class SaccadeOrientation:
    def __init__(self):
        self.Mean = []
        self.Median = []
        self.Stddev = []
        self.Var = []
        self.Max = []
        self.Min = []
        self.Sum = []
        
        self.Last = []
        self.Prior = []

        self.RatioPriorLast = []
        

    def countStatisticsPandas(self, fixationCoords):
        # tan uhlu = protilehla / prilehle
        # uhel = arctan( protilehla / prilehla) - stupne

        horizontalDistance = fixationCoords[0].diff()[1::].reset_index(drop=True)
        verticalDistance = fixationCoords[1].diff().reset_index(drop=True)

        horizontalDistance = horizontalDistance[1::].reset_index(drop=True)
        verticalDistance = verticalDistance[1::].reset_index(drop=True)

# https://docs.scipy.org/doc/numpy/reference/generated/numpy.arctan2.html#numpy.arctan2
        angles = np.arctan2(verticalDistance/horizontalDistance)
        anglesDeg = np.rad2deg(angles)

        #todo: continue here

        pass

    def countStatistics(self,  seq,  prefix,  suffix):
        directions = []
        
        for i in range(0,len(seq)-1):
            
            vY = seq[i+1][0]['gazePointY'] - seq[i][0]['gazePointY']
            vX = seq[i+1][0]['gazePointX'] - seq[i][0]['gazePointX']

            direction = np.arctan2(vY, vX)

## tady to opravit - prepocitat na..absolutni vychylku? co to udela s prumerama?
            directionDegree = np.rad2deg(direction)
            if(directionDegree < 0): ##opravdu prasacky napsany
                directionDegree = 360 + directionDegree
            directions.append(directionDegree)

#        directions = np.degrees(directions)

        self.Mean.append( np.mean(directions) )
        self.Median.append( np.median(directions) )
        self.Stddev.append( np.std(directions))
        self.Var.append( np.var(directions))
        self.Max.append( np.max(directions))
        self.Min.append( np.min(directions))
        self.Sum.append( np.sum(directions))
        
        self.Prior.append( directions[-2] )
        self.Last.append( directions[-1] )
        
        ratio= -1
        if(directions[-1] > 0):
            ratio = directions[-2]/directions[-1]
        self.RatioPriorLast.append(ratio) 

    ##debug - kdyz pokud bude direction 0, tak to zacne hazet inf

          
        
    def allToString(self):
        self.strMean = np.array(self.Mean).astype('|S10')
        self.strMedian= np.array(self.Median).astype('|S10')
        self.strStddev = np.array(self.Stddev).astype('|S10')
        self.strVar = np.array(self.Var).astype('|S10')
        self.strMax = np.array(self.Max).astype('|S10')
        self.strMin = np.array(self.Min).astype('|S10')
        self.strSum = np.array(self.Sum).astype('|S10')
        
        self.strLast = np.array(self.Last).astype('|S10')
        self.strPrior = np.array(self.Prior).astype('|S10')
        
        self.strRatioPriorLast = np.array(self.RatioPriorLast).astype('|S10')     


#-------------------------------------------------------------------------------
## Saccadic orientation (relative)
#-------------------------------------------------------------------------------
# mean
# minimal
# maximal
# first two
# last two
# first and last
# circular histograms (rose plots)
