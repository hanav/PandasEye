# date: 04/30/18
# author: Hana Vrzakova
# description: Features derived from saccade accelerations.

import numpy as np
import pandas as pd

class SaccadeAcceleration:
    def __init__(self):
        self.Mean = []
        
    def countStatistics(self,  velocities,  durations):
#        allSpeed = []
#        for i in range(0,len(seq)-1):
#            length = self.countEuclidean(seq[i+1],seq[i])
#            duration = seq[i+1][0]['timestamp'] - seq[i][-1]['timestamp']
#            speed = length/duration
#            allSpeed.append(speed)
        
        self.Mean.append((velocities[-1] - velocities[0])/np.sum(durations))
        
        #print "Done - Saccade acceleration"

    def countStatisticsPandas(self, velocities,durations):
        diffVelocity = velocities.iloc[-1] - velocities.iloc[0]
        sumDuration = durations.sum()
        acceleration = diffVelocity/sumDuration

        diffsVelocities = velocities.diff()
        sumsDurations = durations.rolling(2).sum()
        accelerations = diffsVelocities / sumsDurations

        accelerations = accelerations[1::].reset_index(drop=True)

        #these needs at least 2 acceleration points

        accelMean = accelerations.mean()
        accelMedian = accelerations.median()
        accelSTD = accelerations.std()
        accelrVariance = accelerations.var()
        accelMin = accelerations.min()
        accelMax = accelerations.max()

        accelFirst = accelerations.iloc[0]
        accelLast = accelerations.iloc[-1]

        accelMinMax = accelMin/accelMax
        accelFirstLast = accelFirst/accelLast

        dfOutput = pd.DataFrame([accelMean, accelMedian, accelSTD, accelrVariance,
                                 accelMin,accelMax,
                                 accelFirst, accelLast,
                                 accelMinMax,accelFirstLast]).transpose()

        dfOutput.columns = ['accelMean','accelMedian', 'accelSTD', 'accelrVariance',
                            'accelMin', 'accelMax',
                            'accelFirst', 'accelLast',
                            'accelMinMax', 'accelFirstLast']

        return dfOutput

    def allToString(self):
        self.strMean = np.array(self.Mean).astype('|S10')
