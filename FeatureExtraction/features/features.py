# date: 04/30/18
# author: Hana Vrzakova
# description: Segmentation by n-consequtive fixations, 1 fixation overlap,
# feature extraction.

import numpy as np
import pandas as pd
from fixationDuration import FixationDuration
from fixationDistance import FixationDistance

from saccadeDuration import SaccadeDuration
from saccadeOrientation import SaccadeOrientation
from saccadeVelocity import SaccadeVelocity
from saccadeAcceleration import SaccadeAcceleration
from aoiFeatures import AoiFeatures
from pupilDilations import PupilDilations

# from pupilDilations.rawSignal import RawSignal

## absolute features: 

# The properties of the movement are:
# 1. direction
# 2. amplitude
# 3. duration
# 4. velocity
# 5. acceleration
# 6. shape
# 7. AOI order and transition measures
# 8. scanpaths comparison measures

# inteligentni ukladani - ano, chtelo by to

class Features(FixationDuration,FixationDistance,SaccadeDuration,SaccadeVelocity,SaccadeAcceleration,AoiFeatures,PupilDilations):
    def __init__(self):
        self.data = []
        self.aoi = []

        self.prefix = 0
        self.suffix = 0
        
        self.ndType = [('timestamp',int), 
                       ('pupilL', np.float32),  ('validityL',int),
                       ('pupilR', np.float32), ('validityR',int),
                       ('fixationNumber', int),
                       ('gazePointX',int), ('gazePointY',int),
                       ('event','S15'), 
                       ('rawX', int), ('rawY', int)
                       ]

    def load(self, data, aoi, prefix,  suffix):
        self.data = data
        self.aoi = aoi
        self.prefix = prefix
        self.suffix = suffix

    def loadPandas(self, data, aoi, prefix, suffix):
        self.data = data
        self.aoi = aoi
        self.prefix = prefix
        self.suffix = suffix

    def loadRawPandas(self, dfRaw):
        self.dfRaw = dfRaw

    def extractFeatures(self):
        self.fixationDuration = FixationDuration()
        self.fixationDistance = FixationDistance()
        self.saccadeDuration = SaccadeDuration()
        # self.saccadeDirection = SaccadeDirection()
        self.saccadeVelocity = SaccadeVelocity()
        self.saccadeAcceleration = SaccadeAcceleration()
        self.aoiFeatures = AoiFeatures()
        # self.rawSignal = RawSignal()

        for seq in self.data:
            
            self.fixationDuration.countStatistics(seq,  self.prefix,  self.suffix)
            self.fixationDistance.countStatistics(seq,  self.prefix,  self.suffix)
            self.saccadeDuration.countStatistics(seq,  self.prefix,  self.suffix)
            self.saccadeDirection.countStatistics(seq,  self.prefix,  self.suffix)
            self.saccadeVelocity.countStatistics(self.fixationDistance.Distances[-1],  self.saccadeDuration.Durations[-1],  self.prefix,  self.suffix)
            self.saccadeAcceleration.countStatistics(self.saccadeVelocity.Velocities[-1],  self.saccadeDuration.Durations[-1])


    def extractFeaturesPandas(self):

        fixationDuration = FixationDuration()
        featuresFixDuration = fixationDuration.countStatisticsPandas(self.data)

        fixationDistance = FixationDistance()
        featuresFixDistance = fixationDistance.countStatisticsPandas(self.data)

        saccadeDuration = SaccadeDuration()
        featuresSaccDuration = saccadeDuration.countStatisticsPandas(self.data)

        betweenFixVelocity = SaccadeVelocity()
        featuresSaccVelocity = betweenFixVelocity.countStatisticsPandas(fixationDistance.distances, saccadeDuration.durations)

        betweenFixAcceleration = SaccadeAcceleration()
        featuresSaccAcceleration = betweenFixAcceleration.countStatisticsPandas(betweenFixVelocity.velocities,
                                                                     saccadeDuration.durations)

        dfOutput = pd.concat([featuresFixDuration.reset_index(drop=True),featuresFixDistance.reset_index(drop=True),featuresSaccDuration.reset_index(drop=True),featuresSaccVelocity.reset_index(drop=True),featuresSaccAcceleration.reset_index(drop=True)], axis=1)

        return dfOutput

    def extractFeaturesPupilPandas(self):
        pupilDilations = PupilDilations(self.dfRaw)

        featuresRawPupilDilations = pupilDilations.countStatisticsPandas()
        featuresZscorePupilDilations = pupilDilations.countStatisticsZscorePandas()
        featuresAPCPSPupilDilations = pupilDilations.countStatisticsAPCPSPandas()

        dfOutput = pd.concat([ featuresRawPupilDilations.reset_index(drop=True), featuresZscorePupilDilations.reset_index(drop=True), featuresAPCPSPupilDilations.reset_index(drop=True)], axis=1)

        return dfOutput



