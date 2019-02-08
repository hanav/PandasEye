#-------------------------------------------------------------------------------
## Features and their statistics
#-------------------------------------------------------------------------------

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

from pupildilations.rawSignal import RawSignal

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

class Features():
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

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Main feature call
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def extractFeatures(self):
        self.fixationDuration = FixationDuration()
        self.fixationDistance = FixationDistance()
        self.saccadeDuration = SaccadeDuration()
        # self.saccadeDirection = SaccadeDirection()
        self.saccadeVelocity = SaccadeVelocity()
        self.saccadeAcceleration = SaccadeAcceleration()
        self.aoiFeatures = AoiFeatures()
        self.rawSignal = RawSignal()
        

        for seq in self.data:
            
            self.fixationDuration.countStatistics(seq,  self.prefix,  self.suffix)
            self.fixationDistance.countStatistics(seq,  self.prefix,  self.suffix)
            self.saccadeDuration.countStatistics(seq,  self.prefix,  self.suffix)
            self.saccadeDirection.countStatistics(seq,  self.prefix,  self.suffix)
            self.saccadeVelocity.countStatistics(self.fixationDistance.Distances[-1],  self.saccadeDuration.Durations[-1],  self.prefix,  self.suffix)
            self.saccadeAcceleration.countStatistics(self.saccadeVelocity.Velocities[-1],  self.saccadeDuration.Durations[-1])
            
            #self.aoiFeatures.countStatistics(seq,  self.prefix,  self.suffix,  self.aoi)        
##        self.rawSignal(seq)

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
        #todo: saccade direction - between fixations
        #saccOrientation = SaccadeOrientation()
        #saccOrientationFeatures = saccOrientation.countStatisticsPandas(fixationDistance.fixationCoords)

        #todo: saccade velocity - from raw points

        #todo: saccade accelleration - from raw points

        #todo: pupi dilations

        #todo: and final MERGE
        

        return dfOutput

    def extractFeaturesPupilPandas(self):
        pupilDilations = PupilDilations(self.dfRaw)

        featuresRawPupilDilations = pupilDilations.countStatisticsPandas()
        featuresZscorePupilDilations = pupilDilations.countStatisticsZscorePandas()
        featuresAPCPSPupilDilations = pupilDilations.countStatisticsAPCPSPandas()

        dfOutput = pd.concat([ featuresRawPupilDilations.reset_index(drop=True), featuresZscorePupilDilations.reset_index(drop=True), featuresAPCPSPupilDilations.reset_index(drop=True)], axis=1)

        return dfOutput

    def returnAllFeatures(self):

        allFeatures = []
        #allFeatures = map(" ".join, zip(self.fixationDuration.strMean(), self.fixationDuration.strMedian())) #not working, shrinking float to int to iterate, bastard
        #allFeatures = np.core.defchararray.add(myMean, myMedian) #not working
        #allFeatures = zip(itertools.repeat(self.fixationDuration.Mean, self.fixationDuration.Median),  self.fixationDuration.Median) #not working
        #allFeatures = np.concatenate((myMean,  myMedian),  axis=0) #not working
        
#        print "durations:",  len(self.fixationDuration.duration)
#        print "Mean:",  len(self.fixationDuration.Mean)
#        print "Median:",  len(self.fixationDuration.Median)       
#        self.fixationDuration.allToStr() 
#        print "strMean:",  len(self.fixationDuration.strMean)
#        print "strMedian:",  len(self.fixationDuration.strMedian)
#        print "data length:",  len(self.data)
        
        self.fixationDuration.allToString()
        self.fixationDistance.allToString()
        self.saccadeDuration.allToString()
        self.saccadeDirection.allToString()
        self.saccadeVelocity.allToString()
        self.saccadeAcceleration.allToString()
        #self.aoiFeatures.allToString()
        
        for i in range(0,len(self.data)):
            #print self.data[i][0][0]['timestamp']
            #print self.data[i][0]['timestamp']

            row = [
                      #self.data[i][0][0]['timestamp'].astype('|S10'), 
                    self.fixationDuration.strMean[i],       #2
                    self.fixationDuration.strMedian[i],     #3
                    self.fixationDuration.strStddev[i],     #4
                    self.fixationDuration.strVar[i],          #5  
                    self.fixationDuration.strMax[i],        #6
                    self.fixationDuration.strMin[i],        #7  
                    self.fixationDuration.strSum[i],        #8
                      
                    self.fixationDuration.strFirst[i],       #9
                    self.fixationDuration.strLast[i],       #10 
                    self.fixationDuration.strPrior[i],      #11         
                    self.fixationDuration.strPriorPrior[i],      #12     

                    self.fixationDuration.strRatioFirstPrior[i], #13 
                    self.fixationDuration.strRatioPriorLast[i], #14
                    self.fixationDuration.strRatioFirstLast[i], #15 
                      
                    ######################
                      
                      self.fixationDistance.strMean[i], #16
                      self.fixationDistance.strMedian[i], #17
                      self.fixationDistance.strStddev[i], #18
                      self.fixationDistance.strVar[i], #19
                      self.fixationDistance.strMax[i], #20
                      self.fixationDistance.strMin[i],  #21
                      self.fixationDistance.strSum[i], #22
                      
                      self.fixationDistance.strLast[i], #23
                      self.fixationDistance.strPrior[i], #24
                      
                      self.fixationDistance.strRatioPriorLast[i], #25 
                    
                    ######################
                      
                      self.saccadeDuration.strMean[i], #26
                      self.saccadeDuration.strMedian[i], #27
                      self.saccadeDuration.strStddev[i], #28
                      self.saccadeDuration.strVar[i], #29
                      self.saccadeDuration.strMax[i], #30
                      self.saccadeDuration.strMin[i],  #31
                      self.saccadeDuration.strSum[i], #32
                      
                      self.saccadeDuration.strLast[i], #33
                      self.saccadeDuration.strPrior[i], #34

                      self.saccadeDuration.strRatioPriorLast[i], #35                       

                    ######################

                      self.saccadeDirection.strMean[i], #36
                      self.saccadeDirection.strMedian[i], #37
                      self.saccadeDirection.strStddev[i], #38
                      self.saccadeDirection.strVar[i], #39
                      self.saccadeDirection.strMax[i], #40
                      self.saccadeDirection.strMin[i],  #41
                      self.saccadeDirection.strSum[i], #42
                      
                      self.saccadeDirection.strLast[i], #43
                      self.saccadeDirection.strPrior[i],  #44
                      
                      self.saccadeDirection.strRatioPriorLast[i], #45 
                        
                        ######################

                      
                      self.saccadeVelocity.strMean[i], #46
                      self.saccadeVelocity.strMedian[i], #47
                      #self.saccadeVelocity.strStddev[i], 
                      #self.saccadeVelocity.strVar[i], 
                      self.saccadeVelocity.strMax[i], #48
                      self.saccadeVelocity.strMin[i],  #49
                      self.saccadeVelocity.strSum[i], #50

                      self.saccadeVelocity.strPrior[i], #51
                      self.saccadeVelocity.strLast[i], #52

                      self.saccadeVelocity.strRatioPriorLast[i], #53 

                        ######################
#                      
                      self.saccadeAcceleration.strMean[i],  #54
                      
                      #######################
                      
                      self.fixationDuration.strDecreased[i] #55
#                      
#                      self.aoiFeatures.strMiddleTile[i], 
#                      self.aoiFeatures.strVisitedTilesCount[i], 
#                      self.aoiFeatures.strUniqueVisitedTiles[i], 
#                      self.aoiFeatures.FirstAsLast[i], 
#                      self.aoiFeatures.Strategy[i]
                      #self.aoiFeatures.VerticalStrategy[i], 
                      #self.aoiFeatures.HorizontalStrategy[i]
                      
                      ] 
 
## two-fixation datasets 
#            row = [
#                    self.fixationDuration.strMean[i],       #1
#                    self.fixationDuration.strStddev[i],     #2
#                    self.fixationDuration.strVar[i],          #3   
#                    self.fixationDuration.strSum[i],        #4
#                      
#                    self.fixationDuration.strFirst[i],       #5
#                    self.fixationDuration.strLast[i],       #6    
#
#                    self.fixationDuration.strRatioFirstLast[i], #7 
#                      
#                    ######################
#                      
#                      self.fixationDistance.strMean[i], #8
#                    
#                    ######################
#                      
#                      self.saccadeDuration.strMean[i], #9                     
#
#                    ######################
#
#                      self.saccadeDirection.strMean[i], #10
#                        
#                        ######################
#
#                      
#                      self.saccadeVelocity.strMean[i], #11
#
#                    ]
                      
## original dataset, do not erase
#                        str(self.meanFixationDuration[i], #2
#                        str(self.sumFixationDuration[i]), #3
#                        str(self.lastFixationDuration[i]), #4
#                        str(self.penultFixationDuration[i]), #5
#                        str(self.meanPathDistance[i]), #6
#                        str(self.sumPathDistance[i]), #7 - uncorr
#                        str(self.lastSaccadeLength[i]), #8
#                        str(self.lastSaccadeDuration[i]), #9 - uncorr
#                        str(self.sumSaccadeDuration[i]), #10
#                        str(self.meanSaccadeDuration[i]), #11
#                       #self.middleFixationPosition[i][0],
#                       #self.middleFixationPosition[i][1],
#                       #self.middleTileNumber[i],
#                        str(self.meanSaccadeSpeed[i]), #12 - uncorr
#                        str(self.lastSaccadeSpeed[i]), #13 - uncorr
#                        str(self.meanSaccadeAcceleration[i])] #14            

            
            allFeatures.append(row)

        return allFeatures



