# date: 04/30/18
# author: Hana Vrzakova
# description: Features derived from the pupillary responses.

import numpy as np
import pandas as pd

class PupilDilations:
    def __init__(self, pupils):
        self.pupils = pupils['pupil']

    def countStatisticsPandas(self):

        dfMean = self.pupils.mean()
        dfMedian = self.pupils.median()
        dfStd = self.pupils.std()
        dfVariance = self.pupils.var()
        dfMin = self.pupils.min()
        dfMax = self.pupils.max()

        dfFirst = self.pupils.iloc[0]
        dfLast = self.pupils.iloc[-1]

        dfMinMax = dfMin/dfMax
        dfFirstLast = dfFirst/dfLast

        dfOutput = pd.DataFrame([dfMean, dfMedian, dfStd, dfVariance,
                                 dfMin,dfMax,
                                 dfFirst, dfLast,
                                 dfMinMax, dfFirstLast]).transpose()

        dfOutput.columns = ['pupilRawMean','pupilRawMedian', 'pupilRawSTD', 'pupilRawVariance',
                            'pupilRawMin', 'pupilRawMax',
                            'pupilRawFirst', 'pupilRawLast',
                            'pupilRawMinMax', 'pupilRawFirstLast']

        return dfOutput


    def countStatisticsZscorePandas(self):

        # normalize by Z-score and divide by new mean

        pupils = (self.pupils - self.pupils.mean())/ self.pupils.std()

        dfMean = pupils.mean()
        dfMedian = pupils.median()
        dfMin = pupils.min()
        dfMax = pupils.max()

        dfFirst = pupils.iloc[0]
        dfLast = pupils.iloc[-1]

        dfMinMax = dfMin/dfMax
        dfFirstLast = dfFirst/dfLast

        dfOutput = pd.DataFrame([dfMean, dfMedian,
                                 dfMin,dfMax,
                                 dfFirst, dfLast,
                                 dfMinMax, dfFirstLast]).transpose()

        dfOutput.columns = ['pupilZscoreMean','pupilZscoreMedian',
                            'pupilZscoreMin', 'pupilZscoreMax',
                            'pupilZscoreFirst', 'pupilZscoreLast',
                            'pupilZscoreMinMax', 'pupilZscoreFirstLast']

        return dfOutput


    def countStatisticsAPCPSPandas(self):

        # normalize by PCPS
        pcps = (self.pupils - self.pupils.mean())/ self.pupils.mean()

        dfMean = pcps.mean()
        dfMedian = pcps.median()
        dfStd = pcps.std()
        dfVariance = pcps.var()
        dfMin = pcps.min()
        dfMax = pcps.max()

        dfFirst = pcps.iloc[0]
        dfLast = pcps.iloc[-1]

        dfMinMax = dfMin/dfMax
        dfFirstLast = dfFirst/dfLast

        dfOutput = pd.DataFrame([dfMean, dfMedian, dfStd, dfVariance,
                                 dfMin,dfMax,
                                 dfFirst, dfLast,
                                 dfMinMax, dfFirstLast]).transpose()

        dfOutput.columns = ['pupilPCPSMean','pupilPCPSMedian', 'pupilPCPSSTD', 'pupilPCPSVariance',
                            'pupilPCPSMin', 'pupilPCPSMax',
                            'pupilPCPSFirst', 'pupilPCPSLast',
                            'pupilPCPSMinMax', 'pupilPCPSFirstLast']

        return dfOutput