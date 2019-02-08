#-------------------------------------------------------------------------------
## Saccadic duration
#-------------------------------------------------------------------------------
from __future__ import division
import numpy as np
import pandas as pd

class SaccadeDuration():
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
        
        self.Durations = []
        
        
        
    def countStatistics(self, seq,  prefix,  suffix):

        durations  = []
        for i in range(0,len(seq)-1):
            duration = seq[i+1][0]['timestamp'] - seq[i][-1]['timestamp'] 
            durations.append(duration)
            
        self.Durations.append(durations)

        self.Mean.append(np.mean(durations))
        self.Median.append(np.median(durations))
        self.Stddev.append(np.std(durations))
        self.Var.append(np.var(durations))
        self.Max.append(np.max(durations))
        self.Min.append(np.min(durations))
        self.Sum.append(np.sum(durations))
        
        self.Prior.append(durations[0])
        self.Last.append(durations[-1])
        
        ratio = -1
        if(durations[-1] > 0):
            ratio = durations[-2]/durations[-1]
        self.RatioPriorLast.append(ratio)
        
        #print "durations",  durations
        #print "ratio",  self.RatioPriorLast

    def subtract_first_last_timestamp(self, df):
        return pd.Series(data=[df['timestamp'].iloc[0], df['timestamp'].iloc[-1]])

    def countStatisticsPandas(self, df):
        durations = df.groupby(['fixationNumber']).apply(self.subtract_first_last_timestamp)  # index je fixation id, hodnota je fixation duration
        durations.columns = ['fixStart','fixEnd']
        durations['fixEnd'] = durations['fixEnd'].shift(1)
        durations = durations['fixStart'] - durations['fixEnd']
        durations = durations.iloc[1::].reset_index(drop=True)

        # save for velocity between fixations
        self.durations = durations

        sacDurMean = durations.mean()
        sacDurMedian = durations.median()
        sacDurVariance = durations.var()
        sacDurSTD = durations.std()
        sacDurMin = durations.min()
        sacDurMax = durations.max()
        sacDurSum = durations.sum()

        sacDurFirst = durations.iloc[0]
        sacDurLast = durations.iloc[-1]
        saDurPriorLast = durations.iloc[-2]

        sacDurRatioFirstLast = sacDurFirst / sacDurLast
        sacDurratioFirstPrior = sacDurFirst / saDurPriorLast
        sacDurratioPriorLast = saDurPriorLast / sacDurLast

        dfOutput = pd.DataFrame([sacDurMean, sacDurMedian, sacDurVariance, sacDurSTD,
                                 sacDurMin, sacDurMax, sacDurSum,
                                 sacDurFirst, sacDurLast, saDurPriorLast,
                                 sacDurRatioFirstLast, sacDurratioFirstPrior, sacDurratioPriorLast]).transpose()

        dfOutput.columns = ['sacDurMean', 'sacDurMedian', 'sacDurVariance', 'sacDurSTD',
                            'sacDurMin', 'sacDurMax', 'sacDurSum',
                            'sacDurFirst', 'sacDurLast', 'saDurPriorLast',
                            'sacDurRatioFirstLast', 'sacDurratioFirstPrior', 'sacDurratioPriorLast']

        return dfOutput


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
