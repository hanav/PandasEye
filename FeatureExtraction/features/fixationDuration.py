# date: 04/30/18
# author: Hana Vrzakova
# description: Features derived from the fixation durations.

from __future__ import division
import numpy as np
import pandas as pd


class FixationDuration():
    def __init__(self):

        self.Mean = []
        self.Median = []
        self.Stddev = []
        self.Var = []
        self.Max = []
        self.Min = []
        self.Sum = []
        
        self.First = []
        self.Last = []
        self.Prior = []
        self.PriorPrior = []
 
        self.RatioFirstLast = []
        self.RatioFirstPrior = []
        self.RatioPriorLast = []
        
        self.Decreased = []
        
    # for each single sequence, count    
    def countStatistics(self, seq,  prefix,  suffix):
        durations = []
        
        #print "Lenght of sequence:",  len(seq)
        
        for i in range(0,len(seq)):
            duration = seq[i][-1]['timestamp'] - seq[i][0]['timestamp'] 
            durations.append(duration)        

        self.Mean.append(np.mean(durations))
        self.Median.append(np.median(durations))
        self.Stddev.append(np.std(durations))
        self.Var.append(np.var(durations))
        self.Max.append(np.max(durations))
        self.Min.append(np.min(durations))
        self.Sum.append(np.sum(durations))
        
        self.First.append(durations[0])
        self.Last.append(durations[-1])
        self.Prior.append(durations[-2])
        self.PriorPrior.append(durations[-3])
        
        ratio1 = -1
        ratio2 = -1
        ratio3 = -1
        
        if(durations[-2] > 0):
            ratio1 = durations[0]/durations[-2]
        if(durations[-1] > 0):
            ratio2 = durations[-2]/durations[-1]
            ratio3 = durations[0]/durations[-1]
        
        self.RatioFirstPrior.append(ratio1)
        self.RatioPriorLast.append(ratio2)
        self.RatioFirstLast.append(ratio3) 
        
        if(durations[-3] > durations[-2] > durations[-1]):
            decreased = 1
        else:
            decreased = 0
        self.Decreased.append(decreased)
        
        #print "Durations:",  durations
        #print "Ratios:",  self.RatioFirstPrior,  self.RatioPriorLast, self.RatioFirstLast

    def subtract_first_last_timestamp(self, df):
        return df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]

    def countStatisticsPandas(self, df):
        durations = df.groupby(['fixationNumber']).apply(self.subtract_first_last_timestamp) # index je fixation id, hodnota je fixation duration

        #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.describe.html
        #centralTendency = durations.describe() - hezky, ale nepotrebujeme

        fixDurMean = durations.mean()
        fixDurMedian = durations.median()
        fixDurVariance = durations.var()
        fixDurSTD = durations.std()
        fixDurMin = durations.min()
        fixDurMax = durations.max()
        fixDurSum = durations.sum()

        fixDurFirst = durations.iloc[0]
        fixDurLast = durations.iloc[-1]
        fixDurPriorLast = durations.iloc[-2]

        fixDurRatioFirstLast = fixDurFirst / fixDurLast
        fixDurRatioFirstPrior = fixDurFirst / fixDurPriorLast
        fixDurRatioPriorLast = fixDurPriorLast / fixDurLast

        dfOutput = pd.DataFrame([fixDurMean,fixDurMedian,fixDurVariance,fixDurSTD,
                                 fixDurMin,fixDurMax,fixDurSum,
                                 fixDurFirst,fixDurLast,fixDurPriorLast,
                                 fixDurRatioFirstLast, fixDurRatioFirstPrior, fixDurRatioPriorLast]).transpose()

        dfOutput.columns = ['fixDurMean', 'fixDurMedian', 'fixDurVariance', 'fixDurSTD',
                   'fixDurMin', 'fixDurMax', 'fixDurSum',
                   'fixDurFirst', 'fixDurLast', 'fixDurPriorLast',
                   'fixDurRatioFirstLast', 'fixDurRatioFirstPrior', 'fixDurRatioPriorLast']

        return dfOutput
        
    def allToString(self):
        self.strMean = np.array(self.Mean).astype('|S10')
        self.strMedian= np.array(self.Median).astype('|S10')
        self.strStddev = np.array(self.Stddev).astype('|S10')
        self.strVar = np.array(self.Var).astype('|S10')
        self.strMax = np.array(self.Max).astype('|S10')
        self.strMin = np.array(self.Min).astype('|S10')
        self.strSum = np.array(self.Sum).astype('|S10')
        
        self.strFirst = np.array(self.First).astype('|S10')
        self.strLast = np.array(self.Last).astype('|S10')
        self.strPrior = np.array(self.Prior).astype('|S10')
        self.strPriorPrior = np.array(self.PriorPrior).astype('|S10')       
        
        self.strRatioFirstPrior = np.array(self.RatioFirstPrior).astype('|S10')
        self.strRatioPriorLast = np.array(self.RatioPriorLast).astype('|S10')
        self.strRatioFirstLast = np.array(self.RatioFirstLast).astype('|S10')
        
        self.strDecreased = np.array(self.Decreased).astype('|S10') 
        
