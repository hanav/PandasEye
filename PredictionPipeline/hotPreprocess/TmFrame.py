import numpy as np
import pandas as pd

class TmFrame:
    def __init__(self, input, label):
        self.data = input
        self.label = label
        self.meanTMSum = None
        self.meanFingerCount = None


    def countTMSumStats(self):

        self.meanTMSum = self.data.tmsum.mean()
        self.medianTMSum = self.data.tmsum.median()
        self.stdTMSum = self.data.tmsum.std()
        self.varTMSum = self.data.tmsum.var()
        self.minTMSum = self.data.tmsum.min()
        self.maxTMSum = self.data.tmsum.max()
        self.skewTMSum = self.data.tmsum.skew()
        self.kurtTMSum = self.data.tmsum.kurt()

    def countFingerStats(self):
        self.meanFingerCount = self.data.fingercount.mean()
        self.medianFingerCount = self.data.fingercount.median()
        self.stdFingerCount = self.data.fingercount.std()
        self.varFingerCount = self.data.fingercount.var()
        self.minFingerCount = self.data.fingercount.min()
        self.maxFingerCount = self.data.fingercount.max()
        self.skewFingerCount = self.data.fingercount.skew()
        self.kurtFingerCount = self.data.fingercount.kurt()

# tmsum / tmcount = normalized signal

    def returnDataFrameLabel(self):
        df = pd.DataFrame(data=[
            self.label,
            self.meanTMSum,
            self.medianTMSum,
            self.stdTMSum,
            self.varTMSum,
            self.minTMSum,
            self.maxTMSum,
            self.skewTMSum,
            self.kurtTMSum,

            self.meanFingerCount,
            self.medianFingerCount,
            self.stdFingerCount,
            self.varFingerCount,
            self.minFingerCount,
            self.maxFingerCount,
            self.skewFingerCount,
            self.kurtFingerCount
        ])
        df = df.transpose()
        return df

    def returnDataFrame(self):
        df = pd.DataFrame(data=[
            self.meanTMSum,
            self.medianTMSum,
            self.stdTMSum,
            self.varTMSum,
            self.minTMSum,
            self.maxTMSum,
            self.skewTMSum,
            self.kurtTMSum,

            self.meanFingerCount,
            self.medianFingerCount,
            self.stdFingerCount,
            self.varFingerCount,
            self.minFingerCount,
            self.maxFingerCount,
            self.skewFingerCount,
            self.kurtFingerCount
        ])

        df = df.transpose()
        df = df.rename(index=str, columns={
            1: "meanTMSum",
            2: "medianTMSum",
            3:"stdTMSum",
            4:"varTMSum",
            5:"minTMSum",
            6:"maxTMSum",
            7:"skewTMSum",
            8:"kurtTMSum",

            9:"meanFingerCount",
            10:"medianFingerCount",
            11:"stdFingerCount",
            12:"varFingerCount",
            13:"minFingerCount",
            14:"maxFingerCount",
            15:"skewFingerCount",
            16:"kurtFingerCount"} )
        return df