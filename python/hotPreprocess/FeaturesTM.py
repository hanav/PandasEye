import pandas as pd
import numpy as np


class FeaturesTM:
    def __init__(self):
        self.allFeatures = list()

    def CountHaarLikeFeatures(self, df):
        index_split = [2, 4, 6, 8]
        index_arr = df.index
        haarFeatures = list()

        for i in index_split:
            df_arr = np.array(df)
            splits = np.array_split(df_arr, i)
            df_splits = pd.DataFrame(splits)
            df_splits.fillna(0)
            m = df_splits.apply(np.mean, axis=1)
            differences = m.diff()
            haarFeatures.extend(differences)

        haarFeatures = pd.Series(haarFeatures)
        haarFeatures = haarFeatures[np.logical_not(np.isnan(haarFeatures))]
        self.allFeatures.extend(haarFeatures)


    def CountFeatures(self, df):
        self.allFeatures = list()
        # self.CountHaarLikeFeatures(df.tmsum)
        # self.CountHaarLikeFeatures(df.fingercount)
        #
        # featureCount = len(self.allFeatures) / 2
        # names = range(0, featureCount)
        # strNames = map(str, names)
        # [s + "TM_pxSum_" for s in strNames]
        # namesPixelSum = ["TM_pxSum_" + s for s in strNames]
        # namesFingerCount = ["TM_fingerCount_" + s for s in strNames]
        # featureNames = namesPixelSum + namesFingerCount

        meanTMSum = df['tmsum'].mean()
        medianTMSum = df['tmsum'].median()
        varTMSum = df['tmsum'].var()
        maxTMSum = df['tmsum'].max()
        minTMSum = df['tmsum'].min()

        meanTMCount = df['fingercount'].mean()
        medianTMCount = df['fingercount'].median()
        varTMCount = df['fingercount'].var()
        maxTMCount = df['fingercount'].max()
        minTMCount = df['fingercount'].min()

        self.allFeatures = [meanTMSum, medianTMSum, varTMSum, maxTMSum, minTMSum,
                            meanTMCount, medianTMCount, varTMCount, maxTMCount, minTMCount
                            ]

        row = pd.DataFrame([self.allFeatures])
        row.columns = ['TMSum_mean', 'TMSum_median','TMSum_var', 'TMSum_max','TMSum_min',
                       'TMCount_mean', 'TMCount_median', 'TMCount_var', 'TMCount_max', 'TMCount_min' ]

        return row