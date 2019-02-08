import pandas as pd
import numpy as np
# from matplotlib import pyplot as plt

from scipy.signal import butter
from scipy.signal import filtfilt


class FeaturesGSR:
    def __init__(self):
        self.df = None
        self.tonic = None
        self.phasic = None

        self.tonicFeatures = list()
        self.phasicFeatures = list()
        self.allFeatures = list()

    def SubtractBaseline(self):
        self.df = self.df - self.df.iloc[0]
        #plt.plot(self.df)

    def NormalizeDf(self):
        self.df = (self.df - self.df.mean()) / self.df.std()
        #plt.plot(self.df)

    def Smoothing(self):
        #   alpha = 0.08 - band pass filter
        #  exponential smoothing with alpha = 0.08
        b, a = butter(5, Wn=0.66, btype='lowpass')  # order of the filter, critical frequency, type of filter
        self.df = filtfilt(b, a, self.df)
        #plt.plot(self.df)
        return self.df

    def CountTonicDf(self):
        # Butterworth filter for tonic signal
        # f=0.05Hz, T=1/20s - in terms of resampled samples?
        # 0.75 = 0.05 (wanted) / 0.0666666 (our)
        # bf_low < - butter(n=5, W=0.75, type="low")
        # lowPassData < - signal:::filter(bf_low, x=data)

        b, a = butter(5,Wn=0.05, btype='lowpass') #order of the filter, critical frequency, type of filter
        tonic_series = filtfilt(b, a, self.df)
        self.tonic = pd.Series(data=tonic_series)
        #self.tonic.index = self.resampledIndex
        #plt.plot(self.tonic)

    def CountPhasicDf(self):
        # Butterworth filter for phasic signal
        b, a = butter(5, Wn=0.33, btype='highpass')  # order of the filter, critical frequency, type of filter
        phasic_series = filtfilt(b, a, self.df)
        #self.phasic.index = self.resampledIndex
        #plt.plot(self.phasic)

    def PreprocessSignal(self, taskDF):
        df = taskDF.mediangsr
        df.index = taskDF.time
        df = (df - df.mean()) / df.std() # zscore

        # smoothing
        #   alpha = 0.08 - band pass filter
        #  exponential smoothing with alpha = 0.08
        b, a = butter(5, Wn=0.66, btype='lowpass')  # order of the filter, critical frequency, type of filter
        df = filtfilt(b, a, df)
        self.processedDF = df

        self.FilterPhasic()
        self.FilterTonic()

    def FilterPhasic(self):
        b, a = butter(5, Wn=0.33, btype='highpass')  # order of the filter, critical frequency, type of filter
        self.phasic_series = filtfilt(b, a, self.processedDF)

    def FilterTonic(self):
        # tady to nefunguje spravne, protoze ten signal vypada uplne stejne
        b, a = butter(5,Wn=0.05, btype='lowpass') #order of the filter, critical frequency, type of filter
        self.tonic_series = filtfilt(b, a, self.processedDF)

    def detectPeaks(self):
        #http: // stackoverflow.com / questions / 24656367 / find - peaks - location - in -a - spectrum - numpy
        #todo: find better alpha for peaks here
        #todo: smooth the signal a bit, so it's easier to find peaks

        from scipy.signal import convolve
        kernel = [1, 0, -1]
        dY = convolve(self.phasic, kernel, 'valid')

        # Checking for sign-flipping
        S = np.sign(dY)
        ddS = convolve(S, kernel, 'valid')

        # These candidates are basically all negative slope positions
        # Add one since using 'valid' shrinks the arrays
        candidates = np.where(dY < 0)[0] + (len(kernel) - 1)

        # Here they are filtered on actually being the final such position in a run of
        # negative slopes
        peaks = sorted(set(candidates).intersection(np.where(ddS == 2)[0] + 1))

        #plt.plot(self.phasic)
        # If you need a simple filter on peak size you could use:
        alpha = -0.0025
        peaks = np.array(peaks)[self.phasic[peaks] < alpha]
        # plt.scatter(peaks, self.phasic[peaks], marker='x', color='g', s=40)

    # division by 10s
    def CountTonicFeatures(self):
        self.tonic_series = pd.Series(self.tonic_series)
        meanSCL = self.tonic_series.mean() #
        medianSCL = self.tonic_series.median()
        varSCL = self.tonic_series.var()
        maxSCL = self.tonic_series.max()
        minSCL = self.tonic_series.min()
        sumSCL = self.tonic_series.sum() / self.segmentSizeSeconds # Barrel 2017: the sum of the tonic activity per second, computed as the sum of tonic data within an abstract (centered with the mean tonic data of the first 10 s spent reading the article) divided by the time spent reading the article (sumTonic)

        self.allFeatures.extend([meanSCL,medianSCL, varSCL, maxSCL, minSCL, sumSCL])

    def CountPhasicFeatures(self):
        # ups < - floor((floor(2000 / dx)) / 3)
        # downs < - floor(floor(4000 / dx) / 3)
        # phasicPeaks < - findpeaks(as.numeric(smoothed), nups = ups, ndowns = downs)
        self.phasic_series = pd.Series(self.phasic_series)

        meanSCR = self.phasic_series.mean()
        medianSCR = self.phasic_series.median()
        varSCR = self.phasic_series.var()
        maxSCR = self.phasic_series.max()
        minSCR = self.phasic_series.min()
        sumSCR = self.phasic_series.sum() / self.segmentSizeSeconds

        self.allFeatures.extend([meanSCR, medianSCR, varSCR, maxSCR, minSCR, sumSCR ])

    def HaarLikeFeature(self, df1, df2):
        feature = df1.mean() - df2.mean()
        return feature

    def CountHaarLikeFeatures(self, df):
        index_split = [2, 4, 6, 8, 10, 12, 14, 16]
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


    def CountFeatures(self, df, segmentSizeSeconds):
        self.segmentSizeSeconds = segmentSizeSeconds
        self.allFeatures = list()

        self.PreprocessSignal(df)

        #self.df = df.mediangsr
        #self.df.index = df.time
        # http://stackoverflow.com/questions/34342370/analysing-time-series-in-python-pandas-formatting-error-statsmodels
        # error: proste nefunguje
        # res = sm.tsa.seasonal_decompose(self.df)
        # resplot = res.plot()

        #self.SubtractBaseline()
        #self.NormalizeDf()
        #self.Smoothing()

        self.CountTonicFeatures()
        self.CountPhasicFeatures()

        row = pd.DataFrame([self.allFeatures])
        row.columns = ['GSR_tonic_mean','GSR_tonic_median','GSR_tonic_var','GSR_tonic_max','GSR_tonic_min', 'GSR_tonic_sum_s',
                                'GSR_phasic_mean','GSR_phasic_median','GSR_phasic_var','GSR_phasic_max','GSR_phasic_min','GSR_phasic_sum_s']

        return row

    # todo: pridat percentage change
    # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.pct_change.html
