# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.fftpack import fft
from scipy.fftpack import fftfreq


class GSRFrame:
    def __init__(self,input, label):
        self.data = input
        self.label = label
        self.tonic = pd.Series(data=None)
        self.phasic=pd.Series(data=None)
        self.baseline = None

    def preprocessing(self, baseline):
        self.baseline = baseline
        # downsample to 50Hz signal
        series = pd.Series(data = self.data.mediangsr)
        series.index = pd.DatetimeIndex(self.data.timeofday)
        series = series.resample('20L', how='mean', closed='right', label='left')  #20ms ± 50Hz
        series = series.fillna(method='backfill')

        self.resampledIndex = series.index

        #   alpha = 0.08 - band pass filter
        #  exponential smoothing with alpha = 0.08
        b, a = butter(5,Wn=0.66, btype='lowpass') #order of the filter, critical frequency, type of filter
        series = filtfilt(b, a, series)

        # Butterworth filter for tonic signal
        # f=0.05Hz, T=1/20s - in terms of resampled samples?
        # 0.75 = 0.05 (wanted) / 0.0666666 (our)
        # bf_low < - butter(n=5, W=0.75, type="low")
        # lowPassData < - signal:::filter(bf_low, x=data)

        b, a = butter(5,Wn=0.05, btype='lowpass') #order of the filter, critical frequency, type of filter
        tonic_series = filtfilt(b, a, series)
        self.tonic = pd.Series(data=tonic_series)
        self.tonic.index = self.resampledIndex

        # Butterworth filter for phasic signal
        b, a = butter(5, Wn=0.33, btype='highpass')  # order of the filter, critical frequency, type of filter
        phasic_series = filtfilt(b, a, series)
        self.phasic = pd.Series(data=phasic_series)
        self.phasic.index = self.resampledIndex

    def countTonicFeatures(self): # propagate here the baseline value from the first 10s of recordings
        self.meanSCL = self.tonic.mean()
        self.medianSCL = self.tonic.median()
        self.stdSCL = self.tonic.std()
        self.varSCL = self.tonic.var()
        self.minSCL = self.tonic.min()
        self.maxSCL = self.tonic.max()
        self.skewSCL = self.tonic.skew()
        self.kurtSCL = self.tonic.kurt()


        PCPS = ((self.tonic - self.tonic.mean())/self.tonic.std())/self.tonic.mean()
        PCPS.index = self.resampledIndex

        self.meanPCPS = PCPS.mean()
        self.medianPCPS = PCPS.median()
        self.stdPCPS = PCPS.std()
        self.varPCPS = PCPS.var()
        self.minPCPS = PCPS.min()
        self.maxPCPS = PCPS.max()
        self.skewPCPS = PCPS.skew()
        self.kurtPCPS = PCPS.kurt()

        self.ratioPeakValleySCL = self.tonic.max() /self.tonic.min()
        self.ratioPeakValleyPCPS = PCPS.max() / PCPS.min()

        self.diffPeakValleySLC = self.tonic.max() - self.tonic.min()
        self.diffPeakValleyPCPS = PCPS.max() - PCPS.min()

        self.diffFirstLastSLC = self.tonic.iloc[-1] - self.tonic.iloc[0]
        self.diffFirstLastPCPS = PCPS.iloc[-1] - PCPS.iloc[0]

        self.diffFirstLastPCPS

        #self.durationSCLPeakValley = abs((self.tonic.idxmin() - self.tonic.idxmax()).total_seconds() * 1000.0)
        self.durationPCPSPeakValley = abs((PCPS.idxmin() - PCPS.idxmax()).total_seconds() * 1000.0)

    def countPhasicFeatures(self):
        ff_phasic = fft(self.phasic)
        spacing = (self.phasic.index[1] - self.phasic.index[0]).total_seconds()
        ## Get Power Spectral Density
        signalPSD = np.abs(ff_phasic) ** 2
        fftFreq = fftfreq(len(signalPSD), spacing)
        #plt.plot(fftFreq[i], 10 * np.log10(signalPSD[i]));
        hist = np.histogram(signalPSD, bins = fftFreq[fftFreq > 0])


    def returnDataFrame(self):
        df = pd.DataFrame(data=[
            self.meanSCL,
            self.medianSCL,
            self.stdSCL ,
            self.varSCL,
            self.minSCL,
            self.maxSCL,
            self.skewSCL,
            self.kurtSCL,

            self.meanPCPS,
            self.medianPCPS,
            self.stdPCPS,
            self.varPCPS,
            self.minPCPS,
            self.maxPCPS,
            self.skewPCPS,
            self.kurtPCPS,

            self.ratioPeakValleySCL,
            self.ratioPeakValleyPCPS,
            self.diffPeakValleySLC,
            self.diffPeakValleyPCPS,
            self.diffFirstLastSLC,
            self.diffFirstLastPCPS,
            self.durationPCPSPeakValley
        ])
        df = df.transpose()
        df = df.rename(index=str, columns={
            0:"meanSCL",
            1:"medianSCL",
            2:"stdSCL",
            3:"varSCL",
            4:"minSCL",
            5:"maxSCL",
            6:"skewSCL",
            7:"kurtSCL",

            8:"meanPCPS",
            9:"medianPCPS",
            10:"stdPCPS",
            11:"varPCPS",
            12:"minPCPS",
            13:"maxPCPS",
            14:"skewPCPS",
            15:"kurtPCPS",

            16:"ratioPeakValleySCL",
            17:"ratioPeakValleyPCPS",
            18:"diffPeakValleySLC",
            19:"diffPeakValleyPCPS",
            20:"diffFirstLastSLC",
            21:"diffFirstLastPCPS",
            22:"durationPCPSPeakValley"
        } )
        return df