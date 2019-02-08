import os.path
import pandas as pd
import numpy as np
from hotPreprocess.AllData import AllData
from scipy.signal import butter
from scipy.signal import filtfilt

userhome = os.path.expanduser('~')

gsrout = os.path.join(userhome, 'Dropbox', 'HotOrNot','r_icmi', 'out_gsr.csv')
commentStarts = os.path.join(userhome, 'Dropbox', 'HotOrNot','r_icmi', 'commentsStartsEnds_3.csv')

allData = AllData()
allData.LoadCommentsStartEnd(commentStarts)
allData.LoadGSROut(gsrout)

allParticipants = np.intersect1d(np.unique(allData.gsrData.participant), np.unique(allData.commentData.data.participant))
allParticipants = np.array(['P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18',
       'P19', 'P20', 'P23', 'P24', 'P25', 'P28', 'P27', 'P36',
       'P29', 'P30', 'P31', 'P32', 'P33', 'P34', 'P35', 'P37',
       'P39', 'P40', 'P41', 'P6', 'P7', 'P8']) # P21 and P22 excluded

for i in range(len(allParticipants) - 1):
    print("Participant", i, "-", allParticipants[i])
    gsrDF = allData.GetAllGSRDataFrame(allParticipants[i])

    series = pd.Series(data=gsrDF.mediangsr)
    series.index = pd.DatetimeIndex(gsrDF.timeofday)
    series = series.resample('20L', how='mean', closed='right', label='left')
    series = series.fillna(method='backfill')

    b, a = butter(5, Wn=0.66, btype='lowpass')  # order of the filter, critical frequency, type of filter
    series = filtfilt(b, a, series)
    pd.Series(data=series).plot() # pokracovani tady

    b, a = butter(5, Wn=0.05, btype='lowpass')  # order of the filter, critical frequency, type of filter
    tonic_series = filtfilt(b, a, series)
    tonic = pd.Series(data=tonic_series)
    tonic

    # Butterworth filter for phasic signal #todo: continue here, exponential smoothing is missing
    b, a = butter(5, Wn=0.33, btype='highpass')  # order of the filter, critical frequency, type of filter
    phasic_series = filtfilt(b, a, series)
    phasic = pd.Series(data=phasic_series)
    phasic
