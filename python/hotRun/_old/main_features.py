# ------------------------------------------------------------------------------
# cut the columns we need and produce signal_out.csv
# ------------------------------------------------------------------------------


import os.path
import pandas as pd
import numpy as np
from hotPreprocess.AllData import AllData
from hotPreprocess.Participant import Participant

userhome = os.path.expanduser('~')


gsrout = os.path.join(userhome, 'Dropbox', 'HotOrNot','r_icmi', 'out_gsr.csv')
tmout = os.path.join(userhome, 'Dropbox', 'HotOrNot','r_icmi', 'out_tm.csv')
eyelineout = os.path.join(userhome, 'Dropbox', 'HotOrNot','r_icmi', 'out_eye.csv')


commentout = os.path.join(userhome, 'Dropbox', 'HotOrNot','r_icmi', 'comments3.csv')
commentStarts = os.path.join(userhome, 'Dropbox', 'HotOrNot','r_icmi', 'commentsStartsEnds_3.csv')


# ------------------------------------------------------------------------------
# count line entropy from eye-tracking data
# ------------------------------------------------------------------------------
allData = AllData()
allData.LoadTMDataFrame(tmout)
#allData.LoadEyeLineData(eyeresampled) #todo: repair the frequency, apply a fixation filter - eyelineout
allData.LoadEyeLineData(eyelineout)
allData.LoadCommentsStartEnd(commentStarts)
allData.LoadGSROut(gsrout)
allParticipants = np.intersect1d(np.unique(allData.eyeData.data.participant), np.unique(allData.commentData.data.participant))

allParticipants = np.array(['P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18',
       'P19', 'P20', 'P23', 'P24', 'P25', 'P28', # debug 'P27', 'P36'
       'P29', 'P30', 'P31', 'P32', 'P33', 'P34', 'P35', 'P37',
       'P39', 'P40', 'P41', 'P6', 'P7', 'P8']) # P21 and P22 excluded

outputDf = pd.DataFrame(data=None)
timeToCut = 60 #seconds



for i in range(len(allParticipants) - 1):
    print("Participant", i, "-", allParticipants[i])
    cmtDF = allData.GetCommentDataFrame(1, allParticipants[i])  # task 1, "1" # todo: update loading the comment data
    eyeDF = allData.GetETDataFrame(1, allParticipants[i])
    tmDF = allData.GetTmDataFrame("1",allParticipants[i])
    gsrDF = allData.GetGSRDataFrame("1", allParticipants[i])
    gsrBaseline = allData.GetGSRBaseline(allParticipants[i])

    participant = Participant(allParticipants[i])
    participant.LoadTask(tmDF, cmtDF, eyeDF, gsrDF)
    participant.LoadGSRBaseline(gsrBaseline)

    df = participant.CutData(timeToCut)
    df["participant"] = allParticipants[i]

    outputDf = outputDf.append(df, ignore_index=True)

featureFilename = os.path.join(userhome, 'Dropbox', 'HotOrNot','r_icmi', ('out_features_' + str(timeToCut) + 's.csv'))
outputDf.to_csv(path_or_buf = featureFilename, sep=",",  index=False)



print("***********************************\n All good\n")
exit(0)