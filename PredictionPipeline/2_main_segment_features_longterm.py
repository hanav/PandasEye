# date: 04/30/17
# author: Hana Vrzakova
# description: Main script to segment data and extract the features from
# eye gaze, GSR, and TouchMouse, and label each feature vector by
# PAM-derived valence and arousal.

import os.path
import pandas as pd
import numpy as np
import dateutil as du
from hotPreprocess.FeaturesEye import FeaturesEye
from hotPreprocess.FeaturesGSR import FeaturesGSR
from hotPreprocess.FeaturesTM import FeaturesTM

def ConvertTimeofday(dateItem):
    # pinfo = du.parser.parserinfo(dayfirst = True, yearfirst=False) # example: 07/08/15 11:42
    pinfo = du.parser.parserinfo(dayfirst=False, yearfirst=False)  # example: 8/7/2015 11:42:23
    convertedDate = du.parser.parse(dateItem, parserinfo=pinfo)
    return convertedDate

userhome = os.path.expanduser('~')

# SETTINGS
OUTPUT = os.path.join('/Users/icce/Dropbox (Personal)/_thesis_framework/_scripts_hoy/r_icmi/')
outputDurationFile = os.path.join(OUTPUT,"features_longterm_first5min_stats_omitShort_ValenceArousal_npNan.csv")

windowSize = 100 #T = 20ms => 100 samples == 2seconds
overlap = 100 #= no overlap

SAMPLINGPERIOD = 20 #ms
LASTMINUTES = 5
SEGMENTORDER = 'FIRST'
OMITSHORT = True
SEGMENTSIZESECONDS = windowSize * SAMPLINGPERIOD / 1000

# Input data
eyePath = os.path.join('/Users/icce/Dropbox (Personal)/_thesis_framework/_scripts_hoy/r_icmi/resampled_eye.csv')
gsrPath = os.path.join('/Users/icce/Dropbox (Personal)/_thesis_framework/_scripts_hoy/r_icmi/resampled_gsr.csv')
tmPath = os.path.join('/Users/icce/Dropbox (Personal)/_thesis_framework/_scripts_hoy/r_icmi/resampled_tm.csv')

allParticipants = np.array([
    'P6',
    'P7', # - 24, -1, +1
'P8', # 25, -1, -1
'P9',
'P10',
'P11',
#'P12', # we don't have PAM evaluations for him
'P13',
'P14', # - 3, -1, -1
'P15',
'P16',
'P17',
'P18',
'P19',
'P21',
'P22',
'P23',
'P24',
'P25', # -11, -1, 1
'P27',
'P28',
'P29',
'P30',
'P31',
'P32', # 17, -1, -1
'P33',
'P34',
'P35',
'P36',
'P37',
'P38',
'P39',
'P40',
'P41']) # 22, -1, -1


print("Load CMT data")
cmtPath = os.path.join('/Users/icce/Dropbox (Personal)/_thesis_framework/_scripts_hoy/r_icmi/comment_CodeReviewCommentEmotionsWithTimestampsInMS.xlsx')
cmtDf = pd.read_excel(cmtPath)

print("Load questionnaire data")
questionniarePath = os.path.join('/Users/icce/Dropbox (Personal)/_thesis_framework/_scripts_hoy/r_scripts_longterm/5_task_1_PAM_separated_3.csv')
questionnaireDf =  pd.read_csv(questionniarePath, sep=',')

print("Load ET data")
eyeData = pd.read_csv(eyePath)

print("Load GSR data")
gsrData = pd.read_csv(gsrPath)

print("Load TM data")
tmData = pd.read_csv(tmPath)

i=0


output_durations_df = pd.DataFrame(columns=['participant','taskStart','taskEnd','taskDuration','taskDurationSec','taskDurationMin'], index=range(0,len(allParticipants)))

dfAll = pd.DataFrame()

for participant in allParticipants:
    eyeOutput = pd.DataFrame()
    gsrOutput = pd.DataFrame()
    tmOutput = pd.DataFrame()
    dfOutput = pd.DataFrame()

############################################################
    print("Processing touchMouse data...", participant)
    taskDf = tmData[(tmData.participant == participant) & (tmData.task == '1')]
    taskDf['time'] = pd.DatetimeIndex(taskDf.timeofday)

    taskStart = cmtDf['Recording Session Start Time (PDT)'].loc[cmtDf.Participant == participant].iloc[0]
    taskEnd = taskDf['time'].iloc[-1]
    sessionDf = taskDf[(taskDf['time'] >= taskStart) & (taskDf['time'] <= taskEnd)]

    sessionDuration = (sessionDf['timestamp'].iloc[-1] - sessionDf['timestamp'].iloc[0]) / 60000
    print(participant, ": TM session duration: ", sessionDuration, " minutes.")

    if sessionDuration < 5:  # 5 minutes
        # take all samples in the very short review
        if OMITSHORT == False:
            data = sessionDf.index
            firstIndex = sessionDf.index[0]
            lastIndex = sessionDf.index[-1]
        else:
            # skip the very short reviews
            continue
    else:

        if SEGMENTORDER == 'FIRST':
            # take first X minutes of code review
            lastSamples = (LASTMINUTES*60*1000) / SAMPLINGPERIOD
            data = sessionDf.index
            firstIndex = sessionDf.index[0]
            lastIndex = sessionDf.index[0] + lastSamples
        elif SEGMENTORDER == 'LAST':
            # take last X minutes of code review
            lastSamples = (LASTMINUTES*60*1000) / SAMPLINGPERIOD
            data = sessionDf.index
            firstIndex = sessionDf.index[-1] - lastSamples
            lastIndex = sessionDf.index[-1]
        else:
            print("set SEGMENT ORDER parameter")

    # adopted from here: https://stackoverflow.com/questions/35458404/generating-overlapping-sequences
    indeces = [[x for x in range(s, s + windowSize)] for s in range(firstIndex, lastIndex, overlap) if
               s + windowSize <= lastIndex + 1]

    print participant, ":", len(indeces)

    for idx in indeces:
        segment = sessionDf.loc[idx]
        tmFeatures = FeaturesTM()
        features_tm = tmFeatures.CountFeatures(segment)
        tmOutput = tmOutput.append(features_tm)

    ############################################################
    print("Processing eye-tracking data...")
    taskDf = eyeData[(eyeData.participant == participant) & (eyeData.task == 1)]
    taskDf['time'] = pd.DatetimeIndex(taskDf.timeofday)

    taskStart = cmtDf['Recording Session Start Time (PDT)'].loc[cmtDf.Participant == participant].iloc[0]
    taskEnd = taskDf['time'].iloc[-1]
    sessionDf = taskDf[(taskDf['time']>=taskStart) & (taskDf['time'] <= taskEnd)]

    sessionDuration = (sessionDf['timestamp'].iloc[-1] - sessionDf['timestamp'].iloc[0])/60000
    print(participant,": Gaze session duration: ", sessionDuration)

    if sessionDuration < 5: #5 minutes
        # take all samples in the very short review
        data = sessionDf.index
        firstIndex = sessionDf.index[0]
        lastIndex = sessionDf.index[-1]
    else:

        if SEGMENTORDER == 'FIRST':
            # take first X minutes of code review
            lastSamples = (LASTMINUTES*60*1000) / SAMPLINGPERIOD
            data = sessionDf.index
            firstIndex = sessionDf.index[0]
            lastIndex = sessionDf.index[0] + lastSamples

        elif SEGMENTORDER == 'LAST':
            # take last X minutes of code review
            lastSamples = (LASTMINUTES*60*1000) / SAMPLINGPERIOD
            data = sessionDf.index
            firstIndex = sessionDf.index[-1] - lastSamples
            lastIndex = sessionDf.index[-1]

        else:
            print("set SEGMENTORDER param")

    #adopted from here: https://stackoverflow.com/questions/35458404/generating-overlapping-sequences
    indeces = [[x for x in range(s, s + windowSize)] for s in range(firstIndex, lastIndex, overlap) if s + windowSize <= lastIndex + 1]

    print participant,":",len(indeces)

    for idx in indeces:
        segment = sessionDf.loc[idx]
        eyeFeatures = FeaturesEye()
        features_eye = eyeFeatures.CountFeatures(segment)
        eyeOutput = eyeOutput.append(features_eye)

    eyeOutput = eyeOutput.replace(to_replace=0, value=np.nan)
    eyeMedian = eyeOutput.median(skipna=True)
    if np.any(np.isnan(eyeOutput)) == True:
        eyeOutput.fillna(eyeMedian, inplace=True, axis=0)
############################################################
    print("Processing GSR data...", participant)
    taskDf = None
    taskDf = gsrData[(gsrData.participant == participant) & (gsrData.task == '1')]
    taskDf['time'] = pd.DatetimeIndex(taskDf.timeofday)

    taskStart = cmtDf['Recording Session Start Time (PDT)'].loc[cmtDf.Participant == participant].iloc[0]
    taskEnd = taskDf['time'].iloc[-1]
    sessionDf = taskDf[(taskDf['time']>=taskStart) & (taskDf['time'] <= taskEnd)]

    sessionDuration = (sessionDf['timestamp'].iloc[-1] - sessionDf['timestamp'].iloc[0])/60000
    print(participant,": GSR session duration: ", sessionDuration)

    if sessionDuration < 5: #5 minutes
        # take all samples in the very short review
        data = sessionDf.index
        firstIndex = sessionDf.index[0]
        lastIndex = sessionDf.index[-1]
    else:

        if SEGMENTORDER == 'FIRST':
            # take first X minutes of code review
            lastSamples = (LASTMINUTES*60*1000) / SAMPLINGPERIOD
            data = sessionDf.index
            firstIndex = sessionDf.index[0]
            lastIndex = sessionDf.index[0] + lastSamples

        elif SEGMENTORDER == 'LAST':
            # take last X minutes of code review
            lastSamples = (LASTMINUTES*60*1000) / SAMPLINGPERIOD
            data = sessionDf.index
            firstIndex = sessionDf.index[-1] - lastSamples
            lastIndex = sessionDf.index[-1]

        else:
            print("set SEGMENT ORDER param")

    #adopted from here: https://stackoverflow.com/questions/35458404/generating-overlapping-sequences
    indeces = [[x for x in range(s, s + windowSize)] for s in range(firstIndex, lastIndex, overlap) if s + windowSize <= lastIndex + 1]
    print participant,":",len(indeces)

    for idx in indeces:
        segment = sessionDf.loc[idx]
        gsrFeatures = FeaturesGSR()
        features_gsr = gsrFeatures.CountFeatures(segment, SEGMENTSIZESECONDS)
        gsrOutput = gsrOutput.append(features_gsr)

    gsrOutput = gsrOutput.replace(to_replace=0, value=np.nan)
    gsrMedian = gsrOutput.median(skipna=True)
    if np.any(np.isnan(gsrOutput)) == True:
        gsrOutput.fillna(gsrMedian, inplace=True, axis=0)

############################################################
    # Merge: eyeOutput + gsrOutput
    print("Merge output with labels", participant)

    dfOutput = pd.concat([eyeOutput.reset_index(drop=True), gsrOutput.reset_index(drop=True), tmOutput.reset_index(drop=True)], axis=1)

    dfOutput['participant'] = participant

    # pID = questionnaireDf.loc[questionnaireDf['Participant']==participant]
    # pID.index
    diffValence = questionnaireDf['diff_pre_post_valence_Y'].loc[questionnaireDf['Participant']==participant]
    dfOutput['diffValence'] = diffValence.iloc[0]

    valenceCode = questionnaireDf['post_PAM_valence (Y)'].loc[questionnaireDf['Participant'] == participant]
    dfOutput['valenceCode'] = valenceCode.iloc[0]

    if(valenceCode.iloc[0] <= 2):
        dfOutput['valenceCodeBinary'] = -1
    else:
        dfOutput['valenceCodeBinary'] = 1


    arousalCode = questionnaireDf['post_PAM_arousal (Y)'].loc[questionnaireDf['Participant'] == participant]
    dfOutput['arousalCode'] = arousalCode.iloc[0]

    if(arousalCode.iloc[0] <= 2):
        dfOutput['arousalCodeBinary'] = -1
    else:
        dfOutput['arousalCodeBinary'] = 1

    dfAll = dfAll.append(dfOutput)
    i=i+1



dfAll.to_csv(path_or_buf = outputDurationFile, sep=",",index=False)

print("All good, folks!")
exit(0)
