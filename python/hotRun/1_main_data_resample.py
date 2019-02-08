# ------------------------------------------------------------------------------
# resample the best effort frequency to a standard frequency
# ------------------------------------------------------------------------------
import os.path
import pandas as pd
import numpy as np

userhome = os.path.expanduser('~')
outputPath = os.path.join(userhome, 'Dropbox', 'HotOrNot','r_icmi')
gsrPath = os.path.join(userhome, 'Dropbox', 'HotOrNot','r_icmi', 'out_gsr.csv')
tmPath = os.path.join(userhome, 'Dropbox', 'HotOrNot','r_icmi', 'out_tm.csv')
eyePath = os.path.join(userhome, 'Dropbox', 'HotOrNot','r_icmi', 'out_eye.csv')


print("Resample ET data")
eyeData = pd.read_csv(eyePath)

timestampList = list()
participantList = list()
taskList = list()
timeofdayList = list()
validityList = list()
gazexList = list()
gazeyList = list()
linenumList = list()
mousexList = list()
mouseyList = list()

for userID in eyeData.participant.unique():
    #todo: urychlit pres indexy
    userDf = eyeData[eyeData.participant == userID]
    tasks = userDf.task.unique()
    for task in tasks:
        print userID,"-",task
        taskDf = userDf[userDf.task == task]
        taskDf.index = pd.DatetimeIndex(taskDf.timeofday)
        # 20L == 20ms == 50Hz

        resampleValidity = taskDf.validity.resample('20L', how='mean', closed='right', label='left')
        resampleValidity = resampleValidity.fillna(method='backfill')

        resampleGazeX = taskDf.gazex.resample('20L', how='mean', closed='right', label='left')
        resampleGazeX = resampleGazeX.fillna(method='backfill')

        resampleGazeY = taskDf.gazey.resample('20L', how='mean', closed='right', label='left')
        resampleGazeY = resampleGazeY.fillna(method='backfill')

        resampleMouseX = taskDf.mousex.resample('20L', how='mean', closed='right', label='left')
        resampleMouseX = resampleMouseX.fillna(method='backfill')

        resampleMouseY = taskDf.mousey.resample('20L', how='mean', closed='right', label='left')
        resampleMouseY = resampleMouseY.fillna(method='backfill')

        resampleLinenum = taskDf.linenum.resample('20L', how='mean', closed='right', label='left')
        resampleLinenum = resampleLinenum.fillna(method='backfill')

        resampleTimestamp = range(np.int(taskDf.timestamp[0]), np.int(taskDf.timestamp[0]) + resampleValidity.size*20, 20)

        # reconstruct the dataframe
        timestampList.extend(resampleTimestamp)
        participantList.extend ([userID] * resampleValidity.size)
        taskList.extend([task] * resampleValidity.size)
        timeofdayList.extend(resampleValidity.index)

        validityList.extend(resampleValidity)
        gazexList.extend(resampleGazeX)
        gazeyList.extend(resampleGazeY)
        mousexList.extend(resampleMouseX)
        mouseyList.extend(resampleMouseY)
        linenumList.extend(resampleLinenum)

    taskDf

outputDF = pd.DataFrame(data=None, columns=list(eyeData))


outputDF.timestamp =timestampList
outputDF.participant = participantList
outputDF.task = taskList
outputDF.timeofday = timeofdayList
outputDF.validity = validityList
outputDF.gazex =gazexList
outputDF.gazey = gazeyList
outputDF.linenum = linenumList
outputDF.mousex = mousexList
outputDF.mousey = mouseyList


outputFilename = os.path.join(userhome, outputPath, 'resampled_eye.csv')
outputDF.to_csv(path_or_buf = outputFilename, sep=",",  index=False)

print("Resample GSR")
timestampList = list()
participantList = list()
taskList = list()
timeofdayList = list()
mediangsrList = list()

gsrData = pd.read_csv(gsrPath)

for userID in gsrData.participant.unique():
    userDf = gsrData[gsrData.participant == userID]
    tasks = userDf.task.unique()
    for task in tasks:
        print userID,"-",task
        taskDf = userDf[userDf.task == task]
        taskDf.index = pd.DatetimeIndex(taskDf.timeofday)
        # 20L == 20ms == 50Hz
        resampleDf = taskDf.mediangsr.resample('20L', how='mean', closed='right', label='left')
        resampleDf = resampleDf.fillna(method='backfill')
        resampleTimestamp = range(np.int(taskDf.timestamp[0]), np.int(taskDf.timestamp[0]) + resampleDf.size*20, 20)

        # reconstruct the dataframe
        timestampList.extend(resampleTimestamp)
        participantList.extend ([userID] * resampleDf.size)
        taskList.extend([task] * resampleDf.size)
        timeofdayList.extend(resampleDf.index)
        mediangsrList.extend(resampleDf)
    taskDf

outputDF = pd.DataFrame(data=None, columns=list(gsrData))
# timestamp, participant, task, timeofday, mediangsr

outputDF.timestamp =timestampList
outputDF.participant = participantList
outputDF.task = taskList
outputDF.timeofday = timeofdayList
outputDF.mediangsr = mediangsrList

outputFilename = os.path.join(userhome, outputPath, 'resampled_gsr.csv')
outputDF.to_csv(path_or_buf = outputFilename, sep=",",  index=False)
