import os.path
import pandas as pd
import numpy as np

class Resample_GSR:

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