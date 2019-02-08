import os.path
import pandas as pd
import numpy as np
import dateutil as du


def ConvertTimeofday(dateItem):
    # pinfo = du.parser.parserinfo(dayfirst = True, yearfirst=False) # example: 07/08/15 11:42
    pinfo = du.parser.parserinfo(dayfirst=False, yearfirst=False)  # example: 8/7/2015 11:42:23
    convertedDate = du.parser.parse(dateItem, parserinfo=pinfo)
    return convertedDate

userhome = os.path.expanduser('~')
outputPath = os.path.join('/Users/icce/Dropbox/_thesis_framework/_scripts_hoy/r_icmi/')

eyePath = os.path.join(userhome, 'Dropbox', 'HotOrNot','r_icmi', 'resampled_eye.csv') #132 MB
gsrPath = os.path.join(userhome, 'Dropbox', 'HotOrNot','r_icmi', 'resampled_gsr.csv') #
tmPath = os.path.join(userhome, 'Dropbox', 'HotOrNot','r_icmi', 'resampled_tm.csv') #

###todo: moje nove anotace, imabalance classes 21:6
allParticipants = np.array(['P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18',
       'P19', 'P20', 'P21','P22','P23', 'P24', 'P25', 'P27','P28',
       'P29', 'P30', 'P31', 'P32', 'P33', 'P34', 'P35', 'P36','P37', 'P38',
       'P39', 'P40', 'P41', 'P6', 'P7', 'P8','P9'])

negative_diff =['P6',
'P8',
'P9',
'P11',
'P13',
'P14',
'P15',
'P16',
'P19',
'P22',
'P23',
'P27',
'P28',
'P29',
'P31',
'P32',
'P33',
'P37',
'P38',
'P39',
'P41']

positive_diff = ['P10','P21','P25','P30','P34','P40']

print("Load CMT data")
cmtPath = os.path.join('/Users/icce/Dropbox/_thesis_framework/_scripts_hoy/r_icmi/comment_CodeReviewCommentEmotionsWithTimestampsInMS.xlsx')
cmtDf = pd.read_excel(cmtPath)

print("Load ET data")
eyeData = pd.read_csv(eyePath)


i=0

output_durations_df = pd.DataFrame(columns=['participant','taskStart','taskEnd','taskDuration','taskDurationSec','taskDurationMin','numberComments'], index=range(0,len(allParticipants)))

# skoncilo to poslednim indexem.
for participant in allParticipants:

    #todo: for task 1 - later task 2
    taskDf = eyeData[(eyeData.participant == participant) & (eyeData.task == 1)]
    taskDf['time'] = pd.DatetimeIndex(taskDf.timeofday)

    taskStart = cmtDf['Recording Session Start Time (PDT)'].loc[cmtDf.Participant == participant].iloc[0]
    taskEnd = taskDf['time'].iloc[-1]

    taskDuration = taskEnd - taskStart
    #taskDurationSec =  timedelta(hours=taskDuration._h, minutes=taskDuration._m, seconds=taskDuration._s).total_seconds()
    taskDurationSec = taskDuration.components[2]*60 + taskDuration.components[3]
    taskDurationMin = float(taskDurationSec/60.0)

    print(participant,"-",taskDuration)

    output_durations_df['participant'].loc[i] = participant
    output_durations_df['taskStart'].loc[i] = taskStart
    output_durations_df['taskEnd'].loc[i] = taskEnd
    output_durations_df['taskDuration'].loc[i] = taskDuration #trochu lepsi format, treba jenom sekundy (index je v ms)
    output_durations_df['taskDurationSec'].loc[i] = taskDurationSec
    output_durations_df['taskDurationMin'].loc[i] = taskDurationMin

    #output_durations_df['numberComments'].iloc[i] = #todo:...continue here


    #sessionDf = taskDf[(taskDf['time']>=taskStart) & (taskDf['time'] <= taskEnd)]
    #todo and now segment
    #https: // pandas.pydata.org / pandas - docs / stable / generated / pandas.DataFrame.rolling.html

    i=i+1


outputDurationFile = os.path.join(outputPath,"results_task1durations.csv")
output_durations_df.to_csv(path_or_buf = outputDurationFile, sep=",",index=False)

print("All good, folks!")
exit(0)
