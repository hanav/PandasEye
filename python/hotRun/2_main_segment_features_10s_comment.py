# ---------------------------------------------------------------------------------------------
# Segment before and after comments and feature extract
# ---------------------------------------------------------------------------------------------

import os.path
import pandas as pd
import numpy as np
import dateutil as du
from datetime import timedelta
from hotPreprocess.FeaturesEye import FeaturesEye


def ConvertTimeofday(dateItem):
    # pinfo = du.parser.parserinfo(dayfirst = True, yearfirst=False) # example: 07/08/15 11:42
    pinfo = du.parser.parserinfo(dayfirst=False, yearfirst=False)  # example: 8/7/2015 11:42:23
    convertedDate = du.parser.parse(dateItem, parserinfo=pinfo)
    return convertedDate


userhome = os.path.expanduser('~')
OUTPUT = '/Users/icce/Dropbox (Personal)/_thesis_framework/_scripts_hoy/r_icmi/'

# Input data
eyePath = os.path.join('/Users/icce/Dropbox (Personal)/_thesis_framework/_scripts_hoy/r_icmi/resampled_eye.csv')
gsrPath = os.path.join('/Users/icce/Dropbox (Personal)/_thesis_framework/_scripts_hoy/r_icmi/resampled_gsr.csv')
tmPath = os.path.join('/Users/icce/Dropbox (Personal)/_thesis_framework/_scripts_hoy/r_icmi/resampled_tm.csv')

# meta data about comments
cmtPath = os.path.join('/Users/icce/Dropbox (Personal)/_thesis_framework/_scripts_hoy/r_icmi/CodeReviewCommentEmotionsWithTimestampsInMS.xlsx')

# segment 10s before and 10s after the comment
OFFSET = 10


print("Load ET data")
eyeData = pd.read_csv(eyePath)
cmtData = pd.read_excel(cmtPath)

dfColumns = ['participant', 'task', 'commentID','emotion_binary']

dfOutput_beforeComment = pd.DataFrame()

# dfColumns = ['participant', 'task', 'commentID','emotion_binary']
# dfColumns.extend(range(1,9))
# all_outputDF = pd.DataFrame(index=np.arange(0, len(cmtData.CommentID)),
#                                         columns=dfColumns )

i=0
for comment in cmtData.CommentID.unique():
    print("Comment ID: ", comment)

    cmtDf = cmtData[cmtData.CommentID == comment]
    taskDf = eyeData[(eyeData.participant == cmtDf.Participant.iloc[0]) & (eyeData.task == cmtDf.Task.iloc[0])]
    taskDf['time'] = pd.DatetimeIndex(taskDf.timeofday)

    cmtStart = cmtDf['Comment Started (PDT)'].iloc[0]
    cmtEnd = cmtDf['Comment Created (PDT)'].iloc[0]

    preDf = taskDf[ (taskDf.time >=  (cmtStart -  timedelta(seconds=OFFSET))) &  (taskDf.time <= cmtStart)]
    # postDf = taskDf[ (taskDf.time >=  (cmtEnd)) &  (taskDf.time <= (cmtEnd + timedelta(seconds=OFFSET)))]

    # Count gaze-based features
    eyeFeatures = FeaturesEye()
    preRow = eyeFeatures.CountFeatures(preDf)
    # postRow = eyeFeatures.CountFeatures(postDf)
    # allRow = preRow + postRow

    commentMeta = pd.DataFrame([cmtDf.Participant.iloc[0],cmtDf.Task.iloc[0],comment,cmtDf.Emotion_binary.iloc[0]]).transpose()
    commentMeta.columns =['participant', 'task', 'commentID','emotion_binary']

    row = pd.concat([commentMeta,preRow], axis=1).reset_index(drop=True)
    dfOutput_beforeComment = dfOutput_beforeComment.append(row)

    # pre_outputDF.participant.loc[i]  = cmtDf.Participant.iloc[0]
    # pre_outputDF.task.loc[i]  = cmtDf.Task.iloc[0]
    # pre_outputDF.commentID.loc[i]  = comment
    # pre_outputDF.emotion_binary.loc[i]  = cmtDf.Emotion_binary.iloc[0]

    # featureHeader = preRow.columns
    # preRow.columns = range(4,preRow.shape[1]+4)

    # pre_outputDF.loc[i, 4::] = preRow
    # post_outputDF.loc[i, 4::] = postRow
    # all_outputDF.loc[i, 4::] = allRow

    i = i +1

pre_outputFilename = os.path.join(OUTPUT, 'features_shortterm_eye_10s_pre.csv')
# post_outputFilename = os.path.join(userhome, outputPath, 'features_shortterm_eye_10s_post.csv')
# all_outputFilename = os.path.join(userhome, outputPath, 'features_shortterm_eye_10s_all.csv')


dfOutput_beforeComment.to_csv(path_or_buf = pre_outputFilename, sep=",",  index=False)
# post_outputDF.to_csv(path_or_buf = post_outputFilename, sep=",",  index=False)
# all_outputDF.to_csv(path_or_buf = all_outputFilename, sep=",",  index=False)

print("All good, folks!")
exit(0)