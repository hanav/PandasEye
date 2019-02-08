import os.path
import pandas as pd
import numpy as np
import dateutil as du
import datetime
import matplotlib.path as mplPath
import xlrd as xl2
import glob
import fnmatch
from itertools import groupby
import codecs
from matplotlib import pyplot as plt

userhome = os.path.expanduser('~')
outputFolder = os.path.join(userhome, 'Dropbox', 'dizertacka','python', 'project_data','Mouse')
outputFolderArray = os.listdir(outputFolder)
fixationsFolderArray = filter (lambda x:x.endswith("CMD2.txt"), outputFolderArray)

def getPuzzleTile(df):
    x = df.eyeX
    y = df.eyeY
    tile = 0

    if((x > 0 and x < 195) & (y >827 and y < 1055)):
        tile = -1 # goal	0,827	195,827	195,1055	0,1055

    elif((x > 1105 and x < 1395) & (y > 908 and y < 1102)):
        tile = -2 # eyes	1105,908	1395,908	1395,1102	1105,1102

    elif( (x > 270 and x <522) &  (y >123 and y < 364) ):
        tile = 1 # piece00	270,123	522,123	522,364	270,364

    elif( (x > 522 and x <740) & (y >123 and y < 364) ):
        tile = 2 # piece01	522,123	740,123	740,364	522,364

    elif ((x > 740 and x < 983) & (y > 123 and y < 364)):
        tile = 3 # piece02	740,123	983,123	983,364	740,364

    elif ((x > 270 and x < 522) & (y > 364 and y < 588)):
        tile = 4 # piece10	270,364	522,364	522,587	270,587

    elif ((x > 522 and x < 740) & (y > 364 and y < 588)):
        tile = 5 # piece11	522,364	740,364	740,588	522,588

    elif ((x > 740 and x < 983) & (y > 364 and y < 588)):
        tile = 6 # piece12	740,364	983,364	983,588	740,588

    elif ((x > 270 and x < 522) & (y > 588 and y < 844)):
        tile = 7 # piece20	270,587	522,587	522,844	270,842

    elif ((x > 522 and x < 740) & (y > 588 and y < 844)):
        tile = 8 # piece21	522,588	740,588	740,844	522,844

    elif ((x > 740 and x < 983) & (y > 588 and y < 844)):
        tile = 9 # piece22	740,588	983,588	983,844	740,844

    else:
        tile = -8

    return tile


for participant_i in range(0, 7): #len(fixationsFolderArray)):
    participant_prefix = fixationsFolderArray[participant_i].split(".")[0]
    fixationFile  = os.path.join(outputFolder, fixationsFolderArray[participant_i])
    gazeDf = pd.read_csv(fixationFile, sep='\t')
    eventKey = gazeDf['Event']
    gazeDf = gazeDf.convert_objects(convert_numeric=True)


    print(participant_prefix)

    df = pd.DataFrame([])
    df['participant'] = participant_i
    df['timestamp'] = gazeDf.Timestamp
    df['eyeX'] = ( gazeDf['GazepointX (L)'] + gazeDf['GazepointX (R)']) / 2
    df.eyeX = df.eyeX.fillna(0).astype(int)
    df.eyeX = df.eyeX.astype(int)

    df['eyeY'] = (gazeDf['GazepointY (L)'] + gazeDf['GazepointY (R)']) / 2
    df.eyeY = df.eyeY.fillna(0).astype(int)
    df.eyeY = df.eyeY.astype(int)

    df['pupil'] = (gazeDf['Pupil (L)'] + gazeDf['Pupil (R)']) / 2
    df['validity'] = (gazeDf['Validity (L)'] + gazeDf['Validity (R)']) / 2
    df['event'] = 0
    df['event'].loc[eventKey == 'LMouseButton'] = 1

    coords = df.ix[:,'eyeX':'eyeY']
    tiles = coords.apply(getPuzzleTile, axis=1)
    df['location'] = tiles

    #atile = getPuzzleTile(df.eyeX[1000], df.eyeY[1000])

    dfOut = os.path.join(userhome, outputFolder,
                                ("raw_" + participant_prefix + ".csv"))
    df.to_csv(path_or_buf=dfOut, sep=",", index=False)

    # fig = plt.figure(1, dpi=90)
    # ax = fig.add_subplot(111)
    # ax.scatter(df.eyeX, df.eyeY, color="blue")
    # # plt.ylim(max(plt.ylim()), min(plt.ylim())) # flip the axis
    # figureOut = os.path.join(outputFolder, (participant_prefix + ".png"))
    # fig.savefig(figureOut)
    # plt.close(fig)

print("***********************************\n All good\n")
exit(0)