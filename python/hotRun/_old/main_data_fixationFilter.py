import os.path
import pandas as pd
import numpy as np
from hotPreprocess.AllData import AllData

userhome = os.path.expanduser('~')

commentStarts = os.path.join(userhome, 'Dropbox', 'HotOrNot','r_icmi', 'commentsStartsEnds_3.csv')
eyeresampled = os.path.join(userhome, 'Dropbox', 'HotOrNot','r_icmi', 'et_resampled.csv')
allData = AllData()

#eyelineout = os.path.join(userhome, 'Dropbox', 'HotOrNot','r_icmi', 'out_eye.csv')
allData.LoadCommentsStartEnd(commentStarts)
allData.LoadEyeLineData(eyeresampled) #todo: add timeoftheday
allParticipants = np.intersect1d(np.unique(allData.eyeData.data.participant), np.unique(allData.commentData.data.participant))

allParticipants = np.array(['P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18',
                            'P19', 'P20', 'P23', 'P24', 'P25', 'P28',  # debug 'P27', 'P36'
                            'P29', 'P30', 'P31', 'P32', 'P33', 'P34', 'P35', 'P37',
                            'P39', 'P40', 'P41', 'P6', 'P7', 'P8'])  # P21 and P22 excluded

for i in range(len(allParticipants) - 1):
    print("Participant", i, "-", allParticipants[i])
    eyeDF = allData.GetETDataFrame(1, allParticipants[i])

    tmp = eyeDF[~((eyeDF.gazex == 0) & (eyeDF.gazey == 0))]  # remove 0,0 coordinates
    # plt.plot(tmp.gazex, tmp.gazey)
    # plt.plot(tmp.mousex, tmp.mousey)
    # plt.hist(distances, 50, normed=1, facecolor='green', alpha=0.75)

    coord = pd.DataFrame(data=[tmp.gazex, tmp.gazey])
    diffs = np.diff(coord)
    distances = np.sqrt((diffs ** 2).sum(axis=0))
    distances = pd.Series(data=distances)
    distances[distances.count()] = 0 # extra for the last sample

    diff_distances = (distances.shift() - distances).abs()
    diff_distances = pd.DataFrame(data=diff_distances, columns=["diff_dist"])
    diff_distances.loc[0] = 0

    #diff_distances = diff_distances[~diff_distances.isnull()]
    #dd = pd.DataFrame(data=diff_distances)
    diff_distances['minifixation'] = diff_distances.diff_dist < 30
    diff_distances['biggerfixation'] = diff_distances.diff_dist < 50

    diff_distances['id'] = (diff_distances['minifixation'].shift(1) != diff_distances['minifixation']).astype(int).cumsum()

    fixation_idx  = (diff_distances.id[diff_distances['minifixation'] == True]).unique()

    # merge it with other values, compute centroids, compute fixation duration, merge it...eh, on correct indeces


print("***********************************\n All good\n")
exit(0)