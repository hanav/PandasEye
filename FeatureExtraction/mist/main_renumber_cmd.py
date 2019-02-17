import os.path
import pandas as pd
import numpy as np

def extractParticipantPrefix(x):
    return x.split('.')[0]

def mergePath(fileDir,x):
    return os.path.join(fileDir, x)

def findTileAoi(df):
    #print("coords: ", df.x, df.y)
    tileNumber = 'NAN'

    if(pd.isnull(df.x)==False):
        if (df.x >= aoiDF.x1[0] and df.x <= aoiDF.x3[0] and
                    df.y >= aoiDF.y1[0] and df.y <= aoiDF.y3[0]) :
            tileNumber = -1
        elif (df.x >= aoiDF.x1[1] and df.x <= aoiDF.x3[1] and
                      df.y >= aoiDF.y1[1] and df.y <= aoiDF.y3[1]) :
            tileNumber = -2
        elif (df.x >= aoiDF.x1[2] and df.x <= aoiDF.x3[2] and
                      df.y >= aoiDF.y1[2] and df.y <= aoiDF.y3[2]) :
            tileNumber = 1
        elif (df.x >= aoiDF.x1[3] and df.x <= aoiDF.x3[3] and
                      df.y >= aoiDF.y1[3] and df.y <= aoiDF.y3[3]) :
            tileNumber = 2
        elif (df.x >= aoiDF.x1[4] and df.x <= aoiDF.x3[4] and
                      df.y >= aoiDF.y1[4] and df.y <= aoiDF.y3[4]) :
            tileNumber = 3
        elif (df.x >= aoiDF.x1[5] and df.x <= aoiDF.x3[5] and
                      df.y >= aoiDF.y1[5] and df.y <= aoiDF.y3[5]) :
            tileNumber = 4
        elif (df.x >= aoiDF.x1[6] and df.x <= aoiDF.x3[6] and
                      df.y >= aoiDF.y1[6] and df.y <= aoiDF.y3[6]) :
            tileNumber = 5
        elif (df.x >= aoiDF.x1[7] and df.x <= aoiDF.x3[7] and
                      df.y >= aoiDF.y1[7] and df.y <= aoiDF.y3[7]) :
            tileNumber = 6
        elif (df.x >= aoiDF.x1[8] and df.x <= aoiDF.x3[8] and
                      df.y >= aoiDF.y1[8] and df.y <= aoiDF.y3[8]) :
            tileNumber = 7
        elif (df.x >= aoiDF.x1[9] and df.x <= aoiDF.x3[9] and
                      df.y >= aoiDF.y1[9] and df.y <= aoiDF.y3[9]) :
            tileNumber = 8 # goal area
        elif (df.x >= aoiDF.x1[10] and df.x <= aoiDF.x3[10] and
                      df.y >= aoiDF.y1[10] and df.y <= aoiDF.y3[10]) :
            tileNumber = 9 # life view area
        else:
            tileNumber = -8

    return tileNumber


userhome = os.path.expanduser('~')

aoiFile = "/Users/icce/Dropbox (Personal)/_thesis_framework/_dataset_8Puzzles/properly_anonymized_data_GazeAugmented/AOI_codes.csv"
#aoiDF = pd.read_csv(aoiFile, skiprows=0,sep="\t")

aoiDF = pd.read_csv(aoiFile, skiprows=0,sep="\t")

outputPath = os.path.join(userhome, '/Users/icce/Dropbox/dizertacka/python/python_8P_annotated/properly_anonymized_data_GazeAugmented')

fileArray  = os.listdir(outputPath)
folderArray = filter (lambda x:x.endswith("txt") , fileArray)

for i in range(0,(len(folderArray))):

#for i in range(0,1):
    gazefile = folderArray[i]
    gazeFilePath= mergePath(outputPath,gazefile)
    gazeDF = pd.read_csv(gazeFilePath, skiprows=18, sep="\t")

    userID = extractParticipantPrefix(gazefile)
    print("Working on: ", userID)

    all = pd.concat([gazeDF.GazepointX, gazeDF.GazepointY], axis=1)
    all.columns = ['x', 'y']
    all.x = pd.to_numeric(all.x, errors='coerce') #errors=ignore will return mixed array
    all.y = pd.to_numeric(all.y, errors='coerce')
    print("Extracting coordinates...")
    tileNumber = all.apply(findTileAoi, axis=1)
    print("Done!")

    tileDF = pd.DataFrame([tileNumber])
    tileDF = tileDF.transpose()
    tileDF.columns = ["AOI"]
    tileDF = tileDF.replace(['NAN'], '')
    outputDF = pd.concat([gazeDF,tileDF],axis=1)

    outputFile = userID+'_aoi.csv'
    outDir = mergePath(outputPath,outputFile)
    outputDF.to_csv(path_or_buf = outDir, sep=",",  index=False)
    print("saved")

print("All good, folks!")
exit(0)