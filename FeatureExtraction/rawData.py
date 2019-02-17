# date: 06/30/18
# author: Hana Vrzakova
# description: Import of raw and event data

import numpy as np
import pandas as pd

class RawData:
    def __init__(self):
        self.ndType = [('timestamp',int), 
                       ('pupilL', np.float32),  ('validityL',int),
                       ('pupilR', np.float32), ('validityR',int),
                       ('fixationNumber', int),
                       ('gazePointX',int), ('gazePointY',int),
                       ('event','S15'), 
                       ('rawLeftX', int), ('rawLeftY', int), 
                       ('rawRightX', int), ('rawRightY', int)]
        
        self.ndTypeOutput = [('timestamp',int), 
                       ('pupilL', np.float32),  ('validityL',int),
                       ('pupilR', np.float32), ('validityR',int),
                       ('fixationNumber', int),
                       ('gazePointX',int), ('gazePointY',int),
                       ('event','S15'), 
                       ('rawX', int), ('rawY', int)]        
        
        self.data = []
        self.baseL = 0 
        self.baseR = 0
        self.varL = 0
        self.varR = 0
        self.minL = 0
        self.minR = 0

    def load(self, path):
        
        self.dataInput = np.genfromtxt(path, delimiter = '\t',
                        autostrip = True,
                        skip_header = 19,
                        usecols = (0, 7, 8, 14, 15, 16, 17, 18, 19, 2, 3, 9, 10),
                        dtype = self.ndType 
                        )
 
        # average rawX,Y
        gazeX = np.int_(np.floor((self.dataInput['rawLeftX'] + self.dataInput['rawRightX'] )/2))
        gazeY = np.int_(np.floor((self.dataInput['rawLeftY'] + self.dataInput['rawRightY'] )/2))
        
        # takhle se inicializuje
        self.data = np.ndarray(shape=(len(gazeX)), dtype=self.ndTypeOutput)
        
        self.data['timestamp'] = self.dataInput['timestamp']
        self.data['pupilL'] = self.dataInput['pupilL']
        self.data['validityL'] = self.dataInput['validityL']
        self.data['pupilR'] = self.dataInput['pupilR']
        self.data['validityR'] = self.dataInput['validityR']
        self.data['fixationNumber'] = self.dataInput['fixationNumber']
        self.data['gazePointX'] = self.dataInput['gazePointX']
        self.data['gazePointY'] = self.dataInput['gazePointY']
        self.data['event'] = self.dataInput['event']
        self.data['rawX'] = gazeX
        self.data['rawY'] = gazeY
        
        self.deleteUnvalidData()

    def loadDataFrame8P(self, inputPath):
        self.dataInput = pd.read_csv(inputPath, skiprows=18, sep="\t")

        #sequence = self.cmdData.loc[(self.cmdData['fixationNumber'] >= start) & (self.cmdData['fixationNumber'] <= end)]
        eventIdx = self.dataInput['GazepointX (L)'] == " "
        self.eventDf = self.dataInput.loc[eventIdx]
        df = self.dataInput.loc[~eventIdx]


        gazeX = (pd.to_numeric(df['GazepointX (L)']) + pd.to_numeric(df['GazepointX (R)']))/2
        gazeY = (pd.to_numeric(df['GazepointY (L)']) + pd.to_numeric(df['GazepointY (R)'])) / 2
        gazeX.astype(int)
        gazeY.astype(int)

        self.data = pd.DataFrame()

        self.data['timestamp'] = df['Timestamp']
        self.data['validityL'] = df['Validity (L)']
        self.data['validityR'] = df['Validity (R)']
        self.data['rawX'] = gazeX
        self.data['rawY'] = gazeY

        self.data['pupilL'] = pd.to_numeric(df['Pupil (L)'], errors='coerce')
        self.data['pupilR'] = pd.to_numeric(df['Pupil (R)'], errors='coerce')
        self.data['pupil'] = (self.data['pupilL'] + self.data['pupilR']) / 2.0

        self.data['fixationNumber'] = pd.to_numeric(df['Fixation'], errors='coerce')
        self.data['gazePointX'] = pd.to_numeric(df['GazepointX'], errors='coerce')
        self.data['gazePointY'] = pd.to_numeric(df['GazepointY'], errors='coerce')

        self.data = self.deleteInvalidData8P(self.data)

    def loadDataFrame8PEvents(self, inputPath):
        self.dataInput = pd.read_csv(inputPath, skiprows=18, sep="\t")

        # sequence = self.cmdData.loc[(self.cmdData['fixationNumber'] >= start) & (self.cmdData['fixationNumber'] <= end)]

        eventIdx = self.dataInput['GazepointX (L)'] == " "
        self.eventDf = self.dataInput.loc[eventIdx]
        df = self.dataInput.loc[~eventIdx]

        gazeX = (pd.to_numeric(df['GazepointX (L)']) + pd.to_numeric(df['GazepointX (R)'])) / 2
        gazeY = (pd.to_numeric(df['GazepointY (L)']) + pd.to_numeric(df['GazepointY (R)'])) / 2
        gazeX.astype(int)
        gazeY.astype(int)

        self.data = pd.DataFrame()

        self.data['timestamp'] = df['Timestamp']
        self.data['validityL'] = df['Validity (L)']
        self.data['validityR'] = df['Validity (R)']
        self.data['rawX'] = gazeX
        self.data['rawY'] = gazeY

        self.data['pupilL'] = pd.to_numeric(df['Pupil (L)'], errors='coerce')
        self.data['pupilR'] = pd.to_numeric(df['Pupil (R)'], errors='coerce')
        self.data['pupil'] = (self.data['pupilL'] + self.data['pupilR']) / 2.0

        self.data['fixationNumber'] = pd.to_numeric(df['Fixation'], errors='coerce')
        self.data['gazePointX'] = pd.to_numeric(df['GazepointX'], errors='coerce')
        self.data['gazePointY'] = pd.to_numeric(df['GazepointY'], errors='coerce')

        self.data = self.deleteInvalidData8P(self.data)

        self.findNearestFixationBeforeEvent()

    def loadDataFrameWTP(self,inputPath):
        self.dataInput = pd.read_csv(inputPath, sep="\t")

        eventIdx = self.dataInput['GazepointX (L)'] == " "
        self.eventDf = self.dataInput.loc[eventIdx]
        df = self.dataInput.loc[~eventIdx]

        gazeX = (pd.to_numeric(df['GazepointX (L)']) + pd.to_numeric(df['GazepointX (R)'])) / 2
        gazeY = (pd.to_numeric(df['GazepointY (L)']) + pd.to_numeric(df['GazepointY (R)'])) / 2
        gazeX.astype(int)
        gazeY.astype(int)

        self.data = pd.DataFrame()

        self.data['timestamp'] = df['Timestamp']
        self.data['validityL'] = df['Validity (L)']
        self.data['validityR'] = df['Validity (R)']
        self.data['rawX'] = gazeX
        self.data['rawY'] = gazeY

        self.data['pupilL'] = pd.to_numeric(df['Pupil (L)'], errors='coerce')
        self.data['pupilR'] = pd.to_numeric(df['Pupil (R)'], errors='coerce')
        self.data['pupil'] = (self.data['pupilL'] + self.data['pupilR']) / 2.0

        self.data['fixationNumber'] = pd.to_numeric(df['Fixation'], errors='coerce')
        self.data['gazePointX'] = pd.to_numeric(df['GazepointX'], errors='coerce')
        self.data['gazePointY'] = pd.to_numeric(df['GazepointY'], errors='coerce')

        self.data = self.deleteInvalidData8P(self.data)

    def findNearestFixationBeforeEvent(self):
        i = 0

        self.eventDf['eventFixation'] = -1
        for idx in self.eventDf.index:
            fixations = self.data['fixationNumber'].loc[i:idx].unique()
            try:
                eventFixation = fixations[-1]
                self.eventDf['eventFixation'].loc[idx] = eventFixation
                i = idx
            except:
                continue

        self.eventDf = self.eventDf.loc[self.eventDf['eventFixation'].notnull()]

    def deleteInvalidData8P(self,df):
        df = df.drop( df[  (df['validityL'] == 4 ) | (df['validityR'] == 4) ].index )
        return df

    def deleteUnvalidData(self):
        # timestamp == -1
        indeces = np.where(self.data['timestamp'] == -1)
        self.data = np.delete(self.data, indeces[0])

        # validity == 4
        indecesL = np.where(self.data['validityL'] == 4 )
        self.data = np.delete(self.data,indecesL[0])

        indecesR = np.where(self.data['validityR'] == 4)
        self.data = np.delete(self.data,indecesR[0])

    def separateFixations(self):
        indeces = np.where(self.data['fixationNumber'] != -1)
        output = self.data[indeces[0]]
        return output

    def separateFixations8P(self):
        output = self.data.loc[self.data['fixationNumber']!=-1]
        return output


    def separateEvents(self):
        indeces = np.where(self.data['event'] == 'LMouseButton')
        output = self.data[indeces[0]]
        delIndex = np.where(output['timestamp'] <= (5*60*1000)) #umazavame prvnich 5minut, protoze treba trial, nevalidni data
        output = np.delete(output,delIndex[0])
        return output

    def separateEvents8P(self):
        return self.eventDf.loc[self.eventDf['Event']=='LMouseButton']

    def eraseEvents(self):
        index = np.where(self.data['event'] == 'LMouseButton')
        self.data = np.delete(self.data, index[0])

    def countBaseline(self):
        self.baseL = np.mean(self.data['pupilL'])
        self.baseR = np.mean(self.data['pupilR'])
        self.varL = np.var(self.data['pupilL'])
        self.varR = np.var(self.data['pupilR'])      

    def substractBaseline(self):
        self.data['pupilL'] = (self.data['pupilL'] - self.baseL) / self.varL
        self.data['pupilR'] = (self.data['pupilR'] - self.baseR) / self.varR

    def countPCPS(self):
        self.data['pupilL'] = (self.data['pupilL'] - self.baseL) / self.varL
        self.data['pupilR'] = (self.data['pupilR'] - self.baseR) / self.varR
        # pupil changes
        self.data['pupilL'] = (self.data['pupilL'] / self.baseL)
        self.data['pupilR'] = (self.data['pupilR'] / self.baseR)

    def normalizePupils(self):
        self.minL = np.abs(self.data['pupilL'].min())
        self.minR = np.abs(self.data['pupilR'].min())

        self.data['pupilL'] = self.data['pupilL'] + self.minL
        self.data['pupilR'] = self.data['pupilR'] + self.minR

        self.data['pupilL'] = self.data['pupilL'] / np.abs(self.data['pupilL'].max())
        self.data['pupilR'] = self.data['pupilR'] / np.abs(self.data['pupilR'].max())

    def returnBaseVar(self):
        base = (self.baseL + self.baseR)/2
        var = (self.varL + self.varR)/2
        output = np.append(base,var)
        return output
