import os.path
import pandas as pd
import numpy as np
from dateutil.parser import parse

class TMData:
    def __init__(self):
        self.timestamp = pd.DataFrame(data=None)
        self.participant = pd.DataFrame(data=None)
        self.task = pd.DataFrame(data=None)
        self.timeofday = pd.DataFrame(data=None)
        self.tmsum = pd.DataFrame(data=None)
        self.fingercount = pd.DataFrame(data=None)
        self.data = pd.DataFrame(data=None)

    def SetDataFrame(self, df):
        self.data = pd.DataFrame(data=df)
        self.data = self.data.reset_index() #drop=True to not to include extra index column
        self.data['timeofday'] = self.ConvertTimeofday(self.data['timeofday'])
        self.data['block'] = (self.data.tmsum.shift(1) != self.data.tmsum).astype(int).cumsum()

    def AppendData(self,filename):
        data = pd.read_csv(filename, skiprows=0)
        self.timestamp = data.time2
        self.participant = data.participant
        self.task = data.task
        self.timeofday = data.timeofday
        self.tmsum = data.touchmousesum
        self.fingercount = data.fingercount

        self.timestamp = self.timestamp.rename('timestamp')
        self.tmsum = self.tmsum.rename('tmsum')

        print "EyeData: Number of the rows processed: ", len(self.timestamp)

    def LoadData(self, filename):
        #data = pd.read_csv(filename, parse_dates=['timeofday'], nrows=50000)
        data = pd.read_csv(filename)
        self.timestamp = data.timestamp
        self.participant = data.participant
        self.task = data.task
        self.timeofday = data.timeofday
        self.tmsum = data.tmsum
        self.fingercount = data.fingercount

        print "TM data loaded: ", len(self.timeofday)

    def LoadDataFrame(self, filename):
        self.data = pd.read_csv(filename)
        print "TM data frame loaded"

    def SaveCSV(self): #cbind
        data = pd.concat([
            pd.DataFrame(data=self.timestamp),
            pd.DataFrame(data=self.participant),
            pd.DataFrame(data=self.task),
            pd.DataFrame(data=self.timeofday),
            pd.DataFrame(data=self.tmsum),
            pd.DataFrame(data=self.fingercount)], axis=1)

        userhome = os.path.expanduser('~')
        filename = os.path.join(userhome, 'Dropbox', 'HotOrNot', 'r_icmi', 'out_tm.csv')
        data.to_csv(filename, index=False, header=True)

    def GetDataFrame(self, idx):
        df = pd.DataFrame(data=None)
        df['timestamp'] = self.timestamp[idx]
        df['participant'] = self.participant[idx]
        df['task'] = self.task[idx]
        df['timeofday'] = self.timeofday[idx]
        df['tmsum'] = self.tmsum[idx]
        df['fingercount'] = self.fingercount[idx]
        return df

    def GetDataFrame2(self, idx):
        return self.data[idx]

    def ConvertTimeofday(self, dateArray):
        outputDates = list()

        for dateItem in dateArray:
            converted = parse(dateItem)
            outputDates.append(converted)

        convertedDates = pd.DataFrame(data=outputDates)
        return convertedDates
