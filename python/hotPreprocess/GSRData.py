import os.path
import pandas as pd
import numpy as np
from dateutil.parser import parse


class GSRData:
    def __init__(self):
        self.timestamp = pd.DataFrame(data=None)
        self.participant = pd.DataFrame(data=None)
        self.task = pd.DataFrame(data=None)
        self.timeofday = pd.DataFrame(data=None)
        self.mediangsr = pd.DataFrame(data=None)
        self.data = pd.DataFrame(data=None)

    def AppendData(self, filename):
        data = pd.read_csv(filename, skiprows=0)
        self.timestamp = data.time2
        self.participant = data.participant
        self.task = data.task
        self.timeofday = data.timeofday
        self.mediangsr = data.mediangsr

        self.timestamp = self.timestamp.rename('timestamp')

    def LoadData(self, filename):
        data = pd.read_csv(filename)

        self.timestamp = data.timestamp
        self.participant = data.participant
        self.task = data.task
        self.timeofday = data.timeofday
        self.mediangsr = data.mediangsr

        print "GSR data loaded: ", len(self.timeofday)


    def SaveCSV(self):
        data = pd.concat([
            pd.DataFrame(data=self.timestamp),
            pd.DataFrame(data=self.participant),
            pd.DataFrame(data=self.task),
            pd.DataFrame(data=self.timeofday),
            pd.DataFrame(data=self.mediangsr)], axis=1)

        userhome = os.path.expanduser('~')
        filename = os.path.join(userhome, 'Dropbox', 'HotOrNot', 'r_icmi', 'out_gsr.csv')
        data.to_csv(filename, index=False, header=True)

    def GetDataFrame(self,idx):
        df = pd.DataFrame(data=None)
        df['timestamp'] = self.timestamp[idx]
        df['participant'] = self.participant[idx]
        df['task'] = self.task[idx]
        df['timeofday'] = self.timeofday[idx]
        df['mediangsr'] = self.mediangsr[idx]
        return df

    def ConvertTimeofday(self, dateArray):
        outputDates = list()

        for dateItem in dateArray:
            converted = parse(dateItem)
            outputDates.append(converted)

        convertedDates = pd.DataFrame(data=outputDates)
        return convertedDates