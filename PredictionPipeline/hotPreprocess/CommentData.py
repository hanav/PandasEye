import os.path
import pandas as pd
import numpy as np
import dateutil as du

class CommentData:
    def __init__(self):
        self.participant = pd.DataFrame(data=None)
        self.task = pd.DataFrame(data=None)
        self.timeofday = pd.DataFrame(data=None)
        self.hasEmotion = pd.DataFrame(data=None)
        self.data = pd.DataFrame(data=None)

    def SetDataFrame(self,df):
        self.data = pd.DataFrame(data=df)
        self.data = self.data.reset_index()
        self.data['timeofday'] = self.ConvertTimeofday(self.data['timeofday'])

    def AppendData(self,filename):
        data = pd.read_csv(filename, sep=";", skiprows=0)
        self.participant = data.Participant
        self.task = data.Task
        self.timeofday = data.Created
        self.hasEmotion = data.hasEmotions

        self.participant = self.participant.rename('participant')
        self.task = self.task.rename('task')
        self.timeofday = self.timeofday.rename('timeofday')
        self.hasEmotion = self.hasEmotion.rename('hasEmotion')

        print "CommentData: Number of the rows processed: ", len(self.timeofday)

    def LoadData(self, filename):
        data = pd.read_csv(filename, sep=";")
        self.participant = data.participant
        self.task = data.task
        self.timeofday = data.timeofday
        self.hasEmotion = data.hasEmotion

    def LoadDataStartEnd(self, filename):
        self.data = pd.read_csv(filename, sep=";")
        print "Comment start-end loaded: ", len(self.data.start)

    def GetDataFrame(self,idx):
        df = pd.DataFrame()
        df['participant'] = self.participant[idx]
        df['task'] = self.task[idx]
        df['timeofday'] = self.timeofday[idx]
        df['hasEmotion'] = self.hasEmotion[idx]
        return df

    def GetDataFrame2(self,idx):
        return self.data[idx]

    def SaveCSV(self):
        data = pd.concat([
                pd.DataFrame(data=self.participant),
                pd.DataFrame(data=self.task),
                pd.DataFrame(data=self.timeofday),
                pd.DataFrame(data=self.hasEmotion)], axis=1)

        userhome = os.path.expanduser('~')
        filename = os.path.join(userhome, 'Dropbox', 'HotOrNot', 'r_icmi', 'out_comments.csv')
        data.to_csv(filename, index=False, header=True)

    def ConvertTimeofday(self, dateArray):
        pinfo = du.parser.parserinfo(dayfirst=False, yearfirst=False) # example: 8/7/2015 11:42:23

        outputDates = list()

        for dateItem in dateArray:
            converted = du.parser.parse(dateItem, parserinfo=pinfo)
            outputDates.append(converted)

        convertedDates = pd.DataFrame(data=outputDates)
        return convertedDates

