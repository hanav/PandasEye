import os.path
import numpy as np
import pandas as pd
import dateutil as du
import datetime

class EyeData:
    def __init__(self):
        self.timestamp = pd.DataFrame(data=None)
        self.participant = pd.DataFrame(data=None)
        self.task = pd.DataFrame(data=None)
        self.timeofday = pd.DataFrame(data=None)
        self.validity = pd.DataFrame(data=None)
        self.gazex = pd.DataFrame(data=None)
        self.gazey =pd.DataFrame(data=None)
        self.mousex = pd.DataFrame(data=None)
        self.mousey = pd.DataFrame(data=None)
        self.linenum = pd.DataFrame(data=None)
        self.data = pd.DataFrame(data=None)

    def AppendData(self,filename):
        data = pd.read_csv(filename, skiprows=0)
        self.timestamp = data.time2
        self.participant = data.participant
        self.task = data.task
        self.timeofday = data.timeofday
        self.validity = data.areeyesopen
        self.gazex = data.eyegazescreenx
        self.gazey = data.eyegazescreeny
        self.linenum = data.linenum
        self.mousex = data.mousex
        self.mousey = data.mousey

        self.timestamp = self.timestamp.rename('timestamp')
        self.participant = self.participant.rename('participant')
        self.task = self.task.rename('task')
        self.timeofday = self.timeofday.rename('timeofday')
        self.validity = self.validity.rename('validity')
        self.gazex = self.gazex.rename('gazex')
        self.gazey = self.gazey.rename('gazey')
        self.linenum = self.linenum.rename('linenum')
        self.mousex = self.mousex.rename('mousex')
        self.mousey = self.mousey.rename('mousey')

        print "EyeData: Number of the rows processed: ", len(self.timestamp)

    def CutLineData(self,filename):
        data = pd.read_csv(filename, skiprows=0)
        self.participant = data.participant
        self.task = data.task
        self.timestamp = data.time
        self.timeofday = data.timeofday
        self.validity = data.areeyesopen
        self.filepath = data.filepath
        self.linenum = data.linenum
        self.gazex = data.eyegazescreenx
        self.gazey = data.eyegazescreeny
        self.mousex = data.mousex
        self.mousey = data.mousey
        self.scrollx = data.horizontalscrollposition
        self.scrolly = data.verticalscrollposition
        self.dirx = data.eyedirectionx
        self.diry = data.eyedirectiony

        self.timestamp = self.timestamp.rename('timestamp')
        self.participant = self.participant.rename('participant')
        self.task = self.task.rename('task')
        self.timeofday = self.timeofday.rename('timeofday')
        self.validity = self.validity.rename('validity')
        self.filepath = self.filepath.rename('filepath')
        self.linenum = self.linenum.rename('linenum')
        self.gazex = self.gazex.rename('gazex')
        self.gazey = self.gazey.rename('gazey')
        self.mousex = self.mousex.rename('mousex')
        self.mousey = self.mousey.rename('mousey')
        self.scrollx = self.scrollx.rename('scrollx')
        self.scrolly = self.scrolly.rename('scrolly')
        self.dirx = self.dirx.rename('dirx')
        self.diry = self.diry.rename('diry')

        print("Line ET data appended: ", len(self.linenum))

    def LoadData(self,filename):
        data = pd.read_csv(filename)

        self.timestamp = data.timestamp
        self.participant = data.participant
        self.task = data.task
        self.timeofday = data.timeofday
        self.validity = data.validity
        self.gazex = data.gazex
        self.gazey = data.gazey
        self.mousex = data.mousex
        self.mousey = data.mousey

        print "Eye Data loaded:", len(self.timestamp)

    def LoadLineData(self, filename):
        self.data = pd.read_csv(filename)
        print "ET line data loaded"

     #def ResampleEyeData(self):
        self.data

        allParticipants = np.array(['P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18',
                                    'P19', 'P20', 'P23', 'P24', 'P25', 'P28',  # debug 'P27', 'P36'
                                    'P29', 'P30', 'P31', 'P32', 'P33', 'P34', 'P35', 'P37',
                                    'P39', 'P40', 'P41', 'P6', 'P7', 'P8'])  # P21 and P22 excluded

        outputDF = pd.DataFrame(data=None)
        for i in range(len(allParticipants) - 1):
            print(i)
            idx = (self.data.participant == allParticipants[i]) & (self.data.task == 1)
            segment = self.data.loc[idx]
            segment

            gaze_x = self.resampleSerie(segment.gazex, segment.timeofday,'20L')
            gaze_y = self.resampleSerie(segment.gazey, segment.timeofday,'20L')
            mouse_x = self.resampleSerie(segment.mousex, segment.timeofday,'20L')
            mouse_y = self.resampleSerie(segment.mousey, segment.timeofday,'20L')

            df = pd.concat([gaze_x, gaze_y, mouse_x, mouse_y], axis=1) #todo: add new timestamps
            df['timeofday'] = gaze_x.index
            df['participant'] = allParticipants[i]
            df['task'] = 1
            outputDF = outputDF.append(pd.DataFrame(data=df), ignore_index=True)

        userhome = os.path.expanduser('~')
        featureFilename = os.path.join(userhome, 'Dropbox', 'HotOrNot', 'r_icmi',
                                           ('et_resampled.csv'))
        outputDF.to_csv(path_or_buf=featureFilename, sep=",", index=False)
        outputDF

        # downsample to 50Hz signal
        # series_x = pd.Series(data=self.data.gazex)
        # series_y = pd.Series(data=self.data.gazey)
        # series_x.index = pd.DatetimeIndex(self.data.timeofday)
        # series_y.index = pd.DatetimeIndex(self.data.timeofday)
        # series_x = series_x.resample('20L', how='mean', closed='right', label='left')  # 20ms should correspond to 50Hz
        # series_y = series_y.fillna(method='backfill')




    def resampleSerie(self, serieData, timeofday, newFreq):
        newSerie = pd.Series(data = serieData)
        newSerie.index = pd.DatetimeIndex(timeofday)
        newSerie = newSerie.resample(newFreq, how='mean', closed='right', label='left')  # 20ms should correspond to 50Hz
        newSerie = newSerie.fillna(method='backfill')
        return newSerie


    def SaveCSV(self):

        data = pd.concat([
            pd.DataFrame(data=self.timestamp),
            pd.DataFrame(data=self.participant) ,
            pd.DataFrame(data=self.task),
            pd.DataFrame(data=self.timeofday),
            pd.DataFrame(data=self.validity),
            pd.DataFrame(data=self.gazex),
            pd.DataFrame(data=self.gazey),
            pd.DataFrame(data=self.linenum),
            pd.DataFrame(data= self.mousex),
            pd.DataFrame(data=self.mousey)], axis=1)

        userhome = os.path.expanduser('~')
        filename = os.path.join(userhome, 'Dropbox', 'HotOrNot','r_icmi', 'out_eye.csv')
        data.to_csv(filename, index=False, header=True)

    def SaveLineCSV(self):
        data = pd.concat([
        self.timestamp ,
        self.participant ,
        self.task ,
        self.timeofday,
        self.validity,
        self.filepath,
        self.linenum,
        self.gazex,
        self.gazey,
        self.dirx,
        self.diry,
        self.mousex,
        self.mousey,
        self.scrollx,
        self.scrolly], axis=1)

        userhome = os.path.expanduser('~')
        filename = os.path.join(userhome, 'Dropbox', 'HotOrNot', 'r_icmi', 'out_line_eye.csv')
        data.to_csv(filename, index=False, header=True)

    def SetDataFrame(self, df):
        self.data=df

    def GetDataFrame(self, idx):
        df = pd.DataFrame(data=None)
        df['timestamp'] = self.timestamp[idx]
        df['participant'] = self.participant[idx]
        df['task'] = self.task[idx]
        df['timeofday'] = self.timeofday[idx]
        df['validity'] = self.validity[idx]
        df['gazex'] = self.gazex[idx]
        df['gazey'] = self.gazey[idx]
        df['mousex'] = self.mousex[idx]
        df['mousey'] = self.mousey[idx]
        return df

    def ConvertTimeofday(self, dateArray):
        # pinfo = du.parser.parserinfo(dayfirst = True, yearfirst=False) # example: 07/08/15 11:42
        pinfo = du.parser.parserinfo(dayfirst=False, yearfirst=True)  # example: 8/7/2015 11:42:23

        outputDates = list()

        for dateItem in dateArray:
            converted = du.parser.parse(dateItem, parserinfo=pinfo)
            outputDates.append(converted)

        convertedDates = pd.DataFrame(data=outputDates)
        return convertedDates