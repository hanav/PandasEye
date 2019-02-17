from EyeData import EyeData
from GSRData import GSRData
from TMData import TMData
from CommentData import CommentData


class AllData:
    def __init__(self):
        self.eyeData = EyeData()
        self.gsrData = GSRData()
        self.tmData = TMData()
        self.commentData = CommentData()
        self.participants = ()

    # ==========================================#
    def readEyeData(self, filename):
        self.eyeData.AppendData(filename)

    def readEyeLineData(self,filename):
        self.eyeData.CutLineData(filename)

    def saveEyeData(self):
        self.eyeData.SaveCSV()

    def saveEyeLineData(self):
        self.eyeData.SaveLineCSV()

#==========================================#
    def readGSRData(self, filename):
        self.gsrData.AppendData(filename)

    def saveGSRData(self):
            self.gsrData.SaveCSV()

# ==========================================#
    def readTMData(self, filename):
        self.tmData.AppendData(filename)

    def saveTMData(self):
        self.tmData.SaveCSV()
# ==========================================
    def readCommentData(self, filename):
        self.commentData.AppendData(filename)

    def saveCommentData(self):
        self.commentData.SaveCSV()

# ==========================================
    def LoadEyeData(self, filename):
        self.eyeData.LoadData(filename)

    def LoadEyeLineData(self, filename):
        self.eyeData.LoadLineData(filename)

    def LoadGSROut(self, filename):
        self.gsrData.LoadData(filename)
        #self.gsrData
        #print self.gsrData.timestamp[0:30]

    def LoadTMOut(self, filename):
        self.tmData.LoadData(filename)

    def LoadTMDataFrame(self, filename):
        self.tmData.LoadDataFrame(filename)

    def LoadComments(self, filename):
        self.commentData.LoadData(filename)

    def LoadCommentsStartEnd(self, filename):
        self.commentData.LoadDataStartEnd(filename)

# ==========================================
    def GetCommentData(self,taskID,participantID):
        idx = (self.commentData.participant == participantID) & (self.commentData.task == taskID)
        cmtDF = self.commentData.GetDataFrame(idx)
        return cmtDF

    def GetCommentDataFrame(self,taskID,participantID):
        idx = (self.commentData.data.participant == participantID) & (self.commentData.data.task == taskID)
        cmtDF = self.commentData.GetDataFrame2(idx)
        cmtDF = cmtDF.reset_index()
        return cmtDF

    def GetTMData(self,taskID,participantID):
        idx = (self.tmData.participant == participantID) & (self.tmData.task == taskID)
        tmDF = self.tmData.GetDataFrame(idx)
        return tmDF

    def GetTmDataFrame(self, taskID, participantID):
        idx = (self.tmData.data.participant == participantID) & (self.tmData.data.task == taskID)
        tmDF = self.tmData.GetDataFrame2(idx)
        return tmDF

    def GetETData(self, taskID, participantID):
        idx = (self.eyeData.participant == participantID) & (self.eyeData.task == taskID)
        eyeDF = self.eyeData.GetDataFrame(idx)
        return eyeDF

    def GetETDataFrame(self, taskID, participantID):
        idx = (self.eyeData.data.participant == participantID) & (self.eyeData.data.task == taskID)
        eyeDF = self.eyeData.data[idx]
        eyeDF = eyeDF.reset_index()
        return eyeDF

    def GetGSRDataFrame(self, taskID, participantID):
        idx = (self.gsrData.participant == participantID) & (self.gsrData.task == taskID)
        gsrDF = self.gsrData.GetDataFrame(idx)
        gsrDF = gsrDF.reset_index()
        return gsrDF

    def GetAllGSRDataFrame(self, participantID):
        idx = self.gsrData.participant == participantID
        df = self.gsrData.GetDataFrame(idx)
        df = df.reset_index()
        return df

    def GetGSRBaseline(self,participantID):
        idx = self.gsrData.participant == participantID
        df = self.gsrData.GetDataFrame(idx)
        baseline = df.mediangsr.iloc[0:10].mean() # first 10 samples to set up the baseline
        return baseline
