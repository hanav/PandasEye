from Task import Task


class Participant:
    def __init__(self, ID):
        self.id = ID
        self.task = Task()

    def LoadTaskBasic(self, tmDF, cmtDF):
        self.task.SetDataBasic(tmDF, cmtDF)

    def LoadTask(self,tmDF, cmtDF, eyeDF, gsrDF):
        self.task.SetData(tmDF, cmtDF, eyeDF, gsrDF)

    def LoadGSRBaseline(self, gsrBaseline):
        self.task.SetGSRBaseline(gsrBaseline)

    def ExtractTMComments(self):
        self.task.CountCommentStartEnd()
        #self.task.FindTMZeros() aka CommentStart
        #self.task.CountSequenceStartEnd(30)
        #self.task.ExtractSequence()
        #self.task.CountTMFeatures()

    def ExportParticipantCommentTimestamp(self):
        df = self.task.ExportTimestampFrame()
        return df

    def SetGSRBaseline(self):
        pass

    def CutData(self, sequenceDuration):
        featureDF = self.task.CutFrame(sequenceDuration)
        return featureDF



