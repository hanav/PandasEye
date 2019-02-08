from hotPreprocess.GSRData import GSRData
from TMData import TMData
from hotPreprocess.EyeData import EyeData
from hotPreprocess.CommentData import CommentData
import pandas as pd
from datetime import timedelta
import numpy as np


from hotPreprocess.EyeFrame import EyeFrame
from TmFrame import TmFrame
from hotPreprocess.GSRFrame import GSRFrame

class Task:
    def __init__(self):
        self.comment = CommentData()
        self.tm = TMData()
        self.eye = EyeData()
        self.gsr = GSRData()
        self.gsrBaseline = None

        self.grp = pd.DataFrame(data=None)

        self.commentStart = None
        self.sequenceEnd = None
        self.sequenceStart = None

        self.TimestampFrame= pd.DataFrame(data=None)

    def SetDataBasic(self, tmDF, cmtDF):
        self.tm.SetDataFrame(tmDF)
        self.comment.data = cmtDF
        self.comment.data = self.comment.data.reset_index()
        self.comment.data.timeofday = self.comment.ConvertTimeofday(self.comment.data.timeofday)
        self.comment.data

    def SetData(self, tmDF, cmtDF, eyeDF, gsrDF):
        self.tm.SetDataFrame(tmDF)

        self.comment.data = cmtDF
        self.comment.data.start = self.comment.ConvertTimeofday(self.comment.data.start)
        self.comment.data.end = self.comment.ConvertTimeofday(self.comment.data.end)

        self.eye.data  = eyeDF
        self.eye.data.timeofday = self.eye.ConvertTimeofday(self.eye.data.timeofday)

        self.gsr.data = gsrDF
        self.gsr.data.timeofday = self.gsr.ConvertTimeofday(self.gsr.data.timeofday)

    def SetGSRBaseline(self, baseline):
        self.gsrBaseline = baseline

    def CountCommentStartEnd(self):
        commentEnds = self.comment.data.timeofday
        commentEnds = np.unique(commentEnds)
        commentEnds = np.insert(commentEnds, 0,self.tm.data.timeofday.iloc[0])
        commentBegins = []

        for i in range(len(commentEnds)-1):
            previousComment = commentEnds[i]
            currentComment = commentEnds[i+1]
            commentStart = self.FindCommentStart(currentComment,previousComment)
            commentBegins.append(commentStart)

        commentTimestamps = pd.DataFrame(data=None, columns=["start","end"])
        commentTimestamps['start'] = commentBegins
        commentTimestamps['end'] = pd.DataFrame(data=commentEnds[1:len(commentEnds)])

        print commentTimestamps
        self.TimestampFrame = commentTimestamps

    def FindCommentStart(self,currentCommentTimeofDay,previousCommentTimeofDay):
        idx = ((self.tm.data.timeofday > previousCommentTimeofDay) & (self.tm.data.timeofday < currentCommentTimeofDay )) & (self.tm.data.tmsum == 0)

        if not any(idx):
            print "No available zero-data for comment: ", currentCommentTimeofDay
            return None

        tmZeros = self.tm.data[idx]

        grp = tmZeros.groupby('block')
        block_dict = grp.indices
        zero_blocks = block_dict.keys()
        zero_lengths = [len(x) for x in block_dict.values()]

        zero_block_lenghts = pd.DataFrame(data=None)
        zero_block_lenghts['block'] = zero_blocks
        zero_block_lenghts['length'] = zero_lengths

        #print zero_block_lenghts.sort_values(by='block', ascending=False)

        lastTMBlockBeforeComment = tmZeros['block'].max() #
        lastDetectedBlockBeforeComment = zero_block_lenghts['block'].max()

        #maxDetectedBlock = zero_block_lenghts.block[zero_block_lenghts.length.idxmax(1)]
        #print "Max detected block", maxDetectedBlock

        #if lastTMBlockBeforeComment == lastDetectedBlockBeforeComment == maxDetectedBlock:
            #print "everything is fine, go on"
        commentTMdata = tmZeros[tmZeros['block'] == lastDetectedBlockBeforeComment]
        #else:
            #print "nope, you need to take different block"
            #commentTMdata = tmZeros[tmZeros['block'] == maxDetectedBlock]


        commentStart = commentTMdata.timeofday.iloc[0]
        #print self.commentStart
        return commentStart

    def CountSequenceStartEnd(self, durationSeconds):
        self.sequenceEnd = self.commentStart
        self.sequenceStart = self.commentStart - timedelta(seconds=durationSeconds)
        print self.sequenceStart," - ", self.sequenceEnd

    def ExportTimestampFrame(self):
        df = pd.DataFrame(data=self.TimestampFrame)
        df['participant'] = self.comment.data.participant[0]
        df['task'] = self.comment.data.task[0]
        return self.TimestampFrame




        # tmZeros = tmZeros.tmsum
        # validTimestamp = tmChunk < cmtChunk[1]
        # a= tmZeros[tmChunk < cmtChunk[1]]

        # df = pd.DataFrame(data=tmZeros)
        # df['block'] = (df.tmsum.shift(1) != df.tmsum).astype(int).cumsum()
        # zero_blocks = df.block[df.tmsum == 0]
        # z = pd.DataFrame(data=zero_blocks)
        # grp = z.groupby('block') # zdruzeni blocku
        # d_counts = [len(x) for x in d.values()]
        # d_keys = d.keys()
        # dd = pd.DataFrame(data=None)
        # dd['counts'] = d_counts
        # dd['keys']
        # dd.sort('keys', ascending=True)
        # pozor, chybi nam tu timestamps

    def ExtractSequence(self):
        idx = (self.tm.data.timeofday >= self.sequenceStart) & (self.tm.data.timeofday <= self.sequenceEnd)
        self.tmSequence = TMData()
        self.tmSequence.data = self.tm.data[idx]
        #print self.tmSequence.data

    def CutFrame(self, sequenceDuration):

        outputDF = pd.DataFrame(data=None)

        for i in range(0,len(self.comment.data.start)):
        #for i in range(13,14):
            print "Comment c.: ",i ,"z ", len(self.comment.data.start)
            # before comment
            start = self.comment.data.start[i] - timedelta(seconds=sequenceDuration)
            end = self.comment.data.start[i]

            eyeFeaturesBefore = self.CountEyeFrameFeatures(start,end,"beforeComment")
            tmFeaturesBefore = self.CountTMFeatures(start,end,"beforeComment")
            gsrFeaturesBefore = self.CountGSRFeatures(start,end,"beforeComment")

            featuresBefore = pd.concat([eyeFeaturesBefore, tmFeaturesBefore,gsrFeaturesBefore],axis=1)
            featuresBefore['label'] = "beforeComment"


            # signal sequence after the comment
            start = self.comment.data.end[i]
            end = self.comment.data.end[i] + timedelta(seconds=sequenceDuration)

            eyeFeaturesAfter = self.CountEyeFrameFeatures(start,end, "afterComment")
            tmFeatureAfter = self.CountTMFeatures(start,end,"afterComment")
            gsrFeaturesAfter = self.CountGSRFeatures(start, end, "afterComment")

            featuresAfter = pd.concat([eyeFeaturesAfter, tmFeatureAfter,gsrFeaturesAfter], axis=1)
            featuresAfter['label'] = "afterComment"

            outputDF = outputDF.append(pd.DataFrame(data=featuresBefore), ignore_index=True)
            outputDF = outputDF.append(pd.DataFrame(data=featuresAfter), ignore_index=True)

        return outputDF

        # transition matrix if needed
        #a = [2, 1, 3, 1, 2, 3, 1, 2, 2, 2]
        # b = np.zeros((3, 3))
        # for (x, y), counts in Counter(zip(a, a[1:])).iteritems():
        #     b[x - 1, y - 1] = c


    def CountEyeFrameFeatures(self,start,end, label):
        idx = (self.eye.data.timeofday >= start) & (self.eye.data.timeofday <= end)
        eyeFrame = EyeFrame(self.eye.data[idx], label)
        eyeFrame.preprocessing()

        eyeFrame.countLineFeatures()
        eyeFrame.countSaccadeFeatures()
        eyeFrame.countGazeMouseFeatures()
        eyeFrame.countHistograms()
        #eyeFrame.countCoverageFeatures() #todo: check fix, comment: 0, participant 13

        df = eyeFrame.returnDataFrame()
        return df

    def CountTMFeatures(self, start,end, label):
        idx = (self.tm.data.timeofday >= start) & (self.tm.data.timeofday <= end)
        tmFrame = TmFrame(self.tm.data[idx], label)

        tmFrame.countTMSumStats()
        tmFrame.countFingerStats()
        df = tmFrame.returnDataFrame()
        return df

    def CountGSRFeatures(self, start, end, label):
        idx = (self.gsr.data.timeofday >= start) & (self.gsr.data.timeofday <= end)
        gsrFrame = GSRFrame(self.gsr.data[idx], label)

        gsrFrame.preprocessing(self.gsrBaseline)
        gsrFrame.countTonicFeatures()
        #gsrFrame.countPhasicFeatures()
        df = gsrFrame.returnDataFrame()
        return df




