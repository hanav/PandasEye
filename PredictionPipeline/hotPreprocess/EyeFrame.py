import pandas as pd
import scipy.stats
import math
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial import distance


class EyeFrame:
    def __init__(self, input, label):
        self.label = label
        self.data = pd.DataFrame(data=input)
        self.data = self.data.reset_index()
        self.cleanData = pd.DataFrame(data=None)
        self.transitions = None
        self.distances = None

        self.normEntropy = None
        self.entropy = None
        self.transitionSum = None
        self.transitionTypesCount = None

        self.meanTransitionDistance = None
        self.maxTransitionDistance = None
        self.relativeMeanTransitionDistance = None
        self.transitionDensity = None
        self.percentage_short = None
        self.percentage_long = None

    def preprocessing(self):
        pass
    # downsample to 50Hz signal
    # series_x = pd.Series(data=self.data.gazex)
    # series_y = pd.Series(data=self.data.gazey)
    # series_x.index = pd.DatetimeIndex(self.data.timeofday)
    # series_y.index = pd.DatetimeIndex(self.data.timeofday)
    # series_x = series_x.resample('20L', how='mean', closed='right', label='left')  # 20ms should correspond to 50Hz
    # series_y = series_y.fillna(method='backfill')

    # fixation filter

    def countLineFeatures(self):
        self.cleanData = self.data[~np.isnan(self.data.linenum)] # line data only
        self.countNormalizedEntropy()
        self.countLineDistance()
        self.countMeanTransitionDistnace()
        self.countMaxTransitionDistance()
        self.countRatioMeanMaxTransitionDistance()
        self.countTransitionMatrixDensity()
        self.countRatioShortLongTransitions() # float is missing

    def countSaccadeFeatures(self):
        tmp = self.data[~((self.data.gazex == 0) & (self.data.gazey == 0))]     # remove [0,0] -
        #plt.plot(tmp.gazex, tmp.gazey)
        #plt.plot(tmp.mousex, tmp.mousey)

        coord = pd.DataFrame(data=[tmp.gazex,tmp.gazey])
        diffs = np.diff(coord)
        distances = np.sqrt((diffs ** 2).sum(axis=0))
        distances = pd.Series(data=distances)

        distances = distances[distances!= 0] # remove 0-length saccades

        self.saccadeMeanDistance = distances.mean()
        self.saccadeSDDistance = distances.std()
        self.saccadeCount = distances.count()
        self.saccadeRatioLongShort = distances.max() / distances.min() # min = 0, remove duplicates
        self.saccadeDiffLongShort = distances.max() - distances.min()
        # ratio of vertical and horizontal saccades
        # angles between saccades

    def countGazeMouseFeatures(self):
        pass

    def countHistograms(self):
        pass
        # 2D histogram of acceleration and velocity

    def countCoverageFeatures(self):
        tmp = self.data[~((self.data.gazex == 0) & (self.data.gazey == 0))]
        if(not np.all(np.isnan(tmp))):
            hull = ConvexHull(tmp[['gazex', 'gazey']])
            self.hullArea = hull.area
        else:
            self.hullArea = np.nan
        # ratio hullArea / overallArea

        # convex hull area
    # ratio convex hull area / max convex hull area from the whole recording

    def countNormalizedEntropy(self):
        linenums = self.cleanData.linenum

        # count couples of transitions
        self.transitions = zip(linenums, linenums[1:])
        transitionValues = np.array(Counter(self.transitions).values())

        if (len(transitionValues) > 0):
            self.transitionSum = float(sum(transitionValues))
            transitionProbabilities = transitionValues / float(self.transitionSum)
            self.entropy = scipy.stats.entropy(transitionProbabilities, base=2)

            self.transitionTypesCount = len(transitionValues)
            averageProbabilities = 1.0 / self.transitionTypesCount # number of cells in the matrix filled
            maxEntropy = - math.log(averageProbabilities, 2)

            self.normEntropy = self.entropy  / maxEntropy
        else:
            self.normEntropy = np.nan
            self.entropy  = np.nan
            self.transitionSum = np.nan

    def countTransitionMatrixDensity(self):

        if(len(self.cleanData.linenum) > 0):
            linenums = self.cleanData.linenum
            uniqueLines = np.unique(linenums)
            matrixSize = len(uniqueLines) * len(uniqueLines)

            transitionValues = np.array(Counter(self.transitions).values())
            transitionCounts = len(transitionValues)

            self.transitionDensity = np.true_divide(transitionCounts,matrixSize)
        else:
            self.transitionDensity = np.nan

    def countLineDistance(self):
        linenums = self.cleanData.linenum
        transitions = np.asarray(self.transitions)

        if(len(linenums) > 1):
            self.distances = np.abs(transitions[:,1] - transitions[:,0])
        else:
            self.distances = [np.nan]

    def countMeanTransitionDistnace(self):
        if(not np.all(np.isnan(self.distances)) ):
            self.meanTransitionDistance = np.mean(self.distances)
        else:
            self.meanTransitionDistance = np.nan

    def countMaxTransitionDistance(self):
        if(not np.all(np.isnan(self.distances)) ):
            self.maxTransitionDistance = np.max(self.distances)
        else:
            self.meanTransitionDistance = np.nan

    def countRatioMeanMaxTransitionDistance(self):
        if(not np.all(np.isnan(self.distances)) ):
            self.relativeMeanTransitionDistance = float(np.true_divide(np.mean(self.distances),np.max(self.distances)))
        else:
            self.relativeMeanTransitionDistance = np.nan

    def countRatioShortLongTransitions(self):
        if(not np.all(np.isnan(self.transitions)) ):
            transitions = np.asarray(self.transitions)

            shortJumps = transitions[self.distances < 5]
            longJumps = transitions[self.distances >=5]

            self.percentage_short = float(len(shortJumps) /(len(transitions)))
            self.percentage_long = float(len(longJumps) / len(transitions))
        else:
            self.percentage_long = np.nan
            self.percentage_short = np.nan

    def returnDataFrameLabel(self):
        df = pd.DataFrame(data=[
            self.label,
            self.normEntropy,
            self.entropy,
            self.transitionSum,
            self.transitionTypesCount,
            self.meanTransitionDistance,
            self.maxTransitionDistance ,
            self.relativeMeanTransitionDistance,
            self.transitionDensity,
            self.percentage_short,
            self.percentage_long,
            self.saccadeMeanDistance,
            self.saccadeSDDistance,
            self.saccadeCount,
            self.saccadeRatioLongShort,
            self.saccadeDiffLongShort #,self.hullArea
        ])
        df = df.transpose()
        return df

    def returnDataFrame(self):
        df = pd.DataFrame(data=[
            self.normEntropy,
            self.entropy,
            self.transitionSum,
            self.transitionTypesCount,
            self.meanTransitionDistance,
            self.maxTransitionDistance,
            self.relativeMeanTransitionDistance,
            self.transitionDensity,
            self.percentage_short,
            self.percentage_long,
            self.saccadeMeanDistance,
            self.saccadeSDDistance,
            self.saccadeCount,
            self.saccadeRatioLongShort,
            self.saccadeDiffLongShort#,self.hullArea
        ])
        df = df.transpose()
        df = df.rename(index=str, columns={
            0:"line_normEntropy",
            1: "line_entropy",
            2: "line_transitionSum",
            3: "line_transitionTypesCount",
            4:"line_meanTransitionDistance",
            5:"line_maxTransitionDistance",
            6:"line_relativeMeanTransitionDistance",
            7:"line_transitionDensity",
            8:"line_percentage_short",
            9:"line_percentage_long,",
            10:  "saccade_MeanDistance",
            11: "saccade_SDDistance",
            12: "saccade_Count",
            13: "saccade_RatioLongShort",
            14: "saccade_DiffLongShort"#,
            #15: "saccade_hullArea"
        } )
        return df

