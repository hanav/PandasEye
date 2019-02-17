# date: 04/30/17
# author: Hana Vrzakova
# description: Class definition - Pipeline
# input:
# - input_data = DataFrame with features
# - outputDir = Output file directory
# - targetNames = Class labels
# - datasetLog = Experiment tag

from pipeline_load import LoadData
from pipeline_preprocess import Preprocessing
from pipeline_gridsearch import GridSearch
from pipeline_train import ParameterSearch
from pipeline_crossvalidate import CrossValidation
from pipeline_outputResults import ResultsOutput
from pipeline_finalize import FinalizeModel

import pipeline_crossvalidate
import pipeline_visualize


import os.path
import pandas as pd
import numpy as np
import scipy as sp

from sklearn.svm import SVC

# data preprocessing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder

# train-test-grid-search
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict, LeaveOneGroupOut

# SMOTE - todo: doinstalovat
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek

from sklearn.decomposition import PCA

# feature selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, SelectPercentile

# performance evaluation
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score, auc, roc_auc_score, f1_score,matthews_corrcoef

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import dateutil as du
import datetime


class Pipeline(LoadData,Preprocessing,ParameterSearch,CrossValidation,ResultsOutput, GridSearch,FinalizeModel):
    def __init__(self, input_data,outputDir,targetNames, datasetLog):

        self.features = input_data
        self.outputDir = outputDir
        self.resultsFile = os.path.join(outputDir, datasetLog+datetime.datetime.now().strftime("_%Y%m%d_%H%M_")+"test_results.csv")


        self.classifierDir = os.path.join(outputDir, datasetLog+datetime.datetime.now().strftime("_%m%d_%H%M"))
        if not os.path.exists(self.classifierDir):
            os.mkdir(self.classifierDir)

        # datasetLog
        # outputFolder = os.path.join(userhome, 'Dropbox', 'Subtitles', 'output',
        #                             datetime.date.today().strftime("%B %d, %Y"))
        # if not os.path.exists(outputFolder):
        #     os.makedirs(outputFolder)

        self.targetNames = targetNames

        self.features_start = 0
        self.features_end = 0
        self.user_column = 0
        self.label_column = 0

        self.x = None
        self.y = None
        self.users = None
        self.header = None

        self.x_train = None
        self.x_unseen = None
        self.y_train = None
        self.y_unseen = None
        self.user_train = None
        self.user_unseen = None

        self.best_kernel = None
        self.best_C = None
        self.best_gamma = None

        self.outputResults = pd.DataFrame(data=None)

        userhome = os.path.expanduser('~')
        #self.figuresOut = os.path.join(userhome, 'Dropbox', 'HotOrNot','r_icmi', 'figures')
















