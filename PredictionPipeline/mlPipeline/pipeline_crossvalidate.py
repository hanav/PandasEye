# date: 04/30/17
# author: Hana Vrzakova
# description: Functions for crossvalidation
# - kFold crossvalidation - SVM
# - kFold crossvalidation (within a participant) - SVM
# - kFold crossvalidation - RandomForest
# - LOPO crossvalidation - RandomForest
# - kFold crossvalidation - Dummy classifier

import os.path
import pandas as pd
import numpy as np
import scipy as sp
import datetime


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier


# data preprocessing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.preprocessing import Imputer


# train-test-grid-search
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict, LeaveOneGroupOut

# SMOTE
from imblearn.over_sampling import SMOTE

# performance evaluation
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score, auc, roc_auc_score, f1_score,matthews_corrcoef, precision_score, recall_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

class CrossValidation:

    def CrossValidationKFold(self, PARAM_C, PARAM_Gamma, NUM_TRIALS, KSPLITS):
        # print("# Setting up hyper-parameters ")


        classifier = SVC(C=PARAM_C, kernel='rbf', gamma=PARAM_Gamma, probability=False)

        header = pd.DataFrame(data=["outer_counter", "inner_counter", "best_C", "best_G", "no_support_vectors",
                                    "train_accuracy", "train_auc", "train_kappa", "test_accuracy", "test_auc",
                                    "test_kappa", "train_f1", "test_f1", "train_precision","test_precision","train_recall","test_recall","startTime", "endTime", "no_vectors",
                                    "no_features"
                                    ]).transpose()
        self.writeLog(header)

        outer_counter = 0
        for j in range(0,NUM_TRIALS):

            cv = KFold(n_splits=KSPLITS, shuffle=True)
            inner_counter = 0
            for train_index, test_index in cv.split(self.x_train):
                x_train, x_test = self.x_train.iloc[train_index], self.x_train.iloc[test_index]
                y_train, y_test = self.y_train[train_index], self.y_train[test_index]

                print("Model training 1: binary classification")
                startTime = time.ctime()
                classifier.fit(x_train, y_train)

                print("Model parameters")
                no_support_vectors = len(classifier.support_vectors_) / float(len(y_train))

                # evaluations from the binary
                print("Binary: Training & Testing performance")
                train_predict = classifier.predict(x_train)
                test_predict = classifier.predict(x_test)

                print("Binary: predict")
                train_accuracy = accuracy_score(train_predict, y_train)
                test_accuracy = accuracy_score(test_predict, y_test)

                train_kappa = cohen_kappa_score(y_train, train_predict)
                test_kappa = cohen_kappa_score(y_test, test_predict)

                target_names = ['negativeClass', 'positiveClass']
                train_f1 = f1_score(y_train, train_predict, target_names)
                test_f1 = f1_score(y_test, test_predict, target_names)
                train_precision = precision_score(y_train, train_predict, target_names)
                test_precision = precision_score(y_test, test_predict, target_names)
                train_recall = recall_score(y_train, train_predict, target_names)
                test_recall = recall_score(y_test, test_predict, target_names)

                # evaluations from the probabilities
                print("Model training 2: Binary classification: probabilities")


                classifier = SVC(C=PARAM_C, kernel='rbf', gamma=PARAM_Gamma, probability=True)
                classifier.fit(x_train, y_train)
                endTime = time.ctime()

                print("Probabilities: predict")
                train_probabilities = classifier.predict_proba(x_train)
                test_probabilities = classifier.predict_proba(x_test)

                train_auc = roc_auc_score(y_true=y_train, y_score=train_probabilities[:, 1])
                test_auc = roc_auc_score(y_true=y_test, y_score=test_probabilities[:, 1])

                # debug here with simpler/faster train
                row = pd.DataFrame(data=[outer_counter,inner_counter, PARAM_C, PARAM_Gamma, no_support_vectors,
                                         train_accuracy, train_auc, train_kappa, test_accuracy, test_auc, test_kappa,
                                         train_f1, test_f1, train_precision, test_precision, train_recall, test_recall, startTime, endTime, x_train.shape[0],
                                         x_train.shape[1]]).transpose()

                self.writeLog(row)
                inner_counter +=1
            outer_counter +=1
        return

    def CrossValidationKFoldPerson(self, PARAM_C, PARAM_Gamma, NUM_TRIALS, KSPLITS):
        print("# Setting up hyper-parameters ")


        classifier = SVC(C=PARAM_C, kernel='rbf', gamma=PARAM_Gamma, probability=False)

        header = pd.DataFrame(data=["outer_counter", "inner_counter", "best_C", "best_G", "no_support_vectors",
                                    "train_accuracy", "train_auc", "train_kappa", "test_accuracy", "test_auc",
                                    "test_kappa", "train_f1", "test_f1", "train_precision","test_precision","train_recall","test_recall","startTime", "endTime", "no_vectors",
                                    "no_features"
                                    ]).transpose()
        self.writeLog(header)

        outer_counter = 0
        for j in range(0,NUM_TRIALS):

            cv = KFold(n_splits=KSPLITS, shuffle=True)
            inner_counter = 0
            for train_index, test_index in cv.split(self.x_train):
                x_train, x_test = self.x_train.iloc[train_index], self.x_train.iloc[test_index]
                y_train, y_test = self.y_train[train_index], self.y_train[test_index]

                print("Model training 1: binary classification")
                startTime = time.ctime()
                classifier.fit(x_train, y_train)

                print("Model parameters")
                no_support_vectors = len(classifier.support_vectors_) / float(len(y_train))

                # evaluations from the binary
                print("Binary: Training & Testing performance")
                train_predict = classifier.predict(x_train)
                test_predict = classifier.predict(x_test)

                print("Binary: predict")
                train_accuracy = accuracy_score(train_predict, y_train)
                test_accuracy = accuracy_score(test_predict, y_test)

                train_kappa = cohen_kappa_score(y_train, train_predict)
                test_kappa = cohen_kappa_score(y_test, test_predict)

                target_names = ['negative', 'positive']
                train_f1 = f1_score(y_train, train_predict, target_names)
                test_f1 = f1_score(y_test, test_predict, target_names)
                train_precision = precision_score(y_train, train_predict, target_names)
                test_precision = precision_score(y_test, test_predict, target_names)
                train_recall = recall_score(y_train, train_predict, target_names)
                test_recall = recall_score(y_test, test_predict, target_names)

                # evaluations from the probabilities
                print("Model training 2: Binary classification: probabilities")
                classifier = SVC(C=PARAM_C, kernel='rbf', gamma=PARAM_Gamma, probability=True)
                classifier.fit(x_train, y_train)
                endTime = time.ctime()

                print("Probabilities: predict")
                train_probabilities = classifier.predict_proba(x_train)
                test_probabilities = classifier.predict_proba(x_test)

                train_auc = roc_auc_score(y_true=y_train, y_score=train_probabilities[:, 1])
                test_auc = roc_auc_score(y_true=y_test, y_score=test_probabilities[:, 1])

                # debug here with simpler/faster train
                row = pd.DataFrame(data=[outer_counter,inner_counter, PARAM_C, PARAM_Gamma, no_support_vectors,
                                         train_accuracy, train_auc, train_kappa, test_accuracy, test_auc, test_kappa,
                                         train_f1, test_f1, train_precision, test_precision, train_recall, test_recall, startTime, endTime, x_train.shape[0],
                                         x_train.shape[1]]).transpose()

                self.writeLog(row)
                inner_counter +=1
            outer_counter +=1

        return


    def CrossValidationPerson(self, PARAM_C, PARAM_Gamma, NUM_TRIALS, KSPLITS):
        print("# Setting up hyper-parameters ")


        classifier = SVC(C=PARAM_C, kernel='rbf', gamma=PARAM_Gamma, probability=False)

        header = pd.DataFrame(data=["outer_counter", "inner_counter", "best_C", "best_G", "no_support_vectors",
                                    "train_accuracy", "train_auc", "train_kappa", "test_accuracy", "test_auc",
                                    "test_kappa", "train_f1", "test_f1", "train_precision","test_precision","train_recall","test_recall","startTime", "endTime", "no_vectors",
                                    "no_features"
                                    ]).transpose()
        self.writeLog(header)

        logo = LeaveOneGroupOut()

        outer_counter = 0
        for j in range(0,NUM_TRIALS):

            inner_counter = 0



            #newX = self.x_train.iloc[0:self.x_train[0].size, 0:self.x_train.iloc[0].size].values
            for train_index, test_index in logo.split(self.x_train, self.y_train, self.user_train):
                person = np.unique(self.user_train[test_index])
                print "Testing on person: ", person
                #x_train, x_test = self.x_train[train_index], self.x_train[test_index]
                #y_train, y_test = self.y_train[train_index], self.y_train[test_index]

                x_train, x_test = self.x_train.iloc[train_index], self.x_train.iloc[test_index]
                y_train, y_test = self.y_train[train_index], self.y_train[test_index]

                print("Model training 1: binary classification")
                startTime = time.ctime()
                classifier.fit(x_train, y_train)

                print("Model parameters")
                no_support_vectors = len(classifier.support_vectors_) / float(len(y_train))

                # evaluations from the binary
                print("Binary: Training & Testing performance")
                train_predict = classifier.predict(x_train)
                test_predict = classifier.predict(x_test)

                print("Binary: predict")
                train_accuracy = accuracy_score(train_predict, y_train)
                test_accuracy = accuracy_score(test_predict, y_test)

                train_kappa = cohen_kappa_score(y_train, train_predict)
                test_kappa = cohen_kappa_score(y_test, test_predict)

                target_names = ['negative', 'positive']

                train_f1 = f1_score(y_train, train_predict, target_names)
                test_f1 = f1_score(y_test, test_predict, target_names)
                train_precision = precision_score(y_train, train_predict, target_names)
                test_precision = precision_score(y_test, test_predict, target_names)
                train_recall = recall_score(y_train, train_predict, target_names)
                test_recall = recall_score(y_test, test_predict, target_names)

                # evaluations from the probabilities
                print("Model training 2: Binary classification: probabilities")

                classifier = SVC(C=PARAM_C, kernel='rbf', gamma=PARAM_Gamma, probability=True)
                classifier.fit(x_train, y_train)
                endTime = time.ctime()

                print("Probabilities: predict")
                train_probabilities = classifier.predict_proba(x_train)
                test_probabilities = classifier.predict_proba(x_test)

                train_auc = roc_auc_score(y_true=y_train, y_score=train_probabilities[:, 1])
                test_auc = roc_auc_score(y_true=y_test, y_score=test_probabilities[:, 1])

                # debug here with simpler/faster train
                row = pd.DataFrame(data=[outer_counter,inner_counter, PARAM_C, PARAM_Gamma, no_support_vectors,
                                         train_accuracy, train_auc, train_kappa, test_accuracy, test_auc, test_kappa,
                                         train_f1, test_f1, train_precision, test_precision, train_recall, test_recall, startTime, endTime, x_train.shape[0],
                                         x_train.shape[1]]).transpose()

                self.writeLog(row)
                inner_counter +=1
            outer_counter +=1

        return

    def CrossValidationSimple(self):
            #SET: find and replace missing data
            imputer = Imputer(axis=0, copy=False, missing_values='NaN', strategy="most_frequent").fit(self.x)
            self.x = imputer.transform(self.x)  # works ok without line-based features
            scaler = StandardScaler().fit(self.x)
            self.x = scaler.transform(self.x)
            self.x = pd.DataFrame(data=self.x)

            #SET: split data ratio
            x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size = 0.4, random_state = 0)

            #SET: cross-validation kFold
            cvKFold = KFold(n_splits=5, shuffle=True)

            # add to param_grid scores: 'precision', 'recall'
            # add to GridSearch: scoring='%s_weighted' % score

            tuned_parameters = [
                {'C':[1], 'kernel': ['rbf']}
                    #{'C': [1, 250,500,750], 'kernel': ['linear']}
                    #{'C': [400,450,500,550,600], 'kernel':['linear']}
                    #{'C': [400,450,500,550,600], 'gamma': [ 0.001, 0.0001,0.00001], 'kernel': ['rbf']},
                    #{'C': [1, 10, 100, 500, 900], 'gamma': [0.001, 0.0001, 0.00001], 'kernel': ['rbf']},
            ]

            gridSearch = GridSearchCV(SVC(probability=True), tuned_parameters, cv=cvKFold, scoring="precision")
            gridSearch.fit(x_train, y_train)

            trainLeftScore = gridSearch.best_score_
            #trainScore = gridSearch.score(x_train,y_train)
            testScore = gridSearch.score(x_test,y_test)

            clf = gridSearch.best_estimator_
            y_true, y_pred = y_test, clf.predict(x_test)
            report = classification_report(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)

            pass

    def CrossValidationModality(self):
        cv = KFold(n=len(self.y), n_folds=5, shuffle=True)

        etDF = self.x.ix[:, 0:5]
        tmDF = self.x.ix[:, 6:8]
        gsrDF = self.x.ix[:, 9:20]

        self.etResults = pd.DataFrame(data=None)
        for train_idx, unseen_idx in cv:  # we can do this exactly once...
            x_train = etDF.iloc[train_idx]
            y_train = self.y[train_idx]
            x_unseen = etDF.iloc[unseen_idx]
            y_unseen = self.y[unseen_idx]

            x_train, x_unseen = self.ReplaceMissingValues(x_train, x_unseen)

            resultRow = self.GridSearch2(x_train, y_train, x_unseen, y_unseen)
            self.etResults = self.outputResults.append(resultRow)

        self.tmResults = pd.DataFrame(data=None)
        for train_idx, unseen_idx in cv:  # we can do this exactly once...
            x_train = tmDF.iloc[train_idx]
            y_train = self.y[train_idx]
            x_unseen = tmDF.iloc[unseen_idx]
            y_unseen = self.y[unseen_idx]

            x_train, x_unseen = self.ReplaceMissingValues(x_train, x_unseen)

            resultRow = self.GridSearch2(x_train, y_train, x_unseen, y_unseen)
            self.tmResults = self.outputResults.append(resultRow)

        self.gsrResults = pd.DataFrame(data=None)
        for train_idx, unseen_idx in cv:  # we can do this exactly once...
            x_train = gsrDF.iloc[train_idx]
            y_train = self.y[train_idx]
            x_unseen = gsrDF.iloc[unseen_idx]
            y_unseen = self.y[unseen_idx]

            x_train, x_unseen = self.ReplaceMissingValues(x_train, x_unseen)

            resultRow = self.GridSearch2(x_train, y_train, x_unseen, y_unseen)
            self.gsrResults = self.outputResults.append(resultRow)

        averageEtResults = self.etResults.mean()
        averageTmResults = self.tmResults.mean()
        averageGsrResults = self.gsrResults.mean()

    def CrossvalidationUsers(self, outputDir):

            print("# Logo-Crossvalidation")
            outer_cv = LeaveOneGroupOut()

            NUM_TRIALS = 1   # add the outer loop, repeat 30 times with random seed

            best_C = 750
            best_gamma = 'auto'

            dfOutput = pd.DataFrame()

            outer_counter = 0
            for i in range(NUM_TRIALS):
                print i
                test_probabilities = []
                test_y_actual = []

                inner_counter = 0
                for train, test in outer_cv.split(self.x, self.y, groups=self.users):

                    x_train = self.x[train]
                    y_train = self.y[train]

                    x_test = self.x[test]
                    y_test = self.y[test]

                    print("Upsampling with SMOTE")
                    # sm = SMOTE(random_state=i)
                    # x_res, y_res = sm.fit_sample(x_train,y_train)
                    x_res=x_train
                    y_res = y_train

                    print("Binary training")
                    classifier = SVC(C=best_C, kernel='rbf', gamma=best_gamma, class_weight="balanced")
                    #classifier = SVC()
                    #classifier = RandomForestClassifier()
                    #classifier = XGBClassifier()
                    # classifier = AdaBoostClassifier()
                    # classifier = GaussianNB()

                    classifier.fit(x_res,y_res)

                    train_predict = classifier.predict(x_train)
                    test_predict = classifier.predict(x_test)

                    print("Binary: predict")
                    train_accuracy = accuracy_score(train_predict, y_train)
                    test_accuracy = accuracy_score(test_predict, y_test)

                    train_kappa = cohen_kappa_score(y_train, train_predict)
                    test_kappa = cohen_kappa_score(y_test, test_predict)

                    target_names = ['negative', 'positive']
                    train_f1 = f1_score(y_train, train_predict, target_names)
                    test_f1 = f1_score(y_test, test_predict, target_names)
                    train_precision = precision_score(y_train, train_predict, target_names)
                    test_precision = precision_score(y_test, test_predict, target_names)
                    train_recall = recall_score(y_train, train_predict, target_names)
                    test_recall = recall_score(y_test, test_predict, target_names)

                    # print("Binary training - probabilities true")
                    # classifier.probability = True  # we need this to produce the probabilities - ROC curve
                    # classifier.fit(x_res, y_res)
                    # test_y_actual.extend(y_test)
                    # test_probabilities.append(classifier.predict_proba(x_test))

                    # train_auc = roc_auc_score(y_true=y_train, y_score=train_probabilities[:, 1])
                    # test_auc = roc_auc_score(y_true=y_test, y_score=test_probabilities[:, 1])

                    # debug here with simpler/faster train
                    row = pd.DataFrame(data=[outer_counter, inner_counter,
                                             train_accuracy,  train_kappa, test_accuracy,
                                             test_kappa,
                                             train_f1, test_f1, train_precision, test_precision, train_recall,
                                             test_recall, x_train.shape[0],
                                             x_train.shape[1], len(y_test), sum(y_test==1), sum(y_test==0) ]).transpose()

                    # self.writeLog(row)
                    dfOutput = dfOutput.append(row)
                    inner_counter += 1
                outer_counter += 1

            header = ['outer_counter', 'inner_counter',
                                    'train_accuracy', 'train_kappa', 'test_accuracy',
                                    'test_kappa', 'train_f1', 'test_f1', 'train_precision', 'test_precision',
                                    'train_recall', 'test_recall',
                                    'no_rows', 'no_features', 'no_testingSamples', 'no_positiveTest', 'no_negativeTest']

            dfOutput.columns = header

            # count AUC
            # flattened = [val for sublist in test_probabilities for val in sublist]

            print("Mean accuracy: ", dfOutput['train_accuracy'].mean().round(2), " - ", dfOutput['test_accuracy'].mean().round(2))
            print("Mean F1: ", dfOutput['train_f1'].mean().round(2), " - ", dfOutput['test_f1'].mean().round(2))
            print("Mean Precision: ", dfOutput['train_precision'].mean().round(2), " - ", dfOutput['test_precision'].mean().round(2)) # bulling
            print("Mean Recall: ", dfOutput['train_recall'].mean().round(2), " - ", dfOutput['test_recall'].mean().round(2))

            outputFile = os.path.join(outputDir,"loocv"+datetime.datetime.now().strftime("_%Y%m%d_%H%M_") +".csv")
            dfOutput.to_csv(outputFile,sep=",", index=False)

            return

    def CrossvalidationUsersLongterm(self, bestParams, testTag):

            # print("# Logo-Crossvalidation")
            outer_cv = LeaveOneGroupOut()

            NUM_TRIALS = 1   # add the outer loop, repeat 30 times with random seed
            #
            # best_C = 750
            # best_gamma = 'auto'

            array_y_test = []
            array_y_test_scores = []

            dfOutput = pd.DataFrame()
            dfFeatureImportance = pd.DataFrame()

            outer_counter = 0

            dfTestProbabilities = pd.DataFrame()

            for i in range(NUM_TRIALS):
                # print i
                test_probabilities = []
                test_y_actual = []

                inner_counter = 0
                for train, test in outer_cv.split(self.x, self.y, groups=self.users):

                    x_train = self.x[train]
                    y_train = self.y[train]

                    x_test = self.x[test]
                    y_test = self.y[test]

                    # print("Upsampling with SMOTE")
                    sm = SMOTE(random_state=18)
                    x_res, y_res = sm.fit_sample(x_train,y_train)
                    # x_res=x_train
                    # y_res = y_train

                    # print("Binary training")
                    # classifier = RandomForestClassifier()
                    classifier = RandomForestClassifier(n_estimators=bestParams['n_estimators'],
                                                                            max_features=bestParams['max_features'],
                                                                            max_depth=bestParams['max_depth'],
                                                                            min_samples_split=bestParams['min_samples_split'],
                                                                            min_samples_leaf=bestParams['min_samples_split'],
                                                                            bootstrap=bestParams['bootstrap'] )

                    classifier.fit(x_res,y_res)

                    train_predict = classifier.predict(x_res)
                    test_predict = classifier.predict(x_test)

                    y_test_scores = classifier.predict_proba(x_test)
                    # array_y_test.append(y_test)
                    # array_y_test_scores.append(y_test_scores)
                    scores = np.vstack(y_test_scores)

                    user_probabilities = pd.DataFrame([y_test, scores[:, 0], scores[:, 1]]).transpose()
                    user_probabilities['user'] = inner_counter
                    dfTestProbabilities = dfTestProbabilities.append(user_probabilities)

                    # print("Binary: predict")
                    train_accuracy = accuracy_score(train_predict, y_res)
                    test_accuracy = accuracy_score(test_predict, y_test)

                    train_kappa = cohen_kappa_score(y_res, train_predict)
                    test_kappa = cohen_kappa_score(y_test, test_predict)

                    target_names = ['negative', 'positive']
                    train_f1 = f1_score(y_res, train_predict, target_names)
                    test_f1 = f1_score(y_test, test_predict, target_names)

                    train_precision = precision_score(y_res, train_predict, target_names)
                    test_precision = precision_score(y_test, test_predict, target_names)

                    train_recall = recall_score(y_res, train_predict, target_names)
                    test_recall = recall_score(y_test, test_predict, target_names)

                    # print("Feature ranking:")
                    # http: // scikit - learn.org / stable / auto_examples / ensemble / plot_forest_importances.html
                    importances = classifier.feature_importances_
                    dfFeatureImportance = dfFeatureImportance.append([importances])

                    row = pd.DataFrame(data=[outer_counter, inner_counter,
                                             train_accuracy,  train_kappa, test_accuracy,
                                             test_kappa,
                                             train_f1, test_f1, train_precision, test_precision, train_recall,
                                             test_recall, x_train.shape[0],
                                             x_train.shape[1], len(y_test), sum(y_test==1), sum(y_test==0) ]).transpose()

                    dfOutput = dfOutput.append(row)
                    inner_counter += 1
                outer_counter += 1

            dfOutput = dfOutput.append(dfOutput.mean(axis=0), ignore_index=True)
            dfOutput = dfOutput.append(dfOutput.std(axis=0), ignore_index=True)

            dfFeatureImportance = dfFeatureImportance.append(dfFeatureImportance.mean(axis=0), ignore_index=True)
            dfFeatureImportance = dfFeatureImportance.append(dfFeatureImportance.std(axis=0), ignore_index=True)

            header = ['outer_counter', 'inner_counter',
                                    'train_accuracy', 'train_kappa', 'test_accuracy',
                                    'test_kappa', 'train_f1', 'test_f1', 'train_precision', 'test_precision',
                                    'train_recall', 'test_recall',
                                    'no_rows', 'no_features', 'no_testingSamples', 'no_positiveTest', 'no_negativeTest']

            dfOutput.columns = header

            print("Mean accuracy: ", dfOutput['train_accuracy'].mean().round(2), " - ", dfOutput['test_accuracy'].mean().round(2))
            print("Mean F1: ", dfOutput['train_f1'].mean().round(2), " - ", dfOutput['test_f1'].mean().round(2))
            print("Mean Precision: ", dfOutput['train_precision'].mean().round(2), " - ", dfOutput['test_precision'].mean().round(2)) # bulling
            print("Mean Recall: ", dfOutput['train_recall'].mean().round(2), " - ", dfOutput['test_recall'].mean().round(2))

            outputFile = os.path.join(self.classifierDir,testTag + "_loocv_results"+datetime.datetime.now().strftime("_%Y%m%d_%H%M_") +".csv")
            dfOutput.to_csv(outputFile,sep=",", index=False)

            dfFeatureImportance.columns = self.header
            outputFileImportance = os.path.join(self.classifierDir, testTag + "_loocv_importances"+datetime.datetime.now().strftime("_%Y%m%d_%H%M_") +".csv")
            dfFeatureImportance.to_csv(outputFileImportance,sep=",", index=False)

            return

    def CrossvalidationKFoldLongterm(self, bestParams,testTag):
            # print("# KFold-Stratified Crossvalidation")
            outer_cv =  StratifiedKFold(n_splits=5, shuffle=True)

            NUM_TRIALS = 5

            array_y_test = []
            array_y_test_scores = []

            dfOutput = pd.DataFrame()
            dfFeatureImportance = pd.DataFrame()

            outer_counter = 0
            for i in range(NUM_TRIALS):
                # print i
                test_probabilities = []
                test_y_actual = []

                inner_counter = 0
                for train, test in outer_cv.split(self.x_train, self.y_train):

                    x_train = self.x[train]
                    y_train = self.y[train]

                    x_test = self.x[test]
                    y_test = self.y[test]

                    # print(np.bincount(y_train), "-", np.bincount(y_test))
                    # print("Upsampling with SMOTE")
                    sm = SMOTE(random_state=18)
                    x_res, y_res = sm.fit_sample(x_train,y_train)
                    # x_res=x_train
                    # y_res = y_train

                    # classifier = RandomForestClassifier()
                    classifier = RandomForestClassifier(n_estimators=bestParams['n_estimators'],
                                                                            max_features=bestParams['max_features'],
                                                                            max_depth=bestParams['max_depth'],
                                                                            min_samples_split=bestParams['min_samples_split'],
                                                                            min_samples_leaf=bestParams['min_samples_split'],
                                                                            bootstrap=bestParams['bootstrap'] )
                    classifier.fit(x_res,y_res)

                    train_predict = classifier.predict(x_train)
                    test_predict = classifier.predict(x_test)

                    y_test_scores = classifier.predict_proba(x_test)
                    array_y_test.append(y_test)
                    array_y_test_scores.append(y_test_scores)

                    # print("Binary: predict")
                    train_accuracy = accuracy_score(train_predict, y_train)
                    test_accuracy = accuracy_score(test_predict, y_test)

                    train_kappa = cohen_kappa_score(y_train, train_predict)
                    test_kappa = cohen_kappa_score(y_test, test_predict)

                    target_names = ['negative', 'positive']
                    train_f1 = f1_score(y_train, train_predict, target_names)
                    test_f1 = f1_score(y_test, test_predict, target_names)
                    train_precision = precision_score(y_train, train_predict, target_names)
                    test_precision = precision_score(y_test, test_predict, target_names)
                    train_recall = recall_score(y_train, train_predict, target_names)
                    test_recall = recall_score(y_test, test_predict, target_names)

                    # train positive and negative accuracy
                    dfResults = pd.DataFrame()
                    dfResults['true'] = y_train
                    dfResults['predict'] = train_predict
                    dfPositive = dfResults.loc[dfResults['true'] == 1]
                    train_positive_acc = sum(dfPositive['predict']) / float(dfPositive['predict'].count())

                    dfNegative = dfResults.loc[dfResults['true'] == 0]
                    train_negative_acc = dfNegative['predict'].loc[dfNegative['predict'] == 0].count() / float(dfNegative['predict'].count())

                    # test positive and negative accuracy
                    dfResults = pd.DataFrame()
                    dfResults['true'] = y_test
                    dfResults['predict'] = test_predict
                    dfPositive = dfResults.loc[dfResults['true'] == 1]
                    test_positive_acc = sum(dfPositive['predict']) / float(dfPositive['predict'].count())

                    dfNegative = dfResults.loc[dfResults['true'] == 0]
                    test_negative_acc = dfNegative['predict'].loc[dfNegative['predict'] == 0].count() / float(dfNegative['predict'].count())

                    # print("Feature ranking:")
                    # http: // scikit - learn.org / stable / auto_examples / ensemble / plot_forest_importances.html
                    importances = classifier.feature_importances_
                    dfFeatureImportance = dfFeatureImportance.append([importances])
                    # std = np.std([tree.feature_importances_ for tree in classifier.estimators_],
                    #              axis=0)
                    # indices = np.argsort(importances)[::-1]
                    #
                    # for f in range(x_res.shape[1]):
                    #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

                    # print("Binary training - probabilities true")
                    classifier.probability = True  # we need this to produce the probabilities - ROC curve
                    classifier.fit(x_res, y_res)
                    # test_y_actual.extend(y_test)
                    train_probabilities = classifier.predict_proba(x_train)
                    test_probabilities = classifier.predict_proba(x_test)

                    train_auc = roc_auc_score(y_true=y_train, y_score=train_probabilities[:, 1])
                    test_auc = roc_auc_score(y_true=y_test, y_score=test_probabilities[:, 1])

                    # debug here with simpler/faster train
                    row = pd.DataFrame(data=[outer_counter, inner_counter,
                                             train_accuracy, train_auc, train_kappa,
                                             test_accuracy, test_auc, test_kappa,
                                             train_positive_acc, train_negative_acc, test_positive_acc, test_negative_acc,
                                             train_f1, test_f1,
                                             train_precision, test_precision,
                                             train_recall, test_recall,
                                             x_train.shape[0], x_train.shape[1],
                                             len(y_test), sum(y_test==1), sum(y_test==0) ]).transpose()

                    # self.writeLog(row)
                    dfOutput = dfOutput.append(row)
                    inner_counter += 1
                outer_counter += 1

            # auc = roc_auc_score(y_test, y_scores)

            dfOutput = dfOutput.append(dfOutput.mean(axis=0), ignore_index=True)
            dfOutput = dfOutput.append(dfOutput.std(axis=0), ignore_index=True)

            dfFeatureImportance = dfFeatureImportance.append(dfFeatureImportance.mean(axis=0), ignore_index=True)
            dfFeatureImportance = dfFeatureImportance.append(dfFeatureImportance.std(axis=0), ignore_index=True)

            header = ['outer_counter', 'inner_counter',
                                    'train_accuracy', 'train_auc', 'train_kappa',
                                    'test_accuracy', 'test_auc', 'test_kappa',
                                    'train_positive_acc','train_negative_acc', 'test_positive_acc','test_negative_acc',
                                    'train_f1', 'test_f1',
                                    'train_precision', 'test_precision',
                                    'train_recall', 'test_recall',
                                    'no_rows', 'no_features',
                                    'no_testingSamples', 'no_positiveTest', 'no_negativeTest']

            dfOutput.columns = header

            # count AUC
            # flattened = [val for sublist in test_probabilities for val in sublist]

            print("Stratified KFOLD: ")
            print("Mean accuracy: ", dfOutput['train_accuracy'].mean().round(2), " - ", dfOutput['test_accuracy'].mean().round(2))
            print("Mean AUC: ", dfOutput['train_auc'].mean().round(2), "- ", dfOutput['test_auc'].mean().round(2))
            print("Mean F1: ", dfOutput['train_f1'].mean().round(2), " - ", dfOutput['test_f1'].mean().round(2))
            print("Mean Precision: ", dfOutput['train_precision'].mean().round(2), " - ", dfOutput['test_precision'].mean().round(2)) # bulling
            print("Mean Recall: ", dfOutput['train_recall'].mean().round(2), " - ", dfOutput['test_recall'].mean().round(2))

            outputFile = os.path.join(self.classifierDir,testTag + "_stratifiedKfold_results"+datetime.datetime.now().strftime("_%Y%m%d_%H%M_") +".csv")
            dfOutput.to_csv(outputFile,sep=",", index=False)

            dfFeatureImportance.columns = self.header
            outputFileImportance = os.path.join(self.classifierDir, testTag + "_stratifiedKfold_importances"+datetime.datetime.now().strftime("_%Y%m%d_%H%M_") +".csv")
            dfFeatureImportance.to_csv(outputFileImportance,sep=",", index=False)

            return

    def CrossvalidationKFoldLongtermDummy(self,testTag):

            # print("# Dummy KFold-Stratified Crossvalidation")
            outer_cv =  StratifiedKFold(n_splits=5, shuffle=True)

            NUM_TRIALS = 5

            array_y_test = []
            array_y_test_scores = []

            dfOutput = pd.DataFrame()
            dfFeatureImportance = pd.DataFrame()

            outer_counter = 0
            for i in range(NUM_TRIALS):
                # print i
                test_probabilities = []
                test_y_actual = []

                inner_counter = 0
                for train, test in outer_cv.split(self.x_train, self.y_train):

                    x_train = self.x[train]
                    y_train = self.y[train]

                    x_test = self.x[test]
                    y_test = self.y[test]

                    # print(np.bincount(y_train), "-", np.bincount(y_test))

                    # print("Upsampling with SMOTE")
                    sm = SMOTE(random_state=18)
                    x_res, y_res = sm.fit_sample(x_train,y_train)
                    # x_res=x_train
                    # y_res = y_train

                    # print("Binary training")
                    classifier = DummyClassifier(random_state=18)

                    classifier.fit(x_res,y_res)

                    train_predict = classifier.predict(x_train)
                    test_predict = classifier.predict(x_test)

                    y_test_scores = classifier.predict_proba(x_test)
                    array_y_test.append(y_test)
                    array_y_test_scores.append(y_test_scores)

                    # print("Binary: predict")
                    train_accuracy = accuracy_score(train_predict, y_train)
                    test_accuracy = accuracy_score(test_predict, y_test)

                    train_kappa = cohen_kappa_score(y_train, train_predict)
                    test_kappa = cohen_kappa_score(y_test, test_predict)

                    target_names = ['negativeClass', 'positiveClass']
                    train_f1 = f1_score(y_train, train_predict, target_names)
                    test_f1 = f1_score(y_test, test_predict, target_names)
                    train_precision = precision_score(y_train, train_predict, target_names)
                    test_precision = precision_score(y_test, test_predict, target_names)
                    train_recall = recall_score(y_train, train_predict, target_names)
                    test_recall = recall_score(y_test, test_predict, target_names)

                    # train positive and negative accuracy
                    dfResults = pd.DataFrame()
                    dfResults['true'] = y_train
                    dfResults['predict'] = train_predict
                    dfPositive = dfResults.loc[dfResults['true'] == 1]
                    train_positive_acc = sum(dfPositive['predict']) / float(dfPositive['predict'].count())

                    dfNegative = dfResults.loc[dfResults['true'] == 0]
                    train_negative_acc = dfNegative['predict'].loc[dfNegative['predict'] == 0].count() / float(dfNegative['predict'].count())

                    # test positive and negative accuracy
                    dfResults = pd.DataFrame()
                    dfResults['true'] = y_test
                    dfResults['predict'] = test_predict
                    dfPositive = dfResults.loc[dfResults['true'] == 1]
                    test_positive_acc = sum(dfPositive['predict']) / float(dfPositive['predict'].count())

                    dfNegative = dfResults.loc[dfResults['true'] == 0]
                    test_negative_acc = dfNegative['predict'].loc[dfNegative['predict'] == 0].count() / float(dfNegative['predict'].count())


                    classifier.probability = True  # we need this to produce the probabilities - ROC curve
                    classifier.fit(x_res, y_res)
                    train_probabilities = classifier.predict_proba(x_train)
                    test_probabilities = classifier.predict_proba(x_test)

                    train_auc = roc_auc_score(y_true=y_train, y_score=train_probabilities[:, 1])
                    test_auc = roc_auc_score(y_true=y_test, y_score=test_probabilities[:, 1])

                    # debug here with simpler/faster train
                    row = pd.DataFrame(data=[outer_counter, inner_counter,
                                             train_accuracy, train_auc, train_kappa,
                                             test_accuracy, test_auc, test_kappa,
                                             train_positive_acc, train_negative_acc, test_positive_acc, test_negative_acc,
                                             train_f1, test_f1,
                                             train_precision, test_precision,
                                             train_recall, test_recall,
                                             x_train.shape[0], x_train.shape[1],
                                             len(y_test), sum(y_test==1), sum(y_test==0) ]).transpose()

                    # self.writeLog(row)
                    dfOutput = dfOutput.append(row)
                    inner_counter += 1
                outer_counter += 1

            # auc = roc_auc_score(y_test, y_scores)

            dfOutput = dfOutput.append(dfOutput.mean(axis=0), ignore_index=True)
            dfOutput = dfOutput.append(dfOutput.std(axis=0), ignore_index=True)


            header = ['outer_counter', 'inner_counter',
                                    'train_accuracy', 'train_auc', 'train_kappa',
                                    'test_accuracy', 'test_auc', 'test_kappa',
                                    'train_positive_acc','train_negative_acc', 'test_positive_acc','test_negative_acc',
                                    'train_f1', 'test_f1',
                                    'train_precision', 'test_precision',
                                    'train_recall', 'test_recall',
                                    'no_rows', 'no_features',
                                    'no_testingSamples', 'no_positiveTest', 'no_negativeTest']

            dfOutput.columns = header

            # count AUC
            # flattened = [val for sublist in test_probabilities for val in sublist]

            print("Stratified KFOLD: ")
            print("Mean accuracy: ", dfOutput['train_accuracy'].mean().round(2), " - ", dfOutput['test_accuracy'].mean().round(2))
            print("Mean AUC: ", dfOutput['train_auc'].mean().round(2), "- ", dfOutput['test_auc'].mean().round(2))
            print("Mean F1: ", dfOutput['train_f1'].mean().round(2), " - ", dfOutput['test_f1'].mean().round(2))
            print("Mean Precision: ", dfOutput['train_precision'].mean().round(2), " - ", dfOutput['test_precision'].mean().round(2)) # bulling
            print("Mean Recall: ", dfOutput['train_recall'].mean().round(2), " - ", dfOutput['test_recall'].mean().round(2))

            outputFile = os.path.join(self.classifierDir,testTag + "_dummy_stratifiedKfold_results"+datetime.datetime.now().strftime("_%Y%m%d_%H%M_") +".csv")
            dfOutput.to_csv(outputFile,sep=",", index=False)

            return