import os.path
import pandas as pd
import datetime

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

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
from sklearn.externals import joblib

# data supporting
from collections import Counter

class FinalizeModel:

    def TrainTestSaveFinalModel(self, PARAM_C, PARAM_Gamma):
        print("# Train the final model ")
        outer_counter=0
        inner_counter=0

        classifier = SVC(C=PARAM_C, kernel='rbf', gamma=PARAM_Gamma, probability=False)

        header = pd.DataFrame(data=["outer_counter", "inner_counter", "best_C", "best_G", "no_support_vectors",
                                    "train_accuracy", "train_auc", "train_kappa", "test_accuracy", "test_auc",
                                    "test_kappa", "train_f1", "test_f1", "train_precision","test_precision","train_recall","test_recall","startTime", "endTime", "no_vectors",
                                    "no_features"
                                    ]).transpose()
        self.writeLog(header)

        print('Resampled training dataset shape {}'.format(Counter(self.y_train)))
        print('Resampled dataset shape {}'.format(Counter(self.y_unseen)))

        print("Model training 1: binary classification")
        startTime = time.ctime()
        classifier.fit(self.x_train, self.y_train)
        #self.SaveFinalModel(classifier,"binary.pkl")

        no_support_vectors = len(classifier.support_vectors_) / float(len(self.y_train))

        # evaluations from the binary
        print("Binary: predict on unseen data")
        train_predict = classifier.predict(self.x_train)
        test_predict = classifier.predict(self.x_unseen)

        print("Binary: evaluate performance")
        train_accuracy = accuracy_score(train_predict, self.y_train)
        test_accuracy = accuracy_score(test_predict, self.y_unseen)

        train_kappa = cohen_kappa_score(self.y_train, train_predict)
        test_kappa = cohen_kappa_score(self.y_unseen, test_predict)

        target_names = ['intent', 'non-intent']  # todo: zkontrolovat, jestli intent je 1
        train_f1 = f1_score(self.y_train, train_predict, target_names)
        test_f1 = f1_score(self.y_unseen, test_predict, target_names)

        train_precision = precision_score(self.y_train, train_predict, target_names)
        test_precision = precision_score(self.y_unseen, test_predict, target_names)

        train_recall = recall_score(self.y_train, train_predict, target_names)
        test_recall = recall_score(self.y_unseen, test_predict, target_names)

        # evaluations from the probabilities
        print("Model training 2: Binary classification: probabilities")

        classifier = SVC(C=PARAM_C, kernel='rbf', gamma=PARAM_Gamma, probability=True)
        classifier.fit(self.x_train, self.y_train)
        endTime = time.ctime()
        # self.SaveFinalModel(classifier,'probability.pkl')

        print("Probabilities: predict")
        train_probabilities = classifier.predict_proba(self.x_train)
        test_probabilities = classifier.predict_proba(self.x_unseen)

        train_auc = roc_auc_score(y_true=self.y_train, y_score=train_probabilities[:, 1])
        test_auc = roc_auc_score(y_true=self.y_unseen, y_score=test_probabilities[:, 1])

        # debug here with simpler/faster train
        row = pd.DataFrame(data=[outer_counter,inner_counter, PARAM_C, PARAM_Gamma, no_support_vectors,
                                 train_accuracy, train_auc, train_kappa, test_accuracy, test_auc, test_kappa,
                                 train_f1, test_f1, train_precision, test_precision, train_recall, test_recall, startTime, endTime, self.x_train.shape[0],
                                 self.x_train.shape[1]]).transpose()

        self.writeLog(row)
        return

    def SaveFinalModel(self, clf, filename):
        print("# Save the final model ")
        ouputdir = self.CreateClassifierPath(filename)
        joblib.dump(clf, ouputdir)
        return

    def CreateClassifierPath(self, filename):

        if(os.path.isdir(self.classifierDir)==False):
            os.mkdir(self.classifierDir)

        outputFilename = os.path.join(self.classifierDir,filename)
        return outputFilename

    def FinalTrainTestRandomForest(self,outputDir):
        array_y_test = []
        array_y_test_scores = []

        # print("Upsampling with SMOTE")
        # sm = SMOTE(random_state=18)
        # x_res, y_res = sm.fit_sample(self.x_train, self.y_train)
        x_res= self.x_train
        y_res = self.y_train

        print("Binary training")
        classifier = RandomForestClassifier(n_estimators=897,
                                            max_features='sqrt',
                                            max_depth=10,
                                            min_samples_split=2,
                                            min_samples_leaf=4,
                                            bootstrap=True)

        classifier.fit(x_res, y_res)

        train_predict = classifier.predict(self.x_train)
        test_predict = classifier.predict(self.x_unseen)

        y_test_scores = classifier.predict_proba(self.x_unseen)
        array_y_test.append(self.y_unseen)
        array_y_test_scores.append(y_test_scores)

        print("Binary: predict")
        train_accuracy = accuracy_score(train_predict, self.y_train)
        test_accuracy = accuracy_score(test_predict, self.y_unseen)

        train_kappa = cohen_kappa_score(self.y_train, train_predict)
        test_kappa = cohen_kappa_score(self.y_unseen, test_predict)

        target_names = ['intent', 'non-intent']  # todo: zkontrolovat, jestli intent je 1
        train_f1 = f1_score(self.y_train, train_predict, target_names)
        test_f1 = f1_score(self.y_unseen, test_predict, target_names)
        train_precision = precision_score(self.y_train, train_predict, target_names)
        test_precision = precision_score(self.y_unseen, test_predict, target_names)
        train_recall = recall_score(self.y_train, train_predict, target_names)
        test_recall = recall_score(self.y_unseen, test_predict, target_names)

        print("Feature ranking:")
        # http: // scikit - learn.org / stable / auto_examples / ensemble / plot_forest_importances.html
        dfFeatureImportance = pd.DataFrame()
        importances = classifier.feature_importances_
        dfFeatureImportance = dfFeatureImportance.append([importances])
        # std = np.std([tree.feature_importances_ for tree in classifier.estimators_],
        #              axis=0)
        # indices = np.argsort(importances)[::-1]
        #
        # for f in range(x_res.shape[1]):
        #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        print("Binary training - probabilities true")
        classifier.probability = True  # we need this to produce the probabilities - ROC curve
        classifier.fit(x_res, y_res)
        # test_y_actual.extend(y_test)
        train_probabilities = classifier.predict_proba(self.x_train)
        test_probabilities = classifier.predict_proba(self.x_unseen)

        train_auc = roc_auc_score(y_true=self.y_train, y_score=train_probabilities[:, 1])
        test_auc = roc_auc_score(y_true=self.y_unseen, y_score=test_probabilities[:, 1])

        # debug here with simpler/faster train
        outer_counter = 0
        inner_counter = 0
        row = pd.DataFrame(data=[outer_counter, inner_counter,
                                 train_accuracy, train_auc, train_kappa,
                                 test_accuracy, test_auc, test_kappa,
                                 train_f1, test_f1,
                                 train_precision, test_precision,
                                 train_recall, test_recall,
                                 self.x_train.shape[0], self.x_train.shape[1],
                                 len(self.y_unseen), sum(self.y_unseen == 1), sum(self.y_unseen == 0)]).transpose()

        header = ['outer_counter', 'inner_counter',
                  'train_accuracy', 'train_auc', 'train_kappa',
                  'test_accuracy', 'test_auc', 'test_kappa',
                  'train_f1', 'test_f1',
                  'train_precision', 'test_precision',
                  'train_recall', 'test_recall',
                  'no_rows', 'no_features',
                  'no_testingSamples', 'no_positiveTest', 'no_negativeTest']

        row.columns = header

        outputFile = os.path.join(outputDir,
                                  "stratifiedKfold" + datetime.datetime.now().strftime("_%Y%m%d_%H%M_") + "_finalResult.csv")
        row.to_csv(outputFile, sep=",", index=False)

        dfFeatureImportance.columns = self.header
        outputFileImportance = os.path.join(outputDir, "stratifiedKfold_importances" + datetime.datetime.now().strftime(
            "_%Y%m%d_%H%M_") + "_finalResult.csv")
        dfFeatureImportance.to_csv(outputFileImportance, sep=",", index=False)
        return


    def FinalTrainTestRandomForest2(self,outputDir):
        array_y_test = []
        array_y_test_scores = []

        # print("Upsampling with SMOTE")
        sm = SMOTE(random_state=18)
        x_res, y_res = sm.fit_sample(self.x_train, self.y_train)
        # x_res= self.x_train
        # y_res = self.y_train

        print("Binary training")
        classifier = RandomForestClassifier(n_estimators=897,
                                            max_features='sqrt',
                                            max_depth=10,
                                            min_samples_split=2,
                                            min_samples_leaf=4,
                                            bootstrap=True,
                                            class_weight = 'balanced')

        classifier.fit(x_res, y_res)

        train_predict = classifier.predict(self.x_train)
        test_predict = classifier.predict(self.x_unseen)

        y_test_scores = classifier.predict_proba(self.x_unseen)
        array_y_test.append(self.y_unseen)
        array_y_test_scores.append(y_test_scores)

        print("Binary: predict")
        train_accuracy = accuracy_score(train_predict, self.y_train)
        test_accuracy = accuracy_score(test_predict, self.y_unseen)

        train_kappa = cohen_kappa_score(self.y_train, train_predict)
        test_kappa = cohen_kappa_score(self.y_unseen, test_predict)

        target_names = ['intent', 'non-intent']  # todo: zkontrolovat, jestli intent je 1
        train_f1 = f1_score(self.y_train, train_predict, target_names)
        test_f1 = f1_score(self.y_unseen, test_predict, target_names)
        train_precision = precision_score(self.y_train, train_predict, target_names)
        test_precision = precision_score(self.y_unseen, test_predict, target_names)
        train_recall = recall_score(self.y_train, train_predict, target_names)
        test_recall = recall_score(self.y_unseen, test_predict, target_names)

        positive_precision_rate =  len(self.y_unseen[self.y_unseen==1]) / float(len(test_predict[test_predict==1]))
        negative_precision_rate = len(self.y_unseen[self.y_unseen==0]) / float(len(test_predict[test_predict==0]))

        dfResults = pd.DataFrame()
        dfResults['true'] = self.y_unseen
        dfResults['predict'] = test_predict
        dfPositive = dfResults.loc[dfResults['true'] == 1]
        positive_acc = sum(dfPositive['predict']) / float(dfPositive['predict'].count())

        dfNegative = dfResults.loc[dfResults['true'] == 0]
        negative_acc = dfNegative['predict'].loc[dfNegative['predict'] == 0].count() / float(dfNegative['predict'].count())


        print("Feature ranking:")
        # http: // scikit - learn.org / stable / auto_examples / ensemble / plot_forest_importances.html
        dfFeatureImportance = pd.DataFrame()
        importances = classifier.feature_importances_
        dfFeatureImportance = dfFeatureImportance.append([importances])
        # std = np.std([tree.feature_importances_ for tree in classifier.estimators_],
        #              axis=0)
        # indices = np.argsort(importances)[::-1]
        #
        # for f in range(x_res.shape[1]):
        #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        print("Binary training - probabilities true")
        classifier.probability = True  # we need this to produce the probabilities - ROC curve
        classifier.fit(x_res, y_res)
        # test_y_actual.extend(y_test)
        train_probabilities = classifier.predict_proba(self.x_train)
        test_probabilities = classifier.predict_proba(self.x_unseen)

        train_auc = roc_auc_score(y_true=self.y_train, y_score=train_probabilities[:, 1])
        test_auc = roc_auc_score(y_true=self.y_unseen, y_score=test_probabilities[:, 1])

        # debug here with simpler/faster train
        outer_counter = 0
        inner_counter = 0
        row = pd.DataFrame(data=[outer_counter, inner_counter,
                                 train_accuracy, train_auc, train_kappa,
                                 test_accuracy, test_auc, test_kappa,
                                 train_f1, test_f1,
                                 train_precision, test_precision,
                                 train_recall, test_recall,
                                 self.x_train.shape[0], self.x_train.shape[1],
                                 len(self.y_unseen), sum(self.y_unseen == 1), sum(self.y_unseen == 0)]).transpose()

        header = ['outer_counter', 'inner_counter',
                  'train_accuracy', 'train_auc', 'train_kappa',
                  'test_accuracy', 'test_auc', 'test_kappa',
                  'train_f1', 'test_f1',
                  'train_precision', 'test_precision',
                  'train_recall', 'test_recall',
                  'no_rows', 'no_features',
                  'no_testingSamples', 'no_positiveTest', 'no_negativeTest']

        row.columns = header

        outputFile = os.path.join(outputDir,
                                  "stratifiedKfold" + datetime.datetime.now().strftime("_%Y%m%d_%H%M_") + "_finalResult.csv")
        row.to_csv(outputFile, sep=",", index=False)

        dfFeatureImportance.columns = self.header
        outputFileImportance = os.path.join(outputDir, "stratifiedKfold_importances" + datetime.datetime.now().strftime(
            "_%Y%m%d_%H%M_") + "_finalResult.csv")
        dfFeatureImportance.to_csv(outputFileImportance, sep=",", index=False)
        return


    def DummyTrainTest(self):

            classifier = DummyClassifier(strategy='stratified')
            classifier.fit(self.x_train, self.y_train)

            train_predict = classifier.predict(self.x_train)
            test_predict = classifier.predict(self.x_unseen)

            print("Binary: predict")
            train_accuracy = accuracy_score(train_predict, self.y_train)
            test_accuracy = accuracy_score(test_predict, self.y_unseen)

            train_kappa = cohen_kappa_score(self.y_train, train_predict)
            test_kappa = cohen_kappa_score(self.y_unseen, test_predict)

            target_names = ['intent', 'non-intent']  # todo: zkontrolovat, jestli intent je 1
            train_f1 = f1_score(self.y_train, train_predict, target_names)
            test_f1 = f1_score(self.y_unseen, test_predict, target_names)
            train_precision = precision_score(self.y_train, train_predict, target_names)
            test_precision = precision_score(self.y_unseen, test_predict, target_names)
            train_recall = recall_score(self.y_train, train_predict, target_names)
            test_recall = recall_score(self.y_unseen, test_predict, target_names)

            print("Binary training - probabilities true")
            train_probabilities = classifier.predict_proba(self.x_train)
            test_probabilities = classifier.predict_proba(self.x_unseen)

            train_auc = roc_auc_score(y_true=self.y_train, y_score=train_probabilities[:, 1])
            test_auc = roc_auc_score(y_true=self.y_unseen, y_score=test_probabilities[:, 1])

            # debug here with simpler/faster train
            outer_counter = -1
            inner_counter = -1
            row = pd.DataFrame(data=[outer_counter, inner_counter,
                                     train_accuracy, train_auc, train_kappa,
                                     test_accuracy, test_auc, test_kappa,
                                     train_f1, test_f1,
                                     train_precision, test_precision,
                                     train_recall, test_recall,
                                     self.x_train.shape[0], self.x_train.shape[1],
                                     len(self.y_unseen), sum(self.y_unseen == 1), sum(self.y_unseen == 0)]).transpose()

            header = ['outer_counter', 'inner_counter',
                      'train_accuracy', 'train_auc', 'train_kappa',
                      'test_accuracy', 'test_auc', 'test_kappa',
                      'train_f1', 'test_f1',
                      'train_precision', 'test_precision',
                      'train_recall', 'test_recall',
                      'no_rows', 'no_features',
                      'no_testingSamples', 'no_positiveTest', 'no_negativeTest']

            row.columns = header
            #
            # outputFile = os.path.join(outputDir,
            #                           "stratifiedKfold" + datetime.datetime.now().strftime(
            #                               "_%Y%m%d_%H%M_") + "_finalResult.csv")
            # row.to_csv(outputFile, sep=",", index=False)

            return