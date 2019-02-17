# date: 04/30/17
# author: Hana Vrzakova
# description: Functions for parameter search

import os.path
import pandas as pd
import numpy as np
import scipy as sp

from sklearn.svm import SVC

# data preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer

# train-test-grid-search
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict, LeaveOneGroupOut


# performance evaluation
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score, auc, roc_auc_score, f1_score, precision_score, recall_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time


class ParameterSearch:

    # Exhaustive GridSearch and training only on the full set
    def GridSearchFull(self, note):

            print("# Tuning hyper-parameters ")
            cvKfold = KFold(n_splits=5, shuffle=True)
            C_range = [1,10,90,150,450, 650, 750]
            gamma_range = [0.000001, 0.00001, 0.0001, 0.001, 0.01]
            tuned_parameters = [{'gamma': gamma_range, 'C': C_range} ]

            print("Exhaustive Grid search")
            gridSearch = GridSearchCV(SVC(probability=True),tuned_parameters,  cv=cvKfold, scoring="accuracy")
            gridSearch.fit(self.x, self.y)

            print("Model parameters")
            classifier = gridSearch.best_estimator_

            best_score = gridSearch.best_score_
            best_C = gridSearch.best_params_['C']
            best_G = gridSearch.best_params_['gamma']
            no_support_vectors = len(gridSearch.best_estimator_.support_vectors_)

            print("Prediction performance")
            # evaluations from the binary predictions
            train_predict = classifier.predict(self.x)
            train_accuracy = accuracy_score(train_predict, self.y)
            train_kappa = cohen_kappa_score(train_predict, self.y)

            # evaluations from the probabilities
            classifier.probability = True  # we need this to produce the probabilities - ROC curve
            train_probabilities = classifier.predict_proba(self.x)
            train_auc = roc_auc_score(y_true=self.y, y_score=train_probabilities[:, 1])

            print("# Nested tuning hyper-parameters ")

            svr = SVC(kernel="rbf")
            C_range = [best_C]
            gamma_range = [best_G]
            tuned_parameters = [{'gamma': gamma_range, 'C': C_range} ]

            # Number of random trials
            NUM_TRIALS = 30
            nested_scores = np.zeros(NUM_TRIALS)
            nested_kappas = np.zeros(NUM_TRIALS)

            # Loop for each trial
            for i in range(NUM_TRIALS):
                # Choose cross-validation techniques for the inner and outer loops,
                # independently of the dataset.
                # E.g "LabelKFold", "LeaveOneOut", "LeaveOneLabelOut", etc.
                inner_cv = KFold(n_splits=3, shuffle=True, random_state=i)
                outer_cv = KFold(n_splits=3, shuffle=True, random_state=i)

                clf = GridSearchCV(estimator=svr, param_grid=tuned_parameters, cv=inner_cv)
                #nested_predict = cross_val_predict(clf, X=self.x, y=self.y, cv=outer_cv)
                #nested_kappas[i] = cohen_kappa_score(nested_predict, self.y)
                nested_score = cross_val_score(clf, X=self.x, y=self.y, cv=outer_cv)
                nested_scores[i] = nested_score.mean()

            acc_mean = nested_scores.mean()
            acc_std = nested_scores.std()

            self.outputResults = pd.DataFrame(data=[ time.ctime(), note, best_score, best_C, best_G, no_support_vectors,
                                                                    train_accuracy, train_kappa, train_auc, acc_mean, acc_std
            ]).transpose()

    def GridSearch(self):
            print("# Tuning hyper-parameters ")

            self.cv = KFold(n = len(self.y_train), n_folds=5, shuffle=True)
            #self.cv = LeaveOneLabelOut(self.user_train)
            #self.cv  =LabelKFold(self.y_train,n_folds=2)
            #self.cv = LeaveOneOut(len(self.y_train))

            # tuned_parameters = [{'kernel': ['linear'],
            #                      'C': [1, 10, 100, 1000]}]

            C_range = np.logspace(-2, 10, 13)
            gamma_range = np.logspace(-9, 3, 13)

            tuned_parameters = [{'gamma': gamma_range,'C': C_range}
                #{'kernel': ['rbf'], 'gamma': gamma_range,
                #                                 'C': C_range},
                #{'kernel': ['linear'], 'C': C_range}
                                                #{'kernel':['poly'],'degree':[3,4,5]}
                                ]

            # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
            #                 'C': [1, 10, 100, 1000]}]

            # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
            #                      'C': [1, 10, 100, 1000]},
            #                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
            #                       {'kernel':['poly'], 'degree':[1,2,3]}]

            gridSearch = GridSearchCV(SVC(),
                                tuned_parameters,  # RBF
                                cv= self.cv,  # 5-fold, LOLO
                                scoring="accuracy")
            gridSearch.fit(self.x_train, self.y_train)

            classifier = gridSearch.best_estimator_

            best_score = gridSearch.best_score_
            best_C = gridSearch.best_params_['C']
            best_G = gridSearch.best_params_['gamma']
            no_support_vectors = len(gridSearch.best_estimator_.support_vectors_)

            classifier.probability = True
            classifier.classes_

            train_predict = classifier.predict(self.x_train)
            test_predict = classifier.predict(self.x_unseen)
            train_accuracy = accuracy_score(train_predict, self.y_train)
            test_accuracy = accuracy_score(test_predict, self.y_unseen)

            train_probabilities = classifier.predict_proba(self.x_train)
            test_probabilities = classifier.predict_proba(self.x_unseen)
            train_auc = roc_auc_score(y_true=self.y_train, y_score=train_probabilities[:,1])
            test_auc = roc_auc_score(y_true=self.y_unseen, y_score=test_probabilities[:,1])

            train_kappa = cohen_kappa_score(self.y_train, train_probabilities[:,1])
            test_kappa = cohen_kappa_score(self.y_train, train_probabilities[:,1])


            row = pd.DataFrame(data=[ best_score, best_C, best_G, no_support_vectors,
                                                                    train_accuracy, train_auc, test_accuracy, test_auc
            ]).transpose()
            row

    # Exhaustive GridSearch and training and separate testing on 0.4 set
    def GridSearch2(self):
            print("Split into training and testing set")
            x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.4, random_state=0)

            print("# Tuning hyper-parameters ")
            cvKfold = KFold(n_splits=5, shuffle=True)
            #C_range = np.logspace(-2, 10, 13)
            #gamma_range = np.logspace(-9, 3, 13)

            #C_range = [1,10,100,1000,10000] #finer search
            #gamma_range = [0.0001, 0.001, 0.01, 0.1] # finer search

            C_range = [5,50,150,350,500,750]
            gamma_range = [0.0001, 0.001, 0.01, 0.1]

            tuned_parameters = [{'gamma': gamma_range, 'C': C_range} ]

            print("Exhaustive Grid search")
            gridSearch = GridSearchCV(SVC(probability=True),tuned_parameters,  cv=cvKfold, scoring="accuracy")
            gridSearch.fit(x_train, y_train)

            print("Model parameters")
            classifier = gridSearch.best_estimator_

            best_score = gridSearch.best_score_
            best_C = gridSearch.best_params_['C']
            best_G = gridSearch.best_params_['gamma']
            no_support_vectors = len(gridSearch.best_estimator_.support_vectors_)

    # evaluations from the binary
            print("Prediction performance")
            train_predict = classifier.predict(x_train)
            test_predict = classifier.predict(x_test)

            train_accuracy = accuracy_score(train_predict, y_train)
            test_accuracy = accuracy_score(test_predict, y_test)

            train_kappa = cohen_kappa_score(y_train, train_predict)
            test_kappa = cohen_kappa_score(y_test, test_predict)

            target_names=['neutral','emotion']
            confusionMatrix_stats = classification_report(y_train, train_predict, target_names=target_names)

    # evaluations from the probabilities
            classifier.probability = True  # we need this to produce the probabilities - ROC curve
            train_probabilities = classifier.predict_proba(x_train)
            test_probabilities = classifier.predict_proba(x_test)
            train_auc = roc_auc_score(y_true=y_train, y_score=train_probabilities[:, 1])
            test_auc = roc_auc_score(y_true=y_test, y_score=test_probabilities[:, 1])

            row = pd.DataFrame(data=[best_score, best_C, best_G, no_support_vectors,
                                     train_accuracy, train_auc, train_kappa, test_accuracy, test_auc, test_kappa
                                     ]).transpose()
            return row

    def TrainTest(self):
            self.classifier.probability = True
            self.classifier.classes_

            train_predict = self.classifier.predict(self.x_train)
            test_predict = self.classifier.predict(self.x_unseen)
            train_accuracy = accuracy_score(train_predict, self.y_train)
            test_accuracy = accuracy_score(test_predict, self.y_unseen)

            train_probabilities = self.classifier.predict_proba(self.x_train)
            test_probabilities = self.classifier.predict_proba(self.x_unseen)
            train_auc = roc_auc_score(y_true=self.y_train, y_score=train_probabilities)
            test_auc = roc_auc_score(y_true=self.y_unseen, y_score=test_probabilities)

    def SimpleTrain(self,PARAM_C, PARAM_Gamma): #stand-alone command

        header = pd.DataFrame(data=["best_C", "best_G", "no_support_vectors",
                                 "train_accuracy", "train_auc", "train_kappa", "test_accuracy", "test_auc", "test_kappa", "train_f1", "test_f1","startTime","endTime","no_vectors",
                                "no_features"
                                 ]).transpose()
        self.writeLog(header)


        print("Split into training and testing set")
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.4, random_state=0)

        print("# Setting up hyper-parameters ")
        param_C = PARAM_C
        param_Gamma = PARAM_Gamma

        print("Model training 1: binary classification")
        startTime = time.ctime()
        classifier = SVC(C = param_C, kernel='rbf', gamma=param_Gamma, probability=False)
        classifier.fit(x_train,y_train)

        print("Model parameters")
        #best_score = classifier.best_score_
        best_C = param_C
        best_G = param_Gamma
        no_support_vectors = len(classifier.support_vectors_)/float(len(y_train))

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
        train_precision = precision_score(y_train, train_predict, target_names)
        train_recall = recall_score(y_train, train_predict, target_names)
        train_f1  = f1_score(y_train, train_predict, target_names)

        test_precision = precision_score(y_test, test_predict, target_names)
        test_recall = recall_score(y_test, test_predict, target_names)
        test_f1  = f1_score(y_test, test_predict, target_names)

        # evaluations from the probabilities
        print("Model training 2: Binary classification: probabilities")

        classifier = SVC(C = param_C, kernel='rbf', gamma=param_Gamma, probability=True)
        classifier.fit(x_train,y_train)
        endTime = time.ctime()

        print("Probabilities: predict")
        train_probabilities = classifier.predict_proba(x_train)
        test_probabilities = classifier.predict_proba(x_test)

        train_auc = roc_auc_score(y_true=y_train, y_score=train_probabilities[:, 1])
        test_auc = roc_auc_score(y_true=y_test, y_score=test_probabilities[:, 1])

        row = pd.DataFrame(data=[best_C, best_G, no_support_vectors,
                                 train_accuracy, train_auc, train_kappa, test_accuracy, test_auc, test_kappa, train_f1, test_f1,startTime, endTime, self.x.shape[0], self.x.shape[1] ]).transpose()

        self.writeLog(row)

        return



