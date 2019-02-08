import pandas as pd
import numpy as np


# train-test-grid-search
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict, LeaveOneGroupOut

#classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# performance evaluation
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score,matthews_corrcoef, precision_score, recall_score

from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint



import time

class GridSearch:

    # def CrossValidationKFold(self, PARAM_C, PARAM_Gamma, NUM_TRIALS, KSPLITS):
    #     print("# Setting up hyper-parameters ")
    #
    #
    #     classifier = SVC(C=PARAM_C, kernel='rbf', gamma=PARAM_Gamma, probability=False)
    #
    #     header = pd.DataFrame(data=["outer_counter", "inner_counter", "best_C", "best_G", "no_support_vectors",
    #                                 "train_accuracy", "train_auc", "train_kappa", "test_accuracy", "test_auc",
    #                                 "test_kappa", "train_f1", "test_f1", "train_precision","test_precision","train_recall","test_recall","startTime", "endTime", "no_vectors",
    #                                 "no_features"
    #                                 ]).transpose()
    #     self.writeLog(header)
    #
    #     outer_counter = 0
    #     for j in range(0,NUM_TRIALS):
    #
    #         cv = KFold(n_splits=KSPLITS, shuffle=True)
    #         inner_counter = 0
    #         for train_index, test_index in cv.split(self.x_train):
    #             x_train, x_test = self.x_train[train_index], self.x_train[test_index]
    #             y_train, y_test = self.y_train[train_index], self.y_train[test_index]
    #
    #             print("Model training 1: binary classification")
    #             startTime = time.ctime()
    #             classifier.fit(x_train, y_train)
    #
    #             print("Model parameters")
    #             no_support_vectors = len(classifier.support_vectors_) / float(len(y_train))
    #
    #             # evaluations from the binary
    #             print("Binary: Training & Testing performance")
    #             train_predict = classifier.predict(x_train)
    #             test_predict = classifier.predict(x_test)
    #
    #             print("Binary: predict")
    #             train_accuracy = accuracy_score(train_predict, y_train)
    #             test_accuracy = accuracy_score(test_predict, y_test)
    #
    #             train_kappa = cohen_kappa_score(y_train, train_predict)
    #             test_kappa = cohen_kappa_score(y_test, test_predict)
    #
    #             target_names = ['intent', 'non-intent']  # todo: zkontrolovat, jestli intent je 1
    #             train_f1 = f1_score(y_train, train_predict, target_names)
    #             test_f1 = f1_score(y_test, test_predict, target_names)
    #             train_precision = precision_score(y_train, train_predict, target_names)
    #             test_precision = precision_score(y_test, test_predict, target_names)
    #             train_recall = recall_score(y_train, train_predict, target_names)
    #             test_recall = recall_score(y_test, test_predict, target_names)
    #
    #             # evaluations from the probabilities
    #             print("Model training 2: Binary classification: probabilities")
    #             # todo: tady udelame jenom jeden model - tento - a pridame threshold na 0.5
    #
    #             classifier = SVC(C=PARAM_C, kernel='rbf', gamma=PARAM_Gamma, probability=True)
    #             classifier.fit(x_train, y_train)
    #             endTime = time.ctime()
    #
    #             print("Probabilities: predict")
    #             train_probabilities = classifier.predict_proba(x_train)
    #             test_probabilities = classifier.predict_proba(x_test)
    #
    #             train_auc = roc_auc_score(y_true=y_train, y_score=train_probabilities[:, 1])
    #             test_auc = roc_auc_score(y_true=y_test, y_score=test_probabilities[:, 1])
    #
    #             # debug here with simpler/faster train
    #             row = pd.DataFrame(data=[outer_counter,inner_counter, PARAM_C, PARAM_Gamma, no_support_vectors,
    #                                      train_accuracy, train_auc, train_kappa, test_accuracy, test_auc, test_kappa,
    #                                      train_f1, test_f1, train_precision, test_precision, train_recall, test_recall, startTime, endTime, x_train.shape[0],
    #                                      x_train.shape[1]]).transpose()
    #
    #             self.writeLog(row)
    #             inner_counter +=1
    #         outer_counter +=1
    #
    #     return

    # def CrossValidationPerson(self, PARAM_C, PARAM_Gamma, NUM_TRIALS, KSPLITS):
    #     print("# Setting up hyper-parameters ")
    #
    #
    #     classifier = SVC(C=PARAM_C, kernel='rbf', gamma=PARAM_Gamma, probability=False)
    #
    #     header = pd.DataFrame(data=["outer_counter", "inner_counter", "best_C", "best_G", "no_support_vectors",
    #                                 "train_accuracy", "train_auc", "train_kappa", "test_accuracy", "test_auc",
    #                                 "test_kappa", "train_f1", "test_f1", "train_precision","test_precision","train_recall","test_recall","startTime", "endTime", "no_vectors",
    #                                 "no_features"
    #                                 ]).transpose()
    #     self.writeLog(header)
    #
    #     logo = LeaveOneGroupOut()
    #
    #     outer_counter = 0
    #     for j in range(0,NUM_TRIALS):
    #
    #         inner_counter = 0
    #
    #         for train_index, test_index in logo.split(self.x_train, self.y_train, self.user_train):
    #             x_train, x_test = self.x_train[train_index], self.x_train[test_index]
    #             y_train, y_test = self.y_train[train_index], self.y_train[test_index]
    #
    #             print("Model training 1: binary classification")
    #             startTime = time.ctime()
    #             classifier.fit(x_train, y_train)
    #
    #             print("Model parameters")
    #             no_support_vectors = len(classifier.support_vectors_) / float(len(y_train))
    #
    #             # evaluations from the binary
    #             print("Binary: Training & Testing performance")
    #             train_predict = classifier.predict(x_train)
    #             test_predict = classifier.predict(x_test)
    #
    #             print("Binary: predict")
    #             train_accuracy = accuracy_score(train_predict, y_train)
    #             test_accuracy = accuracy_score(test_predict, y_test)
    #
    #             train_kappa = cohen_kappa_score(y_train, train_predict)
    #             test_kappa = cohen_kappa_score(y_test, test_predict)
    #
    #             target_names = ['intent', 'non-intent']  # todo: zkontrolovat, jestli intent je 1
    #
    #             train_f1 = f1_score(y_train, train_predict, target_names)
    #             test_f1 = f1_score(y_test, test_predict, target_names)
    #             train_precision = precision_score(y_train, train_predict, target_names)
    #             test_precision = precision_score(y_test, test_predict, target_names)
    #             train_recall = recall_score(y_train, train_predict, target_names)
    #             test_recall = recall_score(y_test, test_predict, target_names)
    #
    #             # evaluations from the probabilities
    #             print("Model training 2: Binary classification: probabilities")
    #             # todo: tady udelame jenom jeden model - tento - a pridame threshold na 0.5
    #
    #             classifier = SVC(C=PARAM_C, kernel='rbf', gamma=PARAM_Gamma, probability=True)
    #             classifier.fit(x_train, y_train)
    #             endTime = time.ctime()
    #
    #             print("Probabilities: predict")
    #             train_probabilities = classifier.predict_proba(x_train)
    #             test_probabilities = classifier.predict_proba(x_test)
    #
    #             train_auc = roc_auc_score(y_true=y_train, y_score=train_probabilities[:, 1])
    #             test_auc = roc_auc_score(y_true=y_test, y_score=test_probabilities[:, 1])
    #
    #             # debug here with simpler/faster train
    #             row = pd.DataFrame(data=[outer_counter,inner_counter, PARAM_C, PARAM_Gamma, no_support_vectors,
    #                                      train_accuracy, train_auc, train_kappa, test_accuracy, test_auc, test_kappa,
    #                                      train_f1, test_f1, train_precision, test_precision, train_recall, test_recall, startTime, endTime, x_train.shape[0],
    #                                      x_train.shape[1]]).transpose()
    #
    #             self.writeLog(row)
    #             inner_counter +=1
    #         outer_counter +=1
    #
    #     return

    def GridSearchBruteForce(self):
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
            gridSearch.fit(self.x_train, self.y_train)

            trainLeftScore = gridSearch.best_score_
            #trainScore = gridSearch.score(x_train,y_train)
            testScore = gridSearch.score(self.x_test,self.y_test)

            clf = gridSearch.best_estimator_
            y_true, y_pred = self.y_test, clf.predict(self.x_test)
            report = classification_report(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)


    def comparisonBruteForceRandom(self):
        # build a classifier
        clf = RandomForestClassifier(n_estimators=20)

        # Utility function to report best scores
        def report(grid_scores, n_top=3):
            top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
            for i, score in enumerate(top_scores):
                print("Model with rank: {0}".format(i + 1))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    score.mean_validation_score,
                    np.std(score.cv_validation_scores)))
                print("Parameters: {0}".format(score.parameters))
                print("")

        # specify parameters and distributions to sample from
        param_dist = {"max_depth": [3, None],
                      "max_features": sp_randint(1, 11),
                      "min_samples_split": sp_randint(1, 11),
                      "min_samples_leaf": sp_randint(1, 11),
                      "bootstrap": [True, False],
                      "criterion": ["gini", "entropy"]}

        # run randomized search
        n_iter_search = 20
        random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                           n_iter=n_iter_search)

        start = time()
        random_search.fit(self.x_train, self.y_train)
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
              " parameter settings." % ((time() - start), n_iter_search))
        report(random_search.cv_results_)

        # use a full grid over all parameters
        param_grid = {"max_depth": [3, None],
                      "max_features": [1, 3, 10],
                      "min_samples_split": [1, 3, 10],
                      "min_samples_leaf": [1, 3, 10],
                      "bootstrap": [True, False],
                      "criterion": ["gini", "entropy"]}

        # run grid search
        grid_search = GridSearchCV(clf, param_grid=param_grid)
        start = time()
        grid_search.fit(self.x_train, self.y_train)

        print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
              % (time() - start, len(grid_search.cv_results_)))
        report(grid_search.cv_results_)


    def GridSearchRandom(self):
        #classifier
        clf = SVC()

        # ranges for hyperparameters
        tuned_parameters = { "C": [3,900],
                                            "gamma":[0.01,0.000001],
                                            "kernel":["rbf","linear"]
        }

        # random grid search
        n_iter_search = 4 #tady zmenit
        random_search = RandomizedSearchCV(clf, param_distributions=tuned_parameters, n_iter=n_iter_search)

        # start the search
        random_search.fit(self.x_train, self.y_train)

        # results
        params = random_search.best_params_

        return params


    def RandomGridSearch(self):
        #classifier
        print("Randomized grid search started...")
        clf = SVC()

        # ranges for hyperparameters
        tuned_parameters = { "C": [3,900],
                                            "gamma":[0.01,0.000001],
                                            "kernel":["rbf","linear"]
        }

        # random grid search
        n_iter_search = 4 #tady zmenit
        random_search = RandomizedSearchCV(clf, param_distributions=tuned_parameters, n_iter=n_iter_search)

        # start the search
        random_search.fit(self.x, self.y)

        # results
        params = random_search.best_params_

        return params


    def RandomGridSearchRandomForest(self):
        print("Randomized grid search started: Random Forest")

        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=200,
                                                    stop=2000, num=50)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=22)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]

        # #TEST
        # n_estimators = [200]
        # max_features = ['auto']
        # max_depth = [10,15]
        # min_samples_split = [2,4]
        # min_samples_leaf = [1]
        # bootstrap = [True,False]

        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        print(random_grid)

        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        rf = RandomForestClassifier()

        # Random search of parameters, using 3 fold crossvalidation,
        # search across 100 different combinations, and use all available cores

        # rf_random = RandomizedSearchCV(estimator=rf, scoring = 'roc_auc',
        #                                param_distributions=random_grid, n_iter=50, cv=3,
        #                                verbose=0, random_state=42, n_jobs=-1)

        # TEST
        rf_random = RandomizedSearchCV(estimator=rf,
                                       param_distributions=random_grid, n_iter=5, cv=2,
                                       verbose=0, random_state=42, n_jobs=-1,
                                       scoring='roc_auc')

        # Fit the random search model
        rf_random.fit(self.x_train, self.y_train)

        predictions = rf_random.predict(self.x_train)
        accuracy = accuracy_score(self.y_train, predictions)

        print("RandomGridSearch: RandomForest accuracy was ",  accuracy)

        bestParams = rf_random.best_params_
        return bestParams
