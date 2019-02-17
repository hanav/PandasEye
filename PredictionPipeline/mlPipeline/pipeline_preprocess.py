# date: 04/30/17
# author: Hana Vrzakova
# description: Functions for data preprocessing
# - feature scaling
# - imputing missing values
# - splitting to a training and testing set (stratified shuffling)
# - splitting to a training and testing set (person specified)
# - feature selection (percentile)
# - downsampling the majority class
# - upsampling the minority class

import pandas as pd
import numpy as np

from scipy.stats import itemfreq

# data preprocessing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import normalize

# train-test-grid-search
from sklearn.model_selection import train_test_split

# SMOTE
from imblearn.over_sampling import RandomOverSampler

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import RandomUnderSampler

#  feature selection
from sklearn.feature_selection import SelectFromModel, SelectKBest, SelectPercentile

# data supporting
from collections import Counter

class Preprocessing:

    def IrisPreprocessing(self):
        scaler = StandardScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        # self.x_train = normalize(self.x_train, norm='l2')

    def FeaturesPreprocessing(self, x_train, x_unseen):
        x_train, x_unseen = self.ReplaceMissingValues(x_train, x_unseen)
        x_train, x_unseen = self.ScaleFeatures(x_train, x_unseen)
        return x_train, x_unseen

    def SplitTrainTest(self, testRatio):
            self.x_train, self.x_unseen, self.y_train, self.y_unseen= train_test_split(self.x, self.y, test_size= testRatio,
                                                                                               random_state=18)
            self.y_train = np.array(self.y_train)
            self.y_unseen = np.array(self.y_unseen)
            return

    def SplitTrainTestShuffleStratified(self, testRatio):
            self.x_train, self.x_unseen, self.y_train, self.y_unseen= train_test_split(self.x, self.y, test_size= testRatio, shuffle=True, stratify= self.y,
                                                                                               random_state=18)
            self.y_train = np.array(self.y_train)
            self.y_unseen = np.array(self.y_unseen)
            return

    def SplitTrainTestUser(self, testRatio):
            users = np.unique(self.users)

            out_x_train = pd.DataFrame()
            out_x_unseen = pd.DataFrame()

            out_y_train = []
            out_y_unseen = []

            out_user_train = []
            out_user_unseen = []

            for i in range(0, len(users)):
                user = users[i]

                idx = np.where(self.users == user)
                x_user = self.x.iloc[idx[0]]
                y_user = self.y[idx[0]]
                user_user = self.users[idx[0]]

                x_train, x_unseen, y_train, y_unseen, user_train, user_unseen = train_test_split(
                    x_user, y_user, user_user, test_size=testRatio, random_state=18)

                out_x_train = out_x_train.append(x_train)
                out_x_unseen = out_x_unseen.append(x_unseen)

                out_y_train.extend(y_train)
                out_y_unseen.extend(y_unseen)

                out_user_train.extend(user_train)
                out_user_unseen.extend(user_unseen)

            self.x_train = out_x_train
            self.x_unseen = out_x_unseen


            self.y_train = np.array(out_y_train)
            self.y_unseen = np.array(out_y_unseen)

            self.user_train = np.array(out_user_train)
            self.user_unseen = np.array(out_user_unseen)

            return

    def ReplaceMissingValues(self, x_train, x_unseen):
            imputer = Imputer(axis=0, copy=False, missing_values='NaN', strategy="most_frequent").fit(x_train)
            x_train = imputer.transform(x_train)
            x_unseen = imputer.transform(x_unseen)
            return x_train, x_unseen

    def ScaleFeatures(self, x_train, x_unseen):
            scaler = StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_unseen = scaler.transform(x_unseen)
            return x_train, x_unseen

    def ScaleFeatures(self):
            scaler = StandardScaler().fit(self.x)
            self.x = scaler.transform(self.x)
            return

    def NormalizeFeatures(self):
            self.x_train = normalize(self.x_train, norm='l2')
            self.x_unseen = normalize(self.x_unseen, norm='l2')
            self.x_unseen

    def NormalizeAllFeatures(self):
        self.x = normalize(self.x, norm='l2')
        return

    def ImputeMissingData(self):
        print("# Fill missing values")

        if (np.any(np.isnan(self.x)) == True):
            print("We have some NaNs.")
            self.x = self.x.fillna(0, inplace=True)

        if (np.all(np.isfinite(self.x)) == False):
            print("We have some Infinites.")
            self.x = self.x.replace(-np.inf, np.nan)
            self.x = self.x.fillna(0)

        return

    def Normalize(self):
        print("# Normalize the feature set")
        scaler = StandardScaler().fit(self.x)  # todo: robustscaler
        self.x = scaler.transform(self.x)
        self.x = pd.DataFrame(data=self.x)
        return

    def Normalize01(self):
        normalizedArray = normalize(self.x)
        self.x = pd.DataFrame(data=normalizedArray)
        return

    def LogDistribution(self):
        print("Take natural logarithm")
        # http: // pandas.pydata.org / pandas - docs / version / 0.14.1 / generated / pandas.DataFrame.apply.html
        self.x = self.x.apply(np.log, axis=0)
        return

    def SelectFeatures(self, param_percentile):
        # https: // stackoverflow.com / questions / 15796247 / find - important - features - for -classification
        # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RandomizedLogisticRegression.html
        print("#Feature selection - Percentile: 35%")

        clf = SelectPercentile(percentile=param_percentile)
        new_x = clf.fit_transform(self.x, self.y)
        featureCount = self.x.shape

        feature_order = []
        for feature in new_x[0]:
            row = self.x.iloc[0]
            feature_ids = row == feature
            feature_selected = self.header[feature_ids][0]

            feature_order.append(feature_selected)

        print("Selected features are: ", feature_order)

        self.x = pd.DataFrame(new_x)
        return

    def DownsampleMajority(self):
        print('Original dataset shape {}'.format(Counter(self.y)))
        rus = RandomUnderSampler(return_indices=True)
        X_resampled, y_resampled, idx_resampled = rus.fit_sample(self.x, self.y)

        print('Resampled dataset shape {}'.format(Counter(y_resampled)))
        self.x = X_resampled
        self.y = y_resampled
        return

    def DownsampleMajorityTrain(self):
        print('Original dataset shape {}'.format(Counter(self.y_train)))
        # Apply the random under-sampling
        rus = RandomUnderSampler(return_indices=True)
        X_resampled, y_resampled, idx_resampled = rus.fit_sample(self.x_train, self.y_train)

        print('Resampled dataset shape {}'.format(Counter(y_resampled)))
        self.x_train = X_resampled
        self.y_train = y_resampled
        return

    def UpsampleMinority(self):
        print("# SMOTE upsampling")
        print('Original dataset shape {}'.format(Counter(self.y)))

        sm = SMOTE(random_state=42)
                # sm = SMOTEENN()
                # sm = SMOTETomek()
        x_res, y_res = sm.fit_sample(self.x, self.y)
                # self.PlotSMOTE(self.x, self.y, x_res, y_res)
                # self.PlotSMOTEPCA(self.x, self.y, x_res, y_res)
        self.x = x_res
        self.y = y_res
        print('Resampled dataset shape {}'.format(Counter(y_res)))

        return

    def UpsampleMinorityTrain(self):
        print("# SMOTE upsampling")
        print('Original dataset shape {}'.format(Counter(self.y_train)))

        sm = SMOTE(random_state=42)
                # sm = SMOTEENN()
                # sm = SMOTETomek()

        x_res, y_res = sm.fit_sample(self.x_train, self.y_train)
                # self.PlotSMOTE(self.x, self.y, x_res, y_res)
                # self.PlotSMOTEPCA(self.x, self.y, x_res, y_res)
        self.x_train = x_res
        self.y_train = y_res
        print('Resampled dataset shape {}'.format(Counter(y_res)))

        return


    def UpsampleMinorityPersonWhole(self):
        print("# SMOTE upsampling - whole")
        print('Complete dataset before: {}'.format(Counter(self.y)))

        users = np.unique(self.users)

        out_x = pd.DataFrame()
        out_y = []
        out_user = []

        for i in range(0, len(users)):
            user = users[i]

            print("SMOTE on user: ", user)

            idx = np.where(self.users == user)

            x_user = self.x.iloc[idx[0]]
            y_user = self.y[idx[0]]
            user_user = self.users[idx[0]]

            sm = SMOTETomek(random_state=42)
            # sm = SMOTE(random_state=42)
            x_res, y_res = sm.fit_sample(x_user, y_user)

            print('Original dataset shape {}'.format(Counter(y_user)))
            print('Resampled dataset shape {}'.format(Counter(y_res)))

            out_x = out_x.append(pd.DataFrame(x_res))
            out_y.extend(y_res)
            out_user.extend([user] * len(y_res))

        self.x = out_x.reset_index(drop=True)
        self.y = np.array(out_y)
        self.users = np.array(out_user)
        print('Complete dataset after: {}'.format(Counter(self.y)))
        return

    def UpsampleMinorityPerson(self):
        print("# SMOTE upsampling")

        users = np.unique(self.user_train)

        out_x_train = pd.DataFrame()
        out_y_train = []
        out_user_train = []

        for i in range(0,len(users)):
            user = users[i]

            print("SMOTE on user: ", user )

            idx = np.where(self.user_train == user)

            x_user_train = self.x_train.iloc[idx[0]]
            y_user_train = self.y_train[idx[0]]
            user_user = self.user_train[idx[0]]

            sm = SMOTETomek(random_state=42)
            #sm = SMOTE(random_state=42)
            x_res, y_res = sm.fit_sample(x_user_train, y_user_train)

            print('Original dataset shape {}'.format(Counter(y_user_train)))
            print('Resampled dataset shape {}'.format(Counter(y_res)))

            out_x_train = out_x_train.append(pd.DataFrame(x_res))
            out_y_train.extend(y_res)
            out_user_train.extend( [user] * len(y_res))

        self.x_train = out_x_train
        self.y_train = np.array(out_y_train)
        self.user_train = np.array(out_user_train)

        return

    def AddPolynomialFeatures(self):
                poly = PolynomialFeatures(2)
                polynomialFeatures = poly.fit_transform(self.x)
                self.x = pd.DataFrame(data=polynomialFeatures)
                return
