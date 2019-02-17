# date: 04/30/17
# author: Hana Vrzakova
# description: Functions for importing and labeling data.

from sklearn.preprocessing import LabelEncoder
import numpy as np

class LoadData:

    def LoadIris(self):
        self.x_train = self.features.data
        self.y_train = self.features.target

    def LoadFeatures(self,labelColumn):
        print "Loading features"
        self.y = self.features.iloc[:, labelColumn]
        self.x = self.features.iloc[:, (labelColumn+1):]
        le = LabelEncoder()
        le.fit(self.y)
        self.y = le.transform(self.y)
        return

    def LoadFeatures(self,featuresStart, featuresEnd, labelColumn):
        print "Loading features"
        self.y = self.features.iloc[:, labelColumn]
        self.x = self.features.iloc[:, featuresStart:featuresEnd]

        le = LabelEncoder()
        le.fit(self.y)
        self.y = le.transform(self.y)

        if np.any(np.isnan(self.x)) == True:
            self.x.fillna(self.x.mean(), inplace=True, axis=0)

        if np.all(np.isfinite(self.x)) == False:
            infID = np.where(np.isfinite(self.x) == False)
            self.x = self.x.replace([np.inf, -np.inf], np.nan)
            self.x.fillna(self.x.mean(), inplace=True)
        return


    def LoadFeatures(self,featuresStart, featuresEnd, labelColumn, userColumn):
        self.features_start = featuresStart
        self.features_end = featuresEnd
        self.user_column = userColumn
        self.label_column = labelColumn

        print "Loading features"
        self.y = self.features.iloc[:, labelColumn]
        self.x = self.features.iloc[:, featuresStart:featuresEnd]
        self.users = self.features.iloc[:, userColumn]
        self.header = self.x.columns.values

        le = LabelEncoder()
        le.fit(self.y)
        self.y = le.transform(self.y)

        le.fit(self.users)
        self.users = le.transform(self.users)

        if np.any(np.isnan(self.x)) == True:
            self.x.fillna(self.x.mean(), inplace=True, axis=0)

        if np.all(np.isfinite(self.x)) == False:
            infID = np.where(np.isfinite(self.x) == False)
            self.x = self.x.replace([np.inf, -np.inf], np.nan)
            self.x.fillna(self.x.mean(), inplace=True)
        return



    def LoadFeaturesHot(self,featuresStart, featuresEnd, labelColumn, userColumn):
        # Paper 5
        self.features_start = featuresStart
        self.features_end = featuresEnd
        self.user_column = userColumn
        self.label_column = labelColumn

        print "Loading features"
        self.y = self.features.iloc[:, labelColumn]
        self.x = self.features.iloc[:, featuresStart:featuresEnd]
        self.users = self.features.iloc[:, userColumn]
        self.header = self.x.columns.values

        le = LabelEncoder()
        le.fit(self.y)
        self.y = le.transform(self.y)

        le.fit(self.users)
        self.users = le.transform(self.users)
        self.features['users_encoded'] = self.users

        if np.any(np.isnan(self.x)) == True:
            self.x.fillna(self.x.mean(), inplace=True, axis=0)

        if np.all(np.isfinite(self.x)) == False:  # mame tam nejaky Inf nebo velky cislo
            infID = np.where(np.isfinite(self.x) == False)
            self.x = self.x.replace([np.inf, -np.inf], np.nan)
            self.x.fillna(self.x.mean(), inplace=True)  # mean toho sloupce, bezva
        return


    def LoadFeaturesJournal(self, labelColumn):
        print "Loading features and multiple labels"
        self.y = self.features.iloc[:, labelColumn]
        self.x = self.features.iloc[:, (labelColumn+1):]

        self.groups = self.features.participant
        le = LabelEncoder()
        le.fit(self.y)
        self.y = le.transform(self.y)
        return