# data preprocessing
from sklearn.preprocessing import LabelEncoder
import numpy as np

class LoadData:

    def LoadIris(self):
        self.x_train = self.features.data
        self.y_train = self.features.target

    def LoadFeatures(self,labelColumn):
        '''Kdyz mame jenom jednu skupinu labels'''
        print "Loading features"
        self.y = self.features.iloc[:, labelColumn]
        self.x = self.features.iloc[:, (labelColumn+1):]
        le = LabelEncoder()
        le.fit(self.y)
        self.y = le.transform(self.y)
        return

    def LoadFeatures(self,featuresStart, featuresEnd, labelColumn):
        '''Kdyz mame jenom jednu skupinu labels'''
        print "Loading features"
        self.y = self.features.iloc[:, labelColumn]
        self.x = self.features.iloc[:, featuresStart:featuresEnd]

        # care about labels in  Y
        le = LabelEncoder()
        le.fit(self.y)
        self.y = le.transform(self.y)

        # care about X data frame
        # fill in Inf values with NAN values
        # fill in NAN values with mean of the whole thing - should be column-wise

        #self.x = self.x.replace([np.inf, -np.inf], np.nan)
        #

        if np.any(np.isnan(self.x)) == True:
            self.x.fillna(self.x.mean(), inplace=True, axis=0)

        if np.all(np.isfinite(self.x)) == False: # mame tam nejaky Inf nebo velky cislo
            infID = np.where(np.isfinite(self.x) == False)
            self.x = self.x.replace([np.inf, -np.inf], np.nan)
            self.x.fillna(self.x.mean(), inplace=True) # mean toho sloupce, bezva


        return


    def LoadFeatures(self,featuresStart, featuresEnd, labelColumn, userColumn):
        '''Kdyz mame jenom jednu skupinu labels'''
        self.features_start = featuresStart
        self.features_end = featuresEnd
        self.user_column = userColumn
        self.label_column = labelColumn

        print "Loading features"
        self.y = self.features.iloc[:, labelColumn]
        self.x = self.features.iloc[:, featuresStart:featuresEnd]
        self.users = self.features.iloc[:, userColumn]
        self.header = self.x.columns.values

        # care about labels in  Y
        le = LabelEncoder()
        le.fit(self.y)
        self.y = le.transform(self.y)

        # take care about labels in user's column
        le.fit(self.users)
        self.users = le.transform(self.users)

        # care about X data frame
        # fill in Inf values with NAN values
        # fill in NAN values with mean of the whole thing - should be column-wise

        #self.x = self.x.replace([np.inf, -np.inf], np.nan)
        #

        if np.any(np.isnan(self.x)) == True:
            self.x.fillna(self.x.mean(), inplace=True, axis=0)

        if np.all(np.isfinite(self.x)) == False: # mame tam nejaky Inf nebo velky cislo
            infID = np.where(np.isfinite(self.x) == False)
            self.x = self.x.replace([np.inf, -np.inf], np.nan)
            self.x.fillna(self.x.mean(), inplace=True) # mean toho sloupce, bezva
        return



    def LoadFeaturesHot(self,featuresStart, featuresEnd, labelColumn, userColumn):
        '''Kdyz mame jenom jednu skupinu labels'''
        self.features_start = featuresStart
        self.features_end = featuresEnd
        self.user_column = userColumn
        self.label_column = labelColumn

        print "Loading features"
        self.y = self.features.iloc[:, labelColumn]
        self.x = self.features.iloc[:, featuresStart:featuresEnd]
        self.users = self.features.iloc[:, userColumn]
        self.header = self.x.columns.values

        # care about labels in  Y
        le = LabelEncoder()
        le.fit(self.y)
        self.y = le.transform(self.y)

        # take care about labels in user's column
        le.fit(self.users)
        self.users = le.transform(self.users)
        self.features['users_encoded'] = self.users

        # outputdir = '/Users/icce/Dropbox (Personal)/_thesis_framework/_scripts_hoy/r_icmi/features_participants_check.csv'


        # care about X data frame
        # fill in Inf values with NAN values
        # fill in NAN values with mean of the whole thing - should be column-wise

        #self.x = self.x.replace([np.inf, -np.inf], np.nan)

        # Nad celyma datama

        if np.any(np.isnan(self.x)) == True:
            self.x.fillna(self.x.mean(), inplace=True, axis=0)

        if np.all(np.isfinite(self.x)) == False:  # mame tam nejaky Inf nebo velky cislo
            infID = np.where(np.isfinite(self.x) == False)
            self.x = self.x.replace([np.inf, -np.inf], np.nan)
            self.x.fillna(self.x.mean(), inplace=True)  # mean toho sloupce, bezva

        # for user in self.users:
        #
        #     subsetX = self.x.loc[user == self.users]
        #     subsetXMedian = subsetX.median(skipna=True)
        #     subsetXMean = subsetX.mean(skipna=True)
        #
        #
        #     if np.any(np.isnan(subsetX)) == True:
        #         subsetX.fillna(subsetXMedian, inplace=True, axis=0)
        #
        #     if np.all(np.isfinite(subsetX)) == False: # mame tam nejaky Inf nebo velky cislo
        #         infID = np.where(np.isfinite(subsetX) == False)
        #         subsetX = subsetX.replace([np.inf, -np.inf], np.nan)
        #         subsetX.fillna(subsetXMedian, inplace=True) # median toho sloupce, bezva
        #

        return


    def LoadFeaturesJournal(self, labelColumn):
        ''' Kdyz mame dalsi skupiny labels, tak tudyma'''
        print "Loading features and multiple labels"
        self.y = self.features.iloc[:, labelColumn]
        self.x = self.features.iloc[:, (labelColumn+1):]

        self.groups = self.features.participant
        le = LabelEncoder()
        le.fit(self.y)
        self.y = le.transform(self.y)
        return