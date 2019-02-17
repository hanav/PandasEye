import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull

class FeaturesEye:

    def CleanData(self,df):
        # remove invalid data
        df = df[df.validity > 0]
        # remove gazepoint[0,0]
        df = df[(df.gazex > 0) & (df.gazey > 0)]
        return df

    def EuclideanDistance(self,df):
        coords = pd.DataFrame(data=[df.gazex, df.gazey])
        diffs = np.diff(coords)
        distances = np.sqrt((diffs ** 2).sum(axis=0))
        distances = pd.Series(data=distances)
        distances = distances[distances != 0]
        return distances

    def VerticalDistance(self,df):
        diffs = np.diff(df.gazey)
        diffs = diffs[diffs != 0]
        return diffs

    def HorizontalDistance(self,df):
        diffs = np.diff(df.gazex)
        diffs = diffs[diffs != 0]
        return diffs

    def AreaCovered(self,df):
        if(not df.empty):
            hull = ConvexHull(df[['gazex', 'gazey']])
            hullArea = hull.area
        else:
            hullArea = np.nan

        return hullArea


    def CountVelocityFeatures(self, velocities):
        velMean = velocities.mean()
        velMedian = velocities.median()
        velVar = velocities.var()
        velMax = velocities.max()
        velMin   = velocities.min()

        return [velMean, velMedian, velVar, velMax, velMin]

    def CountFeatures(self, df):
        df = self.CleanData(df)

        dfDist = pd.DataFrame(
             [self.EuclideanDistance(df), self.VerticalDistance(df), self.HorizontalDistance(df)])

        dfDist = dfDist.transpose()
        dfDist.columns = ['EuclideanDistance','VerticalDistance','HorizontalDistance']

        if(dfDist.empty==False):
            meanDistances = dfDist.mean(axis=0, skipna=True)
            medianDistances = dfDist.mean(axis=0, skipna=True)
            varDistances = dfDist.var(axis=0, skipna=True)
            maxDistances = dfDist.max(axis=0, skipna=True)
            minDistances = dfDist.min(axis=0, skipna=True)
            sumDistances = dfDist.sum(axis=0, skipna=True)


            velocities = dfDist['EuclideanDistance'].diff()
            velocitiesVertical = dfDist['VerticalDistance'].diff()
            velocitiesHorizontal = dfDist['HorizontalDistance'].diff()

            featuresVelocities = self.CountVelocityFeatures(velocities)
            featuresVelocitiesVertical = self.CountVelocityFeatures(velocitiesVertical)
            featuresVelocitiesHorizontal = self.CountVelocityFeatures(velocitiesHorizontal)

        else:
            meanDistances = [np.nan,np.nan,np.nan]
            medianDistances = [np.nan,np.nan,np.nan]
            varDistances = [np.nan,np.nan,np.nan]
            maxDistances = [np.nan,np.nan,np.nan]
            minDistances = [np.nan,np.nan,np.nan]
            sumDistances = [np.nan,np.nan,np.nan]

            featuresVelocities = [np.nan,np.nan,np.nan,np.nan,np.nan]
            featuresVelocitiesVertical = [np.nan,np.nan,np.nan,np.nan,np.nan]
            featuresVelocitiesHorizontal = [np.nan,np.nan,np.nan,np.nan,np.nan]

        arr = np.concatenate((meanDistances, medianDistances, varDistances, maxDistances, minDistances,sumDistances, featuresVelocities, featuresVelocitiesVertical, featuresVelocitiesHorizontal))
        row = pd.DataFrame([arr])
        row.columns = ['Eye_EuclidDist_mean','Eye_VericalDist_mean','Eye_HorizDist_mean',
                                 'Eye_EuclidDist_median', 'Eye_VericalDist_median', 'Eye_HorizDist_median',
                                  'Eye_EuclidDist_var', 'Eye_VericalDist_var', 'Eye_HorizDist_var',
                                  'Eye_EuclidDist_max', 'Eye_VericalDist_max', 'Eye_HorizDist_max',
                                 'Eye_EuclidDist_min', 'Eye_VericalDist_min', 'Eye_HorizDist_min',
                                 'Eye_EuclidDist_sum', 'Eye_VericalDist_sum', 'Eye_HorizDist_sum',

                                 'Eye_Velocity_mean', 'Eye_Velocity_median', 'Eye_Velocity_var', 'Eye_Velocity_max','Eye_Velocity_min',
                                 'Eye_VerticalVelocity_mean', 'Eye_VerticalVelocity_median', 'Eye_VerticalVelocity_var', 'Eye_VerticalVelocity_max', 'Eye_VerticalVelocity_min',
                                 'Eye_HorizVelocity_mean', 'Eye_HorizVelocity_median', 'Eye_HorizVelocity_var', 'Eye_HorizVelocity_max', 'Eye_HorizVelocity_min'
                       ]

        return row

