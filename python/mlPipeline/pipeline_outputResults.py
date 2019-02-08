import pandas as pd
import numpy as np
import os

class ResultsOutput:

    def writeLog(self,row):

        if not os.path.exists(self.outputDir):
            print("Creating directory:")
            os.makedirs(self.outputDir)
            resultsFile = os.path.join(self.outputDir,"all_results.csv")

        with open(self.resultsFile, 'a') as f:
            row.to_csv(f, header=False, index=False)
        return
