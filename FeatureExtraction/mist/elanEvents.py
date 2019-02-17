# class elan events, to be continued
import numpy as np
from saveOutput import SaveOutput

class ElanEvents():
    
    def __init__(self,  path):
        self.path = path
        self.ndType = [('label', 'S15'), ('whiteSpace',int), ('timestamp', int)]
        self.data = []

    def separateElanEvents(self):
        self.data = np.genfromtxt(self.path, 
                delimiter = '\t',
                autostrip = True,
                usecols = (0, 1, 2),
                dtype = self.ndType 
               )

        delIndex = np.where(self.data['timestamp'] <= (5*60*1000)) #umazavame prvnich 5minut, protoze treba trial, nevalidni data
        self.data = np.delete(self.data, delIndex[0])

        return self.data
