#Class Direction

import numpy as np

class Direction:
    def __init__(self):
        self.data = []
        self.ndType = [('timestamp', int),
                       ('direction', 'S10'), 
                       ('moveLatency', int),  
                       ('intermoveLatency', int)]        

    def load(self, path):
        ndType = [('timestamp', 'S10'),
                       ('direction', 'S10'), 
                       ('moveLatency', int),  
                       ('intermoveLatency', int)]   

        dir1 = np.genfromtxt(path[0], delimiter = '\t',
                        autostrip = True,
                        skip_header = 2,
                        skip_footer = 2,
                        usecols = (3,6,11,12),
                        dtype = ndType
                        )

        dir2 = np.genfromtxt(path[1], delimiter = '\t',
                            autostrip = True,
                            skip_header = 2,
                            skip_footer = 2,
                            usecols = (3,6,11,12),
                            dtype = ndType
                            )

        dir3 = np.genfromtxt(path[2], delimiter = '\t',
                            autostrip = True,
                            skip_header = 2,
                            skip_footer = 2,
                            usecols = (3,6,11,12),
                            dtype = ndType
                            )
        data = []
        data = np.append(dir1, dir2, 0)
        data = np.append(data, dir3, 0)
        data = np.sort(data, order = 'timestamp')

        index = np.where(data['timestamp'] == '')
        data = np.delete(data, index[0])

        convertedTimestamps = self.convertToMilliseconds(data['timestamp'])
        delayedTimestamps = self.delayTimestamps(convertedTimestamps,data['intermoveLatency'])

        self.data = np.zeros(dtype = self.ndType, shape = data.shape)
        self.data['timestamp'] = delayedTimestamps
        self.data['direction'] = data['direction']
        self.data['moveLatency'] = data['moveLatency']
        self.data['intermoveLatency'] = data['intermoveLatency']

    def delayTimestamps(self, inputTimestamps, delay):
        timestamps =  inputTimestamps + delay
        return timestamps

    def convertToMilliseconds(self, inputList):
        outputList = np.zeros(inputList.size)

        for x in range(0, len(inputList)):
            timestamp = inputList[x]    
            splitted = timestamp.split(':')
            minutes = splitted[0]
            splitted2 = splitted[1].split('.') #bug na pul hodiny, clovek to zmeni a zapomene
            seconds = splitted2[0]
            microseconds = splitted2[1]

            microseconds = int(microseconds)
            seconds = int(seconds)
            minutes = int(minutes)
            microseconds1 = microseconds + (1000*seconds) + (1000*60*minutes)

            np.put(outputList,x,microseconds1)
            
        return outputList

    def alignTimestamps(self,event):
        indeces = np.searchsorted(event['timestamp'], self.data['timestamp'], side = 'left')
        deleteIndex = []

        for i in range(0, len(self.data)):
            if indeces[i] < len(event):
                #print self.data[i]['timestamp'], '-', event[indeces[i]]['timestamp']
                self.data[i]['timestamp'] = event[indeces[i]]['timestamp']
            else:
                deleteIndex.append(i)

        #vyhodime indexy, co neslo zarovnat
        self.data = np.delete(self.data, deleteIndex)
