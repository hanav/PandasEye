#Class rawData - gaze data

import numpy as np

class Aoi:
    def __init__(self):
        self.data = []
        self.ndType =[('tileNumber',int), 
                      ('x1', int), 
                      ('y1',int), 
                      ('x3',int), 
                      ('y3',int)] 

    def load(self, path):
        ndType = [('tileNumber',int),
                    ('coord1', 'S10'), ('coord2', 'S10'),
                    ('coord3', 'S10'), ('coord4', 'S10')]


        data = np.genfromtxt(path, delimiter = '\t',
                    autostrip = True,
                    dtype = ndType)

        firstCoord = self.parseCoords(data['coord1']) 
        thirdCoord = self.parseCoords(data['coord3'])

        self.data = np.zeros(dtype = self.ndType, shape = data.shape)
        self.data['tileNumber'] = data['tileNumber']
        self.data['x1'] = firstCoord['x']
        self.data['y1'] = firstCoord['y']
        self.data['x3'] = thirdCoord['x']
        self.data['y3'] = thirdCoord['y']

    def parseCoords(self,inputList):
        parseType = [('x',int),('y',int)]
        parseShape = len(inputList)
        outputList = np.zeros(shape = parseShape, dtype = parseType)
        
        for i in range(0, len(inputList)):
            coord = inputList[i]
            splitted = coord.split(",")
            xCoord = splitted[0]
            yCoord = splitted[1]
            x = int(xCoord) 
            y = int(yCoord)
            xy = (x,y)
            np.put(outputList,i,xy)
        return outputList

    def estimateTile(self, sequence, prefix):
        lastFixation = self.findLastFixation(sequence, prefix)

        try:
            tile = self.findTileAoi(lastFixation['gazePointX'],lastFixation['gazePointY'])
        except:
            tile = 0
        return tile

    def findLastFixation(self,sequence, prefix):
        evtSample = sequence[prefix-1]

        index = np.where(sequence['fixationNumber'] == -1)
        fixations = np.delete(sequence,index[0])

        index = np.searchsorted(fixations['timestamp'],evtSample['timestamp'],side = 'left')
        try:
            outputSample = fixations[index]
        except:
            outputSample = []
        return outputSample

    def findTileAoi(self, x, y):
        tileNumber = 0

        if (x >= self.data[0]['x1'] and x <= self.data[0]['x3'] and 
            y >= self.data[0]['y1'] and y <= self.data[0]['y3']) :
            tileNumber = 1
        elif (x >= self.data[1]['x1'] and x <= self.data[1]['x3'] and 
              y >= self.data[1]['y1'] and y <= self.data[1]['y3']) :
            tileNumber = 2
        elif (x >= self.data[2]['x1'] and x <= self.data[2]['x3'] and 
              y >= self.data[2]['y1'] and y <= self.data[2]['y3']) :
            tileNumber = 3
        elif (x >= self.data[3]['x1'] and x <= self.data[3]['x3'] and 
              y >= self.data[3]['y1'] and y <= self.data[3]['y3']) :
            tileNumber = 4
        elif (x >= self.data[4]['x1'] and x <= self.data[4]['x3'] and 
              y >= self.data[4]['y1'] and y <= self.data[4]['y3']) :
            tileNumber = 5
        elif (x >= self.data[5]['x1'] and x <= self.data[5]['x3'] and 
              y >= self.data[5]['y1'] and y <= self.data[5]['y3']) :
            tileNumber = 6
        elif (x >= self.data[6]['x1'] and x <= self.data[6]['x3'] and 
              y >= self.data[6]['y1'] and y <= self.data[6]['y3']) :
            tileNumber = 7
        elif (x >= self.data[7]['x1'] and x <= self.data[7]['x3'] and 
              y >= self.data[7]['y1'] and y <= self.data[7]['y3']) :
            tileNumber = 8
        elif (x >= self.data[8]['x1'] and x <= self.data[8]['x3'] and 
              y >= self.data[8]['y1'] and y <= self.data[8]['y3']) :
            tileNumber = 9
        elif (x >= self.data[9]['x1'] and x <= self.data[9]['x3'] and 
              y >= self.data[9]['y1'] and y <= self.data[9]['y3']) :
            tileNumber = 10 # goal area
        elif (x >= self.data[10]['x1'] and x <= self.data[10]['x3'] and 
              y >= self.data[10]['y1'] and y <= self.data[10]['y3']) :
            tileNumber = 11 # life view area        


        return tileNumber
