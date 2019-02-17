# date: 04/30/18
# author: Hana Vrzakova
# description: AOI-based features.

import numpy as np

class AoiFeatures:
    def __init__(self):
        self.aoi = []
        self.prefix = 0
        self.suffix = 0
        
        self.MiddleTile = []
        self.VisitedTilesCount = []
        self.UniqueVisitedTiles = []
        self.FirstAsLast = []
        
        #self.Strategy = [] - mostly vertical, mostly horizontal, others
        
        self.VerticalStrategy = []
        self.HorizontalStrategy = []
        self.Strategy = []

    def countStatistics(self,  seq,  prefix,  suffix,  aoi):
        
        #print seq[0]['gazePointX'],seq[1]['gazePointX'],seq[2]['gazePointX']  
        
        self.prefix = prefix
        self.suffix = suffix
        self.aoi = aoi

        self.MiddleTile.append(self.countMiddleTileNumber(seq))
        self.VisitedTilesCount.append(self.countNumberVisitedTiles(seq))

        self.UniqueVisitedTiles.append(self.countUniqueVisitedTiles(seq))

        self.FirstAsLast.append(self.sameFirstAsLast())

        self.Strategy.append(self.countStrategy())

        #self.VerticalStrategy.append(self.sameColumn())
        #self.HorizontalStrategy.append(self.sameRow())


    def allToString(self):
        self.strMiddleTile = np.array(self.MiddleTile).astype('|S10')
        self.strVisitedTilesCount = np.array(self.VisitedTilesCount).astype('|S10')
        self.strUniqueVisitedTiles = np.array(self.UniqueVisitedTiles).astype('|S10')
        self.strFirstAsLast = np.array(self.FirstAsLast).astype('|S10')
        self.Strategy = np.array(self.Strategy).astype('|S10')
        
        #self.strVerticalStrategy = np.array(self.VerticalStrategy).astype('|S10')
        #self.strHorizontalStrategy = np.array(self.HorizontalStrategy).astype('|S10')
        

    def countMiddleTileNumber(self, seq):
        if self.prefix <0 and self.suffix >0:
            tileNumber = self.aoi.findTileAoi(seq[self.prefix][0]['gazePointX'],
                                      seq[self.prefix][0]['gazePointY'])
        else:
            tileNumber = -1
        
        return tileNumber

    def countMiddleFixationPosition(self,seq):
        if self.prefix <0 and self.suffix >0:
            positionX = seq[self.prefix][0]['gazePointX']
            positionY = seq[self.prefix][0]['gazePointY']
            pos = [positionX,positionY]
        else:
            positionX = -1
            positionY = -1
            pos = [positionX,  positionY]
        return pos
    
    def countNumberVisitedTiles(self, seq):
        self.tiles = []
        for i in range(0,  len(seq)):
            tilePos = self.aoi.findTileAoi(seq[i][0]['gazePointX'],
                                                     seq[i][0]['gazePointY'])
            if(i == 0):
                self.tiles.append(tilePos) #hnus - do self.tiles se davaji uz unikatni
                
            if( (i >0) & (self.tiles[-1] != tilePos)):
                self.tiles.append(tilePos)

        #procistit 
        countVisitedTiles = len(self.tiles)
        return countVisitedTiles

    def countUniqueVisitedTiles(self, seq):
        uniqueTiles = []
        for fixation in seq:
            tilePos = self.aoi.findTileAoi(fixation[0]['gazePointX'],
                                      fixation[0]['gazePointY'])
            if(tilePos != 0): #0-okoli mimo 
                uniqueTiles.append(tilePos)
         
        uniqueVisited = np.unique(uniqueTiles)
        
        # same row, same column
        countUniqueTiles = len(uniqueVisited)
        return countUniqueTiles
        
    # jak osetrit jedno-polickove? + longer than 2 (one jump is not sign)
    # longer vs. shorter jumps
    def sameFirstAsLast(self):
        if(self.tiles[0] == self.tiles[-1]):
            return 1
        else:
            return 0
        
    def countStrategy(self):
        tiles = sorted(self.tiles)
        setTiles = set(tiles)
        
        column1 = set([1,  4,  7])
        column2 = set([2,  5,  8])
        column3 = set([3,  6,   9])
        
        row1 = set([1,  2 , 3])
        row2 = set([4,  5,  6])
        row3 = set([7,  8,  9])
        
        corner1 = set([1,  2,  4])
        corner2 = set([2,  3,  6])
        corner3 = set([4,  7,  8])
        corner4 = set([6,  8,  9])

        inner1 = set([2,  4,  5])
        inner2 = set([2,  5,  6])
        inner3 = set([4,  5,  8])
        inner4 = set([5,  6,  8])
        
        leftDiagonal1 = set([1,  5,  9])
        leftDiagonal2 = set([4,  8])
        leftDiagonal3 = set([2,  6])
        
        rightDiagonal1 = set([3,  5,  7])
        rightDiagonal2 = set([2,  4])
        rightDiagonal3 = set([6,  8])
        
        bow1 = set([1,  4,  5])
        bow2 = set([3,  5,  6])
        bow3 = set([4,  5,  7])
        bow4 = set([5,  6,  9])
        
        hourglass1 = set([1,  2,  5])
        hourglass2 = set([2,  3,  5])
        hourglass3 = set([5,  7,  8])
        hourglass4 = set([5,  8,  9])
    
        strategy = -1

 

        if (len(setTiles) == 1) and (tiles[0] != 0):
            strategy = 'A' #bodovky ale ne kolem dokola
        elif setTiles.issubset(row1) | setTiles.issubset(row2) | setTiles.issubset(row3):
            strategy = 'B'       # horizontal strategy
        elif setTiles.issubset(column1) | setTiles.issubset(column2) | setTiles.issubset(column3):
            strategy = 'C'        # vertical strategy
        
        elif setTiles.issubset(leftDiagonal1) | setTiles.issubset(leftDiagonal2) | setTiles.issubset(leftDiagonal3):
            strategy = 'D'
        elif setTiles.issubset(rightDiagonal1) | setTiles.issubset(rightDiagonal2) | setTiles.issubset(rightDiagonal3):
            strategy = 'E'    
            
        elif setTiles.issubset(inner1) | setTiles.issubset(inner2) | setTiles.issubset(inner3) | setTiles.issubset(inner4):
            strategy = 'F'
        elif setTiles.issubset(corner1) | setTiles.issubset(corner2) | setTiles.issubset(corner3) | setTiles.issubset(corner4):
            strategy = 'G'
            
        elif setTiles.issubset(hourglass1) | setTiles.issubset(hourglass2) | setTiles.issubset(hourglass3) | setTiles.issubset(hourglass4):
            strategy = 'H'    
        elif setTiles.issubset(bow1) | setTiles.issubset(bow2) | setTiles.issubset(bow3) | setTiles.issubset(bow4):
            strategy = 'I'
        
        else:
            strategy = 'X'
            
#        elif (len(setTiles) == 1) and (tiles[0] == 0):
#            strategy = 'J' # background only
#        elif (0) in setTiles:
#            strategy = 'K' # strategy has a jump outside of tiles (background = 0)
#                
#        else:
#            strategy = 'L'        # something else strategy
        
##        print "Sequence: ",  setTiles
##        print "Unique visited: ",  self.UniqueVisitedTiles[-1]
##        print "Strategy code: ",  strategy,  "\n"

        return strategy
        
#    def sameColumn(self):
#
#        tiles = sorted(self.tiles)
#        setTiles = set(tiles)
#        
##        print self.tiles
##        print tiles,  "\n"
#        
#        column1 = set([1,  4,  7])
#        column2 = set([2,  5,  8])
#        column3 = set([3, 6,   9])
#        
#        #osetrit bodovky
#        if setTiles.issubset(column1) | setTiles.issubset(column2) | setTiles.issubset(column3):
#            return 1
#        else:
#            return 0
#    
#    def sameRow(self):
#        tiles = sorted(self.tiles)
#        setTiles = set(tiles)
#
#        row1 = set([1,  2 , 3])
#        row2 = set([4,  5,  6])
#        row3 = set([7,  8,  9])
#        
#        #osetrit bodovky
#        if setTiles.issubset(row1) | setTiles.issubset(row2) | setTiles.issubset(row3):
#            return 1
#        else:
#            return 0
            
