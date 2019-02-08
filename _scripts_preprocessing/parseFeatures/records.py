# Records
import os
import numpy as np
from pupildilations.pupillaryResponse import PupillaryResponse
from mist.saveOutput import SaveOutput

class RecordSet(PupillaryResponse, SaveOutput):
    def __init__(self):
        self.outDir = ""
        self.dirMeanbook = {}
        self.dirCodebook = {}

        self.tileMeanbook = {}
        self.tileCodebook = {}

        self.dirMean = []
        self.tileMean = []

        self.noDir = []

        self.prefix = 0
        self.suffix = 0
        self.step = 0

    def loadPrefixSuffix(self,prefix, suffix):
        self.prefix = prefix
        self.suffix = suffix
        self.step = self.prefix + self.suffix

    def loadOutputDir(self, dir):
        self.outDir = dir

    def loadDirectories(self,dir):
        dir = dir + "\\All\\"
        self.loadDirFile(dir,"Right")
        self.loadDirFile(dir,"Left")
        self.loadDirFile(dir,"Up")
        self.loadDirFile(dir,"Down")
        self.loadNoDirFile(dir,"No_dir_sequence")

    def loadDirFile(self,dir,key):
        data = self.loadFile(dir + key + ".txt")
        for i in range(0,len(data)):
            sample = data[i]
            self.dirMeanbook.setdefault(key,[]).append(sample)

    def loadNoDirFile(self, dir, key):
        data = self.loadFile(dir + key + ".txt")
        for i in range(0, len(data)):
            sample = data[i]
            self.noDir.append(sample)

    def loadTiles(self,dir):
        combinations = ['1_Right', '1_Down', '2_Left', '2_Right', '2_Down', '3_Left', '3_Down',
                        '4_Up', '4_Right', '4_Down', '5_Up', '5_Right', '5_Down', '5_Left', '6_Left', '6_Down', '6_Up',
                        '7_Right', '7_Up', '8_Up', '8_Right', '8_Left', '8_Up', '9_Left']
        dir = dir + "\\All\\"
        for key in combinations:
            self.loadTileFile(dir, key)

    def loadTileFile(self,dir,key):
        data = self.loadFile(dir + key + ".txt")

        if len(data) == self.step:
            self.tileMeanbook.setdefault(key,[]).append(data)
        else:
            for i in range(0,len(data)):
                sample = data[i]
                self.tileMeanbook.setdefault(key,[]).append(sample)

    def loadFile(self,path):
        data = []
        if os.path.exists(path):
            data = np.loadtxt(path)
        return data

    def createCodebooks(self):
        self.createDirCodebook()
        self.createTileCodebook()

    def saveClasses(self):
        dir = self.outDir

        dirMean = self.createDir(dir, "Mean")
        self.saveDictionary(dirMean, self.dirCodebook)
        self.saveDictionary(dirMean, self.tileCodebook)
        pathSample = self.createPathName(dirMean,"Dir_Mean")
        self.saveSample(pathSample, self.dirMean)
        pathSample = self.createPathName(dirMean, "Tile_Mean")
        self.saveSample(pathSample, self.tileMean)

        dirAll = self.createDir(dir, "All")
        self.saveDictionary(dirAll, self.dirMeanbook)
        self.saveDictionary(dirAll, self.tileMeanbook)
        pathSample = self.createPathName(dirAll, "No_dir_sequence")
        self.saveSample(pathSample, self.noDir)

        dirFig = self.createDir(dir, "Figures")
        self.saveImage(dirFig, self.dirCodebook, self.dirMean, "All_directions", self.step, self.prefix)
        self.saveImage(dirFig, self.tileCodebook, self.tileMean, "All_tiles", self.step, self.prefix)

