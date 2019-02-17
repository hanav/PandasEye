#nepotrebny kod

#xmlReader
def checkPath(self, inputPath):
    try:
            isfile(inputPath)
    except Exception:
            print "Soubor na adrese '%s' neexistuje." %inputPath
            # quit()
    return path
    
#record.py

    def extractMeanFixationStarts(self):
        fixStarts = self.pupilsIntent.returnFixStarts()
        nofixStarts = self.pupilsNon.returnFixStarts()
        arr = fixStarts + nofixStarts
        meanStarts = np.floor(np.mean(arr, axis=0))
        return meanStarts
        
    def signalAnalysis(self):
        self.signals = SignalAnalysis()
        self.signals.load(self.fixSequences, self.nofixSequences)
        baseVar = self.rawData.returnBaseVar()
        self.signals.normalization(baseVar)
        self.signals.transformation()
        #self.signals.histograms()
        
    def cutDirSamples(self):
        for i in range(0,len(self.events)):
            moveEvent = self.events[i]
            index = np.searchsorted(self.rawData.data['timestamp'],moveEvent['timestamp'], side = 'left')

            evtIndex = index 
            beginIndex = evtIndex - self.prefix
            endIndex = evtIndex + self.suffix

            # sekvence dat
            sequence = self.rawData.data[beginIndex:endIndex]
            self.pupilResp.data.append(sequence)

            # udaj o pocatecni dlazdici
            #tilePosition = self.aoiBorders.estimateTile(sequence, self.prefix) 
            #self.pupilResp.tileNumber.append(tilePosition)

            # udaj o smeru
            #self.pupilResp.direction.append(moveEvent)
            
    def cutFreeSamples(self):
        for i in range(0, len(self.events) -1):
            moveEvent0 = self.events[i]
            index0 = np.searchsorted(self.rawData.data['timestamp'],moveEvent0['timestamp'], side = 'left')
            moveEvent1 = self.events[i+1]
            index1 = np.searchsorted(self.rawData.data['timestamp'],moveEvent1['timestamp'], side = 'left')

            interval = index1 - index0
            if interval > self.step:
                indices = np.arange(index0, index1, step=self.step)
                for i in range(0,len(indices)-1): 
                    sequence = self.rawData.data[indices[i]:indices[i+1]]
                    self.freeSamples.append(sequence)
                    
                    
    def cutNonFixations(self):
        for sequence in self.freeSamples:
            fixNumbers = self.fixations[np.where((self.fixations['timestamp'] >= sequence['timestamp'][0]) 
                                & (self.fixations['timestamp'] <= sequence['timestamp'][0 + self.prefix]))]['fixationNumber']

            if(len(fixNumbers) > 0):  
                lastFixNumber = fixNumbers[-1]
                idx = np.where(self.fixations['fixationNumber'] == lastFixNumber)
                lastFixation = self.fixations[idx]
                self.nonFixSequences.append(lastFixation)
            else:
                emptyList = []
                self.nonFixSequences.append(emptyList)
                
    def createClasses(self):
        self.pupilResp.sortSequences()
        self.pupilResp.createTileCodebook()
        self.pupilResp.createDirCodebook()
        
    #        
    def saveClasses(self):
        dir = self.recordPath.outPath

        dirMean = self.createDir(dir,"Mean")
        self.saveDictionary(dirMean, self.pupilResp.dirCodebook)
        self.saveDictionary(dirMean, self.pupilResp.tileCodebook)
        pathSample = self.createPathName(dirMean, "Dir_Mean")
        self.saveSample(pathSample, self.pupilResp.dirMean)
        pathSample = self.createPathName(dirMean, "Tile_Mean")
        self.saveSample(pathSample, self.pupilResp.tileMean)

        dirAll = self.createDir(dir,"All")
        self.saveDictionary(dirAll, self.pupilResp.dirMeanbook)
        self.saveDictionary(dirAll, self.pupilResp.tileMeanbook)
        pathSample = self.createPathName(dirAll, "No_dir_sequence")
        self.saveSample(pathSample, self.freeSamples)
        pathSample = self.createPathName(dirAll, "Whole_sequence")
        pupils = (self.rawData.data['pupilL'] + self.rawData.data['pupilR']) / 2
        self.saveSample(pathSample, pupils)
        
        dirFig = self.createDir(dir,"Figures")
        self.saveImage(dirFig, self.pupilResp.dirCodebook, self.pupilResp.dirMean, "All_directions", self.step, self.prefix)
        self.saveImage(dirFig, self.pupilResp.tileCodebook, self.pupilResp.tileMean, "All_tiles", self.step, self.prefix)
        self.saveImageSequence(dirFig, self.rawData.data,"Raw_data")
        
    def saveSequences(self, number, dir):
        pathSample = self.createPathName(dir, "samples_"+str(number))
        pathLabelled = self.createPathName(dir, "labelled_"+str(number))

        dirPupils = self.averagePupil(self.pupilResp.data)
        freePupils = self.averagePupil(self.freeSamples)

        labelDir = self.labelList(dirPupils,1)
        labelFree = self.labelList(freePupils,0)
        
    #
    def saveFixations(self, number, dir):
        pathSample = self.createPathName(dir, "pupils_"+str(number))

        intentPupils = self.pupilsIntent.pupils
        nonPupils = self.pupilsNon.pupils

        labelIntent = self.labelFixations(intentPupils,1)
        labelNon = self.labelFixations(nonPupils,0)
        allSamples = labelIntent + labelNon
        self.saveArray(pathSample, allSamples)  
        self.allSamples = np.concatenate((labelDir,labelFree),0)

        self.saveArray(pathSample,self.allSamples)
        
    #
    def saveImage(self,number, dir):
        dirPupils = self.averagePupil(self.pupilResp.data)
        nodirPupils = self.averagePupil(self.freeSamples)
        dirMean = self.pupilResp.countMean(dirPupils)
        nodirMean = self.pupilResp.countMean(nodirPupils)
        meanPupils =[]
        meanPupils.append(dirMean)
        meanPupils.append(nodirMean)

        self.saveImageArray(dir,meanPupils,"Mean"+str(number), self.step, self.prefix)  
         
    #
    def averagePupil(self,array):
        pupils = []
        for i in range(0,len(array)):
            pupil = (array[i]['pupilL'] + array[i]['pupilR']) / 2
            pupils.append(pupil)
        return pupils
        
    #
    def labelList(self,inputList,label):
        rows = np.size(inputList,0)
        labels = np.ndarray(shape=(rows,1), dtype=np.int)
        labels.fill(label)
        outputList = np.concatenate((labels, inputList),1)
        return outputList
        
    #
    def stringLabelList(self,inputList,label):
        rows = np.size(inputList,0)
        labels = np.ndarray(shape=(rows,1), dtype="S10")
        labels.fill(label)
        outputList = np.concatenate((labels, inputList),1)
        return outputList
        
    #
    def stringLabelFixations(self,inputList, label):
        outputList = []
        for i in range(0,len(inputList)):
            seq = np.append(label,inputList[i])
            outputList.append(seq)
        return outputList   
