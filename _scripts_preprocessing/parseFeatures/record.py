#Class Record
# Pozn. vypnuta normalizace

import numpy as np
import pandas as pd

from recordPath import RecordPath
from rawData import RawData
from mist.aoi import Aoi
from pupildilations.pupillaryResponse import PupillaryResponse
from mist.saveOutput import SaveOutput
from features.features import Features
from pupildilations.pupilSequences import PupilSequences
from mist.elanEvents import ElanEvents

class Record(SaveOutput):
    def __init__(self):
        self.rawData = RawData()
        self.fixations = []
        self.events = []
        self.aoiBorders = Aoi()

        self.recordPath = RecordPath()
        self.outputDir = []
        self.pupilResp = PupillaryResponse()

        self.prefix = 0 #pocet fixaci pred eventem
        self.suffix = 0 #pocet fixaci po eventu
        

        self.eventFixation = [] #cisla fixaci podilejicich se na event sequencich
        self.fixationSequencesNumbers = []
        self.fixationSequence = []     #vsechny samples obsazene v sequencich (kvuli pupilam)
        self.fixationArray = []             #samples pouze uvnitr fixaci - k cemu? skoro se nepouziva
        
        self.eventSequence = []
        self.eventArray = []
        
        self.nonSequence = []
        self.nonArray = []
        
        self.allSamples = []

        self.ndType = [('timestamp',int), 
                       ('pupilL', np.float32),  ('validityL',int),
                       ('pupilR', np.float32), ('validityR',int),
                       ('fixationNumber', int),
                       ('gazePointX',int), ('gazePointY',int),
                       ('event','S15'), 
                        ('rawX', int), ('rawY', int)]

# nacteme, kolik fixaci pred a po budeme preparovat
# kontrola, ze jsou hodnoty v poradku
    def loadPrefixSuffix(self,prefix, suffix):
        
        if prefix >= suffix:
            print "Prefix cannot be bigger than suffix (no time travelling here).\n"
            quit()
        self.prefix = prefix
        self.suffix = suffix

#nacteme, odkud budeme sosat data
    def loadPath(self, xmlPaths):
        self.recordPath = xmlPaths

#
    def loadData(self):
        self.rawData.load(self.recordPath.cmdPath)

        if(not self.recordPath.elanPath):
            elanEvents = ElanEvents(self.recordPath.elanPath)
            self.events = elanEvents.separateElanEvents()
        else:
            self.events = self.rawData.separateEvents()
            
        self.fixations = self.rawData.separateFixations()
        self.aoiBorders.load(self.recordPath.aoiPath)

    #def loadData8P(self, cmdPath, aoiPath, eventFilePath):
    def loadData8P(self, cmdPath, aoiPath, eventFilePath):
        self.rawData.loadDataFrame8P(cmdPath)
        self.cmdData = self.rawData.data
        self.events = pd.read_csv(eventFilePath, sep=",") #todo: nenacita to moji eventFixation
        #self.events = self.rawData.separateEvents8P()
        self.fixations = self.rawData.separateFixations8P()
        self.aoiBorders = pd.read_csv(aoiPath, skiprows=0, sep="\t")

        # relict when producing event fixations
        #self.events.to_csv(path_or_buf = eventFilePath , sep=",", index=False)

    def exportEvents8P(self, cmdPath, aoiPath, eventFilePath):
        self.rawData.loadDataFrame8PEvents(cmdPath)
        self.events = self.rawData.separateEvents8P()
        self.events.to_csv(path_or_buf = eventFilePath , sep=",", index=False)

    def loadDataWTP(self, cmdPath, aoiPath, eventFilePath):
        self.rawData.loadDataFrameWTP(cmdPath)
        self.cmdData = self.rawData.data
        #self.events = pd.read_csv(eventFilePath, sep=",") #todo: nenacita to moji evenFixation
        self.events = self.rawData.separateEventsWTP()
        self.fixations = self.rawData.separateFixations8P()
        self.aoiBorders = pd.read_csv(aoiPath, skiprows=0, sep="\t")

        # relict when producing event fixations
        self.events.to_csv(path_or_buf = eventFilePath , sep=",", index=False)
    #
# vymazeme eventy a spocitame baseline
    def prepareRawData(self):
        self.rawData.eraseEvents()
        self.rawData.countBaseline()
#
# vypreparujeme event a non-intention fixace
    def cutFixations(self):
        #print "cutFixationSequences:..."
        self.cutFixationSequences()

        #print "extractEventNumbers:..."
        self.extractEventFixationNumbers()
        
        #print "saveEventFixations:..." - tady se to musi upravit
        self.saveEventFixations()

    def cutFixations8P(self):
        fixationNumbers = pd.Series(self.fixations['fixationNumber'].unique())
        fixationNumbers = fixationNumbers.dropna()
        fixationNumbers = fixationNumbers.astype(int)

        features = Features()
        outputDf = pd.DataFrame()

        # generate sequences & check they are in range and continuous
        for fixationNumber in fixationNumbers:
            print "Fixation: ", fixationNumber
            idxSeq = self.returnSeqFixationNumbers(fixationNumber, self.prefix, self.suffix)

            if np.unique(np.diff(idxSeq)) != 1:
                print "Skips in sequence: ", idxSeq
                continue

            #todo: tohle nefunguje spravne - 47 neni ve fixationNumbers a stejne to nerekne
            results = [any(fixationNumbers == x) for x in idxSeq]
            #results = all(check)

            # results = [False for number in idxSeq if (number in fixationNumbers) == False]
            if (False in results):
                print "Not existing in the dataset:",idxSeq
                continue



            seqDf = self.cutFixationSequencePandas(idxSeq)
            dfSequenceRaw = self.cutFixationSequenceRawPandas(idxSeq)

            if len(seqDf['fixationNumber'].unique()) != (self.suffix-self.prefix+1):
                print "Sequence lenght is wrong:", len(seqDf['fixationNumber'].unique())
                continue

            features.load(seqDf,self.aoiBorders,  self.prefix,  self.suffix)
            features.loadRawPandas(dfSequenceRaw)

            row = features.extractFeaturesPandas()
            rowPupils = features.extractFeaturesPupilPandas()
            row = pd.concat([row, rowPupils], axis=1)

            # is event embedded in the fixation sequence?
            actionTag = 0
            #todo: tady chceme primo fixace pred eventem, timestampy v tom bohuzel delaji bordel
            
            eventsInBetween = self.events['eventFixation'].between(idxSeq[0], idxSeq[-1])

            if eventsInBetween.any():

                eventIdx = eventsInBetween[eventsInBetween==True].index[0] #index eventu ktery je v sequenci
                eventFixation = self.events['eventFixation'].loc[eventIdx]

                #tohle je hezky, ale rozbiji to hledani
                # if math.isnan(fixationsBeforeEvent.iloc[-1]) == True: #event je mimo fixaci, proto chci hned tu prvni pred eventem
                #     firstFixationBeforeEvent = fixationsBeforeEventUnique[-1]
                # else: # event je vnoreny ve fixaci, proto chci jeste tu predchozi pred eventem
                #     #print "event je ve fixaci"
                #     try:
                #         firstFixationBeforeEvent = fixationsBeforeEventUnique[-2]
                #     except:
                #         # this is the first fixation in the sequence, there is no other fixation before this one
                #         # we won't have any prior fixation, so we set up this particular embedded fixation
                #         #print "First fixation of the sequence & event embdedded in this fixation"
                #         firstFixationBeforeEvent = fixationsBeforeEventUnique[-1]

                # try it the stupid old way bellow.

                #firstFixationBeforeEventTimestamp = seqDf['timestamp'].loc[ seqDf['fixationNumber'] ==  firstFixationBeforeEvent]
                #firstFixationBeforeEventTimestamp = firstFixationBeforeEventTimestamp.iloc[-1]
                #print "event", eventTimestamp, "- fixation", firstFixationBeforeEvent, "-", firstFixationBeforeEventTimestamp
                #print idxSeq[self.suffix], "-", firstFixationBeforeEvent

                #if idxSeq[self.suffix] == firstFixationBeforeEvent:
                print "Sequence: ", idxSeq, "- fixation before event: ", eventFixation
                if idxSeq[self.suffix] == eventFixation:
                    print "- right spot"
                    actionTag = 1
                else:
                    print "- not exactly right spot but in the sequence: "
                    actionTag = 0.5

            row['action'] = actionTag

            outputDf = outputDf.append(row)

        self.events['eventFixation'].count() #369
        outputDf['action'].loc[outputDf['action'] == 1].count() #113

        return outputDf



# Tady se realizuje prekryv
    def cutFixationSequences(self):
        fixationNumbers = np.array(self.returnFixationNumbers())  # sekvence s cisly fixaci    

        for fixationNumber in fixationNumbers:
            idxSeq = self.returnSeqFixationNumbers(fixationNumber,  self.prefix,  self.suffix)

            if self.isExistingSequence(idxSeq)==False: #pokud by se tam mela vyskytnout neplatna sekvence, preskoc ji
                continue

            partSequence = self.cutFixationSequence(idxSeq)
            partArray = self.cutFixationArray(idxSeq)
            
            self.fixationSequence.append(partSequence)
            self.fixationArray.append(partArray)
            self.fixationSequencesNumbers.append( idxSeq)
        
#
# najdeme a ulozime cisla eventovych fixaci
    def extractEventFixationNumbers(self):
        for event in self.events:
            index = np.where(self.fixations['timestamp'] < event['timestamp']) # puvodne tu bylo <=, kvuli GA interakci, pro mys nam staci <           
            idx = index[0][-1]
            fixationNumber = self.fixations[idx]['fixationNumber']
            
            #actionDelta = event['timestamp'] - self.fixations[idx]['timestamp']
            #print   actionDelta
            #if(actionDelta < 0) :
            #    print "Action time: ",  event['timestamp'],  "- Fixation end at: ",  self.fixations[idx]['timestamp']
            self.eventFixation.append(fixationNumber)

#
# ulozime eventove sekvence fixaci (prekryv)
# tady se to musi upravit, protoze v sekvenci vubec nemusi byt eventova fixace
# navic muze byt zaporny prefix, ktery to posle do kytek:)

    def saveEventFixations(self):
        idxArray = []
        
        fixationSequences = np.array(self.fixationSequencesNumbers)
        
        for i in range(0, len(fixationSequences)): # pro kazdou fixacni sekvenci se koukni, jestli..

            if(self.checkOverlay(self.fixationArray[i]) == 1):
                continue
            
            ##################################
            
            eventFixation = fixationSequences[i, 0]
  
            if self.prefix <0 and self.suffix < 0:
                eventFixation =  fixationSequences[i, 0] - self.prefix
                # protoze prefix je zaporny, tak toto pricte vsechny n-fixaci k nule, kde je event
                #eventFixation = fixationSequences[i, 0-(self.prefix)]
                # prohledavame na zaporne ose,  eventova fixace neni soucasti sekvence
                # eventova pozice bude vepredu, pricte se suffix
                
            elif self.prefix >0 and self.suffix >0 :
                 eventFixation =  fixationSequences[i, 0] - self.prefix
                 # protoze prefix je kladny, odecte se n-fixaci k nule, kde je event
                 #eventFixation = fixationSequences[i, 0-(self.prefix)]
                # prohledavame na kladne ose, eventova fixace neni soucasti sekvence
                # eventova pozice bude vzadu, odecte se prefix
                
                
            if self.prefix <0 and self.suffix >0:
                eventFixation = fixationSequences[i, 0-(self.prefix)]




            if ( eventFixation in self.eventFixation)==True:   #na pozici, kde ma byt eventova fixace, je eventova fixace
                self.eventArray.append(self.fixationArray[i])                                #pokud ano, pridej je do eventArray a eventSequence
                self.eventSequence.append(self.fixationSequence[i])
                continue
                #print "event detected"
                #print fixationSequences[i], "-",  eventFixation
            
            else:
                self.nonArray.append(self.fixationArray[i])                                  # pokud ne, pridej je na druhou hromadku
                self.nonSequence.append(self.fixationSequence[i])
                
        print len(self.eventArray), "-",   len(self.nonArray)

#
# tady se spocita fixacni sekvence z prefixu a suffixu
    def returnSeqFixationNumbers(self, fixationNumber,  prefix,  suffix):
            fixBegin = fixationNumber + prefix + 1 #zmena tady, puvodne -prefix, protoze byl kladny
            fixEnd = fixationNumber + suffix + 1
            idxSeq = range(fixBegin,fixEnd+1) # range vraci sekvenci o jedno kratsi
            return idxSeq

# 
# podivame se, jestli vsechny fixace v budouci sekvenci jsou obsazena
# v fixations, ze kterych jsme vysekali ty neplatne fixace
# tzn. jsou vsechny fixace validni?
    def isExistingSequence(self,idxSeq):
        results = [False for number in idxSeq if (number in self.fixations['fixationNumber'])==False]
        if (False in results):
            return False
        return True
#
# Najde timestamp prvni fixace a posledni timestamp posledni fixace a Vybere vsechny sample uvnitr sekvence fixaci
    def cutFixationSequence(self, idxSequence):
        sequence = []
        idxBegin = np.where(self.rawData.data['fixationNumber'] == idxSequence[0])
        idxEnd = np.where(self.rawData.data['fixationNumber'] == idxSequence[-1])
        sequence = self.rawData.data[idxBegin[0][0]:(idxEnd[0][-1]+1)]
        return sequence

    def cutFixationSequencePandas(self, idxSequence):
        startFixation = idxSequence[0]
        endFixation = idxSequence[-1]
        #tohle je zajimave chovani, kdyz mu rekneme at vybere fixace 2 - 5, tak zachodi vsechny 'nan' fixace mezi nimi
        sequence = self.cmdData.loc[ (self.cmdData['fixationNumber'] >=startFixation) & (self.cmdData['fixationNumber'] <= endFixation) ]
        return sequence

    def cutFixationSequenceRawPandas(self, idxSequence):
        startFixation = idxSequence[0]
        endFixation = idxSequence[-1]

        startTimestamp = self.cmdData.loc[ self.cmdData['fixationNumber'] == startFixation,'timestamp']
        endTimestamp = self.cmdData.loc[ self.cmdData['fixationNumber'] == endFixation,'timestamp']

        # a proto si tady vyrezeme vlastni raw - df mezi timestampa
        sequence = self.cmdData.loc[ (self.cmdData['timestamp'] >=startTimestamp.iloc[0]) & (self.cmdData['timestamp'] <= endTimestamp.iloc[-1]) ]
        return sequence


#
# Vybere samply pouze uvnitr fixaci
    def cutFixationArray(self, idxSequence):
        fixationArray = []
        for i in range(idxSequence[0],(idxSequence[-1]+1)): 
            idx = np.where(self.fixations['fixationNumber'] == i)
            fixationArray.append(self.fixations[idx])
        return fixationArray
        

#
    def extractFeatures(self):
        # self.featuresIntent = Features()
        # self.featuresIntent.load(self.eventArray,self.aoiBorders,  self.prefix,  self.suffix)
        # self.featuresIntent.extractFeatures()
        #
        # self.featuresNon = Features()
        # self.featuresNon.load(self.nonArray,self.aoiBorders,  self.prefix,  self.suffix)
        # self.featuresNon.extractFeatures()


        # pupillary responses are here:
        self.pupilsIntent = PupilSequences()
        self.pupilsIntent.load(self.eventSequence)
        self.pupilsIntent.extractPupils()

        #self.pupilsNon = PupilSequences()
        #self.pupilsNon.load(self.nonSequence)
        #self.pupilsNon.extractPupils()
#
# vrati poradova cisla fixaci
    def returnFixationNumbers(self):    
        fixNumbers = []
        
        for row in self.fixations:
            fixNumber = row['fixationNumber']
            if ((fixNumber in fixNumbers) == False):
                fixNumbers.append(fixNumber)
        return fixNumbers
        
#
    def labelFixations(self,inputList, label):
        outputList = []
        for i in range(0,len(inputList)):
            seq = np.append(label,inputList[i])
            outputList.append(seq)
        return outputList      
        
# tato funkce se pouziva na ukladani
    def returnFeatures(self):
        
        intFeatures = self.featuresIntent.returnAllFeatures()   
        nonFeatures = self.featuresNon.returnAllFeatures()
        
        labelIntent = self.labelFixations(intFeatures,"intent")
        labelNon = self.labelFixations(nonFeatures,"non")

        features = labelIntent + labelNon 

        return features

# ----------------------------------------------------------------------------------------------
    def returnPupils(self):
        labelIntent = self.labelFixations(self.pupilsIntent.returnPupils(),"intent")
        labelNon = self.labelFixations(self.pupilsNon.returnPupils(),"non")
        pupils = labelIntent + labelNon
        return pupils

#
    def returnDiff1(self):
        diff1 = self.pupilsIntent.returnDiff1() + self.pupilsNon.returnDiff1()
        return diff1
#
    def returnDiff2(self):
        diff2 = self.pupilsIntent.returnDiff2() + self.pupilsNon.returnDiff2()
        return diff2
#
    def returnSpectrum(self):
        spectrum = self.pupilsIntent.returnSpectrum() + self.pupilsNon.returnSpectrum()
        return spectrum
#
    def returnCepstrum(self):
        cepstrum = self.pupilsIntent.returnCepstrum() + self.pupilsNon.returnCepstrum()
        return cepstrum
        
# for debug purposes
#     def printDirFixations(self):
#         falseCount = 0
#         durations = 0
#         deltas = 0
#         sampleCounts = 0
#
#         #print self.recordPath.cmdPath
#         #print "Fixation No.\tEvent[ms]\tFixation[ms]\tDelta[ms]\tDuration[ms]\tEmbedded\tSamples"
#
#         delta = (moveEvent['timestamp'] - fixSequence[0]['timestamp'])
#         deltas += delta
#         duration = fixSequence[-1]['timestamp']-fixSequence[0]['timestamp']
#         durations += duration
#         sampleCount = len(fixSequence['timestamp'])
#         sampleCounts += sampleCount
#         embedded = delta < duration
#         if embedded == 0:
#             falseCount += 1

            #print fixations[idx]['fixationNumber'], "\t", + \
            #      moveEvent['timestamp'], "\t", +\
            #      fixSequence[0]['timestamp'], "\t", +\
            #      delta,"\t", + \
            #      duration, "\t", + \
            #      embedded, "\t", + \
            #      sampleCount

        #print "Average\t\t\t",(deltas/count),"\t",(durations/count),"\t",((falseCount *100 ) / count),"%\t",(sampleCounts/count)

# tato funkce se tu uz vubec nepouziva
    def saveFeatures(self,number,dir):
        pathSample = self.createPathName(dir, "features_"+str(number))

        intFeatures = self.featuresIntent.returnAllFeatures()   
        nonFeatures = self.featuresNon.returnAllFeatures()

        labelIntent = self.labelFixations(intFeatures,"intent")
        labelNon = self.labelFixations(nonFeatures,"non")
        self.allSamples = labelIntent + labelNon # tohle je zakomentovane pro ucely strategy zkoumani
        self.allSamples = ">" + '\n' + nonFeatures
        self.saveArray(pathSample,self.allSamples)
        print "saved"
        
        #odladit pupily
        #save pupils
        #pathSample = self.createPathName(dir, "pupils_"+str(number))

        #labelIntent = self.labelFixations(self.pupilsIntent.returnPupils(),"intent")
        #labelNon = self.labelFixations(self.pupilsNon.returnPupils(),"non")
        #allPupils = labelIntent + labelNon
        #self.saveArray(pathSample,allPupils)

# Navratove funkce  
    def returnAllSamples(self):
        return self.allSamples
        
    # nekdy se muzou fixace prekryvat, protoze je tracker zbytecne rozdelil, pro tento pripad,
    # preskocime celou sekvenci
    def checkOverlay(self,  seq):
        
        for i in range(0,len(seq)-1):
            
            vY = seq[i+1][0]['gazePointY'] - seq[i][0]['gazePointY']
            vX = seq[i+1][0]['gazePointX'] - seq[i][0]['gazePointX']
                
            if vY == 0 and vX == 0:
                return 1

        return 0    
