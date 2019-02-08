# Modul ParsedList
from gazeCode import CMDCode, EventCode

class ParsedList:
    "Parsovaci trida seznamu"
    window = 20
    def __init__(self,listCMD,listDIR):
        self.cmdList = listCMD
        self.dirList = listDIR
        self.eventList = []
        self.sequenceList    = []
        self.parsedList       = []
        self.outputString     = []
# ------------------------------------------------------------------------------
    def selectEvents(self, row):
        if(row[CMDCode.EventKey] == '1'):
            return 1
        return 0
            
    def findMovingStamps(self):
        self.eventList = filter(self.selectEvents, self.cmdList) # zamen za cmdList
        print "Pocet polozek v eventListu: ", len(self.eventList)

# ------------------------------------------------------------------------------
    def cutWindowStamps(self):
        sequenceCount = 0
        for x in range(0, len(self.eventList)):
            subl = next(subl for subl in self.cmdList if self.eventList[x][0] in subl)
            index = self.cmdList.index(subl)
            
            self.sequenceList.append(str(sequenceCount))
            for y in range( index - self.window, index):
                if (self.cmdList[y][CMDCode.Validity_L] == '0') and (self.cmdList[y][CMDCode.Validity_R] == '0') :
                    self.sequenceList.append(self.cmdList[y])
            sequenceCount = sequenceCount + 1
             
# ------------------------------------------------------------------------------
    def findNearestStamp(self):
        #konverze do intu timestampu directions listu
        self.toInt(self.dirList, 1)
        self.toInt(self.cmdList, 0)

        listSource2 = []
        startIndex = 0
        stopIndex = len(self.cmdList)
        for x in range(startIndex,stopIndex):
            subl = self.cmdList[x]
            listSource2.append(subl[0]) 

        #vyhledani nejblizsiho prvku a ulozeni do noveho seznamu
        print "Vyhledani nejblizsiho prvku a ulozeni do seznamu..."
        startIndex = 1
        stopIndex = len(self.dirList)
        listIndexes = []
        for x in range(startIndex, stopIndex):
            try:
                subl = self.dirList[x]
                target = subl[1]     
                nearest = min((abs(target - i), i) for i in listSource2)[1]
                index = listSource2.index(nearest)
                self.parsedList.append(self.cmdList[index])
            except Exception, e:
                print e
# ------------------------------------------------------------------------------
    def saveCmdList(self, outputPath):
        outputString = self.createOutputList(self.sequenceList)
        self.saveString(outputString, outputPath)

    def createOutputList(self,parsedList):
        print "Vytvarim vysledny seznam k ulozeni..."
        outputString = []
        print parsedList[0]
        for x, row in enumerate(parsedList):
            try:
                outputString.append("%s %s %s %s %s %s %s %s %s" 
                                      %(row[CMDCode.Timestamp], 
                                        row[CMDCode.GazepointX_L], 
                                        row[CMDCode.GazepointY_L],
                                        row[CMDCode.Pupil_L],
                                        row[CMDCode.GazepointX_R],
                                        row[CMDCode.GazepointY_R],
                                        row[CMDCode.Pupil_R],
                                        row[CMDCode.GazepointX],
                                        row[CMDCode.GazepointY]))
            except Exception:
                outputString.append(row)

        return outputString

    def saveString(self, outputString, outputPath):
        try:
            print "Ukladam..."
            outputTxt = '\n'.join(outputString)
            outputFile = open(outputPath,'w') 
            outputFile.write(outputTxt)
            outputFile.close()
        except Exception,e:
            print "Nepovedlo se ulozit:", e
# ------------------------------------------------------------------------------
    def toInt(self, nestedList, position):
        for x in range(0,len(nestedList)):
            subl = nestedList[x]
            try:
                subl[position] = int(subl[position])
            except Exception,e:
                subl[position] = -1
            nestedList[x] = subl

