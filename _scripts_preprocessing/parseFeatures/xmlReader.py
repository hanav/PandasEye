#Class XmlReader

from xml.dom.minidom import parseString, parse
from recordPath import RecordPath

class XmlReader:
    def __init__(self):
        self.recordPaths = []

    def readPaths(self,  inputFile):
        #xmlDoc = parse("parserConfig.xml")
        xmlDoc = parse(inputFile)
    
        aoiPath = xmlDoc.getElementsByTagName('aoiPath')
        aoiContent = self.getContent(aoiPath[0].toxml(),'<aoiPath>','</aoiPath>')

        for player in xmlDoc.getElementsByTagName('player'):
            cmdPath = player.getElementsByTagName('cmdPath')
            gzdPath = player.getElementsByTagName('gzdPath')
            evdPath = player.getElementsByTagName('evdPath')
            elanPath = player.getElementsByTagName('elanPath')
            dirPath = player.getElementsByTagName('directionsPath')
            outPath = player.getElementsByTagName('outputPath')
            fixPath = player.getElementsByTagName('fixPath')
            
            cmdContent = self.getContent(cmdPath[0].toxml(),'<cmdPath>','</cmdPath>')
            gzdContent = self.getContent(gzdPath[0].toxml(),'<gzdPath>','</gzdPath>')
            evdContent = self.getContent(evdPath[0].toxml(),'<evdPath>','</evdPath>')
            elanContent = self.getContent(elanPath[0].toxml(),'<elanPath>','</elanPath>')
            fixContent = self.getContent(fixPath[0].toxml(),'<fixPath>','</fixPath>')
            outContent = self.getContent(outPath[0].toxml(),'<outputPath>','</outputPath>')
            dirContent1 = self.getContent(dirPath[0].toxml(),'<directionsPath>','</directionsPath>')
            dirContent2 = self.getContent(dirPath[1].toxml(),'<directionsPath>','</directionsPath>')
            dirContent3 = self.getContent(dirPath[2].toxml(),'<directionsPath>','</directionsPath>')
            dirContent = [dirContent1, dirContent2, dirContent3]

            self.recordPaths.append(RecordPath(cmdContent, gzdContent, fixContent, outContent, evdContent, elanContent, aoiContent,  dirContent))

        self.numberRecords = len(self.recordPaths)


    def getContent(self,xmlData,tagName, tagName2):
        xmlContent = xmlData.replace(tagName,'').replace(tagName2,'')
        return xmlContent 
