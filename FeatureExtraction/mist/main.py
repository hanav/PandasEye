# Author: Hana Vrzakova
# Date: 23.1.2011
# Date: 10.8.2012
# Date: 29.8.2012
# Date: 16.4.2014
# Date: 24.1.2015
# Date: 4.4. 2017
# Date: 9.4.2018

import sys
import os
import numpy as np
from ETPreprocessing.parseFeatures.record import Record
from ETPreprocessing.parseFeatures.xmlReader import XmlReader
from ETPreprocessing.parseFeatures.mist.saveOutput import SaveOutput
from ETPreprocessing.parseFeatures.mist.mess import Mess


def quit():
    print "Konec parseFeatures"
    raw_input()
    exit()

def cutSamples(paths, prefix, suffix, dir):
    features = []
    pupils = []
    diff1 = []
    diff2 = []
    spectrum = []
    cepstrum = []
    all = []

    #for i in range(0,len(paths)):
    for i in range(1, 2):
        print i
        record = Record()
        record.loadPrefixSuffix(prefix,suffix)
        record.loadPath(paths[i])
        record.loadData()
        record.prepareRawData()

        record.cutFixations()

        record.extractFeatures()
    
        #save one person
        personDir = dir + str(i) + "\\"
        # extract the name from path
        
        saveObj = SaveOutput()
        outDir = saveObj.createPathName(personDir,"all")
        temp =  record.returnFeatures()
        featureArray = np.array(temp)
        saveObj.saveArray(outDir , featureArray)
        myMess = Mess()
        myMess.binaryTrainTest(featureArray,personDir,(0.6)) #split them into 2/3 for training and 1/3 testing
    
        features.append(record.returnFeatures())
#        pupils.append(record.returnPupils())
#        diff1.append(record.returnDiff1())
#        diff2.append(record.returnDiff2())
#        spectrum.append(record.returnSpectrum())
#        cepstrum.append(record.returnCepstrum())

        print "...done..."

    all = [features, pupils,diff1,diff2,spectrum,cepstrum]
    return all

# uklidit do saveOutput
def saveAll(array, dir,name):
    obj = SaveOutput()
    pathName = obj.createPathName(dir,name)
    obj.saveArray(pathName, array)
    return


############################################################

    # opravit pupily
    # ukladani po featurech - quick analysis
    # ukladani feature fixations
    # ukladat basic stats (abs. + rel. hodnoty) + spocitat vahy, ulozit
    # vykreslit timeline pro featury - ukladat timestamps - zacatek a konec sekvence (pozdeji)

print("Start computing...")

    # C:\Python27\python D:\Dropbox\dizertacka\python\parseFeatures\main.py ga -5 -3
    #prefix = -6 #pocet fixaci pred eventem, cislujeme od nuly => [0,1,2,event,4,5] = 3 + event + 2 = 6fixaci na analyzu
    #suffix = -2 #pocet fixaci po eventu - pada to na suffixu

modality = sys.argv[-3]
prefix = int(sys.argv[-2])
suffix = int(sys.argv[-1])


if(os.name == 'posix'):
    userhome = os.path.expanduser('~')
    input_dir = os.path.join(userhome, 'Dropbox','dizertacka','python','parseFeatures')
    output_dir = os.path.join(userhome, 'Dropbox','dizertacka','python','parseFeatures', 'hokus', (modality + "_" + str(prefix) + "_" + str(suffix)))

    if(modality == 'ga'):
        xmlDir = os.path.join(input_dir, 'parserConfig_mac.xml')
    elif(modality == 'mo'):
        xmlDir = os.path.join(input_dir, 'parserMouseAbsolute.xml')
    elif(modality == 'dt'):
        xmlDir = os.path.join(input_dir, 'parserDwellTime.xml')

elif(os.name == 'Windows'):
    input_dir = "..\\thesis_late_results\\" + modality + "_" + str(prefix) + "_" + str(suffix) + "\\"
    output_dir = "D:\\Dropbox\\dizertacka\\python\\umap_late_results\\" + modality + "_" + str(prefix) + "_" + str(
            suffix) + "\\"
    xmlDir = "D:\\Dropbox\\dizertacka\\python\\parseFeatures\\parserMouseAbsolute.xml"
        # xmlPaths.readPaths("D:\\Dropbox\\dizertacka\\python\\parseFeatures\\parserDwellTime.xml")
        # xmlPaths.readPaths("E:\\dropbox\\Dropbox\\dizertacka\\python\\parseFeatures\\parserConfig.xml")
elif(os.name == 'linux'):
    pass


xmlPaths = XmlReader()
xmlPaths.readPaths(xmlDir)
paths = xmlPaths.recordPaths
recordArray = cutSamples(paths, prefix, suffix, output_dir)


#    #projit postupne
features = reduce(lambda x,y: x+y, recordArray[0]) # reduce dimension and create list
    #pupils = reduce(lambda x,y: x+y, recordArray[1])
    #histArray1 = createHistogram(recordArray[2])
    #histArray2 = createHistogram(recordArray[3])
    #spectrum = reduce(lambda x,y: x+y,recordArray[4])
    #cepstrum = reduce(lambda x,y: x+y,recordArray[5])

    #all = np.concatenate((features,spectrum,cepstrum),axis=1)

    #saveAll(pupils,dir,"pupilsAll")
    #saveAll(features,dir,"featuresAll")
    #saveAll(histArray1,dir,"hist1")
    #saveAll(histArray2,dir,"hist2")
    #saveAll(cepstrum, dir, "cepstrumAll")
    #saveAll(spectrum, dir, "spectrumAll")
    #saveAll(all, dir,"mixAll")

    #saveAllRecords(recordArray, dir)
saveObj = SaveOutput()
outDir = saveObj.createPathName(output_dir,"all")
featureArray = np.array(features)
saveObj.saveArray(outDir , featureArray)

myMess = Mess()
myMess.binaryTrainTest(features,output_dir,(0.6))

    #records = RecordSet()
    #records.loadPrefixSuffix(prefix, suffix)
    #records.loadOutputDir(dir)

    #for i in range(0,len(xmlPaths.recordPaths)):
    #for i in range(0,1):
    #    records.loadDirectories(xmlPaths.recordPaths[i].outPath)
    #    records.loadTiles(xmlPaths.recordPaths[i].outPath)
    #records.createCodebooks()
    #records.saveClasses()
print("...done...")

print("Skript ended sucessfully.")

