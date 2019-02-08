import os.path
import numpy as np
import matplotlib.pyplot as plt

class SaveOutput:
    def __init__(self):
        pass

    def saveDictionary(self, directory, codebook):
        for key, sample in codebook.iteritems():
            path = self.createPathName(directory, key)
            self.saveSample(path, sample)

    def createPathName(self, dir, name):
        outPath = dir + name + ".csv"
        #print outPath
        return outPath  

    def createDir(self, path, name):
        outDir = path + '\\' + name + '\\'
        self.checkDir(outDir)
        return outDir

    def saveSample(self, path, sample):
        #print sample
        self.checkDir(path)
        np.savetxt(path, 
                   sample, 
                   fmt = '%f',
                   delimiter = "\t",  
                   newline = "\n")


    def saveLabelArray(self, path, array):
        self.checkDir(path)
        file = open(path,'w')

        
        outList = []
        for i in range(0,len(array)):
            list = array[i].tolist()       
            line = [] 
            label = str(int(list[0]))
            line.append(label)
            for j in range(1,len(list)):
                element = str(j) + ":" + str(list[j])
                line.append(element)  
            strLine = " ".join(line)
            file.write(strLine + "\n")
        file.close()              
            
    def saveArray(self,path, array):
        self.checkDir(path)
        file = open(path,'w')

        outList = []

        #vygenerujeme hlavicky a zapiseme do souboru
        numberElements = len(array[1].tolist())
        headder = range(1,  numberElements+1)
        line = []
        for element in headder:
            strElement = "att"+str(element)
            line.append(strElement)
        outline = ' '.join(line)
        file.write(outline + '\n')
        

        for i in range(0,len(array)):
        
            list = array[i].tolist()
            line = []

            for element in list:
                strEl = str(element)
                line.append(strEl)

            outline =' '.join(line)
            file.write(outline + '\n') 
            #file.write(outline + ' ') # zakomentovano kvuli strategies- zmeneno kvuli strategies
        file.close()

    def checkDir(self, outDir):
        dir = os.path.dirname(outDir)
        if not os.path.exists(dir):
            os.makedirs(dir)

    def createImagePath(self, dir, title):
        outPath = dir + title + ".png"
        return outPath

    def saveImage(self, dir, book, mean, title, step, border):
        xrange = range(step)

        #grafy
        toPlot = []
        toLabel = []
        for key, item in book.iteritems():
            toPlot.append(plt.plot(xrange,item, linewidth = 0.5))
            toLabel.append(key)
        toPlot.append(plt.plot(xrange, mean, linewidth = 1.5))
        toLabel.append("Average")
        plt.axvline(x=border)

        #legenda
        leg = plt.legend(toPlot,toLabel,
                   loc="best", fancybox=True, ncol=2)
        leg.get_frame().set_alpha(0.5)
        
        #titulky
        plt.title("Pupillary responses - " + title)
        plt.ylabel("Pupil diameter[mm]")
        plt.xlabel("Samples")
        plt.grid(True)

        #ulozeni a zobrazeni
        imgPath = self.createImagePath(dir, title)
        plt.savefig(imgPath, format="png", dpi=(200))
        plt.clf()
        plt.close()

    def saveImageArray(self, dir, array, title, step, border):
        xrange = range(step)

        toPlot = []
        for i in range(0,len(array)):
            toPlot.append(plt.plot(xrange, array[i], linewidth = 0.5))
        plt.axvline(x=border)

        #legenda
        leg = plt.legend(toPlot,[["Intent"],["No Intent"]],
                   loc="best", fancybox=True, ncol=2)
        leg.get_frame().set_alpha(0.5)
        
        #titulky
        plt.title("PCPS - " + title)
        plt.ylabel("PCPS[%]")
        plt.xlabel("Samples[n]")
        plt.grid(True)

        #ulozeni a zobrazeni
        imgPath = self.createImagePath(dir, title)
        plt.savefig(imgPath, format="png", dpi=(200))
        plt.clf()
        plt.close()


    def saveImageSequence(self, dir, sequence, title):
        xrange = sequence['timestamp']
        meanPupil = (sequence['pupilL'] + sequence['pupilR']) / 2

        fig = plt.figure(num=None, figsize=(80, 6), dpi=200, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1,1,1) # two rows, one column, first plot
        ax.plot(xrange, meanPupil, label = "Whole game")
        
        leg = plt.legend(loc="best", fancybox=True)
        leg.get_frame().set_alpha(0.5)

        plt.title("Pupillary responses - " + title)
        plt.ylabel("Pupil diameter[mm]")
        plt.xlabel("Samples")
        plt.grid(True)

        #fig.show()
        imgPath = self.createImagePath(dir, title)
        fig.savefig(imgPath, format="png")
        fig.clf()
        plt.close()












