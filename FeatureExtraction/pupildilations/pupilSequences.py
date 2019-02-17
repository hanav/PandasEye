import numpy as np
import matplotlib.pyplot as plt
#plt.use('TkAgg')
import pylab

class PupilSequences:
    def __init__(self):
        self.data = []
        self.pupils = []
        self.rawPupils = []
        self.rawPupils_x = []
        self.rawPupilsLength = []
        self.cepstrum = []
        self.spectrum = []
        self.diff1 = []
        self.diff2 = []       

        self.fixationCount = 0
        self.fixStarts = [0,18,37,50]
        self.kernel = np.array([1,1,1], dtype=np.float64)
        self.kernel = self.kernel/3

    def load(self, data): 
        self.data = data
        self.fixationCount = 3

    def extractPupils(self):
        i = 0
        
        for seq in self.data:
            pupil = (seq['pupilL'] + seq['pupilR'])/2

            starts = self.findFixationStarts(seq)
            warpPupil = self.warpPupil(pupil,starts)
            normPupil = self.normalizeEventSubs(warpPupil)
            #normPupil = self.normalizeEventZscore(warpPupil)
            #normPupil = self.normalizeEventPCPS(warpPupil)
            smoothPupil = np.convolve(normPupil,self.kernel,'same')

            spectrum = self.countSpectrum(smoothPupil)
            cepstrum = self.countCepstrum(spectrum)
            diff1 = self.countDiff(smoothPupil)
            diff2 = self.countDiff(diff1)

            self.rawPupils.append(pupil)
            x_values = range(0,len(pupil))
            x_values = (x_values - np.median(x_values))*33
            plt.plot(x_values, pupil)

            self.rawPupils_x.append(x_values)

            self.rawPupilsLength.append(len(pupil))
            self.pupils.append(smoothPupil)
            self.cepstrum.append(cepstrum)
            self.spectrum.append(spectrum)
            self.diff1.append(diff1)
            self.diff2.append(diff2)

    def warpPupil(self,seq,starts):
        warped = []
        for i in range(0,len(self.fixStarts)-1):
            realStart = starts[i]
            realEnd = starts[i+1]
            warpStart = self.fixStarts[i]
            warpEnd = self.fixStarts[i+1]
            delta = np.true_divide((realEnd - realStart),(warpEnd - warpStart))

            for j in range(warpStart,warpEnd):
                warpedIdx = np.rint((realStart + (j - warpStart)*(delta)))
                warped.append(int(warpedIdx))
        return seq[warped]

    def normalizeEventSubs(self,seq):
        mean = np.mean(seq)
        result = seq - mean
        return result

    def normalizeEventZscore(self,seq):
        mean = np.mean(seq)
        std = np.var(seq)
        result = (seq - mean)/std
        return result

    def normalizeEventPCPS(self,seq):
        mean = np.mean(seq)
        std = np.var(seq)
        result = ((seq - mean)/std)/mean
        return result

    def countPCPS(self, seq):
        mean = np.mean(seq)
        result = (seq - mean)/mean
        return result

    def countSpectrum(self,seq):       
        spectrum = np.abs(np.fft.fft(seq))
        return spectrum

    def countCepstrum(self,spectrum):
        # y = real(ifft(log(abs(fft(x)))))
        logSpectrum = np.log(spectrum)
        ifft = np.fft.ifft(logSpectrum)
        cepstrum = ifft.real
        return cepstrum

    def countDiff(self,seq):
        diff1 = np.diff(seq)
        diff1 = np.append(0,diff1)
        return diff1

    def findFixationStarts(self,seq):
        starts = []
        fixNumber = seq[0]['fixationNumber']

        for i in range(0,self.fixationCount):
            fixNo = fixNumber + i
            idx = np.where(seq['fixationNumber'] == fixNo)
            fixStart = idx[0][0]
            starts.append(fixStart)
        starts.append(len(seq)-1)
        return starts

    def returnFixStarts(self):
        return self.fixStarts

    def returnAllFeatures(self):
        allFeatures = []
        for i in range(0,len(self.data)):
            row = np.append(self.spectrum[i],self.cepstrum[i])
            allFeatures.append(row)
        return allFeatures
        
    def returnPupils(self):
        return self.pupils

    def returnDiff1(self):
        return self.diff1

    def returnDiff2(self):
        return self.diff2

    def returnCepstrum(self):
        return self.cepstrum

    def returnSpectrum(self):
        return self.spectrum
    

    def drawPupils(self,pupils, sequenceLengths):
        plt.xlim([0, max(sequenceLengths)])
        plt.plot(pupils)
        plt.ylabel('pupil diameter [mm]')
        plt.xlabel('time [ms]')

    def drawNormPupils(self, normPupils):
        meanPupil = np.mean(normPupils)
        plt.plot(normPupils)
        plt.plot(meanPupil, linewidth=2.0, color='r')
