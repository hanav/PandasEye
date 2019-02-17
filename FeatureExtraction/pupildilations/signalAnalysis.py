import numpy as np
# import matplotlib.pyplot as plt
#plt.use('TkAgg')

class SignalAnalysis:
    def __init__(self):
        self.all = []

        self.rawPupils = []
        self.normPupils = []

        self.spectrum = []
        self.cepstrum = []
        self.diff1 = []
        self.diff2 = []

    def load(self, dir, non):
        for sequence in dir:
            self.all.append(sequence)
        for sequence in non:
            self.all.append(sequence)
        self.extractPupils()

    def extractPupils(self):
        for sequence in self.all:
            pupils = (sequence['pupilL'] + sequence['pupilR']) / 2
            pupils = self.denoise(pupils)
            self.rawPupils.append(pupils)

    def denoise(self,sequence):
        kernel = np.array([1,1,1], dtype=np.float64)
        kernel = kernel/3
        smooth = np.convolve(sequence, kernel, 'same')
        return smooth

    def normalization(self,baseVar):
        for sequence in self.rawPupils:
            #self.normalizationSession(sequence,baseVar)
            #self.normalizationEvent(sequence)
            self.normalizationPCPS(sequence)

    def normalizationSession(self, sequence, baseVar):
        normSequence = (sequence - baseVar[0]) / baseVar[1]
        self.normPupils.append(normSequence)

    def normalizationEvent(self, sequence):
        base = np.mean(sequence)    
        var = np.var(sequence)
        normSequence = (sequence - base) / var
        self.normPupils.append(normSequence)

    def normalizationPCPS(self,sequence):
        base = np.mean(sequence)    
        var = np.var(sequence)
        normSequence = ((sequence - base) / var) / base
        self.normPupils.append(normSequence)

    #http://stackoverflow.com/questions/2791114/using-cepstrum-for-pda
    def transformation(self):
        for sequence in self.normPupils:
            spect = self.countSpectrum(sequence)
            self.spectrum.append(spect)

            #ceps = self.countCepstrum(sequence)
            #self.cepstrum.append(ceps)

            #diff1 = self.countDiff1(sequence)
            #self.diff1.append(diff1)

            #diff2 = self.countDiff2(diff1)
            #self.diff2.append(diff1)

    def histograms(self):
        pass
        #1.derivace
        #2.derivace
        #histogram

    def countSpectrum(self,sequence):
        spect = np.fft.fft(sequence)
        absSpect = np.abs(spect)
        return absSpect
