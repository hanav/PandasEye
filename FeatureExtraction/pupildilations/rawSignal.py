import numpy as np

class RawSignal:
    def __init__(self):
        pass
        
    def countSpectrum(self,  seq):

        spectX = np.fft.fft(seq[0]['rawX'], 50)
        spectXabs= np.abs(spectX)
        freqBinX = np.fft.fftfreq(len(spectX))

        spectY = np.fft.fft(seq[0]['rawY'])
        spectYabs = np.abs(spectY)
