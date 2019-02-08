#-------------------------------------------------------------------------------
## Raw signal 
#-------------------------------------------------------------------------------

class RawSignal:
    def __init__(self):
        pass
        
    def countSpectrum(self,  seq):
        #print "vzorky ve fix:", len(seq[0])
        
        spectX = np.fft.fft(seq[0]['rawX'], 50)
        spectXabs= np.abs(spectX)
        freqBinX = np.fft.fftfreq(len(spectX))
        
        #print spectX, "\n", (freqBinX * 50),  "\n len:", len(freqBinX) 
        
        spectY = np.fft.fft(seq[0]['rawY'])
        spectYabs = np.abs(spectY)
