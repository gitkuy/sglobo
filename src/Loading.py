from enum import Enum, auto
from src.HelperFunctions import getIndex
import numpy as np
import os

dirname = os.path.dirname(__file__)

def stripNewline(array):
    while array[-1] == "\n":
        array.pop()

## Loading class
#  a class which loads the startFile (__init__) containing settings for the calc and external data
class Loading:

    def __init__(self, startFileName):

        startFileLines = []
        fid = open(startFileName, "r")
        for x in fid:
            startFileLines.append(x)
        fid.close()
        stripNewline(startFileLines)
        self.branchname = startFileLines[0][14:-1]

        fixedInd = getIndex(startFileLines, "fixed parameter") + 1 # +1 because index starts with 0
        gaussInd = getIndex(startFileLines, "gaussian prior") + 1
        flatInd = getIndex(startFileLines, "flat prior") + 1

        general = np.genfromtxt(
            startFileName,
            dtype="U128,U128,U128,U128",
            delimiter=";",
            skip_header=2,
            skip_footer=len(startFileLines) - fixedInd -1, # careful! empty lines in the footer do not count!
        )
        self.nFiles = getIndex(general,'volume')
        self.dataFileNames = general[0:self.nFiles]
        self.V = general[getIndex(general,'volume')][1]        
        self.qX = [float(general[getIndex(general,'qX')][1]),float(general[getIndex(general,'qX')][2])]
        self.qN = [float(general[getIndex(general,'qN')][1]),float(general[getIndex(general,'qN')][2])]
        self.traceDir = os.path.join(dirname,general[getIndex(general,'mcsavefile')][1].strip())
        self.nChains = int(general[getIndex(general,'nChains')][1])
        self.burnIn = int(general[getIndex(general,'BurnIn')][1])
        self.nMC = int(general[getIndex(general,'Nmcmc')][1])
        self.shtdwn = int(general[getIndex(general,'shutdown')][1])
        
        ## load all parameters
        # fixed parameters: only one field (whole formula to be executed)
        if gaussInd-fixedInd > 2:
            fixedTemp = np.genfromtxt(
                startFileName,
                dtype="U128",
                delimiter=";",
                skip_header=fixedInd,
                skip_footer=len(startFileLines) - gaussInd - 1, # loads actually a line too much
            )
            self.fixed = fixedTemp[:-1]
        else:
            self.fixed = {}
        # variables with gaussian prior: loading [name, mean, std]
        if flatInd-gaussInd > 2:
            gaussTemp = np.genfromtxt(
                startFileName,
                dtype="U128,f8,f8",
                delimiter=";",
                skip_header=gaussInd,
                skip_footer=len(startFileLines) - flatInd, # loads actually a line too much
            )
            self.gauss = gaussTemp[:-1] # tweak to avoid having a corrupted (<- no idea why) 1D-array and to be able to concatenate
        else:
            self.gauss = {}
        # variables with flat prior: loading [name, testValue, lowerBorder, upperBorder]
        self.flat = np.genfromtxt(
            startFileName,
            dtype="U128,f8,f8,f8",
            delimiter=";",
            skip_header=flatInd,
        )

    ## loads experimental Data from a file
    #
    def loadData(self,n):
        fileName = os.path.join(dirname, self.dataFileNames[n][0])
        if 'X' in self.dataFileNames[n][1]:
            data = np.genfromtxt(fileName, dtype="f8,f8,f8")
            q, I, Err = [], [], []
            for x in data:
                q.append(x[0])
                I.append(x[1])
                Err.append(x[2])
            q = np.asarray(q)/10
            I = np.asarray(I)
            Err = np.asarray(Err)
            qErr = np.zeros(q.shape)
            qmin, qmax = self.qX[0], self.qX[1]
            #print(q)
        else:
            data = np.genfromtxt(fileName, dtype="f8,f8,f8,f8")
            q, I, Err, qErr = [], [], [], []
            for x in data:
                q.append(x[0])
                I.append(x[1])
                Err.append(x[2])
                qErr.append(x[3])
            q = np.asarray(q)
            I = np.asarray(I)
            Err = np.asarray(Err)
            qErr = np.asarray(qErr)
            qmin, qmax = self.qN[0], self.qN[1]

        # throw away negative values and trim q range
        LM = np.where(((q >= qmin) & (q <= qmax)) & (I >= 0))
        self.q, self.I, self.Err, self.qErr = q[LM], I[LM], Err[LM], qErr[LM]
