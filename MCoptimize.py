#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
from src.Loading import Loading
from src.Model import FFModel
from src.HelperFunctions import setUpLogger, endSound, round_sig

import theano
import theano.tensor as tt
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm

import platform
import os
from os.path import dirname, join
from logging import info
import shutil, sys
import argcomplete, argparse
import datetime

def MCoptimize(startFile):

    if __name__ == "__main__":
        # initialize classes
        DAT = Loading(startFile)
        MOD = FFModel(DAT)
        
        # make directory, logfile, copy startfile
        traceDir = DAT.traceDir+'_'+str(datetime.datetime.now()).replace(' ','_').replace(':','h')[:-10]
        os.makedirs(traceDir)
        setUpLogger(join(traceDir,"trace.log"))
        shutil.copy(startFile, join(traceDir, "startFile.txt"))
        
        with pm.Model() as tmodel:
            # set up parameter priors
            params = []
            for x in DAT.gauss:
                params.append(
                    pm.Normal(
                        x[0],
                        mu=x[1],
                        sigma=x[2],
                    )
                )            
            for x in DAT.flat:
                params.append(
                    pm.Uniform(
                        x[0],
                        lower=x[2],
                        upper=x[3],
                    )
                )

            PAR = tt.as_tensor_variable(params)
            inten = MOD.intensity(PAR,tt) # model function
            pm.Normal("likelihood", mu=inten, sd=MOD.Err, observed=MOD.Iex) # data likelihood function
            db = pm.backends.Text(traceDir) # save chains
            # sampling:
            trace = pm.sample(DAT.nMC, 
                step = pm.NUTS(), 
                tune=DAT.burnIn, trace = db, chains=DAT.nChains
            )
            info(f"output created in {traceDir}")

            traceRaw = pm.backends.text.load(traceDir)
            trace = traceRaw[DAT.burnIn:]
            names = np.concatenate(( [x[0] for x in DAT.gauss], [x[0] for x in DAT.flat] ))
            traceDirRel = traceDir[traceDir.find('traces'):]
            parMeans, parStds = {}, {}
            for x in names:
                parMeans[x] = round_sig(trace.get_values(x).mean(),3)
                parStds[x] = round_sig(trace.get_values(x).std(),2)
            # save tracemeans into file
            parMeansStr = str(parMeans.values())
            fid  = open(os.path.join(traceDir, "means.txt"), 'w') 
            fid.write(parMeansStr[13:-2])

        if DAT.shtdwn:
            if platform.system() == "Windows":
                os.system("shutdown -s -t 120")
            elif platform.system() == "Linux":
                print("shutting down in 120 sec")
                os.system("sleep 120 && poweroff")
        else:
            endSound()

# #for debugging:
# MCoptimize('startFiles/PMPC_all_start.txt')

def main(argv):
    inputfile = ""

    parser = argparse.ArgumentParser()
    parser.add_argument("inputfile")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    MCoptimize(args.inputfile)


if __name__ == "__main__":
    main(sys.argv[1:])

