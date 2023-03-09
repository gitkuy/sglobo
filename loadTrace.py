#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

from src.Loading import Loading
from src.Model import FFModel
from src.HelperFunctions import round_sig

import theano
import theano.tensor as tt
from scipy.stats import chisquare
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import os
import argcomplete, argparse, sys

def loadTrace(traceDir,report=False):
    startFile = os.path.join(traceDir, "startFile.txt")
    DAT = Loading(startFile)
    MOD = FFModel(DAT)

    with pm.Model() as tmodel:
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
        trace_raw = pm.backends.text.load(traceDir)

    trace = trace_raw[DAT.burnIn:] # cut away burn-in (doesnt seem to work automatically)
    names = np.concatenate(( [x[0] for x in DAT.gauss], [x[0] for x in DAT.flat] ))

    pm.traceplot( trace, var_names = names[:8] )#, grid=True)
    plt.legend()
    f1 = plt.gcf()
    if len(names) > 8:
        pm.traceplot( trace, var_names = names[8:] )
        f2 = plt.gcf()
    else:
        f2 = []
    pm.pairplot( trace, 
        var_names = names[:names.shape[0]-DAT.nFiles],#[[2,6]],#[[2,4,6,7,8,9,11]],#[:names.shape[0]-DAT.nFiles],#,#*2],
        kind='hexbin', 
        gridsize=50,
        textsize=9,
        # figsize=(4,3),
    )
    f3 = plt.gcf()
    # plt.xlabel('$r$'),#$V_{BW}~[\AA^3]$')#'$d_{PCN}~[\AA]$')#'$\sigma_{CH3}~[\AA]$')#
    # plt.ylabel('$\sigma_{CH3}$'),#$d_{Chol}~[\AA]$')#'$V_{H}~[\AA^3]$')#'$d_{CG}~[\AA]$')#'$d_{CholCH3}~[\AA]$')#
    # plt.tight_layout()

    parMeans, parStds = {}, {}
    for x in names:
        parMeans[x] = round_sig(trace.get_values(x).mean(),3)
        parStds[x] = round_sig(trace.get_values(x).std(),2)

    if report:
        return parMeans, parStds, f1, f2, f3, trace
    else:
        print('Trace means:')
        for x in names:
            print(f"{x}:\t{parMeans[x]}\t\u00B1 {parStds[x]}")
        plt.show()

# for debugging:
# loadTrace('traces/AD2_sans_test')

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("traceDir")
    parser.add_argument("-r", "--report", type=bool, default=False)
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    loadTrace(args.traceDir,args.report)


if __name__ == "__main__":
    main(sys.argv[1:])
