import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import argcomplete, argparse, sys
import datetime

from src.Model import FFModel
from src.Loading import Loading
from src.HelperFunctions import round_sig
from plotMeans import plotMeans

from matplotlib.colors import BoundaryNorm

def LSoptimize(startFile):

    DAT = Loading(startFile)
    p0 = [x[1] for x in DAT.flat]# take "plotting values" as start values:

       # calculate modelfunction
    MOD = FFModel(DAT)
    q = MOD.qex
    Iex = MOD.Iex
    Err = MOD.Err

    def fun_LS(q,*p):
        return MOD.intensity(p,np)
    bounds = [ [x[2] for x in DAT.flat], [x[3] for x in DAT.flat] ]
    pOpt,pCov = curve_fit(fun_LS, q, np.concatenate([Iex,[0]]), p0,  np.concatenate([Err,[0.1]]),
        bounds=bounds, # no bounds for 'lm'! use 'trf' or 'dogbox'
        method='trf',
        max_nfev=15e3,
        )

    # def fun_DE(*p):
    #     return (MOD.intensity(p,np) - np.concatenate([Iex,[0]]))**2/np.concatenate([Err,[0.1]])**2
    # bounds = np.array([ [x[2] for x in DAT.flat], [x[3] for x in DAT.flat] ]).T
    # pOpt,pCov = differential_evolution(fun_DE,
    #     bounds=bounds, # no bounds for 'lm'! use 'trf' or 'dogbox'
    #     # method='trf',
    #     # max_nfev=15e3,
    #     )

    stds = [np.sqrt(pCov[k][k]) for k in range(len(p0))]

    ## plot the covariance matrix
    sl = slice(len(p0)) # determine which parameters you want to plot
    varnames = [x[0] for x in DAT.flat]
    pCorr = [x[sl]/stds[sl]/y for x,y in zip(pCov[sl],stds[sl])] # correlation matrix: pCov normalized to the parameter variances
    # define the colormap
    cmap = plt.get_cmap('PuOr')
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    # define the bins and normalize and forcing 0 to be part of the colorbar!
    mx = np.ceil(10*np.max(np.abs(pCorr)))/10
    step = 0.01
    bounds = np.arange(-1,1,step)
    # norm = BoundaryNorm(bounds, cmap.N)
    # plot a heatmap
    hm=plt.imshow(pCorr,interpolation='none',cmap=cmap)#,norm=norm)
    plt.xticks(ticks=np.arange(len(varnames)),labels=varnames,rotation=45)
    plt.yticks(ticks=np.arange(len(varnames)),labels=varnames)
    cbar=plt.colorbar(hm,orientation='horizontal',shrink=0.5)
    # cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(),rotation=45)
    plt.tight_layout()

   ## plot solution into terminal
    print('Found solution (parName value error):')
    for k in range(len(DAT.flat)):
        print(DAT.flat[k][0],round_sig(pOpt[k],2),round_sig(stds[k],1))

   ## save solution into startfile-like resultfile
    with open(startFile, 'r') as file :
        filedata = file.read()

    # proper format the string
    filedata = filedata.replace('\t',' ')
    filedata = filedata.replace(';','; ')
    filedata = filedata.replace('  ',' ')
    # replace the start-values with the result-values
    for m in range(len(DAT.flat)):
        i0 = filedata.find(DAT.flat[m][0],filedata.find('flat prior;'))
        i1 = filedata.find(';',i0+1)
        i2 = filedata.find(';',i1+1)
        filedata = (DAT.flat[m][0]+'; '+str(round_sig(pOpt[m],3))).join(filedata.rsplit(filedata[i0:i2],1))
    
    # generate resultfilename
    resultFile = startFile[len(startFile) - startFile[::-1].find('/'):] # cut away filepath
    if resultFile.find('start') >= 0:
        resultFile = resultFile.replace('start','result')
    else:
        resultFile = resultFile.replace('.txt','result.txt')
    resultFile = 'fits/'+resultFile.replace('.txt',str(datetime.datetime.now()).replace(' ','_').replace(':','h')[:-10]+'.txt')

    # write into resultfile
    with open(resultFile, 'w') as file :
        filedata = file.write(filedata)

    plotMeans(resultFile)

#LSoptimize('startFiles/DPPC_VHvar_start.txt')

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("startFile")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    LSoptimize(args.startFile)


if __name__ == "__main__":
    main(sys.argv[1:])
