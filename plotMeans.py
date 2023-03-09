import numpy as np
from src.Model import FFModel
from src.Loading import Loading
from src.HelperFunctions import chisquare
from scipy.special import erf
import matplotlib.pyplot as plt
import os
import argcomplete, argparse, sys

def plotMeans(FolFile,report=False,pOutput=False):
    # input: 
    #   FolFile must be either the trace-folder to plot the mean solution of a trace
    #       or a path to a start-file containing start-values 
    if FolFile[:3] == 'tra':
        startFile = os.path.join(FolFile, "startFile.txt")
        params = np.array(np.genfromtxt(os.path.join(FolFile, "means.txt"),delimiter=','))
        DAT = Loading(startFile)
    elif FolFile[:3] == 'sta' or FolFile[:3] == 'fit':
        DAT = Loading(FolFile)
        params = np.concatenate(( # DAT.gauss is still somehow 2-dimensional if there is 1 entry, see Loading.py 
            [x[1] for x in DAT.gauss],
            [x[1] for x in DAT.flat]
        ))    
    else:
        print('Define input parameters: start/means\nPlotting terminated')
        sys.exit(2)

 #########################################

    # calculate modelfunction
    MOD = FFModel(DAT)
    Isim = MOD.intensity(params,np)[:-1]

    # plot data & fit
    q = MOD.qex
    Iex = MOD.Iex
    Err = MOD.Err
    if MOD.N > 1:
        f1, axes = plt.subplots(MOD.N, 1, sharex=True,figsize=(6, 7),num='Data loglog')
        f2, axes2 = plt.subplots(MOD.N, 1, sharex=True,figsize=(6, 7),num='Data FFlin')
    else:
        f1 = plt.figure(num='Data loglog')
        axes = [plt.axes()]
        f2 = plt.figure(num='Data FFlin')
        axes2 = [plt.axes()]
    scX,scN = 1, 1
    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown']
    for sNo in range(MOD.N):
        r1 = MOD.dataLs[0:sNo+1].sum()
        r2 = MOD.dataLs[0:sNo+2].sum()
        if MOD.x[sNo] == 'SAXS':
            # log-log representation:
            axes[sNo].errorbar(q[r1:r2], scX*Iex[r1:r2], yerr=scX*Err[r1:r2], fmt="cx", label="Data",zorder=0)
            axes[sNo].loglog(q[r1:r2], scX*Isim[r1:r2], 'k', label="Model",zorder=10)
            # Nagle/Kucerka represenation:
            axes2[sNo].errorbar(q[r1:r2], q[r1:r2]*np.sqrt(Iex[r1:r2]-MOD.inc[sNo]), yerr=q[r1:r2]*np.sqrt(Err[r1:r2]), fmt="x", label="Data",zorder=0)
            axes2[sNo].plot(q[r1:r2], q[r1:r2]*np.sqrt(Isim[r1:r2]-MOD.inc[sNo]), label="Model",zorder=10)
            scX /= 10
        else:
            # log-log representation:
            axes[sNo].errorbar(q[r1:r2], scN*Iex[r1:r2], yerr=scN*Err[r1:r2], fmt="x", label="Data",zorder=0)
            axes[sNo].loglog(q[r1:r2], scN*Isim[r1:r2], 'k', label="Model",zorder=10)  
            # Nagle/Kucerka represenation:
            axes2[sNo].errorbar(q[r1:r2], q[r1:r2]*np.sqrt(Iex[r1:r2]-MOD.inc[sNo]), yerr=q[r1:r2]*np.sqrt(Err[r1:r2]), fmt="x", label="Data",zorder=0)
            axes2[sNo].plot(q[r1:r2], q[r1:r2]*np.sqrt(Isim[r1:r2]-MOD.inc[sNo]), label="Model",zorder=10)
            scN /= 10
        axes[sNo].set_ylabel('I [a.u.]')#,fontsize=16)
        axes2[sNo].set_ylabel('|F$_{bilayer}$| [a.u.]')

        # save the fit
        fid  = open('tmp/Fit_'+MOD.x[sNo]+'.txt', 'w') 
        fitdata = np.transpose([q[r1:r2], Isim[r1:r2]])
        for x in fitdata:
            for y in x:
                fid.write(str(y)+'\t')
            fid.write('\n')
        fid.close()
        
        # save data in Nagle/Kucerka-format
        fid  = open('tmp/FF_'+MOD.x[sNo]+'.txt', 'w') 
        fitdata = np.transpose([q, q[r1:r2]*np.sqrt(Iex[r1:r2]-MOD.inc[sNo]), q[r1:r2]*np.sqrt(Err[r1:r2])])
        for k in range(r2-r1):
            fid.write( str(q[r1+k])+'\t'+str(q[r1+k]*np.sqrt((Iex[r1+k]-MOD.inc[sNo])))+'\t'+str(q[r1+k]*np.sqrt(MOD.scal[sNo]*Err[r1+k])) )#/MOD.scal[sNo]
            fid.write('\n')
        fid.close()

    plt.legend(loc=0)

    # axes[sNo].tick_params(labelsize=14)
    axes2[sNo].title.set_text(MOD.x[sNo])
    # axes[sNo].grid()#b=True, which='both')
    # axes2[sNo].grid()#b=True, which='both')
    axes[sNo].set_xlabel('q [$\AA^-1$]')   
    axes2[sNo].set_xlabel('q [$\AA^-1$]')
    f1.tight_layout()
    f1.savefig("tmp/fits.pdf")
    f2.tight_layout()
    f2.savefig("tmp/formfactors.pdf")

    chi2 = chisquare(abs(Isim - Iex), Err)

 ####################################

    # plot SDP-profile
    MOD.ffprecalc(params,np)
    z = np.array(MOD.z) + (MOD.V[0]+MOD.V[1])/MOD.A
    c = MOD.V/MOD.A
    sig = MOD.sig

    # Erf-molecular group (area normalized to 1)
    def symErf(start,end,sig,x):
        D = end - start
        x0 = (start+end)/2
        return (erf((x-x0+D/2)/(np.sqrt(2)*sig)) - erf((x-x0-D/2)/(np.sqrt(2)*sig)))/(2*D)

    def gaussDis(mu,sig,x):
        return 1/np.sqrt(2*np.pi*sig**2)*np.exp(-(x-mu)**2/(2*sig**2))

    nz = 10000
    zPl = np.linspace(-45,45,num=2*nz)
    # Molecular group volume probability densities
    Pk = np.zeros((z.shape[0]+3,zPl.shape[0]))
    Pk[0] = 2*c[0]*gaussDis(0,sig[0],zPl)
    Pk[1] = 2*z[0]*symErf(-z[0],z[0],sig[1],zPl)-Pk[0]
    # Pk[2] = c[2]*symErf(-2*z[1]+z[0],-z[0],sig[2],zPl) + c[2]*symErf(z[0],2*z[1]-z[0],sig[2],zPl) # CG-backbone (ERF)
    Pk[2] = c[2]*gaussDis(-z[1],sig[2],zPl) + c[2]*gaussDis(z[1],sig[2],zPl) # CG-backbone (GAUSS)
    Pk[3] = c[3]*gaussDis(-z[2],sig[3],zPl) + c[3]*gaussDis(z[2],sig[3],zPl)
    Pk[4] = c[4]*gaussDis(-z[3],sig[4],zPl) + c[4]*gaussDis(z[3],sig[4],zPl)
    Pk[5] = (z[3]+sig[4]-z[0])*(symErf(-sig[4]-z[3],-z[0],sig[1],zPl) + symErf(z[0],z[3]+sig[4],sig[1],zPl)) - Pk[2] - Pk[3] - Pk[4]# "bound water"
    Pk[-1]=1-Pk[:-1].sum(axis=0)


    # SLD-profiles
    SDP = []
    for sNo in range(MOD.N):
        SLDlip = MOD.SL[sNo]/MOD.V
        SLDlip = np.append(SLDlip, MOD.SLW[sNo]/MOD.VbW)
        sdpN = np.zeros(zPl.shape)
        for k in range(Pk.shape[0]-1):
            sdpN += SLDlip[k]*Pk[k]
        sdpN = sdpN+MOD.SLW[sNo]/MOD.VW*Pk[-1]
        SDP.append(sdpN)

    n_ax = 2
    f3, axes = plt.subplots(n_ax, 1, sharex=True,figsize=(6,n_ax*3+1),num='Structures')
    colors = ['forestgreen','limegreen','goldenrod','crimson','magenta','royalblue','aqua']
    for k in range(len(Pk)):
        axes[0].plot(zPl,Pk[k],c=colors[k])
    # axes[0].plot(zPl,Pk.sum(axis=0),'k:') # plot the sum of all functions to check volume balance
    # axes[0].plot(zPl, Pk[:5].sum(axis=0), 'k:') # plot the sum of all lipid functions to check volume balance
    axes[0].legend(('Methyl','Methylene','Backbone','Phosphate','Choline','Bound Water','Bulk Water'),loc=5)#'Water'),loc=0)#'
    axes[0].set_ylabel('Volume probability')

    # calculate Luzzati-thickness:
    watFun = Pk[5][nz:]+Pk[-1][nz:]
    def fun(i):
        return np.sum(watFun[:i]) - np.sum(1 - watFun[i:])
    iL = 0
    while fun(iL) < 0:
        iL += 1
    dB = 2*zPl[nz+iL]
    nW = np.sum(Pk[5][nz:])*(zPl[1]-zPl[0])*MOD.A/MOD.VW
    ylts = axes[0].get_ylim()
    axes[0].plot([zPl[nz-iL],zPl[nz-iL]],ylts,':k')
    axes[0].plot([zPl[nz+iL],zPl[nz+iL]],ylts,':k')


    xDat = 0
    for sNo in range(MOD.N):
        if MOD.x[sNo] != 'SAXS':
            axes[1].plot(zPl,SDP[sNo],'--',label = MOD.x[sNo])
            fid  = open('tmp/NSLD_'+MOD.x[sNo]+'.txt', 'w') 
            SLDdata = np.transpose([zPl, SDP[sNo]])
            for x in SLDdata[::100]:
                for y in x:
                    fid.write(str(y)+'\t')
                fid.write('\n')
            fid.close()
        else:
            xDat = 1
            
    plt.xlabel('z [$\AA$]')#, fontsize=18)
    plt.xlim([0,35])
    plt.ylabel('SLD [fm/$\AA^3$]')#, fontsize=18)
    # plt.legend(loc=0)
    ylts = axes[1].get_ylim()
    axes[1].plot([zPl[nz-iL],zPl[nz-iL]],ylts,':k')
    axes[1].plot([zPl[nz+iL],zPl[nz+iL]],ylts,':k')
    if xDat:
        ax2 = axes[1].twinx()
        ax2.set_ylabel('ED [1/$\AA^3$]', color='c')
        ax2.tick_params(axis='y', labelcolor='c')
        for sNo in range(MOD.N):
            if MOD.x[sNo] == 'SAXS':
                ax2.plot(zPl,SDP[sNo],'c--',label = MOD.x[sNo])
                fid  = open('tmp/ED_'+MOD.x[sNo]+'.txt', 'w') 
                SLDdata = np.transpose([zPl, SDP[sNo]])
                for x in SLDdata[::100]:
                    for y in x:
                        fid.write(str(y)+'\t')
                    fid.write('\n')
                fid.close()
    f3.tight_layout()

    if pOutput:
        dHH = 'undetermined'
        for sNo in range(MOD.N):
            if MOD.x[sNo] == 'SAXS':
                arr = SDP[sNo][nz:]
                dHH = 2*zPl[100+ np.where(arr == np.amax(arr))[0][0]] 
        sig0 = np.sqrt(2/np.pi)*c[0]

        pars = [
            MOD.D["VL"],# VL
            MOD.D["VH"],# VH
            MOD.D["rCG"],# rCG
            MOD.D["rPCN"],# rPCN
            MOD.D["r"],# r
            0.8,# r12
            dB,# DB
            dHH,# DHH
            z[0]*2,# 2DC
            (dHH-z[0]*2)/2,# DH1
            MOD.D["Alip"],# A
            z[1],# zCG
            MOD.D["sigCG"],# sigCG
            z[2],# zPCN
            MOD.D["sigPhos"],# sigPCN
            z[3],# zCholCH3
            MOD.D["sigChol"],# sigCholCH3
            MOD.D["sigCH2"],# sigHC
            MOD.D["sigCH3"],# sigCH3
            MOD.D["sigPoly"],# sigPoly
            MOD.D["VbW"],
            nW, # number of bound waters
            (MOD.D["sigCH3"]-sig0)/(z[0]/3-sig0), # Y (interdigitation parameter)
        ]
        for x in pars:
            if isinstance(x,str):
                print(x)
            else:
                print(np.round(x,2))


    if report:
        values = [
            ['Bilayer thickness',np.round(dB,2)],
            ['Chain thickness',np.round(z[0],2)],
            ['Bound water',np.round(nW,2)],
        ]
        return MOD, DAT, chi2, values, f1, f3
    else:
        # Print Chi squared
        print(f"chi^2 of fit:\t\t{np.round(chi2,2)}")
        # print bilayer properties:
        print(f"Bilayer thickness:\t{np.round(dB,2)}")    
        print(f"Chain thickness:\t{np.round(z[0],2)}")        
        print(f"Bound water:\t\t{np.round(nW,2)}")  
        plt.show()

# for debugging:
# plotMeans('startFiles/DPPC_all_start.txt',pOutput=1)
# plotMeans('traces/MSPC_all_2020-02-03_14h17')
# plotMeans('fits/DPPC_all_result2021-05-26_10h16.txt')

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("FolFile")
    parser.add_argument("-r", "--report", type=bool, default=False)
    parser.add_argument("-p", "--pOutput", type=bool, default=False)
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    plotMeans(args.FolFile,args.report,args.pOutput)


if __name__ == "__main__":
    main(sys.argv[1:])
