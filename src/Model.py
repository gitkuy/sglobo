# pylint: disable=no-member

import numpy as np
import theano.tensor as tt
from theano import scan
from scipy.special import erf
from src.Loading import Loading
from src.HelperFunctions import chisquare
import os

import matplotlib.pyplot as plt

dirname = os.path.dirname(__file__)

## FFModel Class
# symmetrical membrane SDP-model
#  responsible for calculating the form factor
class FFModel:

    def __init__(self, LoadingInstance):
        self.q, self.qex, self.Iex, self.Err, self.gaussians = [], [], [], [], []

        # solvent scattering lengths
        bLW = -1.675 # H2O scattering length
        bHW = 19.145 # D2O scattering length
        eWX = 10 # H2O/D2O No of electrons

        if isinstance(LoadingInstance, Loading):
            DAT = LoadingInstance
            # write command that allocates fixed parameters
            self.fixStr = '\n'.join([x[0]+' = '+x[1] for x in DAT.fixed])
            # write commands that assign input parameter to the write variable name
            if isinstance(DAT.gauss,np.ndarray):
                self.gaussStr = '\n'.join([DAT.gauss[k][0]+' = params[%i]' % (k) for k in range(DAT.gauss.shape[0])])
                # self.gaussStr = '\n'.join([DAT.gauss[k][0]+' = params[0][%i]' % (k) for k in range(DAT.gauss.shape[0])])
                incr = DAT.gauss.shape[0]
            else:
                incr = 0 # respect the case where there is no gaussian priors
                self.gaussStr = ''
            if isinstance(DAT.flat,np.ndarray):
                self.flatStr = '\n'.join([DAT.flat[k][0]+' = params[%i]' % (k+incr) for k in range(DAT.flat.shape[0])])
                # self.flatStr = '\n'.join([DAT.flat[k][0]+' = params[0][%i]' % (k+incr) for k in range(DAT.flat.shape[0])])

            self.N = DAT.nFiles # number of datasets
            self.dataLs = np.array([0]) # lengths of the data
            self.x = [] # sample names (after the technique)
            self.SL = [] # vector with all lipid moety scattering lengths
            self.SLW = [] # vector with all different background SLDs
            self.Vstr = DAT.V # string for lipid moety volumes
            self.weight = []
            for k in range(DAT.nFiles):
                if '-"-' in DAT.dataFileNames[k][2]:
                    DAT.dataFileNames[k][2]=DAT.dataFileNames[k-1][2]
                exec('self.SL.append(np.array(%s))' % DAT.dataFileNames[k][2])
                
                if 'X' in DAT.dataFileNames[k][1]:
                    self.SLW.append(eWX) # adds the ED to the vector with all different background SLDs
                    self.x.append('SAXS')
                else:
                    x = float(DAT.dataFileNames[k][1])/100
                    self.SLW.append((bLW*(1-x)+bHW *x)) # vector with all different background SLDs
                    self.x.append('SANS %u' % (x*100))
                DAT.loadData(k)
                qErr = DAT.qErr # will be 0 for saxs
                self.dataLs = np.append(self.dataLs,qErr.shape[0])
                self.Iex = np.concatenate([self.Iex,DAT.I])
                self.Err = np.concatenate([self.Err,DAT.Err])
                self.Err = self.Iex*0.1 # fake error to give more weight to the higher q
                # self.weight.append(DAT.q.shape[0]/(DAT.q[-1]-DAT.q[0]))
                
                # extension of q to include smearing from outside the final range
                q_orig = DAT.q#.copy() 
                self.qex = np.concatenate([self.qex,q_orig])
                dqL = q_orig[1]-q_orig[0]
                dqH = q_orig[-1]-q_orig[-2]
                n_add = int(np.amax([np.around(3*qErr[0]/dqL),np.around(3*qErr[-1]/dqH)]))
                self.q.append(np.concatenate( (np.arange(q_orig[0]-n_add*dqL,q_orig[0],dqL), q_orig, np.arange(q_orig[-1]+dqH,q_orig[-1]+(n_add+0.5)*dqH,dqH))))
                
                # gaussians for smearing of sans
                qGauss, dq = np.array([]), np.array([])
                for m in range(q_orig.shape[0]): # qGauss: matrix with a row of q-values for a gaussian of each datapoint. dq: stepsizes for correct weighting
                    qGauss = np.append(qGauss,self.q[k][m:m+2*n_add+1]-self.q[k][m+n_add])
                    dq =np.append( dq, (np.append(0,self.q[k][m+1:m+2*n_add+1]-self.q[k][m:m+2*n_add]) + np.append(self.q[k][m+1:m+2*n_add+1]-self.q[k][m:m+2*n_add],0))/2 )
                qGMesh = np.reshape(qGauss,[-1,2*n_add+1])
                dq = np.reshape(dq,[-1,2*n_add+1])
                dummy, qEMesh = np.meshgrid(np.arange(-n_add,n_add+1), qErr)
                gaussTemp = np.exp(-(qGMesh/qEMesh)**2/2)*dq # make a Gaussian for each q-value
                #gaussTemp[np.abs(gaussTemp)<1e-3]=0 # cut off small values for speed
                self.gaussians.append(np.array([x/sum(x) for x in gaussTemp])) # normalized and saved in object

            # automated weighting of saxs-data
            for k in range(self.N):
                if (DAT.dataFileNames[k][1] == 'X') & (self.N > 1):
                    r1 = self.dataLs[0:k+1].sum()
                    r2 = self.dataLs[0:k+2].sum()
                    self.Err[r1:r2] = self.Err[r1:r2] *np.sqrt(self.weight[k]/(sum(self.weight)-self.weight[k]))
        else:
            raise TypeError("LoadingInstance is not of the type Loading")

    def ffprecalc(self,params,nptt):
        self.D = locals()
        exec(self.gaussStr, {}, self.D)
        exec(self.flatStr, {}, self.D)
        exec(self.fixStr, {}, self.D)
        exec('self.V = np.array(%s)' % self.Vstr)

        self.A = self.D["Alip"]
        self.sig = [self.D["sigCH3"],self.D["sigCH2"],self.D["sigCG"],self.D["sigPhos"],self.D["sigChol"]]
        self.sPoly = self.D["sigPoly"]/100
        self.z = [0, self.D["dCG"], self.D["dCG"]+self.D["dPhos"], self.D["dCG"]+self.D["dPhos"]+self.D["dChol"]]
        
        # Volume of bulk water
        self.VW = self.D["VW"]#30.28 # Angstroem^3
        # Volume of bound water
        self.VbW = self.D["VbW"]
        self.dzBW = self.D["dshell"]

        # check for negative water:
        def gaussDis(mu,sig,x):
            return 1/np.sqrt(2*np.pi*sig**2)*np.exp(-(x-mu)**2/(2*sig**2))
        # def symErf(start,end,sig,x):
        #     D = end - start
        #     x0 = (start+end)/2
        #     return (erf((x-x0+D/2)/(np.sqrt(2)*sig)) - erf((x-x0-D/2)/(np.sqrt(2)*sig)))/(2*D)
        ztest = np.linspace(-10,15)
        pWat = (erf((ztest)/(np.sqrt(2)*self.sig[1])) - erf((ztest-self.z[3]-self.dzBW)/(np.sqrt(2)*self.sig[1])))/2 - ( self.V[2]/self.A*gaussDis(self.z[1],self.sig[2],ztest) + self.V[3]/self.A*gaussDis(self.z[2],self.sig[3],ztest)+ self.V[4]/self.A*gaussDis(self.z[3],self.sig[4],ztest) )
        self.negW = pWat[pWat<0].sum()*(ztest[0]-ztest[1])*self.A/self.VW # No of negative water molecules
        # plt.plot(ztest,(erf((ztest)/(np.sqrt(2)*self.sig[1])) - erf((ztest-self.z[3]-self.dzBW)/(np.sqrt(2)*self.sig[1])))/2)
        # plt.plot(ztest,pWat)
        # plt.show()
        # if self.negW:
        #     print(self.negW)

        self.Rm, self.sigR, self.scal, self.inc = [],[],[],[] # scaling and incoherent background
        for k in range(self.N):
            self.Rm.append(self.D["Rm"+str(k+1)])
            self.sigR.append(self.D["sigR"+str(k+1)])
            self.scal.append(self.D["Scal"+str(k+1)])
            self.inc.append(self.D["Inc"+str(k+1)])


    def gaussFT(self,mu,sigma,q):
        return np.exp(-(sigma*q)**2/2)*np.cos(mu*q)

    def erfFT(self,start,end,sigma,q):
        d = end - start
        mu = (start+end)/2
        return 2*np.sin(q*d/2)/q*np.exp(-(sigma*q)**2/2)*np.cos(mu*q)/d

    def FF(self,sNo):
        q = self.q[sNo]
        dSLD = self.SL[sNo]/self.V - self.SLW[sNo]/self.VW
        dSLDbW = self.SLW[sNo]/self.VbW - self.SLW[sNo]/self.VW
        sig = self.sig

        # Gaussian vesicle - polydispersity 
        # gaussSphere = np.zeros(q.shape[0])
        # gaussCum = 0
        # for k in np.linspace(-3,3,151):
        #     gaussk = np.exp(-k**2/2)
        #     gaussCum += gaussk
        #     Rk =self.Rm[sNo]*(1+k*self.sigR[sNo]/self.Rm[sNo])
        #     gaussSphere += gaussk* Rk**2 / q**2 * np.sin(q*Rk)**2
        # sphereShell = gaussSphere * (4*np.pi)**2 / gaussCum *5e-8

        # # Gaussian vesicle - polydispersity - SASview,Raviv-group
        # gaussSphere = np.zeros(q.shape[0])
        # gaussCum = 0
        # for k in np.linspace(-3,3,151):
        #     gaussk = np.exp(-k**2/2)
        #     gaussCum += gaussk
        #     Rk =self.Rm[sNo]*(1+k*self.sigR[sNo]/self.Rm[sNo])
        #     gaussSphere += gaussk* Rk**2 / q**2 * ( np.sin(q*Rk) / (q*Rk) - np.cos(q*Rk) )**2
        # sphereShell = gaussSphere * (4*np.pi)**2 / gaussCum *5e-8

        # Schulz vesicle - polydispersity (Kucerka, Langmuir 2007)
        s = self.Rm[sNo]/self.sigR[sNo]**2
        zz = self.Rm[sNo]**2/self.sigR[sNo]**2-1
        sphereShell = 8*np.pi**2*(zz+1)*(zz+2)/(s*q)**2*( 1- (1+4*q**2/s**2)**(-(zz+3)/2) * np.cos((zz+3)*np.arctan(2*q/s)) )*5e-8 # 5e-8 is approx the relation to 1/q² for 50 nm radius
        # sphereShell = q**-2

        FF = np.zeros(q.shape[0])
        gaussCum = 0
        for k in np.linspace(-3,3,11):
            Ak = self.A/(1+k*self.sPoly)
            lC = (self.V[0]+self.V[1])/Ak

            gaussk = np.exp(-k**2/2)
            gaussCum += gaussk

            FF += gaussk*(
                2*(dSLD[0]-dSLD[1])*self.V[0]/Ak*self.gaussFT(0,sig[0],q)+ # Terminal methyl subtracted from MN
                dSLD[1]*2*lC*self.erfFT(-lC,lC,sig[1],q)+ # Methylen (MN) chains
                # 2*(dSLD[2]-dSLDbW)*self.V[2]/self.A*self.erfFT(lC,2*self.z[1]+lC,sig[2],q)+ # CG-backbone (ERF)
                2*(dSLD[2]-dSLDbW)*self.V[2]/self.A*self.gaussFT(self.z[1]+lC,sig[2],q)+ # CG-backbone (GAUSS)
                2*(dSLD[3]-dSLDbW)*self.V[3]/self.A*self.gaussFT(self.z[2]+lC,sig[3],q)+ # phosphate
                2*(dSLD[4]-dSLDbW)*self.V[4]/self.A*self.gaussFT(self.z[3]+lC,sig[4],q)+ # choline-CH3
                2*dSLDbW*(self.z[3]+sig[4])*self.erfFT(lC,lC+self.z[3]+self.dzBW,sig[1],q) # hydration layer
            )**2

            if sNo == 0:
                if k == 0:
                    # print(Ak)
                    Fexp = (
                        2*(dSLD[0]-dSLD[1])*self.V[0]/Ak*self.gaussFT(0,sig[0],q)+ # Terminal methyl subtracted from MN
                        dSLD[1]*2*lC*self.erfFT(-lC,lC,sig[1],q)+ # Methylen (MN) chains
                        # 2*(dSLD[2]-dSLDbW)*self.V[2]/self.A*self.erfFT(lC,2*self.z[1]+lC,sig[2],q)+ # CG-backbone (ERF)
                        2*(dSLD[2]-dSLDbW)*self.V[2]/self.A*self.gaussFT(self.z[1]+lC,sig[2],q)+ # CG-backbone (GAUSS)
                        2*(dSLD[3]-dSLDbW)*self.V[3]/self.A*self.gaussFT(self.z[2]+lC,sig[3],q)+ # phosphate
                        2*(dSLD[4]-dSLDbW)*self.V[4]/self.A*self.gaussFT(self.z[3]+lC,sig[4],q)+ # choline-CH3
                        2*dSLDbW*(self.z[3]+sig[4])*self.erfFT(lC,lC+self.z[3]+self.dzBW,sig[1],q) # hydration layer
                    )

        ff = 1e-4*FF/gaussCum # 1e-4 to convert to cm^-1

        if self.x[sNo] == 'SAXS':
            return ff*sphereShell*self.scal[sNo]+self.inc[sNo]
        else:
            ff = ff*sphereShell

            # include detector smearing:
            if isinstance(ff[0], float): # np for plotting            
                FF = []
                shift = self.gaussians[sNo].shape[1]
                for k in range(self.dataLs[sNo+1]):
                    FF = np.append(FF,(ff[k:k+shift]*self.gaussians[sNo][k]).sum())

            else: # tt for fitting
                def writeToArray(idx, vector, vecshift):
                    return vector[idx:idx+vecshift]
                shift = self.gaussians[sNo].shape[1]
                results, updates = scan(
                    fn=writeToArray,
                    sequences=tt.arange(self.gaussians[sNo].shape[0]),
                    non_sequences=[ff,shift])
                FF = (results*self.gaussians[sNo]).sum(axis=1)
            return FF*self.scal[sNo]+self.inc[sNo]

    def intensity(self, params, nptt):
        self.ffprecalc(params,nptt)
        if self.N == 1:
            I = self.FF(0)
        elif self.N == 2:
            I = nptt.concatenate([self.FF(0),self.FF(1)],axis=0)
        elif self.N == 3:
            I = nptt.concatenate([self.FF(0),self.FF(1),self.FF(2)],axis=0)
        elif self.N == 4:
            I = nptt.concatenate([self.FF(0),self.FF(1),self.FF(2),self.FF(3)],axis=0)
        elif self.N == 5:
            I = nptt.concatenate([self.FF(0),self.FF(1),self.FF(2),self.FF(3),self.FF(4)],axis=0)
        # return I # for MCMC
        return nptt.concatenate([ I, [self.negW*chisquare(abs(I - self.Iex), self.Err)] ]) # for fitting with -water penalty
