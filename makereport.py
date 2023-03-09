from reportlab.pdfgen import canvas    
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors

import os
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
import argcomplete, argparse, sys

from src.PdfImage import PdfImage
from plotMeans import plotMeans
from loadTrace import loadTrace
from src.Model import FFModel
from src.Loading import Loading
from src.HelperFunctions import endSound2
# from src.calcpars import calcpars

def Image(imageHandle,x,y,c,opt):
    img = BytesIO()
    # A4: 8.27 x 11.69 inches
    if opt == 0:
        pass
    elif opt == 1:
        imgSize = imageHandle.get_size_inches()
        if imgSize[1]*8/imgSize[0] < 11:
            imageHandle.set_size_inches(8, imgSize[1]*8/imgSize[0])
        else:
            imageHandle.set_size_inches(imgSize[0]*11/imgSize[1], 11)
    elif opt == 2:
        imageHandle.set_size_inches(8.27, 11.69)
    else:
        pass
    
    imageHandle.savefig(img,format='PDF')
    Image = PdfImage(img)
    Image.drawOn(c,x,y)

def makeReport(traceDir):
    # Enable writing 'last' to load the latest made trace
    def all_subdirs_of(b='.'):
        result = []
        for d in os.listdir(b):
            bd = os.path.join(b, d)
            if os.path.isdir(bd): result.append(bd)
        return result
    if traceDir == 'last':
        all_subdirs = all_subdirs_of('./traces/')
        traceDir= max(all_subdirs, key=os.path.getmtime)[2:]

    # Make file and load data and plots
    c = canvas.Canvas(traceDir+"/report.pdf")
    MOD, DAT, chi2, values, meanFits, meanSDPs = plotMeans(traceDir, report = True)
    parMeans, parStds, tracePlt1, tracePlt2, pairPar, trace = loadTrace(traceDir, report = True)
    # calcpars(MOD.fixStr, trace)

 #### header with basic information
    f=open(traceDir+'/trace.log', 'r')
    tracelog = f.readlines()
    if len(tracelog) > 3:
        errStr = 'Yes, check trace.log'
    else:
        errStr = 'No'
    sampleTime = divmod((dt.strptime(tracelog[-1][:19],'%Y-%m-%d %H:%M:%S') - dt.strptime(tracelog[0][:19],'%Y-%m-%d %H:%M:%S')).total_seconds(),60)
    sampleTimeStr = str(int(sampleTime[0]))+' min, '+str(int(sampleTime[1]))+' s'

    c.setFont('Times-Bold', 14, leading = None)
    c.drawString(40,790,"sGlobo - MCMC fit report") 
    c.setFont('Times-Roman', 12, leading = None)

    InfoLeft = ['Date, time of sampling','sGlobo branch','No of Chains','Tuning Samples','Samples per Chain','Sampling Time','Warnings/Errors']
    InfoRight = [traceDir[-16:].replace('_',', ').replace('h',':'), DAT.branchname, str(DAT.nChains), str(DAT.burnIn), str(DAT.nMC), sampleTimeStr, errStr]
    
    for k in range(len(InfoLeft)):
        c.drawString(40,770-14*k,InfoLeft[k]) 
        c.drawRightString(500,770-14*k,InfoRight[k]) 
 ####

 #### Table with info about samples
    t1style = [
            ('TEXTCOLOR',(0,1),(-1,4),colors.grey),
            ('FONT',(0,0), (-1,-1), 'Times-Roman',12),
            ('FONT', (0,0), (-1,0), 'Times-Bold',14),
            ('SIZE', (0,1), (-1,4), 10),
            ]
    data= [['Samples',''],['',''],['No.','Filename'],['Method','Molecular unit (MU) Scattering Lengths/electrons'],['Solvent','']]
    for k in range(len(DAT.dataFileNames)):
        if MOD.x[k]=='SAXS':
            solvent = 'H2O'
        else:
            solvent = MOD.x[k][4:]+'% D2O'
        data.append([k+1,DAT.dataFileNames[k][0]])
        data.append([MOD.x[k][:4],DAT.dataFileNames[k][2]])
        data.append([solvent,DAT.dataFileNames[k][3]])
        t1style.append(('LINEABOVE',(0,5+k*3),(-1,5+k*3),1,colors.black))
    t1 = Table(data,style=t1style, rowHeights=18)  
    t1.wrapOn(c, 400, 100)
    t1.drawOn(c, 40, 650-sum(t1._rowHeights))
 ####
    c.showPage() 
 #### Table with parameters and mean results
    t2style = [('FONT',(0,0), (-1,-1), 'Times-Roman',11),
        ('FONT',(0,4), (0,4), 'Times-Bold',11),
        ('FONT', (0,0), (-1,0), 'Times-Bold',14)]
    data= [['Parameters',''],['',''],['MU Volumes '+MOD.Vstr,''],['',''],['Fixed','']]
    fixStrSep = MOD.fixStr.split('\n')
    for x in fixStrSep:
        data.append([x,'']) 
    t2 = Table(data, style=t2style, rowHeights=16)  
    t2.wrapOn(c, 400, 100)
    t2.drawOn(c, 40, 800-sum(t2._rowHeights))

    t3style = [('LINEBEFORE', (0,0), (0,-1), 1,colors.black),
            ('LINEBEFORE', (3,0), (3,-1), 1,colors.black),
            ('LINEBELOW', (0,1), (-1,1), 1,colors.black),
            ('LINEBELOW', (0,len(DAT.gauss)+3), (-1,len(DAT.gauss)+3), 1,colors.black),
            ('FONT',(0,0), (-1,-1), 'Times-Roman',11),
            ('FONT',(0,0), (0,0), 'Times-Bold',11),
            ('FONT',(0,len(DAT.gauss)+2), (0,len(DAT.gauss)+2), 'Times-Bold',11),
            ('FONT',(3,0), (3,-1), 'Times-Bold',11)]
    varData = [['Gaussian priors','','','',''],['x','\u03BC','\u03C3','mean(x)','std(x)']]
    for x in DAT.gauss:
        varData.append([x[0],x[1],x[2],str(parMeans[x[0]]),'\u00B1 '+str(parStds[x[0]])])
    varData.append(['Flat priors','','','',''])
    varData.append(['x','min','max','mean(x)','std(x)'])
    for x in DAT.flat:
        varData.append([x[0],x[2],x[3],str(parMeans[x[0]]),'\u00B1 '+str(parStds[x[0]])])
    t3 = Table(varData, style=t3style, rowHeights=16)      
    t3.wrapOn(c, 400, 100)
    # t3.drawOn(c, 40+t2._colWidths[0], 800-16*5-sum(t3._rowHeights))
    t3.drawOn(c, 200, 800-16*5-sum(t3._rowHeights))

    t4style = [('LINEBELOW', (0,0), (-1,0), 1,colors.black),
            ('FONT',(0,0), (-1,-1), 'Times-Roman',11),
            ('FONT',(0,0), (0,0), 'Times-Bold',11)]
    t4data = [['Further values:','']]
    for x in values:
        t4data.append(x)
    t4 = Table(t4data, style=t4style, rowHeights=16)
    t4.wrapOn(c, 400, 100)
    t4y = np.maximum(16*5+sum(t3._rowHeights),sum(t2._rowHeights))
    t4.drawOn(c, 40, 780-t4y-sum(t4._rowHeights))

 ####
    c.showPage() 
 #### Plots
    Image(meanFits,0,0,c,2)
    c.setFont('Times-Bold', 12, leading = None)
    c.drawString(40,790,"Data and Fits from Mean-values")
    c.drawString(40,775,"Chi^2 = "+str(np.round(chi2,3)))
 ####
    c.showPage() 
 #### 
    Image(meanSDPs,0,0,c,2)
    c.setFont('Times-Bold', 12, leading = None)
    c.drawString(40,790,"Volume probability and SLD-profiles") 
 ####
    c.showPage() 
 #### 
    Image(pairPar,0,300,c,1)
    c.setFont('Times-Bold', 12, leading = None)
    c.drawString(40,790,"Parameter correlation plots") 
 ####    
    c.showPage() 
 #### 
    Image(tracePlt1,0,0,c,1)
    c.setFont('Times-Bold', 12, leading = None)
    c.drawString(40,800,"Posterior probabilities and traces") 
 ####
    c.showPage() 
 ####    
    if len(varData) > 12:
        Image(tracePlt2,0,0,c,1)
        c.showPage()    
 ####
    c.save()
    endSound2()

# makeReport('last')
# makeReport('traces/fred_C12345_vsim_sdpFred2019-11-21_15h00')

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("traceDir")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    makeReport(args.traceDir)

if __name__ == "__main__":
    main(sys.argv[1:])