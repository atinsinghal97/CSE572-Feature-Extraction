#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math
import pywt

from pandas import DataFrame, read_csv, concat

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from statsmodels.graphics.tsaplots import plot_acf

import scipy.stats
from scipy import signal

import seaborn as sns


# In[2]:


#defining ActivePatient

ActivePatient=1
#ActivePatient='All'


# In[3]:


FeaturesList=list()


# In[4]:


directory = "/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/DataFolder/"

DatenumFile1= directory+"CGMDatenumLunchPat1.csv"
DatenumFile2= directory+"CGMDatenumLunchPat2.csv"
DatenumFile3= directory+"CGMDatenumLunchPat3.csv"
DatenumFile4= directory+"CGMDatenumLunchPat4.csv"
DatenumFile5= directory+"CGMDatenumLunchPat5.csv"

SeriesFile1= directory+"CGMSeriesLunchPat1.csv"
SeriesFile2= directory+"CGMSeriesLunchPat2.csv"
SeriesFile3= directory+"CGMSeriesLunchPat3.csv"
SeriesFile4= directory+"CGMSeriesLunchPat4.csv"
SeriesFile5= directory+"CGMSeriesLunchPat5.csv"


# In[5]:


#Read CSV Files

Datenum1 = pd.read_csv(DatenumFile1, usecols=[*range(0, 30)])
Datenum2 = pd.read_csv(DatenumFile2, usecols=[*range(0, 30)])
Datenum3 = pd.read_csv(DatenumFile3, usecols=[*range(0, 30)])
Datenum4 = pd.read_csv(DatenumFile4, usecols=[*range(0, 30)])
Datenum5 = pd.read_csv(DatenumFile5, usecols=[*range(0, 30)])

Series1 = pd.read_csv(SeriesFile1, usecols=[*range(0, 30)])
Series2 = pd.read_csv(SeriesFile2, usecols=[*range(0, 30)])
Series3 = pd.read_csv(SeriesFile3, usecols=[*range(0, 30)])
Series4 = pd.read_csv(SeriesFile4, usecols=[*range(0, 30)])
Series5 = pd.read_csv(SeriesFile5, usecols=[*range(0, 30)])

if ActivePatient==1:
    Datenum=Datenum1
    Series=Series1
elif ActivePatient==2:
    Datenum=Datenum2
    Series=Series2
elif ActivePatient==3:
    Datenum=Datenum3
    Series=Series3
elif ActivePatient==4:
    Datenum=Datenum4
    Series=Series4
elif ActivePatient==5:
    Datenum=Datenum5
    Series=Series5
elif ActivePatient=='All':
    Datenum=DatenumAll
    Series=SeriesAll


# In[6]:


#Checking data

#Datenum.head()


# In[7]:


#Checking data

#Series.head()


# In[8]:


#Data Cleaning & Merging

#Concat- Keeping columns constant
DatenumAll = pd.concat([Datenum1, Datenum2, Datenum3, Datenum4, Datenum5], ignore_index=True)
SeriesAll= pd.concat([Series1, Series2, Series3, Series4, Series5], ignore_index=True)

#Concat- Horizontally, Increasing Columns
#DatenumAll = pd.concat([Datenum1, Datenum2, Datenum3, Datenum4], axis=1)
#SeriesAll= pd.concat([Series1, Series2, Series3, Series4], axis=1)

rowDate,colDate=DatenumAll.shape
thresholdA1=colDate*0.95
thresholdA1=int(thresholdA1)
DatenumAll=DatenumAll.dropna(axis=0,thresh=thresholdA1)
DatenumAll.fillna(method='pad', inplace=True)

rowSeries,colSeries=SeriesAll.shape
thresholdA2=colSeries*0.95
thresholdA2=int(thresholdA2)
SeriesAll=SeriesAll.dropna(axis=0,thresh=thresholdA2)
SeriesAll.fillna(method='pad', inplace=True)

#print(DatenumAll)
#print(SeriesAll)
# DatenumAll.head()
# SeriesAll.head()


# In[9]:


#DatenumAllPath=directory+"Processed Data/DatenumAll.csv"
#SeriesAllPath=directory+"Processed Data/SeriesAll.csv"

DatenumAll.to_csv(r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Processed Data/DatenumAll.csv", index=None, header=True)
SeriesAll.to_csv(r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Processed Data/SeriesAll.csv", index=None, header=True)


# In[10]:


#Data Cleaning- Individual Files

#Patient1
rowDate1,colDate1=Datenum1.shape
threshold11=colDate1*0.95
threshold11=int(threshold11)
Datenum1=Datenum1.dropna(axis=0,thresh=threshold11)
Datenum1=Datenum1.fillna(method='bfill')
Datenum1.to_csv(r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Processed Data/Datenum1Processed.csv", index=None, header=True)

rowSeries1,colSeries1=Series1.shape
threshold12=colSeries1*0.95
threshold12=int(threshold12)
Series1=Series1.dropna(axis=0,thresh=threshold12)
Series1=Series1.fillna(method='bfill')

#Datenum1.head()
#Series1.head()
#print(Series1)
Series1.to_csv(r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Processed Data/Series1Processed.csv", index=None, header=True)

#Patient2
rowDate2,colDate2=Datenum2.shape
threshold21=colDate2*0.95
threshold21=int(threshold21)
Datenum2=Datenum2.dropna(axis=0,thresh=threshold21)
Datenum2=Datenum2.fillna(method='bfill')
Datenum2.to_csv(r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Processed Data/Datenum2Processed.csv", index=None, header=True)

rowSeries2,colSeries2=Series2.shape
threshold22=colSeries2*0.95
threshold22=int(threshold22)
Series2=Series2.dropna(axis=0,thresh=threshold22)
Series2=Series2.fillna(method='bfill')

#Datenum2.head()
#Series2.head()
#print(Series2)
Series2.to_csv(r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Processed Data/Series2Processed.csv", index=None, header=True)

#Patient3
rowDate3,colDate3=Datenum3.shape
threshold31=colDate3*0.95
threshold31=int(threshold31)
Datenum3=Datenum3.dropna(axis=0,thresh=threshold31)
Datenum3=Datenum3.fillna(method='bfill')
Datenum3.to_csv(r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Processed Data/Datenum3Processed.csv", index=None, header=True)

rowSeries3,colSeries3=Series3.shape
threshold32=colSeries3*0.95
threshold32=int(threshold32)
Series3=Series3.dropna(axis=0,thresh=threshold32)
Series3=Series3.fillna(method='bfill')

#Datenum3.head()
#Series3.head()
#print(Series3)
Series3.to_csv(r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Processed Data/Series3Processed.csv", index=None, header=True)

#Patient4
rowDate4,colDate4=Datenum4.shape
threshold41=colDate4*0.95
threshold41=int(threshold41)
Datenum4=Datenum4.dropna(axis=0,thresh=threshold41)
Datenum4=Datenum4.fillna(method='bfill')
Datenum4.to_csv(r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Processed Data/Datenum4Processed.csv", index=None, header=True)

rowSeries4,colSeries4=Series4.shape
threshold42=colSeries4*0.95
threshold42=int(threshold42)
Series4=Series4.dropna(axis=0,thresh=threshold42)
Series4=Series4.fillna(method='bfill')

#Datenum4.head()
#Series4.head()
#print(Series4)
Series4.to_csv(r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Processed Data/Series4Processed.csv", index=None, header=True)

#Patient5
rowDate5,colDate5=Datenum5.shape
threshold51=colDate5*0.95
threshold51=int(threshold51)
Datenum5=Datenum5.dropna(axis=0,thresh=threshold51)
Datenum5=Datenum5.fillna(method='bfill')
Datenum5.to_csv(r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Processed Data/Datenum5Processed.csv", index=None, header=True)

rowSeries5,colSeries5=Series5.shape
threshold52=colSeries5*0.95
threshold52=int(threshold52)
Series5=Series5.dropna(axis=0,thresh=threshold52)
Series5=Series5.fillna(method='bfill')

#Datenum5.head()
#Series5.head()
#print(Series5)
Series5.to_csv(r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Processed Data/Series5Processed.csv", index=None, header=True)


# In[11]:


if ActivePatient==1:
    Datenum=Datenum1
    Series=Series1
elif ActivePatient==2:
    Datenum=Datenum2
    Series=Series2
elif ActivePatient==3:
    Datenum=Datenum3
    Series=Series3
elif ActivePatient==4:
    Datenum=Datenum4
    Series=Series4
elif ActivePatient==5:
    Datenum=Datenum5
    Series=Series5
elif ActivePatient=='All':
    Datenum=DatenumAll
    Series=SeriesAll
    
#Datenum=Datenum.transpose()
#Series=Series.transpose()


# In[12]:


#Feature Matrix
FeatureMatrix = pd.DataFrame()


# In[13]:


#FFT- Top 4 Instances

def FourierTransform(row):
#     row_in=np.sin(row)
##    FFTVal=row_fft=np.fft.ifft(row)
#     FFTVal = abs(scipy.fftpack.fft(row_in))
    FFTVal = abs(scipy.fftpack.fft(row))
    #FFTsp = np.fft.fft(np.sin(t))
    #FFTFreq = scripy.fft.fftfreq(len(row), d=1.0)
    FFTVal.sort()
    return np.flip(FFTVal)[0:4]

FFT = pd.DataFrame()
FFT['FFTExtracted'] = Series.apply(lambda x: FourierTransform(x), axis=1)
FFT_updated = pd.DataFrame(FFT.FFTExtracted.tolist(), columns=['FFT1', 'FFT2', 'FFT3', 'FFT4'])
#, 'FFT5', 'FFT6', 'FFT7', 'FFT8', 'FFT9', 'FFT10'
FeatureMatrix=FFT_updated
#FeatureMatrix.head()

#print(FFT_updated.columns[0:1])
#FeaturesList.append(FFT_updated.columns)
#print (FeaturesList)
#for x in FeaturesList:
#    print (x)
#FeatureMatrix.head()

#print(type(FFT_updated))
#FFT_updated.head()


# In[14]:


FFT_updated.plot(y='FFT1',figsize=(14,10)).get_figure().savefig("/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Plots/FFT/FFT1.png")
FFT_updated.plot(y='FFT2',figsize=(14,10)).get_figure().savefig("/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Plots/FFT/FFT2.png")
FFT_updated.plot(y='FFT3',figsize=(14,10)).get_figure().savefig("/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Plots/FFT/FFT3.png")
FFT_updated.plot(y='FFT4',figsize=(14,10)).get_figure().savefig("/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Plots/FFT/FFT4.png")
# FFT_updated.plot(y='FFT5',figsize=(14,10)).get_figure().savefig("/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Plots/FFT/FFT5.png")
# FFT_updated.plot(y='FFT6',figsize=(14,10)).get_figure().savefig("/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Plots/FFT/FFT6.png")
# FFT_updated.plot(y='FFT7',figsize=(14,10)).get_figure().savefig("/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Plots/FFT/FFT7.png")
# FFT_updated.plot(y='FFT8',figsize=(14,10)).get_figure().savefig("/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Plots/FFT/FFT8.png")
# FFT_updated.plot(y='FFT9',figsize=(14,10)).get_figure().savefig("/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Plots/FFT/FFT9.png")
# FFT_updated.plot(y='FFT10',figsize=(14,10)).get_figure().savefig("/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Plots/FFT/FFT10.png")


# In[15]:


#RMS

def RMS(series):
    #print (series)

    #Calculating Square
    sq=0
    n=0
    for entry in series:
        sq=sq+entry**2
        n=n+1
    
    #Calculating Mean
    mean=sq/n
    
    #Calculatung Root
    root=math.sqrt(mean)
    
    return root

RMSdf=pd.DataFrame()
RMSdf['RMS'] = Series.apply(lambda x: RMS(x), axis=1) 
#RMSdf.head()
#FeaturesList.append(RMSdf.columns)

FeatureMatrix = FeatureMatrix.merge(RMSdf, left_index=True, right_index=True)
#FeatureMatrix.head()


# In[16]:


FeatureMatrix.plot(y='RMS', figsize=(14,10)).get_figure().savefig("/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Plots/RMS/RMS.png")


# In[17]:


def PSD(series):
    f, psd1=scipy.signal.welch(series)
    return float(psd1[0:1])

psddf=pd.DataFrame()
psddf['PSD'] = Series.apply(lambda x: PSD(x), axis=1) 
PSD_updated = pd.DataFrame(psddf.PSD.tolist(), columns=['PSD'])
#psddf.head()
#print(PSD_updated)

#FeaturesList.append(PSD_updated.columns)

FeatureMatrix = FeatureMatrix.merge(psddf, left_index=True, right_index=True)
#FeatureMatrix.head()


# In[18]:


FeatureMatrix.plot(y='PSD', figsize=(14,10)).get_figure().savefig("/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Plots/PSD/PSD.png")


# In[19]:


#DWT- 7 instances

def DWT(series):
    ca, cb = pywt.dwt(series, 'haar')
    cat = pywt.threshold(ca, np.std(ca)/2, mode='soft')
    cbt = pywt.threshold(cb, np.std(cb)/2, mode='soft')
 
    signal = pywt.idwt(cat, cbt, 'haar')

    DWT8 = ca[:,:-8] #sorted in Ascending

    return DWT8

DWTdf = pd.DataFrame()

DWTdf=DWT(Series)
#DWTdf= DWT(SeriesAll)

DWTdf = pd.DataFrame(DWTdf.tolist(), columns=['DWT1', 'DWT2', 'DWT3', 'DWT4', 'DWT5', 'DWT6', 'DWT7'])
#print (DWTdf)
#print(type(DWTdf))

#FeaturesList.append(DWTdf.columns)

FeatureMatrix = FeatureMatrix.merge(DWTdf, left_index=True, right_index=True)
#FeatureMatrix.head()


# In[20]:


FeatureMatrix.plot(y='DWT1', figsize=(14,10)).get_figure().savefig("/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Plots/DWT/DWT1.png")
FeatureMatrix.plot(y='DWT2', figsize=(14,10)).get_figure().savefig("/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Plots/DWT/DWT2.png")
FeatureMatrix.plot(y='DWT3', figsize=(14,10)).get_figure().savefig("/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Plots/DWT/DWT3.png")
FeatureMatrix.plot(y='DWT4', figsize=(14,10)).get_figure().savefig("/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Plots/DWT/DWT4.png")
FeatureMatrix.plot(y='DWT5', figsize=(14,10)).get_figure().savefig("/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Plots/DWT/DWT5.png")
FeatureMatrix.plot(y='DWT6', figsize=(14,10)).get_figure().savefig("/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Plots/DWT/DWT6.png")
FeatureMatrix.plot(y='DWT7', figsize=(14,10)).get_figure().savefig("/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Plots/DWT/DWT7.png")


# In[21]:


#Final Feature Matrix

FeatureMatrix.head()
#print(type(FeatureMatrix))
#print(FeatureMatrix)
FeatureMatrix.to_csv(r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Processed Data/FeatureMatrix.csv", index=None, header=True)


# In[22]:


#PCA

#Standardizes feature matrix
FeatureMatrix = StandardScaler().fit_transform(FeatureMatrix)

pca = PCA(n_components=5)

#Components
PC = pca.fit(FeatureMatrix)
print(PC.components_)
#PC.components_.to_csv(r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Processed Data/PC-Components.csv", index=None, header=True)


# In[23]:


print(PC.explained_variance_ratio_.cumsum())


# In[24]:


PCTransform = pca.fit_transform(FeatureMatrix)
EigenValues=pd.DataFrame(data=PCTransform,columns = ['PC1', 'PC2','PC3', 'PC4','PC5'])

#print (len(EigenValues.index))
PCAindexList=[]
for x in range (0, len(EigenValues.index)):
    PCAindexList.append(x)
#print (PCAindexList)
EigenValues['Index']=PCAindexList

#EigenValues.head()


# In[25]:


#Graph of variance v/s PC-components
PCList = ['PC1','PC2','PC3','PC4','PC5']

plt.figure(figsize=(14,10))
#plt.grid()
plt.ylabel('Variance Ratio')
plt.stem(PCList,PC.explained_variance_ratio_, use_line_collection=True, label='PCA Variance Ratio')
plt.plot(PCList,PC.explained_variance_ratio_.cumsum(), color='green', label='Cumulative Sum')
plt.legend()
plt.savefig("/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Plots/PCA/PCA_Variance_Ratio.png")


# In[26]:


#Graph of Top 5 PCA Components

ax1 = EigenValues.plot.scatter(x='Index',y='PC1', figsize=(14,10), c="green")
ax2 = EigenValues.plot.scatter(x='Index',y='PC2', figsize=(14,10), c="purple")
ax3 = EigenValues.plot.scatter(x='Index',y='PC3', figsize=(14,10), c="red")
ax4 = EigenValues.plot.scatter(x='Index',y='PC4', figsize=(14,10), c="black")
ax5 = EigenValues.plot.scatter(x='Index',y='PC5', figsize=(14,10), c="blue")

ax1.set_xlabel(" ")
ax2.set_xlabel(" ")
ax3.set_xlabel(" ")
ax4.set_xlabel(" ")
ax5.set_xlabel(" ")

fig1 = ax1.get_figure()
fig2 = ax2.get_figure()
fig3 = ax3.get_figure()
fig4 = ax4.get_figure()
fig5 = ax5.get_figure()

fig1.savefig("/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Plots/PCA/PCA1.png")
fig2.savefig("/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Plots/PCA/PCA2.png")
fig3.savefig("/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Plots/PCA/PCA3.png")
fig4.savefig("/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Plots/PCA/PCA4.png")
fig5.savefig("/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Plots/PCA/PCA5.png")


# In[27]:


#print (FeaturesList)
FeaturesList=['FFT1', 'FFT2', 'FFT3', 'FFT4', 'RMS', 'PSD', 'DWT1', 'DWT2', 'DWT3', 'DWT4', 'DWT5', 'DWT6', 'DWT7']

map=pd.DataFrame(PC.components_, columns=FeaturesList)
#print(map)
map.to_csv(r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Processed Data/NewFeatureMatrix.csv", index=None, header=True)
plt.figure(figsize=(12,6))
sns.heatmap(map,cmap='YlOrRd').figure.savefig("/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 1/Plots/PCAHeatMap.png")
#RdYlBu, magma, twilight

#Possible values are: Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, icefire, icefire_r, inferno, inferno_r, jet, jet_r, magma, magma_r, mako, mako_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, rocket, rocket_r, seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, twilight, twilight_r, twilight_shifted, twilight_shifted_r, viridis, viridis_r, vlag, vlag_r, winter, winter_r


# In[ ]:




