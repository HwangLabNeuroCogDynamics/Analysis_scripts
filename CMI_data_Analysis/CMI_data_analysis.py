#!/usr/bin/env python
# coding: utf-8

# # Fourier transforming 2 second epochs

# In[1]:


import mne
from mne.datasets import sample
from mne import io
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
from mne.preprocessing import ICA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import autoreject
#from autoreject import *
import scipy
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram,linkage
from scipy.stats.mstats import zscore
from fooof import FOOOF

ROOT = '/home/dcellier/RDSS/CMI_data/MIPDB/EEGData/'


# In[20]:


import pickle
sub='A00056054'
def read_object(filename):

                ''' short hand for reading object because I can never remember pickle syntax'''

                o = pickle.load(open(filename, "rb"))     

                return o
twoSec_EO_epoch=read_object(ROOT+sub+'/EEG/preproc/'+sub+'_epoch_eyesOpen')
twoSec_EC_epoch=read_object(ROOT+sub+'/EEG/preproc/'+sub+'_epoch_eyesClosed')


# In[21]:


twoSec_EC_epoch.plot_psd_topomap(ch_type='eeg', normalize=True,cmap='inferno',outlines='head') 
#interactive mode possible in ipython


# In[51]:


twoSec_EC_epoch.info#['ch_names']


# In[178]:



#should be using the scrubbed epochs but its ok for now
psds,freqs=mne.time_frequency.psd_welch(twoSec_EO_epoch,fmin=1,fmax=50,tmax=2,n_overlap=(len(twoSec_EO_epoch.times)*.125))
# default FFT is 256
# n_overlap at 12.5% like sangtae
# length of ea Welch segment defaults to 256
# *** this function is taking issue w/ picks=realData-- don't know why ****
# there is another function, mne.time_frequency.psd_array_welch-- what's the diff?
# psds = psd's, shape (n_epochs,n_channels,n_freqs)
    # should I average over n_epochs now?
        # and then for IAF it would be the avg psd's for occiptal electrodes & alpha freqs ?
        


# In[179]:


# electrode clustering: run on all sbujects, average over
# similarity matrix bween every electrode and every other, builds up hierarchically
# w matlab: given all [Power Spectrum X channel] matrix, pdist function yields something that you feed into dendrogram (find scipy equivalent)

# average in time domain (of epochs) across clusters, then run psd_welch on that. FOOOF takes a matrix of the psd's
# frontal-central cluster and parietal occipital cluster manually for now, run FOOOF and give: peak freq, amp, bandwidth for theta/beta and alpha in parietal/occ cluster
# give the slope and y-int for ea of the 2 clusters
# 13 neural data points per person: beta fareq, beta amp, beta bandwidth, theta freq, theta amp, thea bandwidth, alpha freq, alpha amp, alpha bandwidth, frontal slope, frontal yint, posterior slope, yint   
# demographic data points (age/sex)


# In[180]:


psds.shape


# In[181]:


chFreqs=np.mean(psds,axis=0)
chFreqs.shape


# In[182]:


#np.matrix(psds)
chFreqs


# In[183]:


Y=pdist(zscore(np.corrcoef(chFreqs)))
Y=pdist(chFreqs)
Z= linkage(Y)
Z.shape


# In[187]:


#Z


# In[185]:


#val=pdist(psd,'euclidean')
#val.shape
#val


# In[195]:


fig=plt.figure(figsize=(10,15))
dend=dendrogram(Z,128,leaf_font_size=10,orientation='left',color_threshold=0.5*max(Z[:,2]),labels=np.asarray(twoSec_EO_epoch.info['ch_names']),truncate_mode='mlab')
# changed the threshold from matlab default of 0.7 to 0.5


# In[223]:


dend.keys()
print(dend['ivl'])
from collections import defaultdict


# In[236]:


len(dend['ivl'])


# In[222]:


#cluster_idxs
#dend['color_list']


# In[216]:


#print(cluster_idxs.items())
#print(cluster_idxs.keys())
#cluster_idxs['b']


# In[251]:


def color_map(chs,coordinates,dend): 
    
    ## chs= list of electrodes to be 3d plotted, 
    ## coordinates= the coordinates of the electrodes to be plotted
    ## dend= a dendrogram of clustered electrodes 
    
    cluster_idxs = defaultdict(list) # this is creating a dict of color:channel inds in dend
    for c, pi in zip(dend['color_list'], dend['icoord']):
        for leg in pi[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - int(i)) < 1e-5:
                cluster_idxs[c].append(int(i))
                
    coor_colors=np.zeros(len(chs))
    coor_colors=list(coor_colors)
    for color in cluster_idxs.keys():
        leaf_inds=cluster_idxs[color] # list of leaf inds for that color cluster
        for ind in leaf_inds:
            ch_name=dend['ivl'][ind] # grab the leaf label 
            coor_colors[chs.index(ch_name)]=color
            #and match it to a label in dataChannels, 
            #then insert the color corresponding that ch_name
            # into coor_colors
    
    #%pylab inline
    from mpl_toolkits.mplot3d import Axes3D

    fig=plt.figure()
    fig2=plt.figure()
    ax=fig.add_subplot((111),projection='3d')
    ax2=fig.add_subplot((111))
    for coor in range(len(coor2plot)):
        ax.scatter(xs=coor2plot[coor,0],ys=coor2plot[coor,1],zs=coor2plot[coor,2],c=coor_colors[coor],depthshade=True)
        ax2.scatter(x=coor2plot[coor,0],y=coor2plot[coor,1],c=coor_colors[coor])
        ax.text(x=coor2plot[coor,0],y=coor2plot[coor,1],z=coor2plot[coor,2],s=chs[coor])
        ax2.text(x=coor2plot[coor,0],y=coor2plot[coor,1],s=chs[coor])
    return plt.show(),coor_colors



# In[244]:


#ROOT2 = '/home/dcellier/RDSS/CMI_data/MIPDB/EEGData/'
#raw_file=ROOT2+"A00056054/EEG/raw/raw_format/A00056054001.raw" #"A00051955/EEG/raw/raw_format/A00051955001.raw"#
#raw=mne.io.read_raw_egi(raw_file,montage=mne.channels.read_montage(kind='GSN-HydroCel-129'),preload=True, verbose=True)
#raw2=mne.io.read_raw_egi(raw_file,montage=mne.channels.read_montage(kind='GSN-HydroCel-128'),preload=True, verbose=True)

#chs_indices=mne.pick_types(raw.info,eeg=True)
#chs_list=raw.info['ch_names'][:129]
#len(chs_list)

chs_list=['E1','E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12',
          'E13', 'E14', 'E15', 'E16', 'E17', 'E18', 'E19', 'E20',
          'E21', 'E22', 'E23', 'E24', 'E25', 'E26', 'E27', 'E28', 'E29',
          'E30', 'E31', 'E32', 'E33', 'E34', 'E35', 'E36', 'E37', 'E38', 
          'E39', 'E40', 'E41', 'E42', 'E43', 'E44', 'E45', 'E46',
          'E47','E48', 'E49', 'E50', 'E51', 'E52',
          'E53', 'E54', 'E55', 'E56', 'E57',
          'E58', 'E59', 'E60', 'E61', 'E62', 'E63', 'E64',
          'E65', 'E66', 'E67', 'E68', 'E69', 'E70', 'E71',
          'E72', 'E73', 'E74', 'E75', 'E76', 'E77', 'E78', 'E79',
          'E80', 'E81', 'E82', 'E83', 'E84',
          'E85', 'E86','E87', 'E88', 'E89', 'E90', 'E91', 'E92', 'E93',
          'E94', 'E95', 'E96', 'E97', 'E98', 'E99', 'E100', 'E101',
           'E102', 'E103', 'E104', 'E105', 'E106', 'E107', 'E108',
           'E109', 'E110', 'E111', 'E112', 'E113', 'E114', 'E115',
          'E116', 'E117', 'E118', 'E119', 'E120', 'E121', 
          'E122', 'E123', 'E124', 'E125', 'E126', 'E127', 'E128','E129']

dataChannels = ['E2','E3','E4','E5','E6','E7','E9','E10','E11','E12',
    'E13','E15','E16','E18','E19','E20','E22','E23','E24','E26',
    'E27','E28','E29','E30','E31','E33','E34','E35','E36','E37',
    'E39','E40','E41','E42','E45','E46','E47','E50','E51','E52','E53','E54','E55','E58','E59','E60','E61',
    'E62','E65','E66','E67','E70','E71','E72','E75','E76','E77','E78','E79','E80','E83','E84',
    'E85','E86','E87','E90','E91','E92','E93','E96','E97','E98','E101','E102','E103','E104',
    'E105','E106','E108','E109','E110','E111','E112','E115','E116','E117','E118','E122','E123','E124']
#names=twoSec_EO_epoch.info['ch_names']


#plot
#twoSec_EO_epoch.info['ch_names'].index('E4')
#dataInds

layout=mne.channels.read_montage(kind='GSN-HydroCel-129')
coordinates=layout.pos
coordinates=coordinates[3:] #slicing to exclude FidNz, FidT9, FidT10
len(coordinates)

#new list of coordinates where the indices of the coords match the dataCh inds
coor2plot=np.zeros(len(dataChannels))
coor2plot=list(coor2plot)
for i in chs_list: # iter thru elems in all channels list
    if i in dataChannels: # if the elem is also in dataChannels
        coor2plot[dataChannels.index(i)]=list(coordinates)[chs_list.index(i)]
        # then fill coor2plot with the coordinates of the elem, but make sure it's
        # inserted at an index that matches the index of the corresponding
        # dataChannels label

#coor2plot=[list(coordinates[chs_list.index(i)]) for i in chs_list if i in dataChannels] 
# getting the coordinates of the data channels
coor2plot=np.asarray(coor2plot)
coor2plot[:,2].shape


# In[247]:


#3333333coor2plot


# In[243]:


#coor2plot


# In[231]:


len(chs_list)


# In[257]:


plot,colors=color_map(dataChannels,coor2plot,dend)

input('continue?')
# In[258]:


len(b)


# In[93]:


epEC=twoSec_EC_epoch
epEO=twoSec_EO_epoch


# In[31]:


epEC.average().plot()
epEO.average().plot()


# In[36]:


psdsEC,freqsEC=mne.time_frequency.psd_welch(epEC.average(),fmin=1,fmax=50,tmax=2,n_overlap=(len(twoSec_EC_epoch.times)*.125))
psdsEO,freqsEO=mne.time_frequency.psd_welch(epEO.average(),fmin=1,fmax=50,tmax=2,n_overlap=(len(twoSec_EO_epoch.times)*.125))


# In[37]:


avg_psds_EC=np.mean(psdsEC,axis=0)
avg_psds_EC.shape
avg_psds_EO=np.mean(psdsEO,axis=0)


# In[38]:


freqsEC.shape


# In[42]:


fmEC= FOOOF()


freq_range=[1,50]

fmEC.report(freqsEC,avg_psds_EC,freq_range)


# In[43]:


fmEO= FOOOF()

freq_range=[1,50]

fmEO.report(freqsEO,avg_psds_EO,freq_range)


# In[57]:


theta_band=[4,8]
alpha_band=[8,12]
beta_band=[15,30]


# In[91]:


from fooof import FOOOF, FOOOFGroup
from fooof.synth import gen_group_power_spectra,param_sampler
from fooof.analysis import get_band_peak, get_band_peak_group

get_ipython().run_line_magic('pinfo', 'get_band_peak')


# In[74]:


alphas=get_band_peak(fmEO.peak_params_,alpha_band,ret_one=True)
print('Alpha CF: ',alphas[0])
print('Amp: ', alphas[1])
print('Bandwidth: ', alphas[2])


# In[87]:


betas=get_band_peak(fmEO.peak_params_,beta_band,ret_one=False)
print('Beta CF: ',betas[0])
print('Amp: ', betas[1])
print('Bandwidth: ', betas[2])


# In[88]:


betas


# In[84]:


thetas=get_band_peak(fmEO.peak_params_,theta_band,ret_one=True)
print('Theta CF: ',thetas[0])
print('Amp: ', thetas[1])
print('Bandwidth: ', thetas[2])


# In[172]:





# In[ ]:




