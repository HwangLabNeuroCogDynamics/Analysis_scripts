#!/usr/bin/env python
# coding: utf-8

# In[3]:


import mne
from mne.datasets import sample
from mne import io
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
from mne.preprocessing import ICA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import fnmatch


ROOT = '/home/dcellier/RDSS/dev_1_f/1606_63_Baseline.mff'
save_ROOT='/home/dcellier/RDSS/dev_1_f/'

raw=mne.io.read_raw_egi(ROOT,montage=mne.channels.read_montage(kind='GSN-HydroCel-128'),preload=True, verbose=True)

dataChannels = ['E2','E3','E4','E5','E6','E7','E9','E10','E11','E12','E13','E15','E16','E18','E19','E20','E22','E23','E24','E26','E27','E28','E29','E30','E31','E33','E34','E35','E36','E37','E39','E40','E41','E42','E45','E46','E47','E50','E51','E52','E53','E54','E55','E58','E59','E60','E61','E62','E65','E66','E67','E70','E71','E72','E75','E76','E77','E78','E79','E80','E83','E84','E85','E86','E87','E90','E91','E92','E93','E96','E97','E98','E101','E102','E103','E104','E105','E106','E108','E109','E110','E111','E112','E115','E116','E117','E118','E122','E123','E124']
#this is the list of 'real' channels that Justin sent, all others are external electrodes
extChannels = [e['ch_name'] for e in raw.info['chs'] if (e['ch_name'] not in dataChannels)]
extChannels.remove('VBeg')
extChannels.remove('STI 014')
extChannels.remove('E129')

ch_type_dict={}
for c in extChannels:
	ch_type_dict[c]='ecg'
raw.set_channel_types(ch_type_dict)

raw.plot(block=True,scalings={'eeg':.0003})

two_second_ind=1308
new_events=[]
while two_second_ind <= 179308:
    new_events.append([two_second_ind, 0, 7])
    two_second_ind+=2000

raw.add_events(np.asarray(new_events),'STI 014',replace=True)


two_sec_eps=raw

#two_sec_eps.plot(block=True,scalings={'eeg':.0003})

two_sec_eps.load_data()
two_sec_eps=two_sec_eps.filter(1,50)


two_sec_eps.plot(block=True, scalings={'eeg':.0003}, title='Select channels to interpolate')
bads=two_sec_eps.info['bads']

bads_msg=input("The channels you marked as bad are: "+str(bads)+" Are you sure you want to continue? [y/n]")

if bads_msg=='y':
    two_sec_eps.interpolate_bads()
else:
    print('no interpolation')


#layout=mne.channels.read_montage(kind='GSN-HydroCel-128')
layout=mne.channels.find_layout(two_sec_eps.info)

two_sec_eps, r=mne.set_eeg_reference(two_sec_eps,ref_channels=dataChannels+extChannels) 

ica = ICA(n_components=90,random_state=25,method='infomax')
picks=mne.pick_types(two_sec_eps.info, meg=False, eeg=True, eog=False,ecg=False)
ica.fit(two_sec_eps,picks=picks)

eog_ic=[]
for ch in ['E25','E17','E8','E21','E14','E125','E126','E127','E128']: #insert EOG channels
    #ecg_epochs=create_ecg_epochs(data,ch_name=ch) # ?
    eog_idx,scores=ica.find_bads_eog(two_sec_eps,ch_name=ch)
    eog_ic.append(eog_idx)

reject_ic=[]
for eog_inds in eog_ic:
    for ele in eog_inds:
        if ele not in reject_ic:
            reject_ic.append(ele)

print(reject_ic)                

ica.exclude=[]
ica.exclude.extend(reject_ic) #which IC's to exclude

ica.plot_components(picks=range(50),ch_type=None,title='1-50',inst=two_sec_eps) 

ica.plot_components(picks=range(50,90),ch_type=None,title='50-90',inst=two_sec_eps)

verification_ic=input('The ICs marked for rejection are: ' + str(ica.exclude) +'\n Are you sure you want to proceed? [y/n]: ')

ica.apply(two_sec_eps, exclude=ica.exclude)

two_sec_eps=mne.Epochs(two_sec_eps,new_events,event_id={'twoSec':7},tmin=0,tmax=2,reject_by_annotation=False)
two_sec_eps.plot(block=True,title='Select bad epochs',scalings={'eeg':.0003})

two_sec_eps.plot_psd_topomap(cmap='interactive')

save_msg=input('Are you sure you want to continue onto saving? [y/n]: ')


if save_msg=='y':
    two_sec_eps.save(save_ROOT+'sample_preproc-epo.fif')



