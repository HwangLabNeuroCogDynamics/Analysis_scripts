#!/usr/bin/env python
# coding: utf-8

import mne
from mne.datasets import sample
from mne import io
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
from mne.preprocessing import ICA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import autoreject
from autoreject import *
import os
from os import path
import pickle


ROOT = '/home/dcellier/RDSS/CMI_data/MIPDB/EEGData/'
subject_files=[]
for filename in os.listdir(ROOT): #compiling the subjects downloaded from MIPDB
    if (not filename.endswith('.txt')) and (not filename=='EEG paradigm info Misc_Readme') and (not filename=='subs_without_EEG'):
        print(filename)
        subject_files.append(filename)
        
for f in subject_files: #checking to see who is already preprocessed
    thisSub=f
    preproc_file=ROOT+thisSub+'/EEG/preproc/'+thisSub+'_epoch_eyes'
    if (os.path.exists(preproc_file+'Closed')) and (os.path.exists(preproc_file+'Open')):
        subject_files.remove(thisSub)
# if the subject has already been preprocessed, remove it from the list of people to process


subs={} #compiling the raw files from those who are not yet preprocessed
for files in subject_files:
    thisSub=files
    raw_file=ROOT+thisSub+'/EEG/raw/raw_format/'+thisSub+'001.raw'
    if os.path.exists(raw_file):
        subs[thisSub]=raw_file

print(subs)

for fileN in subs.keys():
    sub=fileN
    raw_file=subs[fileN]
    raw=mne.io.read_raw_egi(raw_file,montage=mne.channels.read_montage(kind='GSN-HydroCel-128'),preload=True, verbose=True)
    #raw2=mne.io.read_raw_egi(raw_file,montage=mne.channels.read_montage(kind='GSN-HydroCel-128'),preload=True, verbose=True)
    
    #raw.info['chs'][44:129]
    dataChannels = ['E2','E3','E4','E5','E6','E7','E9','E10','E11','E12','E13','E15','E16','E18','E19','E20','E22','E23','E24','E26','E27','E28','E29','E30','E31','E33','E34','E35','E36','E37','E39','E40','E41','E42','E45','E46','E47','E50','E51','E52','E53','E54','E55','E58','E59','E60','E61','E62','E65','E66','E67','E70','E71','E72','E75','E76','E77','E78','E79','E80','E83','E84','E85','E86','E87','E90','E91','E92','E93','E96','E97','E98','E101','E102','E103','E104','E105','E106','E108','E109','E110','E111','E112','E115','E116','E117','E118','E122','E123','E124']
    #this is the list of 'real' channels that Justin sent, all others are external electrodes
    extChannels = [e['ch_name'] for e in raw.info['chs'] if ((e['ch_name'] not in dataChannels) and (e in raw.info['chs'][:129]))]
    extChannels
    randChannels=[e for e in raw.info['ch_names'] if ((e not in dataChannels) and (e not in extChannels))]
    #stimChannels=raw.get_data[raw.info['ch_names'].index('STI 014')]
    raw_stimCH=raw.copy() #making a copy that retains its stim channels for finding events later
    raw=raw.drop_channels(randChannels)
    ch_type_dict={}
    for c in extChannels:
        ch_type_dict[c]='ecg'
    raw.set_channel_types(ch_type_dict)
    #aw.info
    text_msg1=raw_input('Ready to move on? [y/n]: ')
    
    # # Cutting and concatenating raw file into eyes open/eyes closed conditions
    
    events = mne.find_events(raw_stimCH, verbose=True)
    print(events)
    raw.info['events']=events # i think that 1=EO and 2=EC
    print('\n\n\n\n~~~~~~~ SEPERATING INTO EYES OPEN AND EYES CLOSED CONDITIONS ~~~~~~~~~\n\n')
    
    firstEv=events[0]
    secondEv=events[1]
    diff=np.around(raw.times[secondEv[0]]-raw.times[firstEv[0]],decimals=-1) #difference in seconds (rounded to nearest 10th) bween 1st and 2nd event
    if diff==20: #if the first event is 20 seconds long its eyes open
        if firstEv[2]==1: # and we want the trigger code (1 or 2) to translate to EC or EO accurately
            task1='EO'
            task2='EC'
        else:
            task1='EC'
            task2='EO'
    elif diff==40: #if the first event is 40 sec long its eyes closed
        if firstEv[2]==2:
            task2='EC'
            task1='EO'
        else:
            task1='EC'
            task2='EO'
    else:
        print('\n\n\n\n ERROR!!!!!!!!!!! \n the timing of the eyes closed/eyes open is not 20/40 seconds. Please skip this subject and look in detail at events')
    eyes_open_idx=[]
    eyes_closed_idx=[]
    for n in range(len(events)):
        event=events[n]
        startTime=event[0]
        if n==(len(events)-1): #if its the last event it might not last all 20 or 40 seconds, so give it whatever data there is
            endTime=len(raw.times)-1 #ie, the index of the last element of raw.times
        else: # otherwise the end time of this event is the start time of the next one
            endTime=events[n+1][0]-1 #ie, the index of the time right before the next event
        if (event[2]==1 and task1=='EO') or (event[2]==2 and task2=='EO'): #assign the event to the correct trigger code
            eyes_open_idx.append((startTime,endTime))
        elif (event[2]==2 and task2=='EC')or (event[2]==1 and task1=='EC'):
            eyes_closed_idx.append((startTime,endTime))
        else:
            print('ERROR \n\n\n trigger codes are not only 1 and 2! see events for details')
    print(eyes_open_idx)
    print(eyes_closed_idx)
    #grabbing out slices of eyes open to later concatenate together, minus first 3 secs of data
    sec2concat=[]
    for idx in eyes_open_idx: #excluding last eyes open event because it doesn't seem to have lasted the whole 20secs
        startTime=raw.times[idx[0]]
        endTime=raw.times[idx[1]]
        eoEvent=raw.copy().crop((startTime+3),(endTime))
        sec2concat.append(eoEvent)
    
    sec2concat[0].append(sec2concat[1:])
    eyes_open_raw=sec2concat[0]
    print(eyes_open_raw)
    #grabbing out slices of eyes closed to later concatenate together, minus first 3 secs of data
    sec2concat=[]
    for idx in eyes_closed_idx:
        startTime=raw.times[idx[0]]
        endTime=raw.times[idx[1]]
        ecEvent=raw.copy().crop((startTime+3),(endTime))
        sec2concat.append(ecEvent)
    
    sec2concat[0].append(sec2concat[1:])
    eyes_closed_raw=sec2concat[0]
    print(eyes_closed_raw)

    
    #eyes_open=mne.Epochs(raw_fiir,events=events,event_id={'eyes open':1},tmin=0,tmax=20,baseline=None,decim=2,picks=our_picks)
    # no baseline correction here
    # downsampled by factor of 2
    # I cut out the first couple of seconds of data for if the event trigger is being sent when they are hearing instructions?
    #eyes_closed=mne.Epochs(raw_fiir,events=events,event_id={'eyes closed':2},tmin=3,tmax=40,decim=2,baseline=None,picks=our_picks)
    #eyes_closed.plot() #is this really the eyes closed data?? Seems very noisy
    
    
    # In[80]:
    
    
    eyes_closed_raw.plot_psd(fmax=45,picks=mne.pick_types(raw.info,meg=False,eeg=True,eog=False,exclude=['E129'])) #sanity check
    #eyes_closed.plot()
    
    
    # # Applying high and low pass filters (1 and 50)
    
    # In[35]:
    
    
    EC_f=eyes_closed_raw.copy()
    EC_f.filter(1,50)
        #can choose different methods of filtering. here it's using the default 'overlap-add' FIR filtering, can
            # also use IIR forward/backward filtering
    #EC_f.plot(n_channels=129)#n_channels=129,group_by#='original')
    
    EO_f=eyes_open_raw.copy()
    EO_f.filter(1,50)
    #EO_f.plot(n_channels=129)#n_channels=129,group_by='original')
    
    
    # # Select bad electrodes from "real" data to interpolate
    
    # In[36]:
    
    
    EC_fi=EC_f.copy()
    EC_DataCH=(EC_fi.copy()).drop_channels(extChannels)
    tryAgain=1
    while tryAgain:
            EC_DataCH.plot(block=True, title='EYES CLOSED') #pauses script while i visually inspect data and select which channels to delete
            bads_EC=EC_DataCH.info['bads']
            text_msg2=raw_input('The channels you marked as bad are: '+str(bads_EC)+' \n Are you ready to interpolate? [y/n]: ')
            if text_msg2=='y':
                    EC_fi.info['bads']=bads_EC
                    EC_fi=EC_fi.interpolate_bads()
                    tryAgain=0
            elif text_msg2=='n':
                    #EC_DataCH.plot(block=True)
                    #text_msg3=raw_input("Which channels do you want to interpolate? Type 'DONE' when finished \n")	
                    #bads=[]
                    #notDone=1
                    #while notDone==1:
                    tryAgain=1
            else:
                    print('invalid entry: '+text_msg2)
                    tryAgain=1
    
    EO_fi=EO_f.copy()
    EO_DataCH=(EO_fi.copy()).drop_channels(extChannels)
    tryAgain=1
    while tryAgain:
            EO_DataCH.plot(block=True, title='EYES OPEN') #pauses script while i visually inspect data and select which channels to delete
            bads_EO=EO_DataCH.info['bads']
            text_msg2=raw_input('The channels you marked as bad are: '+str(bads_EO)+' \n Are you ready to interpolate? [y/n] ')
            if text_msg2=='y':
                    EO_fi.info['bads']=bads_EO
                    EO_fi=EO_fi.interpolate_bads()
                    tryAgain=0
            elif text_msg2=='n':
                    tryAgain=1
            else:
                    print('invalid entry: '+text_msg2)
                    tryAgain=1
    
    
    # # https://github.com/mne-tools/mne-python/blob/master/mne/preprocessing/bads.py automatic detection of bads 
       # uses z-scoring
    
    
    # # Importing Montage
    
    # In[37]:
    
    
    layout=mne.channels.read_montage(kind='GSN-HydroCel-128')
    #layout.plot() 
    
    
    # In[41]:
    
    
    our_picks=mne.pick_types(EC_fi.info,meg=False,eeg=True,eog=False,ecg=False)
    EC_fi=EC_fi.set_montage(layout)
    
    
    # # Re-referencing using global avg referencing
    
    
    # SHOULD THIS COME BEFORE ICA? see manuscript blurb
    ## ?? should we use all scalp electrodes too? not truly global until use all electrodes
    EC_fir, r2 = mne.set_eeg_reference(EC_fi,ref_channels=dataChannels+extChannels) 
    #EC_fir.plot(n_channels=129)
    EO_fir, r2 = mne.set_eeg_reference(EO_fi,ref_channels=dataChannels+extChannels) 
    #EO_fir.plot(n_channels=129)
    
    
    # # ICA
    
    
    
    ## fitting IC's
    print('\n\n\n\n RUNNING ICA ~~~~~~~ \n\n')
    
    datasets=[EC_fir,EO_fir]
    ica_data={}
    
    for data in datasets:
        ica = []
        ica = ICA(n_components=50,random_state=25,method='infomax')
        ica.fit(data,picks=our_picks)
        #ica.info
    
        eog_ic=[]
        for ch in ['E25','E17','E8']: #insert EOG channels
            eog_idx,scores=ica.find_bads_eog(data,ch_name=ch)
            eog_ic.append(eog_idx)
    
        print(eog_ic)
    
        ecg_ic=[]
        for ch in []: # insert ECG channels
            ecg_idx,scores=ica.find_bads_ecg(data,ch_name=ch)
            ecg_ic.append(ecg_idx)
    
        print(ecg_ic)
    
        reject_ic=[]
        for eog_inds in eog_ic:
            for ele in eog_inds:
                if ele not in reject_ic:
                    reject_ic.append(ele)
        for ecg_inds in ecg_ic:
            for ele in ecg_inds:
                if el not in reject_ic:
                    reject_ic.append(el)
        
        print(reject_ic)                
    
        ica.exclude=[]
        ica.exclude.extend(reject_ic) #which IC's to exclude
    
        #ica.exclude #excluding the eog/ecg ic's
        if data==EC_fir:
            name='EC'
        else:
            name='EO'
        ica_data[name]=(ica,data,reject_ic)
    
    
    # In[56]:
    
    repeatICA=1
    while repeatICA:
            EC_firi=[]
            EO_firi=[]
            for icDat in ica_data.keys():
                    data=ica_data[icDat][1]
                    reject_ic=ica_data[icDat][2]
                    ica=ica_data[icDat][0]
                    
                    plot=1
                    while plot:  
                            ica.plot_components(picks=range(50),ch_type=None,title=icDat,inst=data) #needs the channel locations
                            #while len(ica.exclude)==len(reject_ic):
                                    #continue
                            verification_ic=raw_input('The ICs marked for rejection are: ' + str(ica.exclude) +'\n Are you sure you want to proceed? [y/n]: ')
                            if verification_ic=='y':
                                    #print(len(reject_ic))
                                    #print(len(ica.exclude))	
                                    #ica.exclude.extend(bad_ics)
                                    plot=0
                            else:
                                    print('Please select which ICs you would like to reject')
                    ica_data[icDat]=(ica,data,reject_ic)
    
            EC_firi=EC_fir.copy()
    
            EC_firi=(ica_data['EC'][0]).apply(EC_firi,exclude=ica_data['EC'][0].exclude)
            EO_firi=EO_fir.copy()
            EO_firi=ica_data['EO'][0].apply(EO_firi,exclude=ica_data['EO'][0].exclude)
            #raw_fii.plot()
            print(len(ica_data['EC'][0].exclude))
    
            ica_data['EC'][0].plot_overlay(EC_fir,exclude=ica_data['EC'][0].exclude)#,picks=our_picks)
            ica_data['EO'][0].plot_overlay(EO_fir,exclude=ica_data['EO'][0].exclude)#,picks=our_picks)
            cont=raw_input('Continue on to epoching? [y/n]: ')
            if cont=='y':
                    repeatICA=0
    
    
    
    # # Epoching into 2 second, non overlapping time windows
    
    # In[57]:
    
    
    eyes_closed=EC_firi
    eyes_open=EO_firi
    
    
    # In[58]:
    print('\n\n\n\n EPOCHING INTO 2 SEC WINDOWS ~~~~~~~ \n\n')
    
    epoch_array=[]
    for t in eyes_closed.times[0::1000]: #grabbing sample # (index) of ea 2 second data point
        epoch_array.append([int(list(eyes_closed.times).index(t)),int(0),int(7)])  
    
    
    eyes_closed.info['events']=epoch_array
    

    
    realData=mne.pick_types(raw.info,meg=False,eeg=True,eog=False,exclude=extChannels+randChannels,selection=dataChannels)
    

    # SHOULD I DOWNSAMPLE, REJECTION PARAMS??
    twoSec_EC_epoch=mne.Epochs(eyes_closed, baseline=(None,None),events=epoch_array,tmin=0,tmax=2,event_id={'twoSec':7},picks=realData,reject_by_annotation=False,flat=None,reject=None)#,reject={'eeg':40e-6},flat={'eeg'=?}) #have to change this so it's not blurring 				over the eyes closed and eyes open
    
    
    epoch_array_EO=[]
    for t in eyes_open.times[0::1000]: #grabbing sample # (index) of ea 2 second data point
        epoch_array_EO.append([int(list(eyes_open.times).index(t)),int(0),int(7)])  
    eyes_open.info['events']=epoch_array_EO
    twoSec_EO_epoch=mne.Epochs(eyes_open,baseline=(None,None),events=epoch_array,tmin=0,tmax=2,event_id={'twoSec':7},picks=realData,reject_by_annotation=False,flat=None,reject=None)#,reject={'eeg':40e-6},flat={'eeg'=?})
    

    plotEpoch=1
    while plotEpoch:
			thisEp=[]
			thisEp=twoSec_EO_epoch.copy()
			thisEp.plot(block=True,n_epochs=2,n_channels=15, title='eyes open')
			thisEp.plot_psd_topomap(layout=layout,cmap='interactive')
			avg_EO=thisEp.average()
			avg_EO.plot()
			bads=raw_input('Are you sure you want to continue? [y/n]: ')
			if bads=='y':
				twoSec_EO_epoch=thisEp
				plotEpoch=0
			elif bads=='n':
				continue
			else:
				print('oops, please indicate which epochs you would like to reject')
    
    plotEpoch=1
    while plotEpoch:
			thisEp=[]
			thisEp=twoSec_EC_epoch.copy()
			thisEp.plot(block=True,n_epochs=2,n_channels=15, title='eyes closed')
			thisEp.plot_psd_topomap(layout=layout,cmap='interactive')
			avg_EC=thisEp.average()
			avg_EC.plot()
			bads=raw_input('Are you sure you want to continue? [y/n]: ')
			if bads=='y':
				twoSec_EC_epoch=thisEp
				plotEpoch=0
			elif bads=='n':
				continue
			else:
				print('oops, please indicate which epochs you would like to reject')
    #twoSec_EC_epoch.save(ROOT+sub+'/EEG/preproc/'+sub+'_epoch_eyesClosed_epo.fif')
    # twoSec_EO_epoch.save(ROOT+sub+'/EEG/preproc/'+sub+'_epoch_eyesOpen_epo.fif')
    
    ## SAVING OUT ##
    
    print('\n\n\n\n SAVING OUT INFO ~~~~~~~ \n\n'+ROOT+sub+'/EEG/preproc/'+sub+'......DONE!')
    import pickle


    def save_object(obj,filename):
    
			with open(filename,'wb') as output:
        
			
				pickle.dump(obj,output,pickle.HIGHEST_PROTOCOL)

    save_object(twoSec_EC_epoch,filename=ROOT+sub+'/EEG/preproc/'+sub+'_epoch_eyesClosed')
    save_object(twoSec_EC_epoch,filename=ROOT+sub+'/EEG/preproc/'+sub+'_epoch_eyesOpen')
  	#save_object(twoSec_EO_epoch,filename=ROOT+sub+'/EEG/preproc/'+sub+'_epoch_eyesOpen')
	#exit_msg=raw_input("Move on to next sub? [y/n]: ")


    # Autoreject stuff
    
    # In[67]:
    
    
    #avg_EC=twoSec_EC_epoch.average()
    #avg_EC.plot()
    #avg_EO=twoSec_EO_epoch.average()
    #avg_EO.plot()
    
    
    # # automatic selection of epochs to reject
    
    # In[68]:
    
    
    #reject= get_rejection_threshold(twoSec_EC_epoch)
    #reject
    
    
    # In[79]:
    
    
    #twoSec_EC_epoch.load_data()
    #ar=AutoReject()
    #ar.fit(twoSec_EC_epoch)
    #ar.reject_log()
    
    
    # In[70]:
    
    
    #ECep_clean=ar.transform(twoSec_EC_epoch)
    #EC_avg=ECep_clean.average()
    
    
    # In[72]:
    
    
    #set_matplotlib_defaults(plt)
    
    #fig, axes = plt.subplots(2, 1, figsize=(6, 6))
    
    #for ax in axes:
     #   ax.tick_params(axis='x', which='both', bottom='off', top='off')
      #  ax.tick_params(axis='y', which='both', left='off', right='off')
    
    #ylim = range(1,300)
    #avg_EC.pick_types(meg=False,eeg=True,eog=False,exclude=extChannels+randChannels,selection=dataChannels)
    #avg_EC.plot(exclude=[], ylim=ylim, show=False)
    #axes[0].set_title('Before autoreject')
    #EC_avg.pick_types(meg=False,eeg=True,eog=False,exclude=extChannels+randChannels,selection=dataChannels)
    #EC_avg.plot(exclude=[], ylim=ylim)
    #axes[1].set_title('After autoreject')
    #plt.tight_layout()
    
    
    # In[75]:
    
    
    #ar.get_reject_log(twoSec_EC_epoch).plot()
    #ar.get_reject_log(twoSec_EO_epoch).plot()
    
    
    # In[ ]:
    
    
    
