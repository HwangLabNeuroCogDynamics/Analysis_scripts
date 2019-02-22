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
#import autoreject
#from autoreject import *
import os
from os import path
import fnmatch
import pickle

hawkid=input('\n \nPlease input your hawkID with no spaces: ')
ROOT = '/home/'+hawkid+'/RDSS/CMI_data/MIPDB/EEGData/'
save_ROOT='/data/backed_up/shared/CMI_data/MIPDB/preproc/'
subject_files=[]
for filename in os.listdir(ROOT): #compiling the subjects downloaded from MIPDB
    if (not filename.endswith('.txt')) and (not filename=='EEG paradigm info Misc_Readme') and (not filename=='subs_without_EEG') and (not filename=='subs_w_2_RS_EEG'):
        print(filename)
        subject_files.append(filename)

print('\n \n \nSUBS COMPLETE: \n')
new_subject_files= subject_files.copy() 
incomplete_subs={}       
for f in subject_files:#['A00051955','A00053440']:##checking to see who is already preprocessed
    thisSub=f
    preproc_file=save_ROOT+thisSub+'/'
    #print(preproc_file)
    #if not (os.path.exists(preproc_file)):
    	#os.mkdir(preproc_file)
    file_closed = preproc_file+thisSub+'_'+'eyes_closed_epoch_'
    file_open = preproc_file+thisSub+'_'+'eyes_open_epoch_'
    #print(file_closed)
    #print(file_open)
    if len(os.listdir(preproc_file))>=10:#100 or thisSub=='A00053440' or thisSub=='A00059578':#10: #if all 10 preproc files are in the sub folder
    	#print(f)
    	new_subject_files.remove(f)
    	print(f)
    else:
    	for n in range(5): # if not, go through and see which files are incomplete
          if (os.path.exists(file_closed+str(n)+'-epo.fif')) or (os.path.exists(file_open+str(n)+'-epo.fif')): #if this epoch is complete, store it for later
            if thisSub not in incomplete_subs.keys(): #initialize a list to store the epoch label in 
                incomplete_subs[thisSub]=[]
            incomplete_subs[thisSub].append(n)	    

# if the subject has already been preprocessed, remove it from the list of people to process


all_sub_events={} #compiling the raw files from those who are not yet preprocessed
subs={}
for files in new_subject_files:
    thisSub=files
    thisfpath=ROOT+thisSub+'/EEG/raw/csv_format/'
    for raw_dat in os.listdir(thisfpath):
        pattern='%s00*_events.csv' %thisSub
        pattern2='%s 00*_events.csv' %thisSub
        if (fnmatch.fnmatch(raw_dat, pattern)) or (fnmatch.fnmatch(raw_dat, pattern2)):
            thisTrigfile=pd.read_csv(thisfpath+raw_dat)
            not_RS=0
            ev_ar=[]
            for n in range(len(thisTrigfile['type'])):
                trig=thisTrigfile['type'][n]
                time_stamp=thisTrigfile['latency'][n]
                if trig != 'type':
                    if int(trig) not in [90,20,30]:
                        not_RS=1
                    else:
                        ev_ar.append([int(time_stamp),0,int(trig)])
            if not_RS==0 and (thisSub not in all_sub_events.keys()):
                all_sub_events[thisSub]=ev_ar
                RS_file=raw_dat.split('_')[0]
                raw_file=ROOT+thisSub+'/EEG/raw/raw_format/'+RS_file+'.raw'
                if os.path.exists(raw_file):
                    subs[thisSub]=raw_file
            elif not_RS==0 and (thisSub in all_sub_events.keys()):
                rep_subs[f]=ev_ar
                print('\n\n\n\n\n ERRORORORORORR!!!!!!!!!!!!!!!!, there are subs w more than one RS file \n\n\n')
print('\n\n\n')
print(len(subs.keys())==len(new_subject_files))
print(len(subs.keys())==len(all_sub_events.keys()))
print('\n\n\n\n SUBS REMAINING: '+str(len(subject_files))+ '\n')
print(new_subject_files)

for fileN in subs.keys():
    sub=fileN
    raw_file=subs[fileN]
    raw=mne.io.read_raw_egi(raw_file,montage=mne.channels.read_montage(kind='GSN-HydroCel-129'),preload=True, verbose=True)
    #raw2=mne.io.read_raw_egi(raw_file,montage=mne.channels.read_montage(kind='GSN-HydroCel-128'),preload=True, verbose=True)
    
    #raw.info['chs'][44:129]
    dataChannels = ['E2','E3','E4','E5','E6','E7','E9','E10','E11','E12','E13','E15','E16','E18','E19','E20','E22','E23','E24','E26','E27','E28','E29','E30','E31','E33','E34','E35','E36','E37','E39','E40','E41','E42','E45','E46','E47','E50','E51','E52','E53','E54','E55','E58','E59','E60','E61','E62','E65','E66','E67','E70','E71','E72','E75','E76','E77','E78','E79','E80','E83','E84','E85','E86','E87','E90','E91','E92','E93','E96','E97','E98','E101','E102','E103','E104','E105','E106','E108','E109','E110','E111','E112','E115','E116','E117','E118','E122','E123','E124']
    #this is the list of 'real' channels that Justin sent, all others are external electrodes
    extChannels = [e['ch_name'] for e in raw.info['chs'] if ((e['ch_name'] not in dataChannels) and (e in raw.info['chs'][:129]))]
    extChannels
    randChannels=[e for e in raw.info['ch_names'] if ((e not in dataChannels) and (e not in extChannels))]
    #raw=raw.drop_channels(randChannels)
    
    ch_type_dict={}
    for c in extChannels:
        ch_type_dict[c]='ecg'
    raw.set_channel_types(ch_type_dict)
    #raw.save(raw_file+'raw.fif')
    #raw=mne.io.read_raw_fif(raw_file+'raw.fif',preload=True, verbose=True)
    print(all_sub_events[fileN])
    text_msg1=input('Please inspect the events and ensure they are only 20, 30, and 90. Ready to move on? [y/n]: ')
    # # Cutting and concatenating raw file into eyes open/eyes closed conditions
    
    events = mne.find_events(raw, verbose=True)
    print(events)
    real_codes=[y[2] for y in all_sub_events[fileN]]
    trigs=np.ndarray.tolist(np.ndarray.flatten(events))[0::3]
    real_EVs=[x[0] for x in all_sub_events[fileN]]


#real_codes=[y[2] for y in all_sub_events[fileN]]
    if (90 not in real_codes[1:]) and (trigs==real_EVs[1:]) and (len(real_EVs)==12):
        good2go=1
        real_EVs=real_EVs[1:]
        real_codes=real_codes[1:]
    else:
        while 90 in real_codes: # for some subject it looks like they might have begun the RS more than once, just cutting to the last 90
            real_codes=real_codes.copy()[1:]
            real_EVs=real_EVs.copy()[1:]
             
    print('\n\n\n\n~~~~~~~ SEPERATING INTO EYES OPEN AND EYES CLOSED CONDITIONS ~~~~~~~~~\n\n')
    new_events=[]
    ec_label=29
    eo_label=19
    ec_labels=[]
    eo_labels=[]
    if good2go:
        for e in range(len(real_EVs)-1):
            thisEv=real_EVs[e]
            nextEv=real_EVs[e+1]
            print(thisEv)
            print(nextEv)
            if real_codes[e]==20: 
                eo_label+=1
                eo_labels.append(eo_label)
                trig=eo_label# eyes open
            elif real_codes[e]==30:
                ec_label+=1 
                ec_labels.append(ec_label)               
                trig=ec_label# eyes closed
            #thisSection=raw2.times[thisEv:nextEv]
            thisSectionInds=[i for i in range(thisEv,nextEv+1)]
            #print(thisSectionInds)
            twoSecs=thisSectionInds[1500::1000][:-1]
            print(twoSecs)
            for timeInd in twoSecs:
                #data[timeInd]=trig
                new_events.append([timeInd, 0, trig])
    print(new_events)
    raw.add_events(np.asarray(new_events),'STI 014',replace=True)
    events = mne.find_events(raw)#_stimCH, verbose=True)
    print(events)
    allEC_eps={}
    for n in range(len(ec_labels)):

        if (sub in incomplete_subs.keys()) and (n in incomplete_subs[sub]) and (os.path.exists(save_ROOT+sub+'/'+sub+'_'+'eyes_closed_epoch_'+str(n)+'-epo.fif')): #if the sub is an incomplete sub and this epoch is one of the already done epochs
            complete_ep=mne.read_epochs(save_ROOT+sub+'/'+sub+'_'+'eyes_closed_epoch_'+str(n)+'-epo.fif') 
            first_samp=complete_ep.events[0][0] #first event timestamp of the completed epoch
            last_samp=complete_ep.events[-1][0] #last event timestamp of the completed epoch
            first_match=[i[0] for i in events if i[2]==ec_labels[n]][0] # first {eCEV} (ie, 30,34,etc) timestamp of the current event array, make sure the epoch_string is mapping the same EC events as the complete epoch with this same label
            last_match=[i[0] for i in events if i[2]==ec_labels[n]][-1]
            if first_samp != first_match or last_samp != last_match:
            	print("Oops! There is an error discovered during an inspection of this subject's complete vs incomplete EC epochs.\n")
            	print("The completed epoch with this label: "+sub+'_'+'eyes_closed_epoch_'+str(n)+'-epo.fif'+" does not match the time indices of the current working event list. Please check the ec_labels for more info.")
            	exit()
            print('\n \n \n ALREADY COMPLETE: epoch no '+str(n)+'\n\n') 
            continue #then skip over this n

        else:          
            ecEVno=str(n) # 0-4
            label=ec_labels[n] #30-34
            epoch_name='eyes_closed_epoch_'+ecEVno
            ev_id={epoch_name:label}
            eyes_closed=mne.Epochs(raw,events,event_id=ev_id,tmin=0,tmax=2,reject_by_annotation=False)
            allEC_eps[epoch_name]=eyes_closed
    allEO_eps={}
    for n in range(len(eo_labels)):
        if (sub in incomplete_subs.keys()) and (n in incomplete_subs[sub]) and (os.path.exists(save_ROOT+sub+'/'+sub+'_'+'eyes_open_epoch_'+str(n)+'-epo.fif')): #if the sub is an incomplete sub and this epoch is one of the already done epochs
            complete_ep=mne.read_epochs(save_ROOT+sub+'/'+sub+'_'+'eyes_open_epoch_'+str(n)+'-epo.fif') 
            first_samp=complete_ep.events[0][0] #first event timestamp of the completed epoch
            last_samp=complete_ep.events[-1][0] #last event timestamp of the completed epoch
            first_match=[i[0] for i in events if i[2]==eo_labels[n]][0] # first {eoEV} (ie, 30,34,etc) timestamp of the current event array, make sure the epoch_string is mapping the same EC events as the complete epoch with this same label
            last_match=[i[0] for i in events if i[2]==eo_labels[n]][-1]
            if first_samp != first_match or last_samp != last_match:
            	print("Oops! There is an error discovered during an inspection of this subject's complete vs incomplete EO epochs.\n")
            	print("The completed epoch with this label: "+sub+'_'+'eyes_open_epoch_'+str(n)+'-epo.fif'+" does not match the time indices of the current working event list. Please check the eo_labels for more info.")
            	print(first_samp,first_match,last_samp,last_match)
            	exit()
			
            print('\n \n \n ALREADY COMPLETE: epoch no '+str(n)+'\n\n')
            continue #then skip over this n

        else:
            eoEVno=str(n) # 0-4
            label=eo_labels[n] #20-24
            epoch_name='eyes_open_epoch_'+eoEVno
            ev_id={epoch_name:label}
            eyes_open=mne.Epochs(raw,events,event_id=ev_id,tmin=0,tmax=2,reject_by_annotation=False)
            allEO_eps[epoch_name]=eyes_open
    print(allEC_eps.keys())
    print(allEO_eps.keys())
	#eyes_open=mne.Epochs(raw,events,event_id={'twoSec':5},tmin=0,tmax=2,reject_by_annotation=False)
    #eyes_closed.plot_psd(fmax=45,picks=mne.pick_types(raw.info,meg=False,eeg=True,eog=False,exclude=['E129'])) #sanity check
    #eyes_closed.plot()
    
    
    # # Applying high and low pass filters (1 and 50)
    
    # In[35]:
    
    if len(allEC_eps.keys())==len(allEO_eps.keys()):
        ec_KEYS=list(allEC_eps.keys())
        eo_KEYS=list(allEO_eps.keys())     
        for n in range(len(ec_KEYS)):
            save_label_EC=ec_KEYS[n]
            print('\n~~~~~~~~\n~~~~~~~~~~~~~~~~~~~\n~~~~~~~~~~~~~~~~~~~~~~~ RUNNING THROUGH EC and EO NUMBER '+save_label_EC[-1]+' ~~~~~~~~~~~~~~~~~~~~~~~ \n~~~~~~~~~~~~~~~~~~~\n~~~~~~~~\n')
            eyes_closed=allEC_eps[save_label_EC]
            save_label_EO=eo_KEYS[n]
            eyes_open=allEO_eps[save_label_EO]
            
            eyes_closed.load_data()
            EC_f=eyes_closed.copy()
            EC_f.filter(1,50)
                #can choose different methods of filtering. here it's using the default 'overlap-add' FIR filtering, can
                    # also use IIR forward/backward filtering
            #EC_f.plot(n_channels=129)#n_channels=129,group_by#='original')
            
            eyes_open.load_data()
            EO_f=eyes_open.copy()
            EO_f.filter(1,50)
            #EO_f.plot(n_channels=129)#n_channels=129,group_by='original')
            
            
            # # Select bad electrodes from "real" data to interpolate
            
            # In[36]:
            
            
            EC_fi=EC_f.copy()
            tryAgain=1
            EC_DataCH=EC_fi.copy().drop_channels(extChannels)
            EC_DataCH=EC_DataCH.copy().drop_channels(randChannels)
            
            while tryAgain:
                    EC_DataCH.plot(block=True,n_epochs=2,n_channels=15, title='SELECT BAD CHANNELS: eyes closed') #pauses script while i visually inspect data and select which channels to delete
                    bads_EC=EC_DataCH.info['bads']
                    text_msg2=input('The channels you marked as bad are: '+str(bads_EC)+' \n Are you ready to interpolate? [y/n]: ')
                    if text_msg2=='y':
                            EC_fi.info['bads']=bads_EC
                            EC_fi=EC_fi.interpolate_bads()
                            tryAgain=0
                    elif text_msg2=='n':
                            #EC_DataCH.plot(block=True)
                            #text_msg3=input("Which channels do you want to interpolate? Type 'DONE' when finished \n")	
                            #bads=[]
                            #notDone=1
                            #while notDone==1:
                            tryAgain=1
                    else:
                            print('invalid entry: '+text_msg2)
                            tryAgain=1
            
            EO_fi=EO_f.copy()
            EO_DataCH=EO_fi.copy().drop_channels(extChannels)
            EO_DataCH=EO_DataCH.copy().drop_channels(randChannels)
            tryAgain=1
            while tryAgain:
                    EO_DataCH.plot(block=True,n_epochs=2,n_channels=15, title='SELECT BAD CHANNELS: eyes open') #pauses script while i visually inspect data and select which channels to delete
                    bads_EO=EO_DataCH.info['bads']
                    text_msg2=input('The channels you marked as bad are: '+str(bads_EO)+' \n Are you ready to interpolate? [y/n]: ')
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
            
            
            layout=mne.channels.read_montage(kind='GSN-HydroCel-129')
            #layout.plot() 
            
            # # Re-referencing using global avg referencing
            EC_fi.drop_channels(randChannels)
            EO_fi.drop_channels(randChannels)
            
            
            # SHOULD THIS COME BEFORE ICA? see manuscript blurb
            ## ?? should we use all scalp electrodes too? not truly global until use all electrodes
            EC_fir, r2 = mne.set_eeg_reference(EC_fi,ref_channels=dataChannels+extChannels) 
            #EC_fir.plot(n_channels=129)
            EO_fir, r2 = mne.set_eeg_reference(EO_fi,ref_channels=dataChannels+extChannels) 
            #EO_fir.plot(n_channels=129) 
            #EC_fir.drop_channels(randChannels) 
            #EO_fir.drop_channels(randChannels) 
             
            eyes_closed=EC_fir.copy()#drop_channels(extChannels)
            eyes_open=EO_fir.copy()#drop_channels(extChannels)
        
            
            # # ICA
            
            our_picks_EC=mne.pick_types(eyes_closed.info,meg=False,eeg=True,eog=False,ecg=False)
            our_picks_EO=mne.pick_types(eyes_open.info,meg=False,eeg=True,eog=False,ecg=False)
            eyes_closed=eyes_closed.set_montage(layout)
            eyes_open=eyes_open.set_montage(layout)    
            
            ## fitting IC's
            print('\n\n\n\n RUNNING ICA ~~~~~~~ \n\n')
            
            datasets=[eyes_closed,eyes_open]
            ica_data={}
            
            for data in datasets:
                ica = []
                ica = ICA(n_components=90,random_state=25)#,method='infomax')
                if data == eyes_closed:
                    picks=our_picks_EC    
                else: 
                    picks=our_picks_EO   
                ica.fit(data,picks=picks)
                #ica.info
            
                eog_ic=[]
                for ch in ['E25','E17','E8','E21','E14','E125','E126','E127','E128']: #insert EOG channels
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
                if data==eyes_closed:
                    name='EC'
                else:
                    name='EO'
                ica_data[name]=(ica,data,reject_ic)
            
            
            # In[56]:
            
            repeatICA=1
            while repeatICA:
                    EC_firei=[]
                    EO_firei=[]
                    ec_copy=[]
                    eo_copy=[]
                    for icDat in ica_data.keys():
                            data=ica_data[icDat][1]
                            reject_ic=ica_data[icDat][2]
                            ica=ica_data[icDat][0]
                            
                            plot=1
                            while plot:  
                                    ica.plot_components(picks=range(50),ch_type=None,title=icDat,inst=data) #needs the channel locations
                                    ica.plot_components(picks=range(50,90),ch_type=None,title=icDat,inst=data)
                                    #while len(ica.exclude)==len(reject_ic):
                                            #continue
                                    verification_ic=input('The ICs marked for rejection are: ' + str(ica.exclude) +'\n Are you sure you want to proceed? [y/n]: ')
                                    if verification_ic=='y':
                                            #print(len(reject_ic))
                                            #print(len(ica.exclude))	
                                            #ica.exclude.extend(bad_ics)
                                            plot=0
                                    else:
                                            print('Please select which ICs you would like to reject')
                            ica_data[icDat]=(ica,data,reject_ic)
            
                    EC_fire=eyes_closed.copy()
                    ec_copy=eyes_closed.copy()    
                    EC_firei=(ica_data['EC'][0]).apply(EC_fire,exclude=ica_data['EC'][0].exclude)
                    EO_fire=eyes_open.copy()
                    eo_copy=eyes_open.copy()
                    EO_firei=ica_data['EO'][0].apply(EO_fire,exclude=ica_data['EO'][0].exclude)
                    #raw_fii.plot()
                    #print(len(ica_data['EC'][0].exclude))
            
                    #ica_data['EC'][0].plot_overlay(EC_fir.average(),exclude=ica_data['EC'][0].exclude)#,picks=our_picks)
                    #ica_data['EO'][0].plot_overlay(EO_fir.average(),exclude=ica_data['EO'][0].exclude)#,picks=our_picks)
                    EO_firei.plot(title='EO, AFTER ICA',n_epochs=4,n_channels=30)
                    eo_copy.plot(block=True, title='EO, BEFORE ICA',n_epochs=4,n_channels=30)
                    EC_firei.plot(title='EC, AFTER ICA',n_epochs=4,n_channels=30)
                    ec_copy.plot(block=True, title='EC, BEFORE ICA',n_epochs=4,n_channels=30)
        
                    cont=input('Continue on to epoch rejection? [y/n]: ')
                    if cont=='y':
                            eyes_open=EO_firei.drop_channels(extChannels)
                            eyes_closed=EC_firei.drop_channels(extChannels)
                            repeatICA=0
            
                
            plotEpoch=1
            
            while plotEpoch:
                    EO_Ep=eyes_open.copy()
                    print('plotting '+str(np.shape(EO_Ep.get_data())[0])+ ' epochs')
                    EO_Ep.plot(block=True,n_epochs=2,n_channels=30, title='SELECT BAD EPOCHS: eyes open',picks=mne.pick_channels(ch_names=dataChannels,include=[],exclude=extChannels))
                    EO_Ep.plot_psd_topomap(cmap='interactive')
                    #avg_EO=thisEp.average()
                    #avg_EO.plot()
                    bads=input('Are you sure you want to continue? [y/n]: ')
                    if bads=='y':
                        eyes_open=EO_Ep
                        plotEpoch=0
                    elif bads=='n':
                        continue
                    else:
                        print('oops, please indicate which epochs you would like to reject')
               
            plotEpoch=1
            while plotEpoch:
                    EC_Ep=eyes_closed.copy()
                    print('plotting '+str(np.shape(EC_Ep.get_data())[0])+ ' epochs')
                    EC_Ep.plot(block=True,n_epochs=2,n_channels=30, title='SELECT BAD EPCOHS: eyes closed',picks=mne.pick_channels(ch_names=dataChannels,include=[],exclude=extChannels))
                    EC_Ep.plot_psd_topomap(cmap='interactive')
                    #avg_EC=thisEp.average()
                    #avg_EC.plot()
                    bads=input('Are you sure you want to continue onto saving? [y/n]: ')
                    if bads=='y':
                        eyes_closed=EC_Ep
                        plotEpoch=0
                    elif bads=='n':
                        continue
                    else:
                        print('oops, please indicate which epochs you would like to reject')
            
            
            # # Epoching into 2 second, non overlapping time windows
            
            # In[57]:
            
            
            
            
            ## SAVING OUT ##
            eyes_closed.save(save_ROOT+sub+'/'+sub+'_'+save_label_EC+'-epo.fif')
            eyes_open.save(save_ROOT+sub+'/'+sub+'_'+save_label_EO+'-epo.fif')
            if os.path.exists(save_ROOT+sub+'/'+sub+'_'+save_label_EO+'-epo.fif'):
                print('\n\n\n\n SAVING OUT INFO ~~~~~~~ \n\n'+save_ROOT+sub+'/'+sub+'_'+save_label_EO+'-epo.fif......DONE!')
            if os.path.exists(save_ROOT+sub+'/'+sub+'_'+save_label_EC+'-epo.fif'):
                print('\n\n\n\n SAVING OUT INFO ~~~~~~~ \n\n'+save_ROOT+sub+'/'+sub+'_'+save_label_EC+'-epo.fif......DONE!')
            print('\n\n\n\n\n')

    else:
        print("SKIPPING SUB " +fileN+" BECAUSE IT HAS AN UNEQUAL NUM OF EC AND EO")
