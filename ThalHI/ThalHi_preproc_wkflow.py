# # this is the EEG preproc wkflow to be used with alpha study EEG # #

import mne
from mne.datasets import sample
from mne import io
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
from mne.preprocessing import ICA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
from os import path
import pickle
import fnmatch
#import ipywidgets
#from ipywidgets import widgets


#import kai's test data
ROOT = '/home/dcellier/RDSS/ThalHi_data/EEG_data/'
ROOT_raw=ROOT+'eeg_raw/'
ROOT_behav=ROOT+'behavioral_data/'
ROOT_proc='/data/backed_up/shared/ThalHi_data/eeg_preproc/'
subject_files=[]
for filename in os.listdir(ROOT_raw): #compiling the subjects downloaded from MIPDB
	if not filename=='realpeople':
		subject_files.append(filename)
s_file_copy=subject_files[:]
for thisSub in s_file_copy:
	sub_name=thisSub.split('_')[1]
	print(sub_name)
	if os.path.exists(ROOT_proc+sub_name+'/'):
		subject_files.remove(thisSub)
print(subject_files)
for sub in subject_files: # insert for loop through subject list here
	raw_file=ROOT_raw+sub
	sub_name=sub.split('_')[1]
	#print(sub_name)
	pattern=sub_name+'_00*_Task_THHS_2019_*_*_*.csv'#sub+'_alpha_pilot_01_20*_*_*_*.csv'
	behav_files=pd.DataFrame()
	for f in os.listdir(ROOT_behav):
		if fnmatch.fnmatch(f,pattern):
			print(f)
			behav_file=pd.read_csv(ROOT_behav+f,engine='python')
			behav_files=behav_files.append(behav_file,ignore_index=True)
	behav_files.dropna(axis=0,inplace=True,how='any')
	print(behav_files)
	raw=mne.io.read_raw_edf(raw_file,montage=mne.channels.read_montage('biosemi64'),preload=True)
	#raw.plot(n_channels=72)

	# # Re-reference, apply high and low pass filters (1 and 50) # # # # # # # # # #

	raw_f=raw.copy()
	raw_f.filter(1,50)
	raw_f.set_channel_types({'EXG1':'emg','EXG2':'emg','EXG3':'eog','EXG4':'eog','EXG5':'eog','EXG6':'eog',
                        'EXG7':'ecg','EXG8':'emg'})
	
	#raw_f.plot(n_channels=72)
	
		# # # # # # selecting bad electrodes # # # # # #	
	raw_fi=raw_f.copy()	
	raw_DataCH=(raw_fi.copy()).drop_channels(['EXG1', 'EXG2','EXG3','EXG4','EXG5','EXG6','EXG7','EXG8'])
	tryAgain=1
	while tryAgain:
		raw_DataCH.plot(block=True) #pauses script while i visually inspect data and select which channels to delete
		bads=raw_DataCH.info['bads']
		text_msg2=input('The channels you marked as bad are: '+str(bads)+' \n Are you ready to interpolate? [y/n]: ')
		if text_msg2=='y':
			raw_fi.info['bads']=bads
			raw_fi=raw_fi.interpolate_bads()
			tryAgain=0
		elif text_msg2=='n':
			tryAgain=1	
		else:
			print('invalid entry: '+text_msg2)
			tryAgain=1
	raw_fir,r= mne.set_eeg_reference(raw_fi,ref_channels=['EXG1', 'EXG2'])#,'EXG8'])#mastoids, nose -- nose we decided we didn't want to use to reref

	# # Finding Events (triggers) # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

	events = mne.find_events(raw_fir, verbose=True)

	raw_fe=raw_fir.copy() # raw_fe was originally the 2 epoched data
	#raw_fe.plot()
	
	# # Looping through conditions, epoching # # # # # # #  # # # # # # # # # # # # # # # # # # # # # # # #
	
	probe_event_id={'faceProbe_trig':151,'sceneProbe_trig':153}   
	response_event_id={'subResp_trig':157,'subNonResp_trig':155}
	cue_event_id={'IDS_trig':101,'Stay_trig':105, 'EDS_trig':103,
	 ## triggers for emily's data: #cue_event_id={'Donut_Circle_blue_trig': 107,
 #'Donut_Circle_red_trig': 105,
 #'Donut_Polygon_blue_trig': 103,
 #'Donut_Polygon_red_trig': 101,
 #'Filled_Circle_blue_trig': 115,
 #'Filled_Circle_red_trig': 113,
 #'Filled_Polygon_blue_trig': 111,
 #'Filled_Polygon_red_trig': 109}

	cue_tmin, cue_tmax = -0.8, 1.5  # 800 ms before event, and then 2.5 seconds afterwards. cue is 1 sec + 1.5 delay
	probe_tmin, probe_tmax = -0.8,3 # -800 and 2 second probe/response period 
	response_tmin,response_tmax=-0.5,1.5 # probably won't analyze this but might as well have it
	baseline = (None, -0.3) #baseline correction applied with mne.Epochs, this is starting @ beginning of epoch ie -0.8 
	epCond={}
	print('\n\n\n Epoching Conds \n ')
	#for event in cue_event_id.keys():
		#thisID={event:cue_event_id[event]}
	epCond['cue_events']=mne.Epochs(raw_fe, events=events, baseline=baseline, event_id=cue_event_id, tmin=cue_tmin,tmax=cue_tmax,metadata=behav_files)
	#for event in probe_event_id.keys():
		#thisID={event:probe_event_id[event]}
	epCond['probe events']=mne.Epochs(raw_fe, events=events, baseline=baseline, event_id=probe_event_id, tmin=probe_tmin, tmax=probe_tmax,metadata=behav_files)
	#for event in response_event_id.keys():
		#thisID={event:response_event_id[event]}
	epCond['response events']=mne.Epochs(raw_fe, events=events, baseline=(0,None), event_id=response_event_id, tmin=response_tmin, tmax=response_tmax,metadata=behav_files)
		# changed the baseline correction for this one because it doesn't make a whole lot of sense to baseline correct to -500 w a motor response?


	# # Inspect and reject bad epochs # # # # # # # # # # # #  # # # # # # # # # # 
	# # AND ICA on Raw data # # # # # # # # # # # # #  # # # # # # # # # # # # # # # # # # # # # # # # # # 
	our_picks=mne.pick_types(raw_fe.info,meg=False,eeg=True,eog=False)#mne.pick_types(raw_fi.info,meg=False,eeg=True,eog=True,emg=True,stim=True,ecg=True)
 
	layout=mne.channels.read_montage('biosemi64')

	EOG_channels=['EXG3', 'EXG4', 'EXG5', 'EXG6']
	ECG_channels=['EXG7']

	for ev in epCond.keys():
		plotEpoch=1
		while plotEpoch:
			print('plotting ' +ev)
			keep_plotting=1
			while keep_plotting:
				thisEp=[]
				thisEp=epCond[ev].copy()
				thisEp.load_data()
				print(thisEp)
				thisEp.plot(block=True, title="SELECT BAD EPOCHS: "+ev, n_epochs=6,n_channels=15)
				bads=input('Are you sure you want to continue? [y/n]: ')
				if bads=='y':
					#epCond[ev]=thisEp
					keep_plotting=0
				elif bads=='n':
					continue
				else:
					print('oops, please indicate which epochs you would like to reject')
			### ICA ###
			thisCond=thisEp.copy()
			thisCond.set_montage(layout)
			ica=ICA(n_components=64,random_state=25)
			ica.fit(thisCond,picks=our_picks)
			eog_ic=[]
			for ch in EOG_channels:
				#find IC's attributable to EOG artifacts
				eog_idx,scores=ica.find_bads_eog(thisCond,ch_name=ch)
				eog_ic.append(eog_idx)
			ecg_ic=[]
			for ch in ECG_channels: # find IC's attributable to ECG artifacts
				ecg_idx,scores=ica.find_bads_ecg(thisCond,ch_name=ch)
				ecg_ic.append(ecg_idx)
			reject_ic=[]
			for eog_inds in eog_ic:
				for ele in eog_inds:
					if (ele not in reject_ic) and (ele <= 31):
						reject_ic.append(ele)
			for ecg_inds in ecg_ic:
				for ele in ecg_inds:
					if (ele not in reject_ic) and (ele <= 31):
						reject_ic.append(ele) #add these IC indices to the list of IC's to reject
			ica.exclude=[]
			ica.exclude.extend(reject_ic)
			plotICA=1
			while plotICA:
				ica.plot_components(picks=range(32),ch_type=None,cmap='interactive',inst=thisCond)# changed to ch_type=None from ch_type='EEG'because this yielded an error
				#ica.plot_components(picks=range(32,64),ch_type=None,cmap='interactive',inst=thisCond)
				input('The ICs marked for exclusion are: '+str(ica.exclude)+ '\n Press enter.')
				thisCond.load_data()
				thisCond.copy().plot(title=ev+': BEFORE ICA',n_epochs=5,n_channels=30)
				thisCond_copy=thisCond.copy()
				thisCond_Ic=ica.apply(thisCond_copy,exclude=ica.exclude)
				thisCond_Ic.plot(block=True, title=ev+': AFTER ICA',n_epochs=5,n_channels=30)
				verification_ic=input('The ICs marked for rejection are: ' + str(ica.exclude) +'\n Are you sure you want to proceed? [y/n]: ')
				if verification_ic=='y':
					plotICA=0
				else:
					print('Please select which ICs you would like to reject')
			save_ep=input('Save this epoch? Entering "no" will take you back to epoch rejection for this condition. [y/n]: ')
			if save_ep=='y':
				plotEpoch=0
		#ica.plot_components(picks=range(25),ch_type=None,inst=thisEp)  
		thisCond_copy=thisCond.copy()
		ica.apply(thisCond_copy)
		thisCond_copy.drop_channels(['EXG1', 'EXG2','EXG3','EXG4','EXG5','EXG6','EXG7','EXG8'])		
		epCond[ev]=thisCond_copy
	
	print('\n\n\n\n SAVING OUT INFO ~~~~~~~ \n\n')
	save_path=ROOT_proc+sub_name+'/'
	os.mkdir(save_path)

	
	for event in epCond.keys():
		event_name=event.split('_')[0]
		thisEp=epCond[event]
		thisEp.save(save_path+event_name+'-epo.fif')
		os.chmod(save_path+event_name+'-epo.fif',0o660)
	os.chmod(save_path,0o2770)
	ask=1
	exit_loop=0
	while ask:	
		end_msg=input('Continue on to next sub? [y/n]')
		if end_msg=='y':
			continue
			ask=0
		elif end_msg=='n':
			exit_loop=1
			ask=0
		else:
			print('Oops! Please answer y/n')

	if exit_loop:
		break
#raw_f.info
#thispick=mne.pick_types(raw_fi.info,meg=False,eeg=True,eog=True,emg=True,stim=True,ecg=True)

# # Epoching into arbitrary 2-second windows # # # # # # # # # # # # # # # # # # # # # # # # # #
    # IVE COMMENTED THIS OUT BECAUSE OF THE ISSUES W/ RE-CONCAT INTO RAW FILE AFTER 2 SEC EPOCHING

#epoch_array=[]
#for t in raw_f.times[0::1024]:
 #   epoch_array.append([int(t),int(0),int(7)])  
    
#epoch_array=np.asarray(epoch_array)

#twoSec=mne.Epochs(raw_f,events=epoch_array,tmin=0,tmax=2,
	#event_id={'twoSec':7},picks=thispick)

#twoSec.plot(block=True) #selecting bad epochs to throw out
#twoSec.drop_bad()

#eps=[]
#for e in twoSec:
 #   eps.append(e)
#raw_fe=mne.io.RawArray(eps[0],raw_f.info)
#raw_ar=[]
#for ep in eps[1:]:
#    ep=mne.io.RawArray(ep,raw_f.info)
#    raw_ar.append(ep)
#raw_fe.append(raw_ar)

#eps[1]

##raw_f.info['events']
#event_ar=[]
#for event in events:
 #   sample=event[0]
#    if raw_fe.times[sample]==raw_f.times[sample]:
  #     one=sample
  #      event_ar.append([one,event[1],event[2]])
  #  else:
   #     try:
       #     one=list(raw_fe.times).index(raw_f.times[sample])
    #        event_ar.append([one,event[1],event[2]])
      #  except:
       #     continue
    
#raw_fe.info['events']=np.asarray(event_array)
#raw_fe.plot()



#raw_f.set_montage(layout)
#plottables={}
#epAfterICA={}
#for cond in epCond.keys():
 #   thisEp=epCond[cond]  
  #  thisEp_i=thisEp.copy()
   # thisEp_i.load_data()
    #thisEp_i.set_montage(layout)
    #icaCond=ICA(n_components=25,random_state=25)
    #icaCond.fit(thisEp_i,picks=our_picks)
    
    #eog_ic=[]
    #for ch in EOG_channels:  # find IC's attributable to EOG artifacts
     #   eog_idx,scores=icaCond.find_bads_eog(thisEp,ch_name=ch)
      #  eog_ic.append(eog_idx)
    #ecg_ic=[]
    #for ch in ECG_channels: # find IC's attributable to ECG artifacts
     #   ecg_idx,scores=icaCond.find_bads_ecg(thisEp,ch_name=ch)
      #  ecg_ic.append(ecg_idx)
    #reject_ic=[]
    #for eog_inds in eog_ic:
     #   for ele in eog_inds:
      #      if ele not in reject_ic:
       #         reject_ic.append(ele)
    #for ecg_inds in ecg_ic:
     #   for ele in ecg_inds:
      #      if ele not in reject_ic:
       #         reject_ic.append(ele) #add these IC indices to the list of IC's to reject
    
    #icaCond.exclude=[]
    #icaCond.exclude.extend(reject_ic)
    #icaCond.plot_components(picks=range(25),ch_type='eeg',inst=thisEp) 
    #bad_ics=[] #list those identified by visual inspection
    #icaCond.exclude.extend(bad_ics)
    #icaCond.apply(thisEp_i)
    #plottables[cond]=icaCond
    #epAfterICA[cond]=thisEp_i





#for cond in epCond.keys():
 #   thisEp=epCond[cond]
  #  thisEp.load_data()
   # # downsample???
    #thisEp.plot(block=True)
    #thisEp.drop_bad()
    #thisEp.info['bads']=[]
    # selecting bad electrodes
    #bads=[] #select bad electrodes by hand ? 
    #thisEp.info['bads']=bads
    #thisEp.interpolate_bads()
