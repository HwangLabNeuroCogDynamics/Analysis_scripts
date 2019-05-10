import mne
from mne.datasets import sample
from mne import io
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
from mne.preprocessing import ICA
import numpy as np
from numpy import stack
import pandas as pd
import matplotlib.pyplot as plt
import os
import fnmatch
import seaborn as sns
import scipy.io as sio


ROOT='/home/dcellier/RDSS/CMI_data/HBN/EEGdata/'
dat_mat_path='%s/EEG/%s/EEG/raw/mat_format/RestingState.mat' 
dat_csv_path='%s/EEG/%s/EEG/raw/csv_format/RestingState_event.csv'
save_ROOT='/data/backed_up/shared/CMI_data/HBN/preproc_RS/'

print('\n\n ALL SUBS: \n')
fetch_subs={}
for release in os.listdir(ROOT):
	if release != 'weird_subs':
		for sub in os.listdir(ROOT+release+'/EEG/'):
			thisSub=ROOT+dat_mat_path % (release,sub)
			thisSub_events=ROOT+dat_csv_path % (release,sub)
			print(sub)
			fetch_subs[sub]=(thisSub,thisSub_events)
print('\n\n COMPLETE SUBS: \n')
for sub in os.listdir(save_ROOT):
	if sub in fetch_subs.keys():
		print(sub)
		del fetch_subs[sub]
print('\n\n SUBS REMAINING: \n')
for sub in fetch_subs.keys():
	print(sub)

for sub in fetch_subs:
	print('\n\n ~~~~~ RUNNING THROUGH SUBJECT ' +sub+' ~~~~~\n')
	#raw=pd.read_csv(ROOT,header=None,engine='python')
	mat=fetch_subs[sub][0]
	csv_evs=fetch_subs[sub][1]
	raw_array=sio.loadmat(mat)
	events=pd.read_csv(csv_evs,engine='python')

	### Checking .mat info, whether or not the data is in the index it's supposed to be
	raw=raw_array['EEG'][0][0][15]
	if len(raw)!=129:#129, error out on purpose to test
		print("The data array extracted out of sub "+sub+"'s .mat file does not contain data for 129 channels: \n")
		print(raw)
		exit()
	### Checking the .mat events, whether they are populated with event info
	mat_events_labels=[]
	mat_events_inds=[]
	mat_ev_arr=raw_array['EEG'][0][0][25] #extracting an array of arrays which contain event info
	if len(mat_ev_arr)==1: #if the array has more than one array of events we're in trouble
		mat_ev_arr=mat_ev_arr[0] 
	else:
		print("The event array extracted out of sub "+sub+"'s .mat file does not contain the proper number of events: \n")
		print(mat_ev_arr)
		exit()
	if len(mat_ev_arr)>0: #or if the event array within the array is empty we're in trouble
		for ev in mat_ev_arr:
			mat_events_labels.append(ev[0][0])  # grab the 90,20,or30
			mat_events_inds.append(ev[1][0][0]) #grab the time index of the label
	else:
		print("The event array extracted out of sub "+sub+"'s .mat file does not contain the proper number of events: \n")
		print(mat_ev_arr)
		exit()

	### Cleaning the events of non-resting state events
	sections2concat=[]
	if list(events['type']).count('break cnt')>2: #some subs have another task mixed in, hence more than 2 of these break count labels
		break_list=list(events.type[events.type=='break cnt'].index)
		print(break_list)
		for n in range(len(break_list[:-1])): # go through each break and look at the event label right after it, if it's RS keep it, otherwise skip over that section of the event list
			print('\n')
			brk=break_list[n] #the index of the 'break cnt'
			next_brk=break_list[n+1] #the index of the next 'break cnt', ie marking the end of this task
			thisTaskStart=brk+1 #the event right after, should be either the start of RS (90) or the start of the other task (91)
			thisTaskStart_row=events.iloc[thisTaskStart] #grabbing the event df row for thisTaskStart index
			print(brk,next_brk)
			print(thisTaskStart_row)
			if thisTaskStart_row[0].split()[0]=='90': #we have to do .split() because the format is always '90  ', not '90'
				sections2concat.append(events[brk:next_brk]) #if this is RS, grab the beginning and end of it
		new_events=pd.concat(sections2concat) #hopefully there's only 1 
		if list(new_events['type']).count('break cnt')>1: #if there's more than 1 break count that means we've grabbed more than one RS-- that aint right. 
			print("The event array etracted from sub" +sub+"'s .mat or .csv file contains more than one RS section: \n")
			print(events)
			exit()
		else:
			events=new_events # needs to be a pd.DF
			events=events.reset_index(drop=True) 
	sections2concat2=[]
	sections2concat3=[]
	if mat_events_labels.count('break cnt')>2:
		break_list=[i for i,e in enumerate(mat_events_labels) if e=='break cnt']
		for n in range(len(break_list[:-1])):
			brk=break_list[n]
			next_brk=break_list[n+1]
			thisTaskStart_label=mat_events_labels[brk+1]
			if thisTaskStart_label.split()[0]=='90':
				sections2concat2=sections2concat2+mat_events_labels[brk:next_brk]
				sections2concat3=sections2concat3+mat_events_inds[brk:next_brk]
		if sections2concat2.count('break cnt')>1:
				print("The event array etracted from sub" +sub+"'s .mat or .csv file contains more than one RS section: \n")
				print(mat_events_labels)
				exit()
		else:
				mat_events_labels=sections2concat2 
				mat_events_inds=sections2concat3
	print(events)
	print(mat_events_labels)

	### Making the event lists concise -- should ONLY have the RS at this point (hopefully)				
	if mat_events_labels[0]=='break cnt': #cut out that first and last break count thing
		mat_events_labels=mat_events_labels[1:]
		mat_events_inds=mat_events_inds[1:]
	if mat_events_labels[-1]=='break cnt':
		mat_events_labels=mat_events_labels[:-1]
		mat_events_inds=mat_events_inds[:-1]			
	if events['type'][0]=='break cnt':
		events=events[1:]
	if events['type'][(events['type'].index[-1])]=='break cnt':
		events=events[:-1]
	events=events.reset_index(drop=True) #reset indices to zero after chopping off first and last rows

	### Ensure that the events between the .mat and .csv are the same, if not we probably wanna toss this sub
	if len(events)==len(mat_events_labels)==len(mat_events_inds): #make sure these lists are the same length
		csv_events_labels=events['type']
		csv_events_inds=events['sample']
		for n in range(len(events)): # then index each element in both lists and make sure they're the same
			if (csv_events_labels[n]==mat_events_labels[n]) and (csv_events_inds[n]==mat_events_inds[n]):
				this_ev_label=csv_events_labels[n].split()[0]
				this_ev_ind=csv_events_inds[n]
				if this_ev_label not in ['20','90','30']: # throw an error if there's a weird event
					print("The event array extracted out of sub "+sub+"'s .mat or .csv file contains an excessive event. Please inspect this and mark this subject as unusual: \n")
					print(events)
					exit()
			else: # throw an error if there's an event that doesn't match
				print("The event arrays extracted out of sub "+sub+"'s .mat file and .csv event file do not match each other: \n")
				print("Events from the .csv: \n")
				print(events)
				print("\nEvents from the .mat: \n")
				print(mat_events_labels)
				print('\nEvent time inds from the .mat: \n')
				print(mat_events_inds)
				exit()
	else: # throw an error if csv and mat event lists aren't same length
		print("The event arrays extracted out of sub "+sub+"'s .mat file and .csv event file do not match each other: \n")
		print("Events from the .csv: \n")
		print(events)
		print("\nEvents from the .mat: \n")
		print(mat_events_labels)
		print('\nEvent time inds from the .mat: \n')
		print(mat_events_inds)
		exit()
	
	### Moving onto creating raw, then splitting into epochs of EO and EC	
	channels=mne.channels.read_montage(kind='GSN-HydroCel-129')
	channel_pos=channels.pos[3:]
	channel_names=channels.ch_names[3:]
	channels=mne.channels.read_montage(kind='GSN-HydroCel-129',ch_names=channel_names)

	raw_info=mne.create_info(ch_names=channel_names,sfreq=500.0,montage=channels,ch_types='eeg')

	thisRaw=mne.io.RawArray(raw,raw_info,first_samp=0,verbose=True)
	
	dataChannels = ['E2','E3','E4','E5','E6','E7','E9','E10','E11','E12','E13','E15','E16','E18','E19','E20','E22','E23','E24','E26','E27','E28','E29','E30','E31','E33','E34','E35','E36','E37','E39','E40','E41','E42','E45','E46','E47','E50','E51','E52','E53','E54','E55','E58','E59','E60','E61','E62','E65','E66','E67','E70','E71','E72','E75','E76','E77','E78','E79','E80','E83','E84','E85','E86','E87','E90','E91','E92','E93','E96','E97','E98','E101','E102','E103','E104','E105','E106','E108','E109','E110','E111','E112','E115','E116','E117','E118','E122','E123','E124']
	extChannels = [e['ch_name'] for e in thisRaw.info['chs'] if (e['ch_name'] not in dataChannels)]

	ch_type_dict={}
	for ch in extChannels:
		ch_type_dict[ch]='ecg'
	thisRaw.set_channel_types(ch_type_dict)
	thisRaw.filter(1,50)
	scalings={'eeg':100,'ecg':100}
	#thisRaw.plot(block=True,scalings={'eeg':100,'ecg':100})#'auto')

	print('\n ~~~~~~Splitting into EO and EC~~~~~~\n')	
	EO_eps={}
	EC_eps={}

	mat_events_labels=[i.split()[0] for i in mat_events_labels] #getting rid of that weird spacing in the string ('90  ','20  ',etc)
	mat_events=list(zip(mat_events_labels,mat_events_inds))
	
	print(mat_events)
	eo_label=19 #20 is EO and 30 is EC
	ec_label=29
	eo_labels=[]
	ec_labels=[]
	event_array=[]
	for n in range(len(mat_events)-1):
		label=mat_events[n][0]
		timeStamp=mat_events[n][1]
		nextTime=mat_events[n+1][1] 
		if label=='90':
			continue
		elif label=='20':
			eo_label+=1
			trig=eo_label
			eo_labels.append(eo_label)
		elif label=='30':
			ec_label+=1
			trig=ec_label
			ec_labels.append(ec_label)
		thisIndslice=[i for i in range(timeStamp,nextTime+1)] #making a list of indices between the start of this event and the start of the next event
		#print(thisIndslice)
		twoSecs=thisIndslice[1500::1000][:-1] #then pulling out each 2 second index, with the first 3 seconds cut out entirely and the last 2 second cut off
		print(twoSecs)
		for t in twoSecs:
			event_array.append([t, 0, trig])
		
	#print(event_array)
	
	for eoEv in eo_labels:
		epoch_string='EO_epoch_'+str(eo_labels.index(eoEv))
		if os.path.exists(save_ROOT+sub+'/'+sub+'_'+epoch_string+'-epo.fif'): # checking whether this epoch for this subject has already been completed
			complete_ep=mne.read_epochs(save_ROOT+sub+'/'+sub+'_'+epoch_string+'-epo.fif') # and if so, whether the epoch # matches the events we think it does
			first_samp=complete_ep.events[0][0] #first event timestamp of the completed epoch
			last_samp=complete_ep.events[-1][0] #last event timestamp of the completed epoch
			first_match=[i[0] for i in event_array if i[2]==eoEv][0] # first {eoEV} (ie, 20,24,etc) timestamp of the current event array, make sure the epoch_string is mapping the same EO events as the complete epoch
			last_match=[i[0] for i in event_array if i[2]==eoEv][-1]
			if first_samp != first_match or last_samp != last_match:
				print("Oops! There is an error discovered during an inspection of this subject's complete vs incomplete EO epochs.\n")
				print("The completed epoch with this label: "+sub+'_'+epoch_string+'-epo.fif'+" does not match the time indices of the current working event list. Please check the eo_labels for more info.") 
				exit()
			continue
		else:
			ev_id={epoch_string:eoEv}
			EO_eps[epoch_string]=mne.Epochs(thisRaw,event_array,event_id=ev_id,baseline=None,tmin=0,tmax=2,reject_by_annotation=False)
	for ecEv in ec_labels:
		epoch_string='EC_epoch_'+str(ec_labels.index(ecEv))
		if os.path.exists(save_ROOT+sub+'/'+sub+'_'+epoch_string+'-epo.fif'): #this is very paranoid of me but oh well
			complete_ep=mne.read_epochs(save_ROOT+sub+'/'+sub+'_'+epoch_string+'-epo.fif') 
			first_samp=complete_ep.events[0][0] #first event timestamp of the completed epoch
			last_samp=complete_ep.events[-1][0] #last event timestamp of the completed epoch
			first_match=[i[0] for i in event_array if i[2]==ecEv][0] # first {eCEV} (ie, 30,34,etc) timestamp of the current event array, make sure the epoch_string is mapping the same EC events as the complete epoch with this same label
			last_match=[i[0] for i in event_array if i[2]==eCEv][-1]
			if first_samp != first_match or last_samp != last_match:
				print("Oops! There is an error discovered during an inspection of this subject's complete vs incomplete EC epochs.\n")
				print("The completed epoch with this label: "+sub+'_'+epoch_string+'-epo.fif'+" does not match the time indices of the current working event list. Please check the ec_labels for more info.") 
				exit()
			continue
		else:
			ev_id={epoch_string:ecEv}
			EC_eps[epoch_string]=mne.Epochs(thisRaw,event_array,event_id=ev_id,baseline=None,tmin=0,tmax=2,reject_by_annotation=False)

	print(EO_eps)
	print(EC_eps)

	### Now running through the EO epochs to do the preprocessing steps on each ep ~~
	for EO_event in EO_eps:
		eyes_open=EO_eps[EO_event]
		thisEp_name=EO_event
		print('\n~~~~~~~~\n~~~~~~~~~~~~~~~~~~~\n~~~~~~~~~~~~~~~~~~~~~~~ RUNNING THROUGH EO NUMBER '+thisEp_name[-1]+' ~~~~~~~~~~~~~~~~~~~~~~~ \n~~~~~~~~~~~~~~~~~~~\n~~~~~~~~\n')
		
		## CHANNEL INSPECTION/REJECTION/INTERPOLATION
		eyes_open.load_data()
		EO_DataCH=eyes_open.copy().drop_channels(extChannels)
		tryAgain=1
		while tryAgain:
			EO_DataCH.plot(block=True,n_epochs=2,n_channels=15,scalings=scalings, title='SELECT BAD CHANNELS: eyes open') #pauses script while i visually inspect data and select which channels to delete
			bads_EO=EO_DataCH.info['bads']
			text_msg2=input('The channels you marked as bad are: '+str(bads_EO)+' \n Are you ready to interpolate? [y/n]: ')
			if text_msg2=='y':
				eyes_open.info['bads']=bads_EO
				eyes_open=eyes_open.interpolate_bads()
				tryAgain=0
			elif text_msg2=='n':
				tryAgain=1
			else:
				print('invalid entry: '+text_msg2)
				tryAgain=1

		eyes_open, r2 = mne.set_eeg_reference(eyes_open,ref_channels=dataChannels+extChannels)
		
		our_picks_EO=mne.pick_types(eyes_open.info,meg=False,eeg=True,eog=False,ecg=False)

		## ICA
		ica=[]
		ica = ICA(n_components=90,random_state=25)
		ica.fit(eyes_open,picks=our_picks_EO)	

		eog_ic=[]
		for ch in ['E25','E17','E8','E21','E14','E125','E126','E127','E128']: #insert EOG channels
			eog_idx,scores=ica.find_bads_eog(eyes_open,ch_name=ch)
			eog_ic.append(eog_idx)
            
		print(eog_ic)
            
		ecg_ic=[]
		for ch in []: # insert ECG channels
			ecg_idx,scores=ica.find_bads_ecg(eyes_open,ch_name=ch)
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

		repeatICA=1
		while repeatICA:
			plot=1
			while plot:
				ica.plot_components(picks=range(50),ch_type=None,title="EO 1-50",inst=eyes_open) #needs the channel locations
				ica.plot_components(picks=range(50,90),ch_type=None,title="EO 50-90",inst=eyes_open)
				verification_ic=input('The ICs marked for rejection are: ' + str(ica.exclude) +'\n Are you sure you want to proceed? [y/n]: ')
				if verification_ic=='y':
					plot=0
				else:
					print('Please select which ICs you would like to reject')

			EO_without_ICA=eyes_open.copy()
			EO_with_ICA=eyes_open.copy()
			EO_with_ICA=ica.apply(EO_with_ICA,exclude=ica.exclude)

			EO_with_ICA.plot(title='EO, AFTER ICA',n_epochs=4,n_channels=30,scalings=scalings)
			EO_without_ICA.plot(block=True, title='EO, BEFORE ICA',n_epochs=4,n_channels=30,scalings=scalings)
		
			cont=input('Continue on to epoch rejection? [y/n]: ')
			if cont=='y':
				eyes_open=EO_with_ICA.drop_channels(extChannels)
				repeatICA=0

		plotEpoch=1
		while plotEpoch:
			EO_Ep=eyes_open.copy()
			print('plotting '+str(np.shape(EO_Ep.get_data())[0])+ ' epochs')
			EO_Ep.plot(block=True,n_epochs=2,n_channels=30, scalings=scalings, title='SELECT BAD EPOCHS: eyes open',picks=mne.pick_channels(ch_names=dataChannels,include=[],exclude=extChannels))
			EO_Ep.plot_psd_topomap(cmap='interactive')
			bads=input('Are you sure you want to continue onto saving? [y/n]: ')
			if bads=='y':
				eyes_open=EO_Ep
				plotEpoch=0
			elif bads=='n':
				continue
			else:
				print('oops, please indicate which epochs you would like to reject')

		eyes_open.save(save_ROOT+sub+'/'+sub+'_'+thisEp_name+'-epo.fif')
		if os.path.exists(save_ROOT+sub+'/'+sub+'_'+thisEp_name+'-epo.fif'):
			print('\n\n SAVING OUT INFO ~~~~~~~ \n\n'+save_ROOT+sub+'/'+sub+'_'+thisEp_name+'-epo.fif......DONE!')

	eyes_open=[]
	thisEp_name=[]
               
	### Now running through the EC epochs to do the preprocessing steps on each ep ~~
	for EC_event in EC_eps:
		eyes_closed=EC_eps[EC_event]
		thisEp_name=EC_event
		print('\n~~~~~~~~\n~~~~~~~~~~~~~~~~~~~\n~~~~~~~~~~~~~~~~~~~~~~~ RUNNING THROUGH EC NUMBER '+thisEp_name[-1]+' ~~~~~~~~~~~~~~~~~~~~~~~ \n~~~~~~~~~~~~~~~~~~~\n~~~~~~~~\n')
		
		## CHANNEL INSPECTION/REJECTION/INTERPOLATION
		eyes_closed.load_data()
		EC_DataCH=eyes_closed.copy().drop_channels(extChannels)
		tryAgain=1
		while tryAgain:
			EC_DataCH.plot(block=True,n_epochs=2,n_channels=15,scalings=scalings, title='SELECT BAD CHANNELS: eyes closed') #pauses script while i visually inspect data and select which channels to delete
			bads_EC=EC_DataCH.info['bads']
			text_msg2=input('The channels you marked as bad are: '+str(bads_EC)+' \n Are you ready to interpolate? [y/n]: ')
			if text_msg2=='y':
				eyes_closed.info['bads']=bads_EC
				eyes_closed=eyes_closed.interpolate_bads()
				tryAgain=0
			elif text_msg2=='n':
				tryAgain=1
			else:
				print('invalid entry: '+text_msg2)
				tryAgain=1

		eyes_closed, r2 = mne.set_eeg_reference(eyes_closed,ref_channels=dataChannels+extChannels)
		
		our_picks_EC=mne.pick_types(eyes_closed.info,meg=False,eeg=True,eog=False,ecg=False)

		## ICA
		ica=[]
		ica = ICA(n_components=90,random_state=25)
		ica.fit(eyes_closed,picks=our_picks_EC)	

		eog_ic=[]
		for ch in ['E25','E17','E8','E21','E14','E125','E126','E127','E128']: #insert EOG channels
			eog_idx,scores=ica.find_bads_eog(eyes_closed,ch_name=ch)
			eog_ic.append(eog_idx)
            
		print(eog_ic)
            
		ecg_ic=[]
		for ch in []: # insert ECG channels
			ecg_idx,scores=ica.find_bads_ecg(eyes_closed,ch_name=ch)
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

		repeatICA=1
		while repeatICA:
			plot=1
			while plot:
				ica.plot_components(picks=range(50),ch_type=None,title="EC 1-50",inst=eyes_closed) #needs the channel locations
				ica.plot_components(picks=range(50,90),ch_type=None,title="EC 50-90",inst=eyes_closed)
				verification_ic=input('The ICs marked for rejection are: ' + str(ica.exclude) +'\n Are you sure you want to proceed? [y/n]: ')
				if verification_ic=='y':
					plot=0
				else:
					print('Please select which ICs you would like to reject')

			EC_without_ICA=eyes_closed.copy()
			EC_with_ICA=eyes_closed.copy()
			EC_with_ICA=ica.apply(EC_with_ICA,exclude=ica.exclude)

			EC_with_ICA.plot(title='EC, AFTER ICA',n_epochs=4,n_channels=30,scalings=scalings)
			EC_without_ICA.plot(block=True, title='EC, BEFORE ICA',n_epochs=4,n_channels=30,scalings=scalings)
		
			cont=input('Continue on to epoch rejection? [y/n]: ')
			if cont=='y':
				eyes_closed=EC_with_ICA.drop_channels(extChannels)
				repeatICA=0

		plotEpoch=1
		while plotEpoch:
			EC_Ep=eyes_closed.copy()
			print('plotting '+str(np.shape(EC_Ep.get_data())[0])+ ' epochs')
			EC_Ep.plot(block=True,n_epochs=2,n_channels=30, scalings=scalings, title='SELECT BAD EPOCHS: eyes closed',picks=mne.pick_channels(ch_names=dataChannels,include=[],exclude=extChannels))
			EC_Ep.plot_psd_topomap(cmap='interactive')
			bads=input('Are you sure you want to continue onto saving? [y/n]: ')
			if bads=='y':
				eyes_closed=EC_Ep
				plotEpoch=0
			elif bads=='n':
				continue
			else:
				print('oops, please indicate which epochs you would like to reject')

		eyes_closed.save(save_ROOT+sub+'/'+sub+'_'+thisEp_name+'-epo.fif')
		if os.path.exists(save_ROOT+sub+'/'+sub+'_'+thisEp_name+'-epo.fif'):
			print('\n\n SAVING OUT INFO: '+save_ROOT+sub+'/'+sub+'_'+thisEp_name+'-epo.fif......DONE!')
		

	print('done!')

#raw.columns.name='times'

#raw.index.name='channel'

#raw2=raw.stack().reset_index(name='mV')

#g=sns.FacetGrid(raw2,col=u'channel',hue='channel',col_wrap=3,height=3.5)

#g=g.map(plt.plot,'times','mV')
#plt.show()

