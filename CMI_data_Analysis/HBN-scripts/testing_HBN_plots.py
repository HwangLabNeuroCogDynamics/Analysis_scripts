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
save_ROOT='/data/backed_up/shared/CMI_data/HBN/preproc/'

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

	raw=raw_array['EEG'][0][0][15]
	if len(raw_array)!=129:#129, error out on purpose to test
		print("The data array extracted out of sub "+sub+"'s .mat file does not contain data for 129 channels: \n")
		print(raw_array)
		exit()
	
	mat_events_labels=[]
	mat_events_inds=[]
	mat_ev_arr=raw_array['EEG'][0][0][25] #extracting an array of arrays which contain event info
	if len(mat_ev_arr)==1: #if the array has more than one array of events we're in trouble
		mat_ev_arr=mat_ev_arr[0]
	else:
		print("The event array extracted out of sub "+sub+"'s .mat file does not contain the proper number of events: \n")
		print(mat_ev_arr)
		exit()
	if len(mat_ev_array)>0: #or if the event array within the array is empty we're in trouble
		for ev in mat_ev_array:
			mat_events_labels.append(ev[0][0])  # grab the 90,20,or30
			mat_events_inds.append(ev[1][0][0]) #grab the time index of the label
	else:
		print("The event array extracted out of sub "+sub+"'s .mat file does not contain the proper number of events: \n")
		print(mat_ev_arr)
		exit()
	if mat_events_labels[0]=='break cnt': #cut out that first and last break count thing
		mat_events_labels=mat_events_labels[1:]
		mat_events_inds=mat_events_inds[1:]
	if mat_events_labels[-1]=='break cnt':
		mat_events_labels=mat_events_labels[:-1]
		mat_events_inds=mat)events_inds[:-1]			
	if events['type'][0]=='break cnt':
		events=events[1:]
	if events['type'][-1]=='break cnt':
		events=events[:-1]
	events=events.reset_index(drop=True) #reset indices to zero after chopping off first and last rows

	if len(events)==len(mat_events_labels)==len(mat_events_inds): #make sure these lists are the same length
		csv_events_labels=events['type']
		csv_events_inds=events['sample']
		for n in range(len(events)): # then index each element in both lists and make sure they're the same
			if (csv_events_labels[n]==mat_events_labels[n]) and (csv_events_inds[n]==mat_events_inds[n]):
				this_ev_label=csv_events_labels[n].split()
				this_ev_ind=csv_events_inds[n]
				if this_ev_label not in ['20','90','30']:
					print("The event array extracted out of sub "+sub+"'s .mat or .csv file contains an excessive event. Please inspect this and mark this subject as unusual: \n")
					print(events)
					exit()
			else: # throw an error if there's an event that doesn't match
				print("The event arrays extracted out of sub "+sub+"'s .mat file and .csv event file do not match each other: \n")
				print("Events from the .csv: \n")
				print(events)
				print("\nEvents from the .mat: \n")
				print(mat_events_labels)
				print('\nEvent time inds from the .mat: \n")
				print(mat_events_inds)
				exit()
	else: # throw an error if list aren't same length
		print("The event arrays extracted out of sub "+sub+"'s .mat file and .csv event file do not match each other: \n")
		print("Events from the .csv: \n")
		print(events)
		print("\nEvents from the .mat: \n")
		print(mat_events_labels)
		print('\nEvent time inds from the .mat: \n")
		print(mat_events_inds)
		exit()
	EO_events={}
	EC_events={}
 
					if (this_ev_label in ['90','20','30']):
						if this_ev_label=='90':
							start_time=this_ev_ind
						elif this_ev_label=='30':
						elif this_ev_label=='20':
	
	channels=mne.channels.read_montage(kind='GSN-HydroCel-129')
	channel_pos=channels.pos[3:]
	channel_names=channels.ch_names[3:]
	channels=mne.channels.read_montage(kind='GSN-HydroCel-129',ch_names=channel_names)

	raw_info=mne.create_info(ch_names=channel_names,sfreq=500.0,montage=channels,ch_types='eeg')

	thisRaw=mne.io.RawArray(raw,raw_info,first_samp=0,verbose=True)
	thisRaw.filter(1,50)
	
	thisRaw.plot(block=True,scalings={'eeg':100})#'auto')
	
	print('done!')

#raw.columns.name='times'

#raw.index.name='channel'

#raw2=raw.stack().reset_index(name='mV')

#g=sns.FacetGrid(raw2,col=u'channel',hue='channel',col_wrap=3,height=3.5)

#g=g.map(plt.plot,'times','mV')
#plt.show()

