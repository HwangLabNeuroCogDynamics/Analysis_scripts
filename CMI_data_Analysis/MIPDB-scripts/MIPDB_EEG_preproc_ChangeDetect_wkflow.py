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
import csv


ROOT = '/home/dcellier/RDSS/CMI_data/MIPDB/EEGData/'
save_ROOT='/data/backed_up/shared/CMI_data/MIPDB/preproc_CD/'

all_sub_events={}
rep_subs={}
for f in os.listdir(ROOT):
    if f=='channel_locs.txt' or f=='subs_without_EEG' or f=='EEG paradigm info Misc_Readme' or f=='subs_w_2_RS_EEG':
        print('nothin')
    else:
        thisfpath=ROOT+f+'/EEG/raw/csv_format/'
        for raw_dat in os.listdir(thisfpath):
            
            #print(thisfpath)
            pattern='%s00*_events.csv' %f
            pattern2='%s 00*_events.csv' %f
            pattern3='%s_part*00*_events.csv'%f
            if (fnmatch.fnmatch(raw_dat, pattern)) or (fnmatch.fnmatch(raw_dat, pattern2)) or (fnmatch.fnmatch(raw_dat, pattern3)):
                print(raw_dat)
                thisTrigfile=pd.read_csv(thisfpath+raw_dat)
                not_CD=0
                ev_ar=[]
                for n in range(len(thisTrigfile['type'])):
                    trig=thisTrigfile['type'][n]
                    time_stamp=thisTrigfile['latency'][n]
                    if trig != 'type':
                        if int(trig) not in [94,95,96,5,8,9,12,13]:
                            not_CD=1
                            #print(trig)
                        else:
                            ev_ar.append([int(time_stamp),0,int(trig)])
                if not_CD==0 and (f not in all_sub_events.keys()):
                    print("YESSSSSSSSSs")
                    all_sub_events[f]=[]
                if not_CD==0 and (len(all_sub_events[f])<3):
                    print('adding one file')
                    all_sub_events[f].append({raw_dat:ev_ar})
                elif not_CD==0 and (len(all_sub_events[f])>3):
                    print('AHHHHHHHHHHHHHHHH   '+f)
                    rep_subs[f]=ev_ar
                #elif not_CD==1:
                 #   print(raw_dat)

print('\nSUBS MISSING BLOCKS OR BLOCKS TOO SHORT:\n\n')
subs_missing_blocks=[]
new_all_sub_events=all_sub_events.copy()
for sub in all_sub_events:
	sub_dict=all_sub_events[sub]
	if len(sub_dict)<3:
		subs_missing_blocks.append(sub)
		new_all_sub_events.pop(sub)
		print(sub)
	elif len(sub_dict)==3: 
		for list_ele in sub_dict: #for each block dict in the sub_dict list
			for key in list_ele: #for each key in each dict (only 1)
				if len(list_ele[key]) < 60: #if the block has too few events, exclude this sub, kind of randomly picking 60 bc blocks with all 24 trials should have around 72 events but mayb subjects dont respond in time?
					subs_missing_blocks.append(sub)
					new_all_sub_events.pop(sub)
all_sub_events=new_all_sub_events

another_sub_events=all_sub_events.copy()
print("\nCOMPLETE SUBS: \n\n")
for sub in all_sub_events:
	preproc_file=save_ROOT+sub+'/'
	if os.path.exists(preproc_file) and len(os.listdir(preproc_file))>=4:
		#check whether all eps exist, don't know yet
		print(sub)
		another_sub_events.pop(sub)
all_sub_events=another_sub_events

for sub in all_sub_events:
	block_list=all_sub_events[sub] # this is a list of 3 dictionaries, where ea dict key is the file name of the block 1-3 and the value is the event list of these
	for block in block_list: #block is a dictionary
		for block_file in block:#block_file should be a string of the corresponding csv event file name
			csv_evs=block[block_file]
			raw_EEG_path=block_file.split('_events') #returns a list of [filename, '.csv']
			raw_EEG_path=raw_EEG_path[0]+'.raw'
			raw_EEG_path=ROOT+sub+'/EEG/raw/raw_format/'+raw_EEG_path
			raw=mne.io.read_raw_egi(raw_EEG_path,montage=mne.channels.read_montage(kind='GSN-HydroCel-129'),preload=True, verbose=True)
			raw_evs=mne.find_events(raw)
			raw_trigs=[ev[0] for ev in raw_evs] # a list of the time indices of events
			csv_trigs=[ev[0] for ev in csv_evs]
			#print(raw_trigs==csv_trigs)
			#print(raw_trigs)
			#print(csv_trigs)
			if raw_trigs != csv_trigs[1:]: #Slicing off first element because this is usually the one with 94,95 or 96
				print('Uh oh! The events from the csv and the raw eeg file do not match. Please inspect this file for more info: '+block_file)
				exit()

			csv_codes=[ev[2] for ev in csv_evs]
			if csv_codes[0]==94:
				block_num=1
			elif csv_codes[0]==95:
				block_num=2
			elif csv_codes[0]==96:
				block_num=3
			else:
				print('OOPS! looks like the file you are working with is not the correct task')
				print(csv_codes)
				exit()
			csv_iterable=list(zip(csv_codes,csv_trigs))
			
			print('\nBEFORE REMOVAL OF BAD TRIALS:\n')
			print([a for a in csv_iterable])
			print('\n')
			#print('\nREMOVING RESP DURING ITI\n')
			#new_csv_iterable=csv_iterable.copy()
			#for n in range(1,len(csv_iterable)): #need to get the RT of each button press
			#	thisCode=csv_iterable[n][0]
			#	thisTrig=csv_iterable[n][1]
			#	prevCode=csv_iterable[n-1][0]
				#prevTrig=csv_iterable[n-1][1]
			#	tossTrial_resp=0
			#	if prevCode==5 and (thisCode==12 or thisCode==13):
			#		tossTrial_resp=1 # if there's a response during the ITI
			#	if tossTrial_resp:
			#		k=n
			#		if (k+2 <len(csv_iterable)):
			#			nextCode=csv_iterable[k+1][0] # if we aren't at the end of the event list
			#			while nextCode != 5: 
			#				k=k+1
			#				nextCode=csv_iterable[k+1][0]
			#		trials_to_toss=csv_iterable[n-1:k+1] # should yield something like [(5, 36429), (12, 38292), (9, 39427), (13, 40207)]
			#		for trial in trials_to_toss:
			#			new_csv_iterable.remove(trial)
			#		print(trials_to_toss)
			#csv_iterable=new_csv_iterable

			#print('\n')
			#print([a for a in csv_iterable])
			#print('\nREMOVING NON RESP\n')

			#new_csv_iterable=csv_iterable.copy()
			#for n in range(len(csv_iterable)-1): 
			#	thisCode=csv_iterable[n][0]
			#	thisTrig=csv_iterable[n][1]
			#	nextCode=csv_iterable[n+1][0]
			#	tossTrial_no_resp=0
			#	if (thisCode==9 or thisCode==8) and nextCode==5: # if the subject didn't respond between change and next trial start
			#		tossTrial_no_resp=1  
			#	if tossTrial_no_resp:
			#		k=n
			#		print(k)
			#		print(len(csv_iterable))
			#		if (k+2 < len(csv_iterable)): #if this hasn't reached the end
			#			nextCode=csv_iterable[k+1][0]
			#			while nextCode != 5:
			#			    k=k+1
			#			    nextCode=csv_iterable[k+1][0]
			#		trials_to_toss=csv_iterable[n-1:k+1]
			#		print(trials_to_toss)
			#		for trial in trials_to_toss:
			#			new_csv_iterable.remove(trial)
			#print('\n')
			#print(new_csv_iterable)
			#print('\nREMOVING STRAGGLERS\n')

			#csv_iterable=new_csv_iterable
			#last_evs=[a for a in csv_iterable[-3:]]
			#for i in range(3):
		#		print(last_evs)
		#		print('\n')
		#		if (last_evs[2][0] != 12) and (last_evs[2][0] != 13): csv_iterable.remove(last_evs[2])
		#		last_evs=[a for a in csv_iterable[-3:]]

			k=0
			n=0
			new_iterable=[]
			export_to_csv=[]
			while k<len(csv_iterable):
				trialStart=csv_iterable[k][0]
				trialStarttrig=csv_iterable[k][1]
				if not trialStart==5 or k>(len(csv_iterable)-3):	
					k=k+1
				else:
					nextCode=csv_iterable[k+1][0]
					nextTrig=csv_iterable[k+1][1]
					nextnextCode=csv_iterable[k+2][0]
					nextnextTrig=csv_iterable[k+2][1]
					print((trialStart,trialStarttrig, nextCode,nextTrig, nextnextCode,nextnextTrig))
					if (nextCode==8 or nextCode==9) and (nextnextCode==12 or nextnextCode==13): #if this was a response
						thisRT=(nextnextTrig-nextTrig)*0.002
						if (nextCode==8 and nextnextCode==12) or (nextCode==9 and nextnextCode==13):
							thisAcc=1
						else:
							thisAcc=0
						thisITI=(nextTrig-trialStarttrig)*0.002 #change start-trial start ind times sample rate
						thisTrial=n
						n=n+1
						new_iterable.append((trialStart,trialStarttrig))
						new_iterable.append((nextCode,nextTrig))
						new_iterable.append((nextnextCode,nextnextTrig)) # appending these to a new iterable and skipping over weird trials
						export_to_csv.append([[(trialStart,trialStarttrig), (nextCode,nextTrig), (nextnextCode,nextnextTrig)],thisTrial,thisAcc,thisRT,thisITI])
						print(thisAcc,thisRT,thisTrial)
					else:
						q=k+1
						nextTrialstart=0
						while nextTrialstart !=5:
							q=q+1
							nextTrialstart=csv_iterable[q][0]
						print('\n Removing this funkytown trial:')
						print(csv_iterable[k:q])
						print('\n')
					k=k+1

			if not os.path.exists(save_ROOT+sub+'/'):
				os.mkdir(save_ROOT+sub+'/')
				os.chmod(save_ROOT+sub+'/',0o2770)			
			csv_path=save_ROOT+sub+'/'+sub+'_behavior_block_'+str(block_num)+'.csv'	
			if not os.path.exists(csv_path):
				#print('\n')
				#print(export_to_csv)
				#print('\n')
				empty_Csv=pd.DataFrame()
				empty_Csv.to_csv(csv_path)
				with open(csv_path,mode='w') as csv_file:
					fieldnames=['trial_num','trialStart','changeStart','subResponse','targetOri','ITI','RT','acc']
					writer=csv.DictWriter(csv_file,fieldnames=fieldnames)
					writer.writeheader()
					for trial in export_to_csv:
						if trial[0][1][0]==8:
							targetOri='L'
						elif trial[0][1][0]==9:
							targetOri='R'
						writer.writerow({'trial_num':trial[1],'trialStart':trial[0][0],'changeStart':trial[0][1],'subResponse':trial[0][2],'targetOri': targetOri,'ITI':trial[4],'RT':trial[3],'acc':trial[2] })

			csv_iterable=new_iterable
			print('\n')
			print('EVENTS WITH WEIRD STUFF REMOVED: \n')
			print([t for t in csv_iterable])
			print('\n')

			new_csv_events=[]
			for ev in csv_iterable:
				new_csv_events.append([ev[1],0,ev[0]])
			print('NEW (WORKABLE) EVENT ARRAY \n')
			print(new_csv_events)
			print('\n')
			### EPOCHING INTO EVENTS ###

			dataChannels = ['E2','E3','E4','E5','E6','E7','E9','E10','E11','E12','E13','E15','E16','E18','E19','E20','E22','E23','E24','E26','E27','E28','E29','E30','E31','E33','E34','E35','E36','E37','E39','E40','E41','E42','E45','E46','E47','E50','E51','E52','E53','E54','E55','E58','E59','E60','E61','E62','E65','E66','E67','E70','E71','E72','E75','E76','E77','E78','E79','E80','E83','E84','E85','E86','E87','E90','E91','E92','E93','E96','E97','E98','E101','E102','E103','E104','E105','E106','E108','E109','E110','E111','E112','E115','E116','E117','E118','E122','E123','E124']
			#this is the list of 'real' channels that Justin sent, all others are external electrodes
			extChannels = [e['ch_name'] for e in raw.info['chs'] if ((e['ch_name'] not in dataChannels) and (e in raw.info['chs'][:129]))]
			randChannels=[e for e in raw.info['ch_names'] if ((e not in dataChannels) and (e not in extChannels))]
			ch_type_dict={}
			for c in extChannels:
				ch_type_dict[c]='ecg'
			raw.set_channel_types(ch_type_dict)

			events=np.asarray(new_csv_events)
			raw.add_events(events,'STI 014',replace=True)
			events=mne.find_events(raw)
			behav_df=pd.read_csv(csv_path)
			resp_start=mne.Epochs(raw,events,baseline=None, event_id={'response_left':12, 'response_right':13},tmin=-.5, tmax=2, reject_by_annotation=False,metadata=behav_df)
			#ITI_start=mne.Epochs(raw,events,baseline=None,event_id={'ITIStart':5},tmin=0,tmax=(2.8+2.4),reject_by_annotation=False,metadata=behav_df) #ITI PLUS contrast change
			change_start=mne.Epochs(raw, events,baseline=(-.5,0),event_id={'left_target':8,'right_target':9},tmin=-.5,tmax=3,reject_by_annotation=False,metadata=behav_df) # 500 ms of ITI plus change (2400) plus a little of next ITI (600ms)

			resp_path=save_ROOT+sub+'/'+sub+'_block'+str(block_num)+'_'+'response-epo.fif'
			#ITI_path= save_ROOT+sub+'/'+sub+'_block'+str(block_num)+'_'+'ITI-epo.fif'
			change_path= save_ROOT+sub+'/'+sub+'_block'+str(block_num)+'_'+'change-epo.fif'

			run_these=[]
			#if not os.path.exists(ITI_path): 
			#	run_these.append(ITI_start)
			if not os.path.exists(change_path):
				run_these.append(change_start)
			if not os.path.exists(resp_path):
				run_these.append(resp_start)

			for ep in run_these:
				ep.load_data()
				ep.filter(1,50)
				ep_copy=ep.copy().drop_channels(extChannels)
				ep_copy=ep_copy.copy().drop_channels(randChannels)
				
				tryAgain=1
				while tryAgain:
					ep_copy.plot(block=True,n_epochs=2,n_channels=15, title='SELECT BAD CHANNELS: '+str(ep_copy.event_id))
					bads=ep_copy.info['bads']
					text_msg2=input('The channels you marked as bad are: '+str(bads)+' \n Are you ready to interpolate? [y/n]: ')
					if text_msg2=='y': 
						ep.info['bads']=bads
						ep=ep.interpolate_bads()
						tryAgain=0
					elif text_msg2=='n': #if this hasn't reached the end
						tryAgain=1
					else:
						print('invalid entry: '+text_msg2)
						tryAgain=1
		
				layout=mne.channels.read_montage(kind='GSN-HydroCel-129')
				#print(randChannels)
				#ep.plot(block=True)
				ep=ep.drop_channels(randChannels)
				#ep, r = mne.set_eeg_reference(ep,ref_channels=extChannels)

				our_picks=mne.pick_types(ep.info,meg=False,eeg=True,eog=False,ecg=False)

				ep=ep.set_montage(layout)
				#ep.plot(block=True)
				repeatICA=1
				while repeatICA:
					ica = []

					ica = ICA(n_components=90,random_state=25)#,method='infomax')
					ica.fit(ep,picks=our_picks)

					eog_ic=[]
		            
					for ch in ['E25','E17','E8','E21','E14','E125','E126','E127','E128']: #insert EOG channels
						eog_idx,scores=ica.find_bads_eog(ep,ch_name=ch)
						eog_ic.append(eog_idx)
					print(eog_ic)

					ecg_ic=[]
					for ch in []: # insert ECG channels
						ecg_idx,scores=ica.find_bads_ecg(ep,ch_name=ch)
						ecg_ic.append(ecg_idx)

					print(ecg_ic)
					reject_ic=[]
					for eog_inds in eog_ic:
						for ele in eog_inds:
							if (ele not in reject_ic) and (ele <35):
								reject_ic.append(ele)
					for ecg_inds in ecg_ic:
						for ele in ecg_inds:
							if (ele not in reject_ic) and (ele <35):
								reject_ic.append(ele)

					print(reject_ic)

					ica.exclude=[]
					ica.exclude.extend(reject_ic) #which IC's to exclude

					plotICA=1
					while plotICA:
						ica.plot_components(picks=range(35),ch_type=None,title=str(ep.event_id),inst=ep) #needs the channel locations
						#ica.plot_components(picks=range(50,90),ch_type=None,title=str(ep.event_id),inst=ep)
						verification_ic=input('The ICs marked for rejection are: ' + str(ica.exclude) +'\n Are you sure you want to proceed? [y/n]: ')
						if verification_ic=='y':
							plotICA=0
						else:
							print('Please select which ICs you would like to reject')

					ep_w_IC=ep.copy()
					ep_w_no_IC=ep.copy()

					ep_w_IC=ica.apply(ep_w_IC,exclude=ica.exclude)
			
					ep_w_IC.plot(title='AFTER ICA',n_epochs=4,n_channels=30)
					ep_w_no_IC.plot(block=True, title='BEFORE ICA',n_epochs=4,n_channels=30)

					cont=input('Continue on to epoch rejection? [y/n]: ')
					if cont=='y':
						ep=ep_w_IC.drop_channels(extChannels)
						repeatICA=0

				plotEpoch=1
				while plotEpoch:
					ep_copy=ep.copy()
					print('plotting '+str(np.shape(ep_copy.get_data())[0])+ ' epochs')
					ep_copy.plot(block=True,n_epochs=2,n_channels=30, title='SELECT BAD EPOCHS',picks=mne.pick_channels(ch_names=dataChannels,include=[],exclude=extChannels))
					ep_copy.plot_psd_topomap(cmap='interactive')
					bads=input('Are you sure you want to continue onto saving? [y/n]: ')
					if bads=='y':
						ep=ep_copy
						plotEpoch=0
					elif bads=='n':
						continue
					else:
						print('oops, please indicate which epochs you would like to reject')
				
				if not os.path.exists(save_ROOT+sub+'/'):
					os.mkdir(save_ROOT+sub+'/')

				#if ep.event_id==ITI_start.event_id:
					#ep.save(ITI_path)
					#thisPath=ITI_path
				elif ep.event_id == change_start.event_id:
					ep.save(change_path)
					thisPath=change_path
				elif ep.event_id == resp_start.event_id:
					ep.save(resp_path)
					thisPath=resp_path
				else:
					print('woops')

				if os.path.exists(thisPath):
					print("\n SAVING OUT "+thisPath)

		print("\n \n MOVING ONTO NEXT BLOCK, NO. "+str(block_num))
				




