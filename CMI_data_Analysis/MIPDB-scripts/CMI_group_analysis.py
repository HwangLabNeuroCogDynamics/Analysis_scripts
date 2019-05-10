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
from fooof import FOOOF,FOOOFGroup
import os 
import csv
#%pylab inline
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from fooof.synth import gen_group_power_spectra,param_sampler
from fooof.analysis import get_band_peak, get_band_peak_group
import pickle

demos=pd.read_csv('/home/dcellier/RDSS/CMI_data/MIPDB/phenoData/subs_we_care_about_demos.csv')
ROOT_raw_data= '/data/backed_up/shared/CMI_data/raw_data/MIPDB/EEGData'
ROOT_preproc_data= '/data/backed_up/shared/CMI_data/MIPDB/preproc_RS/'
output_filename='/data/backed_up/shared/CMI_data/group_timefreq/'
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
    'E27','E28','E29','E30','E31','E33','E34','E35','E36','E37',    'E39','E40','E41','E42','E45','E46','E47','E50','E51','E52','E53','E54','E55','E58','E59','E60','E61',
    'E62','E65','E66','E67','E70','E71','E72','E75','E76','E77','E78','E79','E80','E83','E84',
    'E85','E86','E87','E90','E91','E92','E93','E96','E97','E98','E101','E102','E103','E104',
    'E105','E106','E108','E109','E110','E111','E112','E115','E116','E117','E118','E122','E123','E124']

def save_object(obj, fileloc):
	with open(fileloc, 'wb') as output:
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def make_csv(filename,subdat):
	with open(filename+'.csv',mode='w') as csv_file:
		max_len=0 
		fieldnames=[]
		for sub_dicts in subdat:  # looping through and making sure all potential dict keys will be added to fieldnames
			for key in sub_dicts.keys():
				if key not in fieldnames:
					fieldnames.append(key)
		writer=csv.DictWriter(csv_file,fieldnames=fieldnames)
		writer.writeheader()
		for dictionary in subdat:
			writer.writerow(dictionary) # erroring out bc some dictionaries are longer than others
		

def color_map(chs,coordinates,dend): 
    
    ## chs= list of electrodes to be 3d plotted, 
    ## coordinates= the coordinates of the electrodes to be plotted
    ## dend= a dendrogram of clustered electrodes 
    
    cluster_idxs = defaultdict(list) #creating a dict of color:channel inds in dend
    for c, pi in zip(dend['color_list'], dend['icoord']):
        for leg in pi[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - int(i)) < 1e-5:
                cluster_idxs[c].append(int(i))
    
    colors_dict=defaultdict(list)          
    coor_colors=np.zeros(len(chs))
    coor_colors=list(coor_colors)
    for color in cluster_idxs.keys():
        leaf_inds=cluster_idxs[color] # list of leaf inds for that color cluster
        for ind in leaf_inds:
            ch_name=dend['ivl'][ind] # grab the leaf label 
            coor_colors[chs.index(ch_name)]=color
            colors_dict[color].append(ch_name)
            #and match it to a label in dataChannels, 
            #then insert the color corresponding that ch_name
            # into coor_colors
    
    fig=plt.figure()
    fig2=plt.figure()
    ax=fig.add_subplot((111),projection='3d')
    ax2=fig2.add_subplot((111))
    coor2plot=coordinates
    for coor in range(len(coor2plot)):
        ax.scatter(xs=coor2plot[coor,0],ys=coor2plot[coor,1],zs=coor2plot[coor,2],c=coor_colors[coor],depthshade=True)
        ax2.scatter(x=coor2plot[coor,0],y=coor2plot[coor,1],c=coor_colors[coor])
        ax.text(x=coor2plot[coor,0],y=coor2plot[coor,1],z=coor2plot[coor,2],s=chs[coor])
        ax2.text(x=coor2plot[coor,0],y=coor2plot[coor,1],s=chs[coor])
    return plt.show(),colors_dict


proc_subs=[]
for sub in os.listdir(ROOT_raw_data): #['A00053440']:##looping through all raw subjects and seeing if they have data in preproc folder
	#print(sub)
	if sub != 'subs_without_EEG' and sub !='channel_locs.txt' and sub != 'subs_w_2_RS_EEG_or_bad' and sub !='EEG paradigm info Misc_Readme' and len(os.listdir(ROOT_preproc_data+sub+'/'))>0:
		proc_subs.append(sub)
		print(sub)

allpsds={}
allEps={}
for finSub in proc_subs:
	allpsds[finSub]={'EO':[],'EC':[]}
	allEps[finSub]={'EO':[],'EC':[]}
	for n in range(0,5):
		if os.path.exists(ROOT_preproc_data+finSub+'/'+finSub+'_eyes_closed_epoch_'+str(n)+'-epo.fif'):
			eyes_closed=mne.read_epochs(ROOT_preproc_data+finSub+'/'+finSub+'_eyes_closed_epoch_'+str(n)+'-epo.fif')
			## combine all EO and EC epochs and then run psd welch on those-- what do i average over?
			psdsEC,freqsEC=mne.time_frequency.psd_welch(eyes_closed,fmin=1,fmax=50,tmax=2,n_overlap=(len(eyes_closed.times)*.125))
			allpsds[finSub]['EC'].append([psdsEC,freqsEC])
			allEps[finSub]['EC'].append(eyes_closed)
		if os.path.exists(ROOT_preproc_data+finSub+'/'+finSub+'_eyes_open_epoch_'+str(n)+'-epo.fif'):
			eyes_open=mne.read_epochs(ROOT_preproc_data+finSub+'/'+finSub+'_eyes_open_epoch_'+str(n)+'-epo.fif')
			psdsEO,freqsEO=mne.time_frequency.psd_welch(eyes_open,fmin=1,fmax=50,tmax=2,n_overlap=(len(eyes_open.times)*.125))
			allpsds[finSub]['EO'].append([psdsEO,freqsEO])
			allEps[finSub]['EO'].append(eyes_open)
	#print(allpsds)
	#rint(allEps)
	#exit()
#print(allpsds)

all_ec_psds=[]
all_eo_psds=[]
for subject in allpsds.keys():
	print(subject)
	subsEC=allpsds[subject]['EC'] # a list of lists of arrays, [psds,freqs] (one list for each EC epoch in this sub's processed folder)
	for l in subsEC:
		pEC=l[0]
		#print(pEC.shape)
		all_ec_psds.append(pEC)

	subsEO=allpsds[subject]['EO']
	for k in subsEO:
		pEO=k[0]
		#print(pEO.shape)
		if not pEO.shape[1] !=90:
			all_eo_psds.append(pEO)
		else:
			print(subject) # there's one subject that has 89 channels for one of its epochs??
	#exit()

elec_group=input("\nClustering or defined electrodes? [c/d]")
if elec_group=='c':
		group_psdsEC=np.concatenate((all_ec_psds),axis=0)
		chFreqsEC=np.mean(group_psdsEC,axis=0)
		Y_EC=pdist(chFreqsEC)
		Z_EC=linkage(Y_EC)
		print(Z_EC.shape)

		group_psdsEO=np.concatenate((all_eo_psds),axis=0)
		chFreqsEO=np.mean(group_psdsEO,axis=0)
		Y_EO=pdist(chFreqsEO)
		Z_EO=linkage(Y_EO)
		print(Z_EO.shape)

		layout=mne.channels.read_montage(kind='GSN-HydroCel-129')
		coords=layout.pos
		coords=coords[3:] #slicing to exclude FidNz, FidT9, FidT10
		#new list of coordinates where the indices of the coords match the dataCh inds
		coor2plot=np.zeros(len(dataChannels))
		coor2plot=list(coor2plot)
		for i in chs_list: # iter thru elems in all channels list
			if i in dataChannels: # if the elem is also in dataChannels
				coor2plot[dataChannels.index(i)]=list(coords)[chs_list.index(i)]
				# then fill coor2plot with the coordinates of the elem, but make sure it's
				# inserted at an index that matches the index of the corresponding
				# dataChannels label

		#coor2plot=[list(coordinates[chs_list.index(i)]) for i in chs_list if i in dataChannels] 
		# getting the coordinates of the data channels
		coor2plot=np.asarray(coor2plot)
		coor2plot[:,2].shape

		threshold=0.01
		notClustered=1
		while notClustered:
			dendEC=dendrogram(Z_EC,128,leaf_font_size=10,orientation='left',color_threshold=threshold*max(Z_EC[:,2]),labels=dataChannels,truncate_mode='mlab')
			plotEC,colorsEC=color_map(dataChannels,coor2plot,dendEC)
			clusterDONE=input("Continue onto EC FOOOF? [y/n]: ")
			if clusterDONE=='y':
				notClustered=0
			elif clusterDONE=='n':
				threshold=threshold+0.05

		mf_color1=input('enter the color of the midfrontal cluster (no spaces) [r , y , b , c(teal) , m(magenta) , g ]: ')
		po_color1=input('enter the color of the parietal/occipital cluster (no spaces) [r , y , b , c(teal) , m(magenta) , g ]: ')

		notmf_EC=[i for i in dataChannels if i not in colorsEC[mf_color1]]
		notpo_EC=[e for e in dataChannels if e not in colorsEC[po_color1]]


		threshold=0.01
		notClustered=1
		while notClustered:
			dendEO=dendrogram(Z_EO,128,leaf_font_size=10,orientation='left',color_threshold=threshold*max(Z_EO[:,2]),labels=dataChannels,truncate_mode='mlab')
			plotEO,colorsEO=color_map(dataChannels,coor2plot,dendEO)
			clusterDONE=input("Continue onto EO FOOOF? [y/n]: ")
			if clusterDONE=='y':
				notClustered=0
			elif clusterDONE=='n':
				threshold=threshold+0.05

		mf_color2=input('enter the color of the midfrontal cluster (no spaces) [r , y , b , c(teal) , m(magenta) , g ]: ')
		po_color2=input('enter the color of the parietal/occipital cluster (no spaces) [r , y , b , c(teal) , m(magenta) , g ]: ')

		notmf_EO=[i for i in dataChannels if i not in colorsEO[mf_color2]]
		notpo_EO=[e for e in dataChannels if e not in colorsEO[po_color2]]	
elif elec_group=='d':
	all_mf_elecs=['E4','E5','E11','E12','E16','E18','E19']#['E1','E16','E10','E23','E11','E3','E4','E124','E123','E19','E27','E24','E12','E20','E5','E118','E117','E116','E111','E112','E110','E104','E105','E106','E6','E13','E7','E30','E31','E80','E36','E35','E29','E28','E34']
	all_po_elecs=['E62','E67','E71','E72','E76','E77']#['E86','E78','E61','E53','E52','E60','E62','E92','E97','E96','E85','E91','E90','E77','E76','E84','E83','E7','E71','E67','E66','E59','E51','E50','E58','E65','E70','E75']
	mf_elecs=[e for e in all_mf_elecs if e in dataChannels] # weeding out any externals that may have accidentally slipped in
	po_elecs=[f for f in all_po_elecs if f in dataChannels] 
	notmf_EO=[i for i in dataChannels if i not in mf_elecs]
	notpo_EO=[i for i in dataChannels if i not in po_elecs]
	notmf_EC=[i for i in dataChannels if i not in mf_elecs]
	notpo_EC=[i for i in dataChannels if i not in po_elecs]
else:
	print('Oops, please choose a method of selecting electrodes subsets.')
	exit()

all_sub_res=[]
for sub in allEps.keys():
	print(sub)
	ec=allEps[sub]['EC']
	eo=allEps[sub]['EO']

	thisSub_group_EC_psds_mf=[]
	thisSub_group_EC_psds_po=[]
	for ec_EP in ec: # fourier transforming each EC epoch and then concatenating into one large array
		ec_mf=ec_EP.copy().drop_channels(notmf_EC)
		ec_po=ec_EP.copy().drop_channels(notpo_EC)
		psdsEC_mf,freqsEC_mf=mne.time_frequency.psd_welch(ec_mf,fmin=1,fmax=50,tmax=2,n_overlap=(len(ec_EP.times)*.125),n_fft=126) ### !!!!!!
		psdsEC_po,freqsEC_po=mne.time_frequency.psd_welch(ec_po,fmin=1,fmax=50,tmax=2,n_overlap=(len(ec_EP.times)*.125),n_fft=126) ### !!!!!!
		if not psdsEC_mf.shape[1] !=(90-len(notmf_EC)):
			thisSub_group_EC_psds_mf.append(psdsEC_mf)
		else:
			print(subject) # there's one subject that has a weird num channels for one of its epochs??
		if not psdsEC_po.shape[1] !=(90-len(notpo_EC)):
			thisSub_group_EC_psds_po.append(psdsEC_po)
		else:
			print(subject) 
		
	thisSub_group_psdsEC_mf=np.concatenate((thisSub_group_EC_psds_mf),axis=0) 
	thisSub_group_psdsEC_po=np.concatenate((thisSub_group_EC_psds_po),axis=0)
	print(thisSub_group_psdsEC_po.shape)
	
	thisSub_group_EO_psds_mf=[]
	thisSub_group_EO_psds_po=[]
	for eo_EP in eo: # fourier transforming each EO epoch and then concatenating
		eo_mf=eo_EP.copy().drop_channels(notmf_EO)
		eo_po=eo_EP.copy().drop_channels(notpo_EO)
		psdsEO_mf,freqsEO_mf=mne.time_frequency.psd_welch(eo_mf,fmin=1,fmax=50,tmax=2,n_overlap=(len(eo_EP.times)*.125),n_fft=126) ### !!!!!!
		psdsEO_po,freqsEO_po=mne.time_frequency.psd_welch(eo_po,fmin=1,fmax=50,tmax=2,n_overlap=(len(eo_EP.times)*.125),n_fft=126)###!!!!!!
		print(psdsEO_mf.shape)
		print(psdsEO_po.shape)
		thisSub_group_EO_psds_mf.append(psdsEO_mf)
		thisSub_group_EO_psds_po.append(psdsEO_po)
	thisSub_group_psdsEO_mf=np.concatenate((thisSub_group_EO_psds_mf),axis=0)
	thisSub_group_psdsEO_po=np.concatenate((thisSub_group_EO_psds_po),axis=0)

	avg_psds_EC_mf=np.mean(thisSub_group_psdsEC_mf,axis=0) # then averaging across all the seperate epochs
	avg_psds_EO_mf=np.mean(thisSub_group_psdsEO_mf,axis=0)
	avg_psds_EC_mf=np.mean(avg_psds_EC_mf,axis=0) # averaging again across channels
	avg_psds_EO_mf=np.mean(avg_psds_EO_mf,axis=0)

	avg_psds_EC_po=np.mean(thisSub_group_psdsEC_po,axis=0)
	avg_psds_EO_po=np.mean(thisSub_group_psdsEO_po,axis=0)
	avg_psds_EC_po=np.mean(avg_psds_EC_po,axis=0)
	avg_psds_EO_po=np.mean(avg_psds_EO_po,axis=0)
#
# ###### THIS SECTION OF CODE WAS INSERTED TO PRODUCE THE PSDs FOR TRADITIONAL TIME-FREQ ANALYSES
#	print('\n')
#	print(avg_psds_EC_mf.shape)
#	this_sex=demos[demos.sub_ID==sub].sub_Sex.iloc[0]
#	this_age=demos[demos.sub_ID==sub].sub_Age.iloc[0]
#	thisDict={'sub':sub,'age':this_age,'sex':this_sex,}
#	for e in range(len(avg_psds_EC_mf)):
#		thisDict['psds_EC_mf'+str(e)]=avg_psds_EC_mf[e]
#		thisDict['psds_EC_po'+str(e)]=avg_psds_EC_po[e]
#		thisDict['psds_EO_mf'+str(e)]=avg_psds_EO_mf[e]
#		thisDict['psds_EO_po'+str(e)]=avg_psds_EO_po[e]
#	all_sub_res.append(thisDict)
		

#make_csv(filename=output_filename+'all_sub_results',subdat=all_sub_res)		
#exit()
#if 1:
#
	freq_range=[1,50]
	
	fmEC_mf=FOOOF()
	fmEC_mf.report(freqsEC_mf,avg_psds_EC_mf,freq_range)
	print('\n\n\n')
	
	fmEO_mf= FOOOF()
	fmEO_mf.report(freqsEO_mf,avg_psds_EO_mf,freq_range)
	print('\n\n\n')

	fmEC_po=FOOOF()
	fmEC_po.report(freqsEC_po,avg_psds_EC_po,freq_range)
	print('\n\n\n')
	
	fmEO_po= FOOOF()
	fmEO_po.report(freqsEO_po,avg_psds_EO_po,freq_range)
	print('\n\n\n')

	theta_band=[4,8]
	alpha_band=[6,12]#[8,12]
	beta_band=[15,30]

	these_EC_mf_alphas=get_band_peak(fmEC_mf.peak_params_,alpha_band,ret_one=True)
	these_EC_mf_betas=get_band_peak(fmEC_mf.peak_params_,beta_band,ret_one=True)
	these_EC_mf_thetas=get_band_peak(fmEC_mf.peak_params_,theta_band,ret_one=True)
	these_EO_mf_alphas=get_band_peak(fmEO_mf.peak_params_,alpha_band,ret_one=True)
	these_EO_mf_betas=get_band_peak(fmEO_mf.peak_params_,beta_band,ret_one=True)
	these_EO_mf_thetas=get_band_peak(fmEO_mf.peak_params_,theta_band,ret_one=True)

	results_EC_mf=fmEC_mf.get_results()
	EC_mf_peaks=results_EC_mf.peak_params
	results_EO_mf=fmEO_mf.get_results()
	EO_mf_peaks=results_EO_mf.peak_params

	these_EC_po_alphas=get_band_peak(fmEC_po.peak_params_,alpha_band,ret_one=True)
	these_EC_po_betas=get_band_peak(fmEC_po.peak_params_,beta_band,ret_one=True)
	these_EC_po_thetas=get_band_peak(fmEC_po.peak_params_,theta_band,ret_one=True)
	these_EO_po_alphas=get_band_peak(fmEO_po.peak_params_,alpha_band,ret_one=True)
	these_EO_po_betas=get_band_peak(fmEO_po.peak_params_,beta_band,ret_one=True)
	these_EO_po_thetas=get_band_peak(fmEO_po.peak_params_,theta_band,ret_one=True)

	results_EC_po=fmEC_po.get_results()
	EC_po_peaks=results_EC_po.peak_params
	results_EO_po=fmEO_po.get_results()
	EO_po_peaks=results_EO_po.peak_params

	all_peaks={}
	for n in range(len(EC_po_peaks)): # i dont know how many peaks it will find per EC/EO, per cluster, per subject, 
					#so I'm trying to generalize in the dictionary I'm constructing here
		peek=EC_po_peaks[n]
		peek_n=str(n+1) #starting w 1
		peek_Peak='PO_EC_Peak_'+peek_n
		peek_BW='PO_EC_BW_'+peek_n
		peek_AMP='PO_EC_AMP_'+peek_n
		all_peaks[peek_Peak]=peek[0]
		all_peaks[peek_AMP]=peek[1]
		all_peaks[peek_BW]=peek[2]
	for n in range(len(EO_po_peaks)): 
		peek=EO_po_peaks[n]
		peek_n=str(n+1) #starting w 1
		peek_Peak='PO_EO_Peak_'+peek_n
		peek_BW='PO_EO_BW_'+peek_n
		peek_AMP='PO_EO_AMP_'+peek_n
		all_peaks[peek_Peak]=peek[0]
		all_peaks[peek_AMP]=peek[1]
		all_peaks[peek_BW]=peek[2]
	for n in range(len(EC_mf_peaks)):
		peek=EC_mf_peaks[n]
		peek_n=str(n+1) #starting w 1
		peek_Peak='MF_EC_Peak_'+peek_n
		peek_BW='MF_EC_BW_'+peek_n
		peek_AMP='MF_EC_AMP_'+peek_n
		all_peaks[peek_Peak]=peek[0]
		all_peaks[peek_AMP]=peek[1]
		all_peaks[peek_BW]=peek[2]
	for n in range(len(EO_mf_peaks)): 
		peek=EO_mf_peaks[n]
		peek_n=str(n+1) #starting w 1
		peek_Peak='MF_EO_Peak_'+peek_n
		peek_BW='MF_EO_BW_'+peek_n
		peek_AMP='MF_EO_AMP_'+peek_n
		all_peaks[peek_Peak]=peek[0]
		all_peaks[peek_AMP]=peek[1]
		all_peaks[peek_BW]=peek[2]

	this_sex=demos[demos.sub_ID==sub].sub_Sex.iloc[0]
	this_age=demos[demos.sub_ID==sub].sub_Age.iloc[0]
	thisDict={'sub':sub,'age':this_age,'sex':this_sex,
			'PO_EC_slope': results_EC_po.background_params[1],'PO_EC_yint':results_EC_po.background_params[0],
			'PO_EO_slope': results_EO_po.background_params[1],'PO_EO_yint':results_EO_po.background_params[0],

			'MF_EC_slope': results_EC_mf.background_params[1],'MF_EC_yint':results_EC_mf.background_params[0],
			'MF_EO_slope': results_EO_mf.background_params[1],'MF_EO_yint':results_EO_mf.background_params[0]}

	bands_peaks={
			'PO_EC_alphaPeak':these_EC_po_alphas[0],'PO_EC_alphaBW':these_EC_po_alphas[2],'PO_EC_alphaAMP':these_EC_po_alphas[1],
			'PO_EO_alphaPeak':these_EO_po_alphas[0],'PO_EO_alphaBW':these_EO_po_alphas[2],'PO_EO_alphaAMP':these_EO_po_alphas[1],	
			'PO_EC_betaPeak':these_EC_po_betas[0],'PO_EC_betaBW':these_EC_po_betas[2],'PO_EC_betaAMP':these_EC_po_betas[1],
			'PO_EO_betaPeak':these_EO_po_betas[0],'PO_EO_betaBW':these_EO_po_betas[2],'PO_EO_betaAMP':these_EO_po_betas[1],
			'PO_EC_thetaPeak':these_EC_po_thetas[0],'PO_EC_thetaBW':these_EC_po_thetas[2],'PO_EC_thetaAMP':these_EC_po_thetas[1],	
			'PO_EO_thetaPeak':these_EO_po_thetas[0],'PO_EO_thetaBW':these_EO_po_thetas[2],'PO_EO_thetaAMP':these_EO_po_thetas[1],
			
			'MF_EC_alphaPeak':these_EC_mf_alphas[0],'MF_EC_alphaBW':these_EC_mf_alphas[2],'MF_EC_alphaAMP':these_EC_mf_alphas[1],
			'MF_EO_alphaPeak':these_EO_mf_alphas[0],'MF_EO_alphaBW':these_EO_mf_alphas[2],'MF_EO_alphaAMP':these_EO_mf_alphas[1],	
			'MF_EC_betaPeak':these_EC_mf_betas[0],'MF_EC_betaBW':these_EC_mf_betas[2],'MF_EC_betaAMP':these_EC_mf_betas[1],
			'MF_EO_betaPeak':these_EO_mf_betas[0],'MF_EO_betaBW':these_EO_mf_betas[2],'MF_EO_betaAMP':these_EO_mf_betas[1],
			'MF_EC_thetaPeak':these_EC_mf_thetas[0],'MF_EC_thetaBW':these_EC_mf_thetas[2],'MF_EC_thetaAMP':these_EC_mf_thetas[1],	
			'MF_EO_thetaPeak':these_EO_mf_thetas[0],'MF_EO_thetaBW':these_EO_mf_thetas[2],'MF_EO_thetaAMP':these_EO_mf_thetas[1],
			}
	save_object(obj=fmEC_mf,fileloc=output_filename+'FOOOFs/'+sub+'_EC_mf_FOOOF')
	save_object(obj=fmEO_mf,fileloc=output_filename+'FOOOFs/'+sub+'_EO_mf_FOOOF')
	save_object(obj=fmEC_po,fileloc=output_filename+'FOOOFs/'+sub+'_EC_po_FOOOF')
	save_object(obj=fmEO_po,fileloc=output_filename+'FOOOFs/'+sub+'_EO_po_FOOOF')
	thisDict.update(bands_peaks)
	thisDict.update(all_peaks)
	all_sub_res.append(thisDict)

make_csv(filename=output_filename+'all_sub_results',subdat=all_sub_res)

#save out sub #, peaks, and theta/alpha/beta nums to a csv
	
	
#['sub','age','sex','slope','yint','alphaPeak','alphaBW','alphaAMP'
#'betaPeak','betaBW','betaAMP','thetaPeak','thetaBW','thetaAMP',
#'peak1','amp1','bw1','peak2','amp2','bw2','peak3','amp3','bw3'
#'peak4','amp4','bw4','peak5','amp5','bw5']




				
