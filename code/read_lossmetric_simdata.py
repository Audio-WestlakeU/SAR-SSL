""" 
    Read downstream results (MAEs of TDOA, DRR, T60, C50, SNR, ABS, SUR, VOL estimation) from log files	for simulated data
"""

import os
import argparse
import numpy as np
import scipy.io
from opt import opt_pretrain
from tensorboard.backend.event_processing import event_accumulator  
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Reading downstream results')

opts = opt_pretrain()
dirs = opts.dir()


def get_ver_time_list(day_range, ver_time_dir):
	"""
		Path: pretrain/time_version or task/time_version/**
	"""
	day_specify = []
	for day in range(int(day_range[0]), int(day_range[1])+1):
		day_str = str(day)
		if len(day_str)==3:
			day_str = '0'+day_str
		day_specify += [day_str]

	time_list = []
	ver_list = []
	ver_time_list = os.listdir(ver_time_dir)
	for ver_time in ver_time_list:
		if ver_time[0:4] in day_specify:
			time_list += [ver_time]
			ver_name_list = os.listdir(ver_time_dir+'/'+ver_time)
			if 'train' in ver_name_list:
				ver_list += [ver_time]
			else: 
				for ver_name in ver_name_list:
					if os.path.splitext(ver_name)[1] != '.mat':
						ver_list +=[ver_time + '-' + ver_name]

	time_list_undup = list(set(time_list))
	time_list_undup.sort()
	ver_list.sort()

	return ver_list, time_list_undup

def get_merged_result_from_log_file(log_dir, LargeNum=500):
	'''
	Function: remove invalied log files, and merge partial results to one
	'''
	log_part_list_valid = os.listdir(log_dir)
	log_part_list = os.listdir(log_dir)
	result = {}

	for log_part in log_part_list:	
		log_name = log_dir + log_part
		ea = event_accumulator.EventAccumulator(log_name)     
		ea.Reload()    
		keys = ea.scalars.Keys()
		if len(keys) == 0:
			log_part_list_valid.remove(log_part)
	log_part_list_valid.sort()

	result_long = np.ones(LargeNum) * LargeNum
	for log_part in log_part_list_valid:	
		idx = log_part_list_valid.index(log_part)
		log_name = log_dir + log_part
		ea = event_accumulator.EventAccumulator(log_name)     
		ea.Reload()    
		keys = ea.scalars.Keys()
		for key in keys:
			if idx == 0:
				result[key] = np.ones(LargeNum) * LargeNum
			nstep = len(ea.scalars.Items(key))
			for step_idx in range(nstep):
				epoch_idx = ea.scalars.Items(key)[step_idx].step-1
				result[key][epoch_idx] = ea.scalars.Items(key)[step_idx].value
			if idx == len(log_part_list_valid)-1:
				result[key][epoch_idx+1:] = LargeNum*2
				result[key] = np.delete(result[key], np.where(result[key]==LargeNum*2))

	result['server'] = log_name.split('.')[-1]

	return result

def read_result(ver_specify, ver_time_dir, stage_list= ['train', 'val', 'test']):
	result = {}
	for stage in stage_list:
		result[stage] = {}
		atts = ver_specify.split('-')
		ver_time = atts[0]
		ver_time_specify_dir = ver_time_dir + ver_time
		ver_names = os.listdir(ver_time_specify_dir)
		if 'train' in ver_names:
			for stage in stage_list:
				log_dir = ver_time_specify_dir + '/' + stage + '/'
				result[stage] = get_merged_result_from_log_file(log_dir)
		else:
			ver_name = ver_specify.replace(ver_time+'-', '')
			for stage in stage_list:
				log_dir = ver_time_specify_dir + '/' + ver_name + '/' + stage + '/'
				result[stage] = get_merged_result_from_log_file(log_dir)

	return result

def get_specifed_result(task, ver_specify, pretrain_ver_name_pattern, ds_ver_name_pattern):
	
	# Get specified pretrain results 
	ver_time_dir = dirs['exp'] + '/pretrain/'
	result = read_result(ver_specify, ver_time_dir)
	
	if ('train' not in result.keys()) | ('val' not in result.keys()) | ('test' not in result.keys()):
		print('lack')
	else:
		if 'loss' not in result['val'].keys():
			print('lack loss recording in validation stage!')

	best_epoch_idx = np.argmin(result['val']['loss'])

	atts_pattern = pretrain_ver_name_pattern.split('-')
	atts = ver_specify.split('-')
	time = atts[atts_pattern.index('TIME')]
	test_loss = result['test']['loss'][best_epoch_idx]
	train_loss = result['train']['loss'][best_epoch_idx]
	val_loss = result['val']['loss'][best_epoch_idx]
	nparam = result['test']['nparam'][0]
	best_epoch = best_epoch_idx+1
	server_id = result['test']['server']
	print('pretrain:', time, str(round(nparam, 2))+'M', best_epoch, round(test_loss,2))
	scipy.io.savemat(ver_time_dir+time+'/loss_'+time+'.mat', result)

	# Get specified downstream results 
	ver_time_dir = dirs['exp'] + '/'+ task + '/'
	result = read_result(ver_specify, ver_time_dir)
	if 'loss' not in result['val'].keys():
		print('lack loss recording in validation stage!')
	
	atts_pattern = ds_ver_name_pattern.split('-')
	atts = ver_specify.split('-')
	time = atts[atts_pattern.index('TIME')]
	for i in range(len(atts_pattern)):
		if i < len(atts):
			globals()[atts_pattern[i]] = atts[i]
		else:
			globals()[atts_pattern[i]] = ''
	ver_name = ds_ver_name_pattern.replace('FT', FT).replace('TIME-', '').replace('TOKEN', TOKEN).replace('HEAD', HEAD).replace('NUM', NUM).replace('BAS', BAS).replace('LR', LR).replace('CV', CV).replace('EMBED', EMBED)

	best_epoch_idx = np.argmin(result['val']['metric'])
	test_loss = result['test']['loss'][best_epoch_idx]
	test_metric = result['test']['metric'][best_epoch_idx]
	best_epoch = best_epoch_idx+1
	print('downstream: ', time, task, ver_name, best_epoch, round(test_loss,2), round(test_metric,2))
	scipy.io.savemat(ver_time_dir+time+'/test_result/loss_metric_' + ver_specify + '.mat', result)

def list2str(lis):
	strings = ''
	for entry in lis:
		strings += entry
		if lis.index(entry) != (len(lis)-1):
			strings += ', '
	return strings


if __name__ == '__main__':
	task = 'TDOA'
	# task = 'DRR'
	# task = 'T60'
	ver_name_pattern_ori = 'TIME-FT-TOKEN-HEAD-NUM-LR-BAS-CV-EMBED' # according to the saved directory+file
	get_specifed_result(task=task, ver_specify='08220022-finetune-all-mlp-800-0.001-8-1-spat-sim_0821R8', pretrain_ver_name_pattern='TIME', ds_ver_name_pattern=ver_name_pattern_ori)
	get_specifed_result(task=task, ver_specify='08220022-finetune-all-mlp-800-0.0005-8-1-spat-sim_0821R8', pretrain_ver_name_pattern='TIME', ds_ver_name_pattern=ver_name_pattern_ori)
	get_specifed_result(task=task, ver_specify='08220022-finetune-all-mlp-800-0.0001-8-1-spat-sim_0821R8', pretrain_ver_name_pattern='TIME', ds_ver_name_pattern=ver_name_pattern_ori)
	get_specifed_result(task=task, ver_specify='08220022-finetune-all-mlp-800-5e-05-8-1-spat-sim_0821R8', pretrain_ver_name_pattern='TIME', ds_ver_name_pattern=ver_name_pattern_ori)
	get_specifed_result(task=task, ver_specify='08220022-scratchlow-all-mlp-800-0.001-8-1-spat-sim_0821R8', pretrain_ver_name_pattern='TIME', ds_ver_name_pattern=ver_name_pattern_ori)
	get_specifed_result(task=task, ver_specify='08220022-scratchlow-all-mlp-800-0.0005-8-1-spat-sim_0821R8', pretrain_ver_name_pattern='TIME', ds_ver_name_pattern=ver_name_pattern_ori)
	get_specifed_result(task=task, ver_specify='08220022-scratchlow-all-mlp-800-0.0001-8-1-spat-sim_0821R8', pretrain_ver_name_pattern='TIME', ds_ver_name_pattern=ver_name_pattern_ori)
	get_specifed_result(task=task, ver_specify='08220022-scratchlow-all-mlp-800-5e-05-8-1-spat-sim_0821R8', pretrain_ver_name_pattern='TIME', ds_ver_name_pattern=ver_name_pattern_ori)

