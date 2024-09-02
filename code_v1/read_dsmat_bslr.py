""" 
    Read downstream results (MAEs of TDOA, DRR, T60, C50, SNR, ABS estimation) from saved mat files
	Usage:  python read_dsmat_bslr.py --time [*]
"""

import os
import argparse
import numpy as np
from opt import opt_pretrain
import scipy.io
import csv

parser = argparse.ArgumentParser(description='Reading downstream results')
parser.add_argument('--time', type=str, default='0', metavar='Time', help='time flag')
parser.add_argument('--task', type=str, nargs='+', default=['TDOA', 'DRR','T60', 'C50', 'SNR', 'ABS'], metavar='Task', help='downstream task (default: all tasks)')
parser.add_argument('--dataset', type=str, nargs='+', default=['locata', 'ace','dechorate', 'butreverb', 'sim'], metavar='DataSet', help='DataSet (default: all datasets)')
parser.add_argument('--save_data', type=str, nargs='+', default=['best_avgtri'], metavar='SaveResult', help='SaveResult (default: best_avgtri)') # ['ori', 'best', 'best_avgtri']

args = parser.parse_args()

opts = opt_pretrain()
dirs = opts.dir()
ver_name_pattern = 'FT-TOKEN-HEAD-NUM-EMBED-MODELDATA' # according to the saved directory+file
ver_time = args.time

real_csv_data = []
sim_csv_data = []
best_csv_data = []
best_avgtri_csv_data = []
for task in args.task:
	ver_names = []

	ver_dir = dirs['exp'] + '/'+ task + '/' + ver_time + '/'
	if os.path.isdir (ver_dir):
		ver_names = os.listdir(ver_dir)
		for ver_name in ver_names:
			if (ver_name.split('.')[-1] == 'mat') & (('lr_bs_cv' in ver_name) | ('lr_bs_tri' in ver_name)) & ('temporal' not in ver_name):
				atts = ver_name.split('-')
				ft = atts[ver_name_pattern.split('-').index('FT')]
				embed = atts[ver_name_pattern.split('-').index('EMBED')]
				modeldata = atts[ver_name_pattern.split('-').index('MODELDATA')] #.replace('real_','')
				num = atts[ver_name_pattern.split('-').index('NUM')]
				# info = atts[ver_name_pattern.split('-').index('INFO')]
				result = scipy.io.loadmat(ver_dir+ver_name)
				lr_set = result['lr_set']
				bs_set = result['bs_set']
				val_losses = result['val_losses']
				val_metrics = result['val_metrics']
				test_losses = result['test_losses']
				test_metrics = result['test_metrics']
				
				ntrial = val_losses.shape[-1]
				for trial_idx in range(ntrial):
					for bs_idx in range(bs_set.shape[1]):
						for lr_idx in range(lr_set.shape[1]):
								lr = lr_set[0, lr_idx]
								bs = bs_set[0, bs_idx]
								val_loss = val_losses[lr_idx, bs_idx, trial_idx]
								val_metric = val_metrics[lr_idx, bs_idx, trial_idx]
								test_loss = test_losses[lr_idx, bs_idx, trial_idx]
								test_metric = test_metrics[lr_idx, bs_idx, trial_idx]

								csv_data_row = [task, ft, modeldata, embed, trial_idx, bs, lr, val_loss, val_metric, test_loss, test_metric]
								
								if 'real' in modeldata:
									real_csv_data += [csv_data_row]
								elif 'sim' in modeldata:
									sim_csv_data += [csv_data_row]
				
				if ('lr_bs_cv' in ver_name): # can be deleted from 2023.11												
					metric = np.mean(val_losses, axis=-1)
					idxes = metric.argmin()
					ncol = metric.shape[1]
					best_lr_idx = idxes//ncol
					best_bs_idx = idxes%ncol
					best_lr = lr_set[0, best_lr_idx]
					best_bs = bs_set[0, best_bs_idx]
					best_val_metric = np.mean(val_metrics, axis=-1)[best_lr_idx, best_bs_idx]
					best_val_loss = np.mean(val_losses, axis=-1)[best_lr_idx, best_bs_idx]
					best_test_metric = np.mean(test_metrics, axis=-1)[best_lr_idx, best_bs_idx]
					best_test_loss = np.mean(test_losses, axis=-1)[best_lr_idx, best_bs_idx]
					room_num = 'invalid'
					if 'sim' in modeldata:
						att = modeldata.split('R')
						if len(att) == 2:
							room_num = int(att[-1])
							modeldata = modeldata.replace('R'+att[-1],'')
							if len(att[0].split('T'))==1:
								trial_idx = 0 
							else:
								trial_idx = att[0].split('T')[-1]

					best_csv_data_row = [ft, task, modeldata, room_num, embed, trial_idx, best_bs, best_lr, best_val_metric, best_test_metric]
					best_csv_data += [best_csv_data_row]

				if ('lr_bs_tri' in ver_name):
					room_num = 'invalid'
					data_num = 'invalid'
					if 'sim' in modeldata:
						att = modeldata.split('R')
						if len(att) == 2:
							room_num = int(att[-1])
							modeldata = modeldata.replace('R'+att[-1],'')
					if 'real' in modeldata:
						data_num = int(num)
					best_val_metric = np.zeros((ntrial))
					best_val_loss = np.zeros((ntrial))
					best_test_metric = np.zeros((ntrial))
					best_test_loss = np.zeros((ntrial))
					for trial_idx in range(ntrial):
						metric = val_losses[:, :, trial_idx]
						idxes = metric.argmin()
						ncol = metric.shape[1]
						best_lr_idx = idxes//ncol
						best_bs_idx = idxes%ncol
						best_lr = lr_set[0, best_lr_idx]
						best_bs = bs_set[0, best_bs_idx]
						best_val_metric[trial_idx] = val_metrics[best_lr_idx, best_bs_idx, trial_idx]
						best_val_loss[trial_idx] = val_losses[best_lr_idx, best_bs_idx, trial_idx]
						best_test_metric[trial_idx] = test_metrics[best_lr_idx, best_bs_idx, trial_idx]
						best_test_loss[trial_idx] = test_losses[best_lr_idx, best_bs_idx, trial_idx]

						if 'real' in modeldata:
							best_csv_data_row = [ft, task, modeldata, room_num, data_num, embed, trial_idx, best_bs, best_lr, best_val_metric[trial_idx], best_test_metric[trial_idx]]
						elif 'sim' in modeldata:
							best_csv_data_row = [ft, task, modeldata, room_num, embed, trial_idx, best_bs, best_lr, best_val_metric[trial_idx], best_test_metric[trial_idx]]
						best_csv_data += [best_csv_data_row]
					
					if 'real' in modeldata:
						best_avgtri_csv_data_row = [ft, task, modeldata, room_num, data_num, embed, np.mean(best_val_metric), np.mean(best_test_metric)]
					elif 'sim' in modeldata:
						best_avgtri_csv_data_row = [ft, task, modeldata, room_num, embed, np.mean(best_val_metric), np.mean(best_test_metric)]
					best_avgtri_csv_data += [best_avgtri_csv_data_row]

save_dir = dirs['exp'] + '/ds_result/'
exist_temp = os.path.exists(save_dir)
if exist_temp==False:
	os.makedirs(save_dir)
	print('make dir: ' + save_dir)
row_name = ['Task', 'FT', 'ModelData', 'Embed', 'Trial(or Cross-Val)', 'BS', 'LR', 'CrossValidation', 'Val loss', 'Val MAE', 'Test loss', 'Test MAE']

# Save original results for each test dataset may including sim, real
if 'ori' in args.save_data:
	for modeldata in args.dataset:
		csv_name = save_dir + ver_time + '_' + modeldata + '.csv'
		if ('real' in modeldata) & (real_csv_data != []):
			with open(csv_name, 'w', newline='') as file:
				writer = csv.writer(file)
				writer.writerow(row_name)
				for csv_row in real_csv_data:
					writer.writerow(csv_row)
		elif ('sim' in modeldata) & (sim_csv_data != []):
			with open(csv_name, 'w', newline='') as file:
				writer = csv.writer(file)
				writer.writerow(row_name)
				for csv_row in sim_csv_data:
					writer.writerow(csv_row)

# Save best results after searching from candidate learing rates and batch sizes
if 'best' in args.save_data:
	if len(best_csv_data_row)==11: # real
		row_name = ['FT', 'TASK', 'ModelData','RoomNum(sim)','DataNum(/epoch-real)', 'Embed', 'Trial(or Cross-Val)', 'BS', 'LR', 'Val MAE', 'Test MAE']
	elif len(best_csv_data_row)==10: # sim
		row_name = ['FT', 'TASK', 'ModelData','RoomNum(sim)', 'Embed', 'Trial(or Cross-Val)', 'BS', 'LR', 'Val MAE', 'Test MAE']
	csv_name = save_dir + ver_time + '_best.csv'
	with open(csv_name, 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(row_name)
		for csv_row in best_csv_data:
			writer.writerow(csv_row)

# Save best results after searching from candidate learing rates and batch sizes and then averaging over trials
if 'best_avgtri' in args.save_data:
	if len(best_avgtri_csv_data_row)==8: # real
		row_name = ['FT', 'TASK', 'ModelData','RoomNum(sim)','DataNum(/epoch-real)', 'Embed', 'Val MAE', 'Test MAE']
	elif len(best_avgtri_csv_data_row)==7: # sim
		row_name = ['FT', 'TASK', 'ModelData','RoomNum(sim)', 'Embed', 'Val MAE', 'Test MAE']
	csv_name = save_dir + ver_time + '_best_avgtri.csv'
	with open(csv_name, 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(row_name)
		for csv_row in best_avgtri_csv_data:
			writer.writerow(csv_row)

if __name__ == '__main__':
    pass
