"""
Run training and test processes for self-supervised learning of spatial acoustic representation
Reference:  Self-Supervised Learning of Spatial Acoustic Representation with Cross-Channel Signal Reconstruction and Multi-Channel Conformer
Author:     Bing Yang
History:    2024-02 - Initial version
Copyright Bing Yang
"""

import os
cpu_num = 8
os.environ["OMP_NUM_THREADS"] = str(cpu_num) # Limit the threads to reduce cpu overloads, will speed up when there are lots of CPU cores on the running machine
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)

from opt import opt_downstream
opts = opt_downstream()
args = opts.parse()
dirs = opts.dir()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

import torch
torch.backends.cuda.matmul.allow_tf32 = True  # The flag below controls whether to allow TF32 on matmul. This flag defaults to False in PyTorch 1.12 and later.
torch.backends.cudnn.allow_tf32 = True  # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
# torch.set_float32_matmul_precision('medium')
# torch.set_num_threads(cpu_num)

import numpy as np
import copy
import scipy.io
from dataset import Parameter
from tensorboardX import SummaryWriter
import dataset as at_dataset
import learner as at_learner
import model as at_model
from dataset import ArraySetup
from common.utils import set_seed, set_random_seed, get_nparams, get_FLOPs, save_config_to_file, vis_TSNE, cross_validation_datadir, one_validation_datadir_simdata

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

set_seed(args.seed)

# Acoustic setting parameters
noise_enable = args.noise_setting['noise_enable']
snr_range = args.noise_setting['snr_range']
noise_type_sim = args.noise_setting['noise_type']
nmic = args.array_setting['nmic']
speed = args.acoustic_setting['sound_speed']	
fs = args.acoustic_setting['fs']
ds_data = args.ds_specifics['data']
real_sim_ratio = args.ds_specifics['real_sim_ratio']
print(args.ds_specifics)

if (args.ds_task == ['TDOA']):
	T = 1.04
else:
	T = 4.112 
print('duration: ', T , 's')

# STFT (for input) & segmenting (for output) parameters
win_len = 512
nfft = 512
win_shift_ratio = 0.5
fre_used_ratio = 1
nf = nfft//2
nt = int((T * fs - win_len*(1-win_shift_ratio)) / (win_len*win_shift_ratio))
print('nt, nf: ', nt, nf)
if args.source_state == 'static':
	seg_len = int(T*fs) 
	seg_shift = 1
elif args.source_state == 'mobile':
	seg_fra_ratio = 12 	# one estimate per segment (namely seg_fra_ratio frames) 
	seg_len = int(win_len*win_shift_ratio*(seg_fra_ratio+1)) # for win_shift_ratio = 0.5
	seg_shift = int(win_len*win_shift_ratio*seg_fra_ratio) # for win_shift_ratio = 0.5
else:
	print('Source state model unrecognized~')
selecting = at_dataset.Selectting(select_range=[0, int(T*fs)])
segmenting = at_dataset.Segmenting(K=seg_len, step=seg_shift, window=None)

# Network
if args.ds_train | args.ds_test:
	dlabel = 1
	net = at_model.SARSSL(sig_shape=(nf, nt, 2, 2), pretrain=False, device=device, 
		downstream_token=args.ds_token, downstream_head=args.ds_head, downstream_embed = args.ds_embed, downstream_dlabel=dlabel)
layer_keys = ['spec_encoder', 'spat_encoder', 'decoder', 'mlp_head','spec_encoder.patch_embed','spec_encoder.embed','spat_encoder.patch_embed','spat_encoder.embed']
nparam, nparam_sum = get_nparams(net, param_key_list=layer_keys)
print('# Parameters (M):', round(nparam_sum, 2), [key+': '+str(round(nparam[key], 2)) for key in nparam.keys()])
nreim = 2
flops_forward_eval, _ = get_FLOPs(net, input_shape=(1, nmic, nf, nt, nreim), duration=T)
print(f"FLOPs_forward: {flops_forward_eval:.2f}G/s")

# Training processing
if (args.ds_train):

	print('Training stage:', args.ds_trainmode)
	num_stop_th = 1
	val_seed = args.seed + 10000
	test_seed = args.seed + 10001
	
	if 'sim' in ds_data: # simulated data
		noise_type = noise_type_sim
		load_noise = False
		print('Number of simulated rooms: ', args.ds_nsimroom)
		sig_on_the_fly = False
		if sig_on_the_fly:
			train_room_range = [100, args.ds_nsimroom+100]
			val_room_range = [50, 70]
			test_room_range = [0, 20]
	else: # real-world data
		addsim4real = True
		if addsim4real:
			real_sim_ratios = {'train':real_sim_ratio, 'val':[1,0] } 	# real 
			# real_sim_ratios = {'train':[0,1], 'val':[0,1] }   		# only sim 

		addvalmonitor = False
		if addsim4real:
			noise_type = noise_type_sim
		else:
			noise_type = ['']
		load_noise = noise_enable
		sig_on_the_fly = True
		if addsim4real:
			train_room_range = [1000, 2000]
			val_room_range = [0, 100] 

	if (args.ds_trainmode=='finetune') | (args.ds_trainmode=='lineareval') | (args.ds_trainmode=='scratchLOW') | (args.ds_trainmode=='scratchUP'):
		log_dir = 'log_task_' + args.ds_trainmode
	else:
		raise Exception('Downstream train mode unrecognized!')
	
	init_state_dict = net.state_dict()
	for task in args.ds_task:
		set_seed(args.seed)
		task_time_dir = dirs['log_task'].replace('TASK', task)
		nepoch = args.ds_setting[task]['nepoch']
		num = args.ds_setting[task]['num']
		bs_set = args.ds_setting[task]['bs_set']
		lr_set = args.ds_setting[task]['lr_set']
		stages = ['train', 'val', 'test', 'test_large']
		if (args.ds_trainmode=='scratchUP'):
			data_num = {'train':num, 'val':4000, 'test':2000, 'test_large':4000} 
			test_bs = 32
			early_stop_patience = 5
			smooth_alpha = 0.6
			nepoch_ensemble = 5
		else:
			data_num = {'train':num, 'val':1000, 'test':1000, 'test_large':4000}
			if 'real' in ds_data:
				if addvalmonitor:
					stages = ['train', 'val', 'test', 'test_large', 'val_real', 'val_sim']
					data_num = {'train':num, 'val':1000, 'test':1000, 'test_large':4000, 'val_real':1000, 'val_sim':1000}
			test_bs = 16
			early_stop_patience = 10 
			smooth_alpha = 0.6
			nepoch_ensemble = 5
			
		if 'sim' in ds_data:
			if sig_on_the_fly:
				room_dir_set = [one_validation_datadir_simdata(dirs['rir'][0], train_room_idx=train_room_range, val_room_idx=val_room_range, test_room_idx=test_room_range)]
				ntrials = len(room_dir_set)
			else:
				ntrials = args.ds_setting[task]['ntrial']
		else:
			if sig_on_the_fly & (task!='TDOA'):
				room_dir_set = cross_validation_datadir(dirs['rir'][0])
				if addsim4real:
					room_dir_addsim2train = one_validation_datadir_simdata(dirs['rir'][1], train_room_idx=train_room_range, val_room_idx=val_room_range, test_room_idx=[0,0]) # add sim
				ntrials = len(room_dir_set)
				# ntrials = 1 #LLSS  
			else:
				ntrials = 1
		nlrs = len(lr_set)
		nbss = len(bs_set)

		atts = dirs[log_dir].replace('TASK', task).replace('NUM', str(num)).replace(task_time_dir,'').split('-')
		result_name_temporal  = atts[0] + '-' + atts[1] + '-' + atts[2] + '-' + atts[3] +'-' + atts[-2] +'-' + atts[-1] +  '-lr_bs_tri_result_temporal.mat'
		# result_name_temporal  = atts[0] + '-' + atts[1] + '-' + atts[2] + '-' + atts[3] +'-' + atts[-2] +'-' + atts[-1] +  '-lr_bs_tri_result_temporal_lr10e-4.mat' # inff
		if os.path.exists(task_time_dir+'/' + result_name_temporal):
			print(result_name_temporal  + ' exist~')
			msg = input('Sure to continue downstream training? (Enter for yes, delete temporal result for restart)')
			data = scipy.io.loadmat(task_time_dir+'/' + result_name_temporal)
			val_losses = data['val_losses']
			test_losses = data['test_losses']
			val_metrics = data['val_metrics']
			test_metrics = data['test_metrics']
			ensemble_epochs = data['ensemble_epoch']
		else:
			val_losses = np.zeros((nlrs, nbss, ntrials))
			test_losses = np.zeros((nlrs, nbss, ntrials))
			val_metrics = np.zeros((nlrs, nbss, ntrials))
			test_metrics = np.zeros((nlrs, nbss, ntrials))
			ensemble_epochs = np.zeros((nlrs, nbss, ntrials, 2))

		for trial_idx in range(ntrials): 
			for bs_idx in range(nbss):
				for lr_idx in range(nlrs):

					set_seed(args.seed)
					lr_init = lr_set[lr_idx]
					bs = bs_set[bs_idx]
					print(task, ': nepoch=',nepoch, 'num=',num, 'lr=',lr_init, 'bs=',bs, 'trial(or cross_validation_dataset)_idx=',trial_idx, 'ntrial=',ntrials)
					task_dir = dirs[log_dir].replace('TASK', task).replace('NUM', str(num)).replace('LR', str(lr_init)).replace('BAS', str(bs)).replace('TRI', str(trial_idx))
					
					if val_losses[lr_idx, bs_idx, trial_idx]==0:
						# Dataset 
						return_data_ds = ['sig', task]
						datasets = {}
						if (args.ds_trainmode=='scratchUP'):
							for stage in stages:
								datasets[stage] = at_dataset.FixMicSigDataset( 
									data_dir_list = dirs['sensig_pre'+stage.split('_')[0]],
									dataset_sz = data_num[stage],
									transforms = [selecting, segmenting],
									return_data = return_data_ds)
						else:
							if sig_on_the_fly & (('sim' in ds_data) | (('real' in ds_data)&(task!='TDOA'))): # real_world ACE- est tasks
								for stage in stages:
									sourceDataset = at_dataset.WSJ0Dataset(
										path = dirs['sousig_'+stage.split('_')[0]], 
										T = T,
										fs = fs)
									noiseDataset = at_dataset.NoiseDataset(
										T = T, 
										fs = fs, 
										nmic = nmic, 
										noise_type = Parameter(noise_type, discrete=True), 
										noise_path = dirs['noisig_'+stage.split('_')[0]], 
										c = speed)
									if addsim4real:
										eps = 10e-5
										if (stage=='train') | (stage=='val'):
											real_sim_ratio_list = real_sim_ratios[stage]
											real_list = room_dir_set[trial_idx][stage.split('_')[0]]
											sim_list = room_dir_addsim2train[stage.split('_')[0]]
											data_dir_list = real_list + sim_list 
											prob_ratio_list = [real_sim_ratio_list[0]/(len(real_list)+eps)*10000]*len(real_list) + [real_sim_ratio_list[1]/(len(sim_list)+eps)*10000]*len(sim_list)
										elif (stage=='test') | (stage == 'test_large'):
											real_list = room_dir_set[trial_idx][stage.split('_')[0]]
											data_dir_list = room_dir_set[trial_idx][stage.split('_')[0]]
											prob_ratio_list = None 
										if addvalmonitor:
											if (stage=='val_real'):
												real_list = room_dir_set[trial_idx][stage.split('_')[0]]
												data_dir_list = real_list +[]
												prob_ratio_list = None
											elif (stage=='val_sim'):
												sim_list = room_dir_addsim2train[stage.split('_')[0]]
												data_dir_list = sim_list+[]
												prob_ratio_list = None
									else:
										real_list = room_dir_set[trial_idx][stage.split('_')[0]]
										data_dir_list = room_dir_set[trial_idx][stage.split('_')[0]]
										prob_ratio_list = None  

									rirDataset = at_dataset.RIRDataset(
										data_dir_list = data_dir_list,
										fs = fs,
										data_prob_ratio_list=prob_ratio_list,
										load_noise=load_noise,
										noise_type_specify=None)
									datasets[stage] = at_dataset.RandomMicSigDataset_FromRIR(
										sourceDataset = sourceDataset,
										noiseDataset = noiseDataset,
										SNR = Parameter(snr_range[0], snr_range[1]), 	
										rirDataset = rirDataset,
										dataset_sz = data_num[stage], 
										transforms = [selecting, segmenting],
										return_data = return_data_ds)

							else: # not on-the-fly (simulated data-all est tasks, LOCATA-TDOA est task)
								for stage in stages:
									sensig_dir_stage = 'sensig_'+stage.split('_')[0]
									if ('sim' in ds_data): # simulated
										
										if (stage == 'train'):
											dir_list = []
											for dir in dirs[sensig_dir_stage]:
												dir_list += [dir+'T'+ str(trial_idx)]
										else:
											dir_list = dirs[sensig_dir_stage]
										datasets[stage] = at_dataset.FixMicSigDataset( 
											data_dir_list = dir_list,
											dataset_sz = data_num[stage],
											transforms = [selecting, segmenting],
											return_data = return_data_ds)  
									else: # real_world LOCATA-TDOA
										dir_list = []
										if addsim4real:
											if (stage == 'train') | (stage == 'val'):
												dir_list = dirs[sensig_dir_stage]
												real_sim_ratio_list = real_sim_ratios[stage]
											elif (stage == 'test') | (stage == 'test_large'):
												dir_list = [dirs[sensig_dir_stage][0]]
												real_sim_ratio_list = None
											if addvalmonitor:
												if (stage=='val_real'):
													dir_list = [dirs[sensig_dir_stage][0]]
													real_sim_ratio_list = None
												elif (stage == 'val_sim'):
													dir_list = [dirs[sensig_dir_stage][1]]
													real_sim_ratio_list = None
										else:
											dir_list = [dirs['sensig_'+stage.split('_')][0]]
											real_sim_ratio_list = None
										datasets[stage] = at_dataset.FixMicSigDataset( 
											data_dir_list = dir_list,
											dataset_sz = data_num[stage],
											dataset_sz_ratio_list = real_sim_ratio_list,
											transforms = [selecting, segmenting],
											return_data = return_data_ds)  

						kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}
						dataloader_train = torch.utils.data.DataLoader(dataset=datasets['train'], batch_size=bs, shuffle=True, **kwargs)
						dataloader_val = torch.utils.data.DataLoader(dataset=datasets['val'], batch_size=test_bs, shuffle=False, **kwargs)
						dataloader_test = torch.utils.data.DataLoader(dataset=datasets['test'], batch_size=test_bs, shuffle=False, **kwargs)
						dataloader_test_large = torch.utils.data.DataLoader(dataset=datasets['test_large'], batch_size=test_bs, shuffle=False, **kwargs)
						if 'real' in ds_data:
							if addvalmonitor:
								dataloader_val_real = torch.utils.data.DataLoader(dataset=datasets['val_real'], batch_size=test_bs, shuffle=False, **kwargs)
								dataloader_val_sim = torch.utils.data.DataLoader(dataset=datasets['val_sim'], batch_size=test_bs, shuffle=False, **kwargs)
										
						# Learner
						net.load_state_dict(init_state_dict)
						learner = at_learner.STFTLearner(net, win_len=win_len, win_shift_ratio=win_shift_ratio, nfft=nfft, fre_used_ratio=fre_used_ratio, fs=fs, task=task, ch_mode='M')
						if len(args.gpu_id)>1:
							learner.mul_gpu()
						if use_cuda:
							learner.cuda()
						else:
							learner.cpu()
						if args.use_amp:
							learner.amp()
						if args.checkpoint_start:
							learner.resume_checkpoint(checkpoints_dir=task_dir, from_latest=True, as_all_state=True) # Train from latest checkpoints
						else:
							if args.ds_trainmode=='finetune':
								learner.resume_checkpoint(checkpoints_dir=dirs['log_pretrain'], from_latest=False, as_all_state=False, param_frozen=False) # fine-tune pretrained networks , ex_key='_orig_mod.'
							elif args.ds_trainmode=='lineareval':
								learner.resume_checkpoint(checkpoints_dir=dirs['log_pretrain'], from_latest=False, as_all_state=False, param_frozen=True)  # ex_key='_orig_mod.'
							
						# Monitor parameters with tensorboard
						train_writer = SummaryWriter(task_dir + '/train/', 'train')
						val_writer = SummaryWriter(task_dir + '/val/', 'val')
						val_sm_writer = SummaryWriter(task_dir + '/val-smooth/', 'val')
						test_writer = SummaryWriter(task_dir + '/test/', 'test')
						test_sm_writer = SummaryWriter(task_dir + '/test-smooth/', 'test')
						if 'real' in ds_data:
							if addvalmonitor:
								val_real_writer = SummaryWriter(task_dir + '/val-real/', 'val')
								val_sim_writer = SummaryWriter(task_dir + '/val-sim/', 'val')
								val_real_sm_writer = SummaryWriter(task_dir + '/val-real-smooth/', 'val')
								val_sim_sm_writer = SummaryWriter(task_dir + '/val-sim-smooth/', 'val')

						# Model Training
						loss_val_list = []
						loss_val_real_list = []
						loss_val_sim_list = []
						lr = lr_init * 1
						cnt_stop = 0
						for epoch in range(learner.start_epoch, nepoch+1, 1):
							print('\nEpoch {}/{}:'.format(epoch, nepoch))

							set_random_seed(epoch)
							loss_train, metric_train = learner.train_epoch(dataloader_train, lr=lr, epoch=epoch, return_metric=True)
							nmetric = 1

							set_random_seed(val_seed)
							loss_val, metric_val = learner.test_epoch(dataloader_val, return_metric=True)

							if 'real' in ds_data:
								if addvalmonitor:
									set_random_seed(val_seed)
									loss_val_real, metric_val_real = learner.test_epoch(dataloader_val_real, return_metric=True)
									set_random_seed(val_seed)
									loss_val_sim, metric_val_sim = learner.test_epoch(dataloader_val_sim, return_metric=True)

							set_random_seed(test_seed)
							loss_test, metric_test = learner.test_epoch(dataloader_test, return_metric=True)

							print('{} estimation, Val loss: {:.4f}, Val metric: {:.4f}'.format(task, loss_val, metric_val))
							print('{} estimation, Test loss: {:.4f}, Test metric: {:.4f}'.format(task, loss_test, metric_test))
								
							loss_val_list += [loss_val]
							loss_val_list_smooth = learner.smooth_data(data_list=loss_val_list, alpha=smooth_alpha)
							if 'real' in ds_data:
								if addvalmonitor:
									loss_val_real_list += [loss_val_real]
									loss_val_real_list_smooth = learner.smooth_data(data_list=loss_val_real_list, alpha=smooth_alpha)
									loss_val_sim_list += [loss_val_sim]
									loss_val_sim_list_smooth = learner.smooth_data(data_list=loss_val_sim_list, alpha=smooth_alpha)
							
							# Save model
							stop_flag, is_best_epoch = learner.early_stopping(current_score=loss_val_list_smooth[-1]*(-1), patience=early_stop_patience)
							learner.save_checkpoint(epoch=epoch, checkpoints_dir=task_dir, is_best_epoch=is_best_epoch, save_extra_hist=True)
							if is_best_epoch:
								best_epoch = copy.deepcopy(epoch)
							
							# Visualize parameters with tensorboardX
							train_writer.add_scalar('loss', loss_train, epoch)
							val_writer.add_scalar('loss', loss_val, epoch)
							val_sm_writer.add_scalar('loss', loss_val_list_smooth[-1], epoch)
							test_writer.add_scalar('loss', loss_test, epoch)
							if 'real' in ds_data:
								if addvalmonitor:
									val_real_writer.add_scalar('loss', loss_val_real, epoch)
									val_sim_writer.add_scalar('loss', loss_val_sim, epoch)
									val_real_sm_writer.add_scalar('loss', loss_val_real_list_smooth[-1], epoch)
									val_sim_sm_writer.add_scalar('loss', loss_val_sim_list_smooth[-1], epoch)
							if nmetric == 1:
								train_writer.add_scalar('metric', metric_train, epoch)
								val_writer.add_scalar('metric', metric_val, epoch)
								test_writer.add_scalar('metric', metric_test, epoch)
								if 'real' in ds_data:
									if addvalmonitor:
										val_real_writer.add_scalar('metric', metric_val_real, epoch)
										val_sim_writer.add_scalar('metric', metric_val_sim, epoch)
							test_writer.add_scalar('lr', lr, epoch)
							if epoch==1:
								test_writer.add_scalar('nparam', nparam_sum, epoch)
							if stop_flag: ## inff
								cnt_stop += 1
								if cnt_stop <= num_stop_th:
									lr = lr / 10
									print('lr decaing')
									learner.early_stop_counter = 0
								else:
									break 

						print('\nTraining finished\n')
						
						# Ensemble model 
						st_epoch = np.maximum(1, best_epoch-nepoch_ensemble+1)
						ed_epoch = copy.deepcopy(best_epoch)
						epochs = [i for i in range(st_epoch, ed_epoch+1, 1)]
						learner.ensembling(checkpoints_dir=task_dir, epochs=epochs)

						# Model validation
						set_random_seed(test_seed)
						best_loss_test, best_metric_test = learner.test_epoch(dataloader_test_large, return_metric=True)
						set_random_seed(val_seed)
						best_loss_val, best_metric_val = learner.test_epoch(dataloader_val, return_metric=True)
						print('{} estimation, Test loss: {:.4f}, Test metric: {:.4f}'.format(task, best_loss_test, best_metric_test))
						print('{} estimation, Val loss: {:.4f}, Val metric: {:.4f}'.format(task, best_loss_val, best_metric_val))
						val_sm_writer.add_scalar('metric', best_metric_val, st_epoch)
						val_sm_writer.add_scalar('metric', best_metric_val, ed_epoch)
						test_sm_writer.add_scalar('metric', best_metric_test, st_epoch)
						test_sm_writer.add_scalar('metric', best_metric_test, ed_epoch)
						print('\nTest finished\n')	

						# Remove some checkpoints
						if (args.ds_trainmode=='scratchUP'):
							epochs_remove = [i for i in range(1, best_epoch-2, 1)] + [i for i in range(best_epoch+1, epoch+1, 1)] 
						else:
							epochs_remove = [i for i in range(1, st_epoch, 1)] + [i for i in range(best_epoch+1, epoch+1, 1)] 
						learner.remove_checkpoint_epochs(checkpoints_dir=task_dir, epochs=epochs_remove)
						print('\nCheckpoints removed\n')	

						# Save temporal results
						val_losses[lr_idx, bs_idx, trial_idx] = best_loss_val
						val_metrics[lr_idx, bs_idx, trial_idx] = best_metric_val
						test_losses[lr_idx, bs_idx, trial_idx] = best_loss_test
						test_metrics[lr_idx, bs_idx, trial_idx] = best_metric_test
						ensemble_epochs[lr_idx, bs_idx, trial_idx, :] = [st_epoch, ed_epoch]
						scipy.io.savemat(task_time_dir+'/' + result_name_temporal, {
											'val_losses':val_losses, 'val_metrics':val_metrics, 
											'test_losses':test_losses, 'test_metrics':test_metrics, 
											'lr_set':lr_set, 'bs_set':bs_set, 'ntrial': ntrials,
											'ensemble_epoch':ensemble_epochs})
		
		# Find best model among searching learning rates and batch size
		metric = np.mean(val_losses, axis=-1)
		idxes = metric.argmin()
		ncol = metric.shape[1]
		best_lr_idx = idxes//ncol
		best_bs_idx = idxes%ncol
		best_lr = lr_set[best_lr_idx]
		best_bs = bs_set[best_bs_idx]
		best_val_metric = np.mean(val_metrics, axis=-1)[best_lr_idx, best_bs_idx]
		best_test_metric = np.mean(test_metrics, axis=-1)[best_lr_idx, best_bs_idx]
		print('\n{} estimation, BS: {}, LR: {}, best val MAE: {:.4f}, best test MAE: {:.4f}\n'.format(task, best_bs, best_lr, best_val_metric, best_test_metric))

		# Save final results and remove temporal results
		atts = task_dir.replace(task_time_dir,'').split('-')
		result_name = atts[0] + '-' + atts[1] + '-' + atts[2] + '-' + atts[3] +'-' + atts[-2] +'-' + atts[-1] +  '-lr_bs_tri_result.mat' 
		# result_name = atts[0] + '-' + atts[1] + '-' + atts[2] + '-' + atts[3] +'-' + atts[-2] +'-' + atts[-1] +  '-lr_bs_tri_result_lr10e-4.mat'  # inff
		scipy.io.savemat(task_time_dir+'/' + result_name, {
							'val_losses':val_losses, 'val_metrics':val_metrics, 
							'test_losses':test_losses, 'test_metrics':test_metrics, 
							'lr_set':lr_set, 'bs_set':bs_set, 'ntrial': ntrials,
							'best_lr_idx':best_lr_idx, 'best_bs_idx':best_bs_idx,
							'ensemble_epoch':ensemble_epochs})
		os.remove(task_time_dir+'/' + result_name_temporal)


if (args.ds_test):

	print('Downstream test stage!', args.ds_trainmode)
	########################
	# test_mode = 'cal_avg'
	# test_mode = 'cal_metric'
	test_mode = 'vis_embed'
	bs_idx = 0
	lr_idx = 1
	########################

	test_bs = 16
	train_seed = args.seed
	test_seed = args.seed+10001

	if 'sim' in ds_data: # simulated data
		noise_type = noise_type_sim
		load_noise = False
		print('Number of simulated rooms: ', args.ds_nsimroom)
		sig_on_the_fly = False
		if sig_on_the_fly:
			train_room_range = [1000, args.ds_nsimroom+1000]
			val_room_range = [50, 70]
			test_room_range = [0, 20]
	else: # real-world data
		load_noise = noise_type_sim
		test_real = True
		test_real = False
		if not test_real:
			train_room_range = [1000, 2000] #sim
			test_room_range = [0, 20] #sim
			noise_type = noise_type_sim
		else:
			noise_type = ['']
		sig_on_the_fly = True

	if (args.ds_trainmode=='finetune') | (args.ds_trainmode=='lineareval') | (args.ds_trainmode=='scratchLOW') | (args.ds_trainmode=='scratchUP'):
		log_dir = 'log_task_' + args.ds_trainmode
	else:
		raise Exception('Downstream train mode unrecognized!')

	maes_test_data = []
	mins_test_data = []
	maxs_test_data = []
	maes_data = []
	means_data = []
	mins_data = []
	maxs_data = []
	losses_test = []
	metrics_test = []
	for task in args.ds_task:
		set_seed(args.seed)
		task_time_dir = dirs['log_task'].replace('TASK', task)
		nepoch = args.ds_setting[task]['nepoch']
		bs_set = args.ds_setting[task]['bs_set']
		lr_set = args.ds_setting[task]['lr_set']
		num = args.ds_setting[task]['num']
		
		if (test_mode == 'cal_metric') | (test_mode == 'cal_avg'):
			if (args.ds_trainmode=='scratchUP') :
				if 'sim' in ds_data:
					data_num = {'train':num, 'test':4000}
				elif 'real' in ds_data:
					data_num = {'train':1, 'test':4000}
			else:
				data_num = {'train':num, 'test':4000} 
				
		elif (test_mode == 'vis_embed'):
			data_num = {'train':8000, 'test':8000} 

		lr_init = lr_set[lr_idx]
		bs = bs_set[bs_idx]
		tasks = [task]

		for tt in tasks:
			set_seed(args.seed)
			print(tt,  'nepoch=',nepoch, 'num=',num, 'lr=',lr_init, 'bs=', bs)

			if 'sim' in ds_data:
				if sig_on_the_fly:
					room_dir_set = [one_validation_datadir_simdata(dirs['rir'][0], train_room_idx=train_room_range, val_room_idx=val_room_range, test_room_idx=test_room_range)]
					ntrials = len(room_dir_set)
				else:
					ntrials = args.ds_setting[task]['ntrial']
			else:
				if sig_on_the_fly & (tt!='TDOA'): 
					if test_real:
						room_dir_set = cross_validation_datadir(dirs['rir'][0])
					else:
						room_dir_set = [one_validation_datadir_simdata(dirs['rir'][1], train_room_idx=train_room_range, val_room_idx=[0,0], test_room_idx=test_room_range)]
					ntrials = len(room_dir_set)
				else:
					ntrials = 1
			
			if test_mode == 'cal_metric':
				loss_test = np.zeros((ntrials))
				metric_test = np.zeros((ntrials))
			if test_mode == 'cal_avg':
				mae_test_data = np.zeros((ntrials))
				min_test_data = np.zeros((ntrials))
				max_test_data = np.zeros((ntrials))
				mae_data = np.zeros((ntrials))
				mean_data = np.zeros((ntrials))
				min_data = np.zeros((ntrials))
				max_data = np.zeros((ntrials))

			for trial_idx in range(ntrials):
				print('trial(or cross_validation_dataset)_idx=', trial_idx, 'ntrial=', ntrials)
				if (args.ds_trainmode=='scratchUP'): # for model loading
					task_dir = dirs[log_dir].replace('TASK', task).replace('NUM', str(num)).replace('LR', str(lr_init)).replace('BAS', str(bs)).replace('TRI', str(0))
				else:
					if 'valsim' in dirs[log_dir]:
						task_dir = dirs[log_dir].replace('TASK', task).replace('NUM', str(num)).replace('LR', str(lr_init)).replace('BAS', str(bs)).replace('TRI', str(0))
					else:
						task_dir = dirs[log_dir].replace('TASK', task).replace('NUM', str(num)).replace('LR', str(lr_init)).replace('BAS', str(bs)).replace('TRI', str(trial_idx))
					
				# Dataset 
				return_data_ds = ['sig', tt]	
				datasets = {}
				stages = ['train', 'test']
				if (args.ds_trainmode=='scratchUP'):
					if 'sim' in ds_data:
						for stage in stages:
							datasets[stage] = at_dataset.FixMicSigDataset( 
								data_dir_list = dirs['sensig_pre'+stage.split('_')[0]],
								dataset_sz = data_num[stage],
								transforms = [selecting, segmenting],
								return_data = return_data_ds,
								)	
					elif 'real' in ds_data:
						if sig_on_the_fly&(tt!='TDOA'):
							for stage in stages:
								sourceDataset = at_dataset.WSJ0Dataset(
									path = dirs['sousig_'+stage.split('_')[0]], 
									T = T,
									fs = fs)
								noiseDataset = at_dataset.NoiseDataset(
									T = T, 
									fs = fs, 
									nmic = nmic, 
									noise_type = Parameter(noise_type, discrete=True), 
									noise_path = dirs['noisig_'+stage.split('_')[0]], 
									c = speed)
								rirDataset = at_dataset.RIRDataset(
									data_dir_list = room_dir_set[trial_idx][stage.split('_')[0]],
									fs = fs,
									load_noise=load_noise,
									noise_type_specify=None)
								datasets[stage] = at_dataset.RandomMicSigDataset_FromRIR(
									sourceDataset = sourceDataset,
									noiseDataset = noiseDataset,
									SNR = Parameter(snr_range[0], snr_range[1]), 	
									rirDataset = rirDataset,
									dataset_sz = data_num[stage], 
									transforms = [selecting, segmenting],
									return_data = return_data_ds)
						else:
							for stage in stages:
								sensig_dir_stage = 'sensig_'+stage.split('_')[0]
								datasets[stage] = at_dataset.FixMicSigDataset( 
									data_dir_list = dirs[sensig_dir_stage],
									dataset_sz = data_num[stage],
									transforms = [selecting, segmenting],
									return_data = return_data_ds)		
				else:
					if sig_on_the_fly & (('sim' in ds_data) | (('real' in ds_data)&(tt!='TDOA'))):
						for stage in stages:
							sourceDataset = at_dataset.WSJ0Dataset(
								path = dirs['sousig_'+stage.split('_')[0]], 
								T = T,
								fs = fs)
							noiseDataset = at_dataset.NoiseDataset(
								T = T, 
								fs = fs, 
								nmic = nmic, 
								noise_type = Parameter(noise_type, discrete=True), 
								noise_path = dirs['noisig_'+stage.split('_')[0]], 
								c = speed)
							rirDataset = at_dataset.RIRDataset(
								data_dir_list = room_dir_set[trial_idx][stage.split('_')[0]],
								fs = fs,
								load_noise=load_noise,
								noise_type_specify=None)
							datasets[stage] = at_dataset.RandomMicSigDataset_FromRIR(
								sourceDataset = sourceDataset,
								noiseDataset = noiseDataset,
								SNR = Parameter(snr_range[0], snr_range[1]), 	
								rirDataset = rirDataset,
								dataset_sz = data_num[stage], 
								transforms = [selecting, segmenting],
								return_data = return_data_ds)
					else: # not on-the-fly (simulated data-all est tasks, LOCATA-TDOA est task)
						for stage in stages:
							sensig_dir_stage = 'sensig_'+stage.split('_')[0]
							if ('sim' in ds_data):  # simulated
								if  (stage == 'train'):
									dir_list = []
									for dir in dirs[sensig_dir_stage]:
										dir_list += [dir+'T'+ str(trial_idx)]
								else:
									dir_list = dirs[sensig_dir_stage]
								datasets[stage] = at_dataset.FixMicSigDataset( 
									data_dir_list = dir_list,
									dataset_sz = data_num[stage],
									transforms = [selecting, segmenting],
									return_data = return_data_ds,
									)  
							else:  # real_world LOCATA-TDOA
								if test_real:
									dir_list = [dirs[sensig_dir_stage][0]]
								else:
									dir_list = [dirs[sensig_dir_stage][1]]
								datasets[stage] = at_dataset.FixMicSigDataset( 
									data_dir_list = dir_list,
									dataset_sz = data_num[stage],
									transforms = [selecting, segmenting],
									return_data = return_data_ds,
									) 
								
				kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}
				dataloader_train = torch.utils.data.DataLoader(dataset=datasets['train'], batch_size=test_bs, shuffle=False, **kwargs)
				dataloader_test = torch.utils.data.DataLoader(dataset=datasets['test'], batch_size=test_bs, shuffle=False, **kwargs)

				learner = at_learner.STFTLearner(net, win_len=win_len, win_shift_ratio=win_shift_ratio, nfft=nfft, fre_used_ratio=fre_used_ratio, fs=fs, task=tt, ch_mode='M')
				if len(args.gpu_id)>1:
					learner.mul_gpu()
				if use_cuda:
					learner.cuda()
				else:
					learner.cpu()
				if args.use_amp:
					learner.amp()

				if (test_mode == 'cal_metric'):
					if (args.ds_trainmode=='scratchUP'):
						learner.resume_checkpoint(checkpoints_dir=task_dir, from_latest=False, as_all_state=True) # Load best checkpoints 
						set_random_seed(test_seed)
						loss_test[trial_idx],  metric_test[trial_idx] = learner.test_epoch(dataloader_test, return_metric=True, return_vis=False)
					else:
						learner.resume_checkpoint_ensemble(checkpoints_dir=task_dir) 
						set_random_seed(test_seed)
						loss_test[trial_idx], metric_test[trial_idx] = learner.test_epoch(dataloader_test, return_metric=True, return_vis=False)

					print('{} estimation, Test loss: {:.4f}, Test metric: {:.4f}'.format(tt, loss_test[trial_idx], metric_test[trial_idx]))

				if test_mode == 'vis_embed':
					
					learner.resume_checkpoint_ensemble(checkpoints_dir=task_dir)  

					set_random_seed(train_seed)
					_, _, vis_train = learner.test_epoch(dataloader_train, return_metric=True, return_vis=True)

					set_random_seed(test_seed)
					_, _, vis_test = learner.test_epoch(dataloader_test, return_metric=True, return_vis=True)

					if ((args.ds_trainmode=='finetune') | (args.ds_trainmode=='lineareval')  | (args.ds_trainmode=='scratchLOW')) & ('sim' in ds_data):
						
						set_random_seed(train_seed)
						vis_train, data_vis_train = vis_TSNE(data=vis_train['embed'].cpu().detach().numpy(), label=vis_train['label'].cpu().detach().numpy())
						vis_train.savefig(task_dir.replace(task_dir.split('/')[-1], 'test_result/tsne_vis_train_'+task_dir.split('/')[-1]) + '.png')
						scipy.io.savemat(task_dir.replace(task_dir.split('/')[-1], 'test_result/tsne_vis_train_'+task_dir.split('/')[-1]) + '.mat', 
								{'data': data_vis_train['data'], 'label':data_vis_train['label']})
						
						set_random_seed(test_seed)
						vis_test, data_vis_test = vis_TSNE(data=vis_test['embed'].cpu().detach().numpy(), label=vis_test['label'].cpu().detach().numpy())
						vis_test.savefig(task_dir.replace(task_dir.split('/')[-1], 'test_result/tsne_vis_test_'+task_dir.split('/')[-1]) + '.png')
						scipy.io.savemat(task_dir.replace(task_dir.split('/')[-1], 'test_result/tsne_vis_test_'+task_dir.split('/')[-1]) + '.mat', 
								{'data': data_vis_test['data'], 'label':data_vis_test['label']})
						
				if test_mode == 'cal_avg':
					mae_test_data[trial_idx], min_test_data[trial_idx], max_test_data[trial_idx], mae_data[trial_idx], mean_data[trial_idx], min_data[trial_idx], max_data[trial_idx] = learner.mae_wotrain(dataloader_train, dataloader_test)
					print('Trial: {}, Data MAE: {:.4f}'.format(trial_idx, mae_test_data[trial_idx]))

			x = 3
			if test_mode == 'cal_metric':
				losses_test += [round(np.mean(loss_test, axis=-1), x)]
				metrics_test += [round(np.mean(metric_test, axis=-1), x)]
				print('{} estimation, Test loss: {:.4f}, Test metric: {:.4f}'.format(tt, np.mean(loss_test, axis=-1), np.mean(metric_test, axis=-1)))

			if test_mode == 'cal_avg':	
				maes_test_data += [round(np.mean(mae_test_data, axis=-1), x)]
				mins_test_data += [round(np.min(min_test_data, axis=-1), x)]
				maxs_test_data += [round(np.max(max_test_data, axis=-1), x)]
				maes_data += [round(np.mean(mae_data, axis=-1), x)]
				means_data += [round(np.mean(mean_data, axis=-1), x)]
				mins_data += [round(np.min(min_data, axis=-1), x)]
				maxs_data += [round(np.max(max_data, axis=-1), x)]
				print('Data MAE: {:.4f}'.format(np.mean(mae_test_data, axis=-1)))
	
	print('Task: ', args.ds_task)
	if test_mode == 'cal_metric':
		print('Test loss: ', losses_test)
		print('Test metric: ', metrics_test)
	if test_mode == 'cal_avg':	
		print('Test MAE: ', maes_test_data)
		print('Test MIN: ', mins_test_data)
		print('Test MAX: ', maxs_test_data)
		print('Train MAE: ', maes_data)
		print('Train Mean: ', means_data)
		print('Train MIN: ', mins_data)
		print('Train MAX: ', maxs_data)
