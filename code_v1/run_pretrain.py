"""	Run training and test processes for self-supervised learning of spatial acoustic representation
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

from opt import opt_pretrain
opts = opt_pretrain()
args = opts.parse()
dirs = opts.dir()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

import torch
torch.backends.cuda.matmul.allow_tf32 = True  # The flag below controls whether to allow TF32 on matmul. This flag defaults to False in PyTorch 1.12 and later.
torch.backends.cudnn.allow_tf32 = True  # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
# torch.set_float32_matmul_precision('medium')
# torch.set_num_threads(cpu_num)

import json
import scipy.io
import soundfile
import numpy as np
from pathlib import Path
from dataset import Parameter
from tensorboardX import SummaryWriter
import dataset as at_dataset
import learner as at_learner
import model as at_model
from dataset import ArraySetup
from data_generation_RealSenSig import RandomRealDataset
from common.utils import set_seed, set_random_seed, set_learning_rate, create_learning_rate_schedule, get_nparams, get_FLOPs, save_config_to_file,vis_time_fre_data

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

set_seed(args.seed)

# Save config file
os.makedirs(dirs['log_pretrain'], exist_ok=True)
file_path = os.path.join(dirs['log_pretrain'],"config.json")
save_config_to_file([args.__dict__, dirs], file_path)

# Acoustic setting parameters
noise_enable = args.noise_setting['noise_enable']
snr_range = args.noise_setting['snr_range']
noise_type_sim = args.noise_setting['noise_type']
nmic = args.array_setting['nmic']
speed = args.acoustic_setting['sound_speed']	
fs = args.acoustic_setting['fs']
mic_dist_range = [0.03, 0.20]

T = 4.112  # Trajectory length (s) 2.064
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
if args.pretrain | args.test:
	net = at_model.SARSSL(sig_shape=(nf, nt, 2, 2), pretrain=True, device=device)
if args.pretrain_frozen_encoder:
	net = at_model.SARSSL(sig_shape=(nf, nt, 2, 2), pretrain=False, device=device, pretrain_frozen_encoder=True)

layer_keys = ['spec_encoder', 'spat_encoder', 'decoder', 'mlp_head','spec_encoder.patch_embed','spec_encoder.embed','spat_encoder.patch_embed','spat_encoder.embed']
nparam, nparam_sum = get_nparams(net, param_key_list=layer_keys)
print('# Parameters (M):', round(nparam_sum, 2), [key+': '+str(round(nparam[key], 2)) for key in nparam.keys()])
nreim = 2
flops_forward_eval, _ = get_FLOPs(net, input_shape=(1, nmic, nf, nt, nreim), duration=T)
print(f"FLOPs_forward: {flops_forward_eval:.2f}G/s")

# Pre-Train
if (args.pretrain):
	
	print('Pre-Training stage!')
	num_stop_th = 1
	nepoch = args.nepoch

	# Dataset
	return_data_pretrain = ['sig', '']
	data_num = {'train': 5120*100, 'val':4000, 'test':4000}
	if args.rir_gen == 'offline':
		# Source signal
		sourceDataset_pretrain = at_dataset.LibriSpeechDataset(
			path = dirs['sousig_pretrain'], 
			T = T, 
			fs = fs, 
			num_source = max(args.sources), 
			return_vad = True, 
			clean_silence = False)

		# Noise signal
		noise_type = ['spatial_white']
		noiseDataset_pretrain = at_dataset.NoiseDataset(
			T = T, 
			fs = fs, 
			nmic = nmic, 
			noise_type = Parameter(noise_type, discrete=True), 
			noise_path = dirs['noisig_pretrain'], 
			c = speed)
			
		# RIR signal
		rirDataset_pretrain = at_dataset.RIRDataset( # unused, need to check
			fs = fs,
			data_dir = dirs['rir_pretrain'])
		dataset_pretrain = at_dataset.RandomMicSigDataset_FromRIR(
			sourceDataset = sourceDataset_pretrain,
			noiseDataset = noiseDataset_pretrain,
			SNR = Parameter(snr_range[0], snr_range[1]), 	
			rirDataset = rirDataset_pretrain,
			dataset_sz = data_num['train'], 
			transforms = [selecting, segmenting])
	elif args.rir_gen == 'wo':
		remove_spkoverlap = True
		prob_mode = ['duration', 'micpair']
		if opts.pretrain_sim:
			data_dir_list = [element for element in dirs['sensig_pretrain'] if 'simulate' in element]
			dataset_pretrain = at_dataset.FixMicSigDataset( 
				data_dir_list = data_dir_list,
				dataset_sz_ratio_list = None,
				dataset_sz = data_num['train'],
				transforms = [selecting, segmenting],
				return_data = return_data_pretrain)
		else:
			# ds_list = ['locata','mcwsj','libricss','chime3','ami','aishell4','m2met', 'realRIR']
			# ds_probs = [1, 5, 5, 5, 8, 8, 8, 40]
			# ds_list_train = ['locata','mcwsj','libricss','chime3','ami','aishell4','m2met','realman','realRIR']
			# ds_probs_train = [1, 5, 5, 0, 8, 8, 8, 8, 40]
			# ds_probs[-1] = np.sum(ds_probs[:-1])

			# ds_list_train =	['dcase', 'mir', 'mesh', 'ace', 'dechorate', 'butreverb'] # RealRIR
			# ds_probs_train = [5, 5, 5, 5, 5, 5]

			ds_list_train = ['locata', 'mcwsj', 'libricss', 'ami', 'aishell4', 'm2met', 'realman', # RealSig
							'dcase', 'mir', 'mesh', 'ace', 'dechorate', 'butreverb'] # RealRIR
			ds_probs_train = [1, 5, 5, 8, 8, 8, 15,
							  5, 5, 5, 5, 5, 5] 

			dataset_pretrain = RandomRealDataset(
					data_dirs = dirs, 
					T = T, 
					fs = fs, 
					mic_dist_range = mic_dist_range, 
					nmic_selected = nmic, 
					dataset = 'train',
					dataset_sz = data_num['train'], 
					transforms = [selecting, segmenting], 
					prob_mode = prob_mode,
					remove_spkoverlap = remove_spkoverlap,
					ds_list = ds_list_train,
					ds_probs = ds_probs_train,
					sound_speed = speed) 
	
	data_dir_list = [element for element in dirs['sensig_preval'] if 'simulate' in element]
	dataset_preval_sim = at_dataset.FixMicSigDataset( 
			data_dir_list = data_dir_list,
			dataset_sz = data_num['val'],
			transforms = [selecting, segmenting],
			return_data = return_data_pretrain)

	ds_list_val = ['aishell4','m2met','realman',  # RealSig
					'dcase', 'butreverb'] # RealRIR
	ds_probs_val = [1, 2, 5, 
					2, 1]
	dataset_preval_real = RandomRealDataset(
			data_dirs = dirs, 
			T = T, 
			fs = fs, 
			mic_dist_range = mic_dist_range, 
			nmic_selected = nmic, 
			dataset = 'val',
			dataset_sz = data_num['val'], 
			transforms = [selecting, segmenting], 
			prob_mode = prob_mode,
			remove_spkoverlap = remove_spkoverlap,
			ds_list = ds_list_val,
			ds_probs = ds_probs_val,
			sound_speed = speed)
	
	# dataset_pretest = at_dataset.FixMicSigDataset( 
	# 	data_dir_list = dirs['sensig_pretest'],
	# 	dataset_sz = data_num['test'],
	# 	transforms = [selecting, segmenting],
	# 	return_data = return_data_pretrain)

	dataset_pretest_locata = RandomRealDataset(
			data_dirs = dirs, 
			T = T, 
			fs = fs, 
			mic_dist_range = mic_dist_range, 
			nmic_selected = nmic, 
			dataset = 'test',
			dataset_sz = data_num['test'], 
			transforms = [selecting, segmenting], 
			prob_mode = prob_mode,
			remove_spkoverlap = remove_spkoverlap,
			ds_list = ['locata'],
			ds_probs = [1],
			sound_speed = speed)
	
	dataset_pretest_ace = RandomRealDataset(
			data_dirs = dirs, 
			T = T, 
			fs = fs, 
			mic_dist_range = mic_dist_range, 
			nmic_selected = nmic, 
			dataset = 'test',
			dataset_sz = data_num['test'], 
			transforms = [selecting, segmenting], 
			prob_mode = prob_mode,
			remove_spkoverlap = remove_spkoverlap,
			ds_list = ['ace'],
			ds_probs = [1],
			sound_speed = speed)

	kwargs = {'num_workers': args.workers, 'pin_memory': True}  if use_cuda else {}
	dataloader_pretrain = torch.utils.data.DataLoader(dataset=dataset_pretrain, batch_size=args.bs[0], shuffle=True, **kwargs)
	dataloader_preval_sim = torch.utils.data.DataLoader(dataset=dataset_preval_sim, batch_size=args.bs[1], shuffle=False, **kwargs)
	dataloader_preval_real = torch.utils.data.DataLoader(dataset=dataset_preval_real, batch_size=args.bs[1], shuffle=False, **kwargs)
	dataloader_pretest_locata = torch.utils.data.DataLoader(dataset=dataset_pretest_locata, batch_size=args.bs[2], shuffle=False, **kwargs)
	dataloader_pretest_ace = torch.utils.data.DataLoader(dataset=dataset_pretest_ace, batch_size=args.bs[2], shuffle=False, **kwargs)

	# Learner

	learner = at_learner.STFTLearner(net, win_len=win_len, win_shift_ratio=win_shift_ratio, nfft=nfft, fre_used_ratio=fre_used_ratio, fs=fs, task=None, ch_mode='M')
	if use_cuda:
		if len(args.gpu_id)>1:
			learner.mul_gpu()
		learner.cuda()
	else:
		learner.cpu()
	if args.use_amp:
		learner.amp()
	if args.checkpoint_start:
		learner.resume_checkpoint(checkpoints_dir=dirs['log_pretrain'], from_latest=True, as_all_state=True) # Train from latest checkpoints
	if args.checkpoint_from_best_epoch:
		# list(Path(dirs['log_pretrain']).glob(f"model*.tar"))
		learner.resume_checkpoint(checkpoints_dir=dirs['log_pretrain'], from_latest=False, as_all_state=True) # Train from best checkpoints
		print(learner.start_epoch)
		max_epoch = 10
		for epoch in range(learner.start_epoch, learner.start_epoch+max_epoch):
			old_name = dirs['log_pretrain'] + '/model' + str(epoch) + '.tar' 
			new_name = dirs['log_pretrain'] + '/model' + str(epoch) + '_.tar' 
			if os.path.exists(old_name):
				print(epoch)
				os.rename(old_name, new_name)

	# Learning rate
	lr_schedule = create_learning_rate_schedule(total_steps=args.nepoch, base = args.lr, decay_type='cosine', warmup_steps=1, linear_end=1e-6)

	# Tensorboard
	train_writer = SummaryWriter(dirs['log_pretrain'] + '/train/', 'train')
	val_real_writer = SummaryWriter(dirs['log_pretrain'] + '/val_real/', 'val')
	val_sim_writer = SummaryWriter(dirs['log_pretrain'] + '/val_sim/', 'val')
	test_locata_writer = SummaryWriter(dirs['log_pretrain'] + '/test_locata/', 'test')
	test_ace_writer = SummaryWriter(dirs['log_pretrain'] + '/test_ace/', 'test')

	# Network training
	for epoch in range(learner.start_epoch, nepoch+1, 1):

		print('\nEpoch {}/{}:'.format(epoch, nepoch))
		
		# lr = args.lr
		# lr = set_learning_rate(epoch=epoch, lr_init=args.lr, step=100, gamma=0.6)
		if opts.pretrain_sim:
			lr = lr_schedule(epoch)	
		else:
			lr = 0.0001

		if (args.rir_gen == 'offline'):
			set_random_seed(epoch)
			loss_train, diff_train, data_vis_train, metrics_train = learner.pretrain_epoch(dataloader_pretrain, lr=lr, epoch=epoch, return_diff=True)

			set_random_seed(args.seed+10001)
			loss_val, diff_val, data_vis_val = learner.pretest_epoch(dataloader_preval_sim, return_diff=True)

			set_random_seed(args.seed+10001)
			loss_val, diff_val, data_vis_val = learner.pretest_epoch(dataloader_preval_real, return_diff=True)

			# set_random_seed(args.seed+10000)
			# loss_test, diff_test, data_vis_test = learner.pretest_epoch(dataloader_pretest, return_diff=True)

		elif (args.rir_gen == 'wo'):
			set_random_seed(epoch+0)
			loss_train, diff_train, data_vis_train = learner.pretrain_epoch(dataloader_pretrain, lr=lr, epoch=epoch, return_diff=True)
			set_random_seed(args.seed+10001) # can be fixed when there is one card, where therea are multiple cards, inputs are fix but network are not
			loss_val_sim, diff_val_sim, data_vis_val_sim = learner.pretest_epoch(dataloader_preval_sim, return_diff=True)
			set_random_seed(args.seed+10001) 
			loss_val_real, diff_val_real, data_vis_val_real = learner.pretest_epoch(dataloader_preval_real, return_diff=True)
			set_random_seed(args.seed+10000)
			loss_test_locata, diff_test_locata, data_vis_test_locata = learner.pretest_epoch(dataloader_pretest_locata, return_diff=True)
			set_random_seed(args.seed+10000)
			loss_test_ace, diff_test_ace, data_vis_test_ace = learner.pretest_epoch(dataloader_pretest_ace, return_diff=True)

		print('Val loss sim: {:.4f}'.format(loss_val_sim) )
		print('Val loss real: {:.4f}'.format(loss_val_real) )
		print('Test loss locata: {:.4f}'.format(loss_test_locata) )
		print('Test loss ace: {:.4f}'.format(loss_test_ace) )

		# Save model
		# is_best_epoch = learner.is_best_epoch(current_score=loss_val_real*(-1))
		early_stop_patience = 100
		stop_flag, is_best_epoch = learner.early_stopping(current_score=loss_val_real*(-1), patience=early_stop_patience)
		learner.save_checkpoint(epoch=epoch, checkpoints_dir=dirs['log_pretrain'], is_best_epoch = is_best_epoch, save_extra_hist=True)

		# Monitor parameters with tensorboard
		train_writer.add_scalar('lr', lr, epoch)

		train_writer.add_scalar('loss', loss_train, epoch)
		val_real_writer.add_scalar('loss', loss_val_real, epoch)
		val_sim_writer.add_scalar('loss', loss_val_sim, epoch)
		test_locata_writer.add_scalar('loss', loss_test_locata, epoch)
		test_ace_writer.add_scalar('loss', loss_test_ace, epoch)
		
		train_writer.add_scalar('diff', diff_train, epoch) 
		val_real_writer.add_scalar('diff', diff_val_real, epoch)
		val_sim_writer.add_scalar('diff', diff_val_sim, epoch)
		test_locata_writer.add_scalar('diff', diff_test_locata, epoch)
		test_ace_writer.add_scalar('diff', diff_test_ace, epoch)

		if epoch == 1:
			train_writer.add_scalar('nparam', nparam_sum, epoch)

		# Save 
		nepoch_save_data = [5, 10, 15, 20, 25, 30, 35, 40]
		if epoch in nepoch_save_data:
			data_path = dirs['log_pretrain'] + '/result/'
			exist_flag = os.path.exists(data_path)
			if exist_flag==False:
				os.makedirs(data_path)
			vis_train = vis_time_fre_data(data_vis_train, ins_idx=1)
			vis_train.savefig(data_path + str(epoch) + '_train')
			# vis_test = vis_time_fre_data(data_vis_test, ins_idx=1)
			# vis_test.savefig(data_path + str(epoch) + '_test')

			# soundfile.write(data_path, mic_sig, fs)
			# scipy.io.savemat(data_path + '_' + str(epoch) + '.mat', data_vis_train)
		
		if stop_flag:
			break

	print('\nPre-Training finished\n')


if (args.pretrain_frozen_encoder):
	print('Frozen encoders and continue pre-training!')
	set_random_seed(args.seed)
	num_stop_th = 1
	nepoch = args.nepoch

	# Dataset
	return_data_pretrain = ['sig', '']
	data_num = {'train': 5120*100, 'val':4000, 'test':4000} 
	data_dir_list = [element for element in dirs['sensig_pretrain'] if 'simulate' in element]
	dataset_pretrain = at_dataset.FixMicSigDataset( 
			data_dir_list = data_dir_list,
			dataset_sz_ratio_list = dirs['sensig_pretrain_ratio'],
			dataset_sz = data_num['train'],
			transforms = [selecting, segmenting],
			return_data = return_data_pretrain
		)
	
	data_dir_list = [element for element in dirs['sensig_preval'] if 'simulate' in element]
	dataset_preval_sim = at_dataset.FixMicSigDataset( 
		data_dir_list = data_dir_list,
		dataset_sz = data_num['val'],
		transforms = [selecting, segmenting],
		return_data = return_data_pretrain
	)
	ds_list_val = ['aishell4','m2met','realman',  # RealSig
							'dcase', 'butreverb'] # RealRIR
	ds_probs_val = [1, 2, 5, 
					2, 1]
	dataset_preval_real = RandomRealDataset(
			data_dirs = dirs, 
			T = T, 
			fs = fs, 
			mic_dist_range = mic_dist_range, 
			nmic_selected = nmic, 
			dataset = 'val',
			dataset_sz = data_num['val'], 
			transforms = [selecting, segmenting], 
			prob_mode = prob_mode,
			remove_spkoverlap = remove_spkoverlap,
			ds_list = ds_list_val,
			ds_probs = ds_probs_val,
			sound_speed = speed)
	
	data_dir_list = [element for element in dirs['sensig_pretest'] if 'simulate' in element]
	dataset_pretest = at_dataset.FixMicSigDataset( 
		data_dir_list = data_dir_list,
		dataset_sz = data_num['test'],
		transforms = [selecting, segmenting],
		return_data = return_data_pretrain
	)
	kwargs = {'num_workers': args.workers, 'pin_memory': True}  if use_cuda else {}
	dataloader_pretrain = torch.utils.data.DataLoader(dataset=dataset_pretrain, batch_size=args.bs[0], shuffle=True, **kwargs)
	dataloader_preval_sim = torch.utils.data.DataLoader(dataset=dataset_preval_sim, batch_size=args.bs[1], shuffle=False, **kwargs)
	dataloader_preval_real = torch.utils.data.DataLoader(dataset=dataset_preval_real, batch_size=args.bs[1], shuffle=False, **kwargs)
	dataloader_pretest = torch.utils.data.DataLoader(dataset=dataset_pretest, batch_size=args.bs[2], shuffle=False, **kwargs)

	# Learner
	learner = at_learner.STFTLearner(net, win_len=win_len, win_shift_ratio=win_shift_ratio, nfft=nfft, fre_used_ratio=fre_used_ratio, fs=fs, task=None, ch_mode='M')
	if use_cuda:
		if len(args.gpu_id)>1:
			learner.mul_gpu()
		learner.cuda()
	else:
		learner.cpu()
	if args.use_amp:
		learner.amp()
	if args.checkpoint_start:
		learner.resume_checkpoint(checkpoints_dir=dirs['log_pretrain_frozen_encoder'], from_latest=True, as_all_state=True) # Train from latest checkpoints
	learner.load_checkpoint_best(checkpoints_dir=dirs['log_pretrain'], as_all_state=False, param_frozen=False) # Train from pretrained frozen encoder 

	# Learning rate
	lr_schedule = create_learning_rate_schedule(total_steps=args.nepoch, base = args.lr, decay_type='cosine', warmup_steps=1, linear_end=1e-6)

	# Fix partial parameters
	# fix_key = 'spec_encoder'
	fix_key = 'encoder'
	cnt = 0
	for key, value in learner.model.named_parameters():
		if fix_key in key:
			cnt = cnt+1
			value.requires_grad = False
	print('# matched keys:', cnt)

	# TensorboardX
	train_writer = SummaryWriter(dirs['log_pretrain_frozen_encoder'] + '/train/', 'train')
	val_real_writer = SummaryWriter(dirs['log_pretrain_frozen_encoder'] + '/val_real/', 'val')
	val_sim_writer = SummaryWriter(dirs['log_pretrain_frozen_encoder'] + '/val_sim/', 'val')
	test_writer = SummaryWriter(dirs['log_pretrain_frozen_encoder'] + '/test/', 'test')

	# Network training
	for epoch in range(learner.start_epoch, nepoch+1, 1):

		print('\nEpoch {}/{}:'.format(epoch, nepoch))
		
		lr = lr_schedule(epoch)
		set_random_seed(epoch)
		loss_train, diff_train, data_vis_train = learner.pretrain_epoch(dataloader_pretrain, lr=lr, epoch=epoch, return_diff=True)
		set_random_seed(args.seed+10001)
		loss_val_sim, diff_val_sim, data_vis_val_sim = learner.pretest_epoch(dataloader_preval_sim, return_diff=True)
		set_random_seed(args.seed+10001)
		loss_val_real, diff_val_real, data_vis_val_real = learner.pretest_epoch(dataloader_preval_real, return_diff=True)
		set_random_seed(args.seed+10000)
		loss_test, diff_test, data_vis_test = learner.pretest_epoch(dataloader_pretest, return_diff=True)

		print('Val loss sim: {:.4f}'.format(loss_val_sim) )
		print('Val loss real: {:.4f}'.format(loss_val_real) )
		print('Test loss: {:.4f}'.format(loss_test) )

		# Save model
		is_best_epoch = learner.is_best_epoch(current_score=loss_val_real*(-1))
		learner.save_checkpoint(epoch=epoch, checkpoints_dir=dirs['log_pretrain_frozen_encoder'], is_best_epoch = is_best_epoch, save_extra_hist=False)

		# Visualize parameters with tensorboardX
		train_writer.add_scalar('lr', lr, epoch)

		train_writer.add_scalar('loss', loss_train, epoch)
		val_real_writer.add_scalar('loss', loss_val_real, epoch)
		val_sim_writer.add_scalar('loss', loss_val_sim, epoch)
		test_writer.add_scalar('loss', loss_test, epoch)
		
		train_writer.add_scalar('diff', diff_train, epoch) 
		val_real_writer.add_scalar('diff', diff_val_real, epoch)
		val_sim_writer.add_scalar('diff', diff_val_sim, epoch)
		test_writer.add_scalar('diff', diff_test, epoch)
		if epoch == 1:
			test_writer.add_scalar('nparam', nparam_sum, epoch)

	print('\nFrozen Pre-Training finished\n')


# Test
if (args.test):
	#########################
	# test_mode = 'all'
	test_mode = 'ins'
	######################### 
	print('Test stage!')
	print('Test data from ', dirs['sensig_pretest'])

	T = 4.112  # Time duration (s) 
	kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}

	# learner
	learner = at_learner.STFTLearner(net, win_len=win_len, win_shift_ratio=win_shift_ratio, nfft=nfft, fre_used_ratio=fre_used_ratio, fs=fs, task=None, ch_mode='M')
	if use_cuda:
		if len(args.gpu_id)>1:
			learner.mul_gpu()
		learner.cuda()
	else:
		learner.cpu()
	if args.use_amp:
		learner.amp()
	epoch = learner.load_checkpoint_best(checkpoints_dir=dirs['log_pretrain'], as_all_state=True) # best checkpoints
	
	if test_mode == 'all':
		for dir_pretest in dirs['sensig_pretest']:
			set_seed(args.seed)
			
			# Dataset
			return_data =  ['sig', '']	
			dataset_pretest = at_dataset.FixMicSigDataset( 
				data_dir_list = [dir_pretest],
				dataset_sz = 2560*2,
				transforms = [selecting, segmenting],
				return_data = return_data)
			dataloader_pretest = torch.utils.data.DataLoader(dataset=dataset_pretest, batch_size=args.bs[2], shuffle=False, **kwargs)

			# Test
			loss_test, diff_test, data_vis_test = learner.pretest_epoch(dataloader_pretest, return_diff=True)

			name_pretest = dir_pretest.split('/')[-1]
			print(name_pretest, ' Test loss: {:.4f}'.format(loss_test) )

	elif test_mode == 'ins':
		for dir_pretest in dirs['sensig_pretest_ins']:
			set_seed(args.seed)
			
			# Dataset
			return_data =  ['sig', 'dp_mic_signal']	
			dataset_pretest = at_dataset.FixMicSigDataset( 
				data_dir_list = [dir_pretest],
				dataset_sz = None,
				transforms = [selecting, segmenting],
				return_data = return_data)
			dataloader_pretest = torch.utils.data.DataLoader(dataset=dataset_pretest, batch_size=len(dataset_pretest), shuffle=False, **kwargs)

			# Test
			loss_test, diff_test, data_vis_test, results_test = learner.pretest_epoch(dataloader_pretest, return_diff=True, return_eval=True)

			name_pretest = dir_pretest.split('/')[-1]
			print(name_pretest, ' Test loss: {:.4f}'.format(loss_test) )
	
			data_path = dirs['log_pretrain'] + '/test_result/'
			exist_flag = os.path.exists(data_path)
			if exist_flag==False:
				os.makedirs(data_path)
			# vis_train = vis_time_fre_data(data_vis_train, ins_idx=1)
			# vis_train.savefig(data_path + str(epoch) + '_train')
			rt = dir_pretest.split('T')[-1]
			keys = data_vis_test.keys()
			for key in keys:
				print(key)
			for ins_idx in range(len(dataset_pretest)):
				vis_test = vis_time_fre_data(data_vis_test, ins_idx=ins_idx)
				vis_test.savefig(data_path + 'rt'+ rt +'_ins' + str(ins_idx) + '_epoch' + str(epoch) + '_test.png')
				soundfile.write(data_path + 'rt'+ rt + '_ins' + str(ins_idx) + '_epoch' + str(epoch) + '_test_pred.wav', results_test['sig_pred'][ins_idx, :, :].cpu().detach().numpy(), fs)
				soundfile.write(data_path + 'rt'+ rt + '_ins' + str(ins_idx) + '_epoch' + str(epoch) + '_test_tar.wav', results_test['sig_tar'][ins_idx, :, :].cpu().detach().numpy(), fs)

			scipy.io.savemat(data_path + 'rt'+ rt + '_ins' + '_epoch' + str(epoch) + '_test.mat', 
		     				{'mask': data_vis_test['mask'].cpu().detach().numpy(),
	    					'pred': data_vis_test['pred'].cpu().detach().numpy(),
						    'tar': data_vis_test['tar'].cpu().detach().numpy(),
						    'dp_tar': data_vis_test['dp_tar'].cpu().detach().numpy(),
							'pesq': results_test['pesq'].cpu().detach().numpy(),
							'pesq_mask_ch': results_test['pesq_mask_ch'].cpu().detach().numpy()})
		