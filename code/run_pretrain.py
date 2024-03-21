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
from dataset import Parameter
from tensorboardX import SummaryWriter
import dataset as at_dataset
import learner as at_learner
import model as at_model
from dataset import ArraySetup
from common.utils import set_seed, set_random_seed, set_learning_rate, create_learning_rate_schedule, get_nparams, vis_time_fre_data

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

set_seed(args.seed)

# Save config file
file_path = dirs['log_pretrain']
os.makedirs(file_path, exist_ok=True)
with open(os.path.join(file_path,"config.json"), "w") as json_file:
	json.dump([args.__dict__, dirs], json_file, indent=4)
	# json.dump({'args':args.__dict__, 'dirs':dirs.__dict__}, json_file, indent=4)

# Acoustic setting parameters
noise_enable = args.noise_setting['noise_enable']
snr_range = args.noise_setting['snr_range']
noise_type_sim = args.noise_setting['noise_type']
nmic = args.array_setting['nmic']
speed = args.acoustic_setting['speed']	
fs = args.acoustic_setting['fs']

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

# Pre-Train
if (args.pretrain):
	
	print('Pre-Training stage!')
	# set_seed(args.seed)
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
		rirDataset_pretrain = at_dataset.RIRDataset( 
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
		dataset_pretrain = at_dataset.FixMicSigDataset( 
			data_dir_list = dirs['sensig_pretrain'],
			dataset_sz_ratio_list = dirs['sensig_pretrain_ratio'],
			dataset_sz = data_num['train'],
			transforms = [selecting, segmenting],
			return_data = return_data_pretrain)
	else:
		raise Exception('rir generation mode unrecognized!')
	dataset_preval = at_dataset.FixMicSigDataset( 
		data_dir_list = dirs['sensig_preval'],
		dataset_sz = data_num['val'],
		transforms = [selecting, segmenting],
		return_data = return_data_pretrain)
	dataset_pretest = at_dataset.FixMicSigDataset( 
		data_dir_list = dirs['sensig_pretest'],
		dataset_sz = data_num['test'],
		transforms = [selecting, segmenting],
		return_data = return_data_pretrain)
	kwargs = {'num_workers': args.workers, 'pin_memory': True}  if use_cuda else {}
	dataloader_pretrain = torch.utils.data.DataLoader(dataset=dataset_pretrain, batch_size=args.bs[0], shuffle=True, **kwargs)
	dataloader_preval = torch.utils.data.DataLoader(dataset=dataset_preval, batch_size=args.bs[1], shuffle=False, **kwargs)
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
		learner.resume_checkpoint(checkpoints_dir=dirs['log_pretrain'], from_latest=True, as_all_state=True) # Train from latest checkpoints
	
	# Learning rate
	lr_schedule = create_learning_rate_schedule(total_steps=args.nepoch, base = args.lr, decay_type='cosine', warmup_steps=1, linear_end=1e-6)

	# Tensorboard
	train_writer = SummaryWriter(dirs['log_pretrain'] + '/train/', 'train')
	val_writer = SummaryWriter(dirs['log_pretrain'] + '/val/', 'val')
	test_writer = SummaryWriter(dirs['log_pretrain'] + '/test/', 'test')

	# Network training
	for epoch in range(learner.start_epoch, nepoch+1, 1):

		print('\nEpoch {}/{}:'.format(epoch, nepoch))
		
		# lr = args.lr
		# lr = set_learning_rate(epoch=epoch, lr_init=args.lr, step=100, gamma=0.6)
		lr = lr_schedule(epoch)

		if (args.rir_gen == 'offline'):
			set_random_seed(epoch)
			loss_train, diff_train, data_vis_train, metrics_train = learner.pretrain_epoch(dataloader_pretrain, lr=lr, epoch=epoch, return_diff=True)

			set_random_seed(args.seed+10001)
			loss_val, diff_val, data_vis_val = learner.pretest_epoch(dataloader_preval, return_diff=True)

			set_random_seed(args.seed+10000)
			loss_test, diff_test, data_vis_test = learner.pretest_epoch(dataloader_pretest, return_diff=True)

		elif (args.rir_gen == 'wo'):
			loss_train, diff_train, data_vis_train = learner.pretrain_epoch(dataloader_pretrain, lr=lr, epoch=epoch, return_diff=True)
			loss_val, diff_val, data_vis_val = learner.pretest_epoch(dataloader_preval, return_diff=True)
			loss_test, diff_test, data_vis_test = learner.pretest_epoch(dataloader_pretest, return_diff=True)

		print('Val loss: {:.4f}'.format(loss_val) )
		print('Test loss: {:.4f}'.format(loss_test) )

		# Save model
		is_best_epoch = learner.is_best_epoch(current_score=loss_val*(-1))
		learner.save_checkpoint(epoch=epoch, checkpoints_dir=dirs['log_pretrain'], is_best_epoch = is_best_epoch, save_extra_hist=False)

		# Monitor parameters with tensorboard
		train_writer.add_scalar('loss', loss_train, epoch)
		val_writer.add_scalar('loss', loss_val, epoch)
		test_writer.add_scalar('loss', loss_test, epoch)
		test_writer.add_scalar('lr', lr, epoch)
		train_writer.add_scalar('diff', diff_train, epoch) 
		val_writer.add_scalar('diff', diff_val, epoch)
		test_writer.add_scalar('diff', diff_test, epoch)

		if epoch == 1:
			test_writer.add_scalar('nparam', nparam_sum, epoch)

		# Save 
		nepoch_save_data = [5, 10, 15, 20, 25, 30, 35, 40]
		if epoch in nepoch_save_data:
			data_path = dirs['log_pretrain'] + '/result/'
			exist_flag = os.path.exists(data_path)
			if exist_flag==False:
				os.makedirs(data_path)
			vis_train = vis_time_fre_data(data_vis_train, ins_idx=1)
			vis_train.savefig(data_path + str(epoch) + '_train')
			vis_test = vis_time_fre_data(data_vis_test, ins_idx=1)
			vis_test.savefig(data_path + str(epoch) + '_test')

			# soundfile.write(data_path, mic_sig, fs)
			# scipy.io.savemat(data_path + '_' + str(epoch) + '.mat', data_vis_train)

	print('\nPre-Training finished\n')


if (args.pretrain_frozen_encoder):
	print('Frozen encoders and continue pre-training!')
	set_random_seed(args.seed)
	num_stop_th = 1
	nepoch = args.nepoch

	# Dataset
	return_data_pretrain = ['sig', '']
	data_num = {'train': 5120*100, 'val':4000, 'test':4000} 

	dataset_pretrain = at_dataset.FixMicSigDataset( 
			data_dir_list = dirs['sensig_pretrain'],
			dataset_sz_ratio_list = dirs['sensig_pretrain_ratio'],
			dataset_sz = data_num['train'],
			transforms = [selecting, segmenting],
			return_data = return_data_pretrain
		)
	dataset_preval = at_dataset.FixMicSigDataset( 
		data_dir_list = dirs['sensig_preval'],
		dataset_sz = data_num['val'],
		transforms = [selecting, segmenting],
		return_data = return_data_pretrain
	)
	dataset_pretest = at_dataset.FixMicSigDataset( 
		data_dir_list = dirs['sensig_pretest'],
		dataset_sz = data_num['test'],
		transforms = [selecting, segmenting],
		return_data = return_data_pretrain
	)
	kwargs = {'num_workers': args.workers, 'pin_memory': True}  if use_cuda else {}
	dataloader_pretrain = torch.utils.data.DataLoader(dataset=dataset_pretrain, batch_size=args.bs[0], shuffle=True, **kwargs)
	dataloader_preval = torch.utils.data.DataLoader(dataset=dataset_preval, batch_size=args.bs[1], shuffle=False, **kwargs)
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
	learner.resume_checkpoint(checkpoints_dir=dirs['log_pretrain'], from_latest=False, as_all_state=False, param_frozen=False) # Train from pretrained frozen encoder 

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
	val_writer = SummaryWriter(dirs['log_pretrain_frozen_encoder'] + '/val/', 'val')
	test_writer = SummaryWriter(dirs['log_pretrain_frozen_encoder'] + '/test/', 'test')

	# Network training
	for epoch in range(learner.start_epoch, nepoch+1, 1):

		print('\nEpoch {}/{}:'.format(epoch, nepoch))
		
		lr = lr_schedule(epoch)

		loss_train, diff_train, data_vis_train = learner.pretrain_epoch(dataloader_pretrain, lr=lr, epoch=epoch, return_diff=True)
		loss_val, diff_val, data_vis_val = learner.pretest_epoch(dataloader_preval, return_diff=True)
		loss_test, diff_test, data_vis_test = learner.pretest_epoch(dataloader_pretest, return_diff=True)

		print('Val loss: {:.4f}'.format(loss_val) )
		print('Test loss: {:.4f}'.format(loss_test) )

		# Save model
		is_best_epoch = learner.is_best_epoch(current_score=loss_val*(-1))
		learner.save_checkpoint(epoch=epoch, checkpoints_dir=dirs['log_pretrain_frozen_encoder'], is_best_epoch = is_best_epoch, save_extra_hist=False)

		# Visualize parameters with tensorboardX
		train_writer.add_scalar('loss', loss_train, epoch)
		val_writer.add_scalar('loss', loss_val, epoch)
		test_writer.add_scalar('loss', loss_test, epoch)
		test_writer.add_scalar('lr', lr, epoch)
		train_writer.add_scalar('diff', diff_train, epoch) 
		val_writer.add_scalar('diff', diff_val, epoch)
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
	epoch = learner.resume_checkpoint(checkpoints_dir=dirs['log_pretrain'], from_latest=False, as_all_state=True) # best checkpoints
	
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
		