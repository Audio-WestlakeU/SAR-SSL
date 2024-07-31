"""	
	Run training and test processes for self-supervised learning of spatial acoustic representation
	Reference:  Self-Supervised Learning of Spatial Acoustic Representation with Cross-Channel Signal Reconstruction and Multi-Channel Conformer
	Author:     Bing Yang
	History:    2024-07 - Initial version
	Copyright Bing Yang

	Examples:
		python run_pretrain.py --pretrain --simu-exp --gpu-id 0,

		# * denotes the time version of pre-training model  
		# --test-mode all: all or ins
		python run_pretrain.py --test --simu-exp --time * --test-mode all --gpu-id 0, 

		python run_pretrain.py --pretrain --gpu-id 0, 
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

import scipy.io
import soundfile
from tensorboardX import SummaryWriter
import dataset as at_dataset
import learner as at_learner
import model as at_model
from common.utils import set_seed, set_random_seed, create_learning_rate_schedule, get_nparams, get_FLOPs, save_config_to_file,vis_time_fre_data

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

set_seed(args.seed)

# Save config file
if args.pretrain:
	os.makedirs(dirs['log_pretrain'], exist_ok=True)
	file_path = os.path.join(dirs['log_pretrain'],"config.json")
	save_config_to_file([args.__dict__, dirs], file_path)

# Acoustic setting parameters
assert args.source_state == 'static', 'Source state model unrecognized~'
nmic = args.acoustic_setting['nmic']
speed = args.acoustic_setting['sound_speed']	
fs = args.acoustic_setting['fs']
T = args.acoustic_setting['T']
mic_dist_range = args.acoustic_setting['mic_dist_range'] 
seeds = {'train': int(args.seed+4e8), 'val': int(args.seed+1e8), 'test': int(args.seed+1)} 
 
# STFT parameters
win_len = 512
nfft = 512
win_shift_ratio = 0.5
fre_used_ratio = 1
nf = nfft//2
nt = int((T * fs - win_len*(1-win_shift_ratio)) / (win_len*win_shift_ratio))
print(f"T: {T:.3f}, nt: {nt}, nf: {nf}")

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
if args.pretrain:
	
	print('Pre-Training stage!')
	num_stop_th = 1
	nepoch = args.nepoch

	# Dataset
	data_num = {'train': 5120*100, 'val':4000*2, 'test':4000*2}

	remove_spkoverlap = True
	prob_mode = ['duration', 'micpair']
	if args.simu_exp:
		dataset_pretrain = at_dataset.FixMicSigDataset( 
			data_dir = dirs['micsig_simu_pretrain'], 
			load_anno=False, 
			load_dp=False,
			fs = fs,
			dataset_sz=data_num['train'],
			transforms = None,
			)
		dataset_preval = at_dataset.FixMicSigDataset( 
			data_dir = dirs['micsig_simu_preval'], 
			load_anno=False, 
			load_dp=False,
			fs = fs,
			dataset_sz=data_num['val'],
			transforms = None,
			)
	else:
		dataset_list_train = ['LOCATA', 'MCWSJ', 'LibriCSS', 'AMI', 'AISHELL4', 'M2MeT', 'RealMAN', # RealSig
							  'DCASE', 'MIR', 'Mesh', 'ACE', 'dEchorate', 'BUTReverb'] # RealRIR
		dataset_probs_train = [1, 5, 5, 8, 8, 8, 15,
							   5, 5, 5, 5, 5, 5] 

		dataset_pretrain = at_dataset.RandomRealDataset(
			data_dirs = dirs['micsig_real_pretrain'], 
			T = T, 
			fs = fs, 
			mic_dist_range = mic_dist_range, 
			nmic_selected = nmic, 
			stage = 'train',
			seed = seeds['train'],
			dataset_sz = data_num['train'], 
			transforms = None, 
			prob_mode = prob_mode,
			remove_spkoverlap = remove_spkoverlap,
			dataset_list = dataset_list_train,
			dataset_probs = dataset_probs_train,
			sound_speed = speed) 

		dataset_list_val = ['AISHELL4', 'M2MeT', 'RealMAN', # RealSig
							'DCASE', 'BUTReverb'] 			# RealRIR
		dataset_probs_val = [1, 2, 5, 
							 2, 1]
		dataset_preval_real = at_dataset.RandomRealDataset(
			data_dirs = dirs['micsig_real_preval'], 
			T = T, 
			fs = fs, 
			mic_dist_range = mic_dist_range, 
			nmic_selected = nmic, 
			stage = 'val',
			seed = seeds['val'],
			dataset_sz = data_num['val'], 
			transforms = None, 
			prob_mode = prob_mode,
			remove_spkoverlap = remove_spkoverlap,
			dataset_list = dataset_list_val,
			dataset_probs = dataset_probs_val,
			sound_speed = speed)

		dataset_pretest_locata = at_dataset.RandomRealDataset(
			data_dirs = dirs['micsig_real_pretest'], 
			T = T, 
			fs = fs, 
			mic_dist_range = mic_dist_range, 
			nmic_selected = nmic, 
			stage = 'test',
			seed = seeds['test'],
			dataset_sz = data_num['test'], 
			transforms = None, 
			prob_mode = prob_mode,
			remove_spkoverlap = remove_spkoverlap,
			dataset_list = ['LOCATA'],
			dataset_probs = [1],
			sound_speed = speed)
		
		dataset_pretest_ace = at_dataset.RandomRealDataset(
			data_dirs = dirs['micsig_real_pretest'], 
			T = T, 
			fs = fs, 
			mic_dist_range = mic_dist_range, 
			nmic_selected = nmic, 
			stage = 'test',
			# seed = seeds['test'],
			dataset_sz = data_num['test'], 
			transforms = None, 
			prob_mode = prob_mode,
			remove_spkoverlap = remove_spkoverlap,
			dataset_list = ['ACE'],
			dataset_probs = [1],
			sound_speed = speed)

	kwargs = {'num_workers': args.workers, 'pin_memory': True}  if use_cuda else {}

	dataloader_pretrain = torch.utils.data.DataLoader(dataset=dataset_pretrain, batch_size=args.bs[0], shuffle=True, **kwargs)
	if args.simu_exp:
		dataloader_preval_sim = torch.utils.data.DataLoader(dataset=dataset_preval, batch_size=args.bs[1], shuffle=False, **kwargs)
	else:
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
	if args.simu_exp:
		val_sim_writer = SummaryWriter(dirs['log_pretrain'] + '/val_sim/', 'val')
	else:
		val_real_writer = SummaryWriter(dirs['log_pretrain'] + '/val_real/', 'val')
		test_locata_writer = SummaryWriter(dirs['log_pretrain'] + '/test_locata/', 'test')
		test_ace_writer = SummaryWriter(dirs['log_pretrain'] + '/test_ace/', 'test')

	
	# Network training
	for epoch in range(learner.start_epoch, nepoch+1, 1):

		print('\nEpoch {}/{}:'.format(epoch, nepoch))
 
		# lr = set_learning_rate(epoch=epoch, lr_init=args.lr, step=100, gamma=0.6)
		if args.simu_exp:
			lr = lr_schedule(epoch)	
		else:
			lr = 0.0001
			
		set_random_seed(seeds['train']+epoch)
		loss_train, diff_train, data_vis_train = learner.pretrain_epoch(dataloader_pretrain, lr=lr, epoch=epoch, return_diff=True)
		if args.simu_exp:
			set_random_seed(seeds['val']) # can be fixed when there is one card, where therea are multiple cards, inputs are fix but network are not
			loss_val_sim, diff_val_sim, data_vis_val_sim = learner.pretest_epoch(dataloader_preval_sim, return_diff=True)
			print('Val loss sim: {:.4f}'.format(loss_val_sim) )
		else:
			set_random_seed(seeds['val']) 
			loss_val_real, diff_val_real, data_vis_val_real = learner.pretest_epoch(dataloader_preval_real, return_diff=True)
			set_random_seed(seeds['test'])
			loss_test_locata, diff_test_locata, data_vis_test_locata = learner.pretest_epoch(dataloader_pretest_locata, return_diff=True)
			set_random_seed(seeds['test'])
			loss_test_ace, diff_test_ace, data_vis_test_ace = learner.pretest_epoch(dataloader_pretest_ace, return_diff=True)

			print('Val loss real: {:.4f}'.format(loss_val_real) )
			print('Test loss locata: {:.4f}'.format(loss_test_locata) )
			print('Test loss ace: {:.4f}'.format(loss_test_ace) )

		# Save model
		# is_best_epoch = learner.is_best_epoch(current_score=loss_val_real*(-1))
		early_stop_patience = 100
		if args.simu_exp:
			current_score = loss_val_sim*(-1)
		else:
			current_score = loss_val_real*(-1)
		stop_flag, is_best_epoch = learner.early_stopping(current_score=current_score, patience=early_stop_patience)
		learner.save_checkpoint(epoch=epoch, checkpoints_dir=dirs['log_pretrain'], is_best_epoch = is_best_epoch, save_extra_hist=True)

		# Monitor parameters with tensorboard
		train_writer.add_scalar('lr', lr, epoch)

		train_writer.add_scalar('loss', loss_train, epoch)
		if args.simu_exp:
			val_sim_writer.add_scalar('loss', loss_val_sim, epoch)
		else:
			val_real_writer.add_scalar('loss', loss_val_real, epoch)
			test_locata_writer.add_scalar('loss', loss_test_locata, epoch)
			test_ace_writer.add_scalar('loss', loss_test_ace, epoch)
		
		train_writer.add_scalar('diff', diff_train, epoch) 
		if args.simu_exp:
			val_sim_writer.add_scalar('diff', diff_val_sim, epoch)
		else:
			val_real_writer.add_scalar('diff', diff_val_real, epoch)
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
		
		if stop_flag:
			break

	print('\nPre-Training finished\n')


if (args.pretrain_frozen_encoder):
	print('Frozen encoders and continue pre-training!')
	assert args.simu_exp == True, 'Frozen encoder mode only for simulated data'
	set_random_seed(args.seed)
	num_stop_th = 1
	nepoch = args.nepoch

	# Dataset
	data_num = {'train': 5120*100, 'val':4000, 'test':4000} 
	dataset_pretrain = at_dataset.FixMicSigDataset( 
		data_dir = dirs['micsig_simu_pretrain'], 
		load_anno=False, 
		load_dp=False,
		fs = fs,
		dataset_sz=data_num['train'],
		transforms = None,
		)
	dataset_preval = at_dataset.FixMicSigDataset( 
		data_dir = dirs['micsig_simu_preval'], 
		load_anno=False, 
		load_dp=False,
		fs = fs,
		dataset_sz=data_num['val'],
		transforms = None,
		)

	kwargs = {'num_workers': args.workers, 'pin_memory': True}  if use_cuda else {}
	dataloader_pretrain = torch.utils.data.DataLoader(dataset=dataset_pretrain, batch_size=args.bs[0], shuffle=True, **kwargs)
	dataloader_preval_sim = torch.utils.data.DataLoader(dataset=dataset_preval, batch_size=args.bs[1], shuffle=False, **kwargs)

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
	val_sim_writer = SummaryWriter(dirs['log_pretrain_frozen_encoder'] + '/val_sim/', 'val')

	# Network training
	for epoch in range(learner.start_epoch, nepoch+1, 1):

		print('\nEpoch {}/{}:'.format(epoch, nepoch))
		
		lr = lr_schedule(epoch)
		set_random_seed(epoch)
		loss_train, diff_train, data_vis_train = learner.pretrain_epoch(dataloader_pretrain, lr=lr, epoch=epoch, return_diff=True)
		set_random_seed(args.seed+10001)
		loss_val_sim, diff_val_sim, data_vis_val_sim = learner.pretest_epoch(dataloader_preval_sim, return_diff=True)
		print('Val loss sim: {:.4f}'.format(loss_val_sim) )

		# Save model
		is_best_epoch = learner.is_best_epoch(current_score=loss_val_sim*(-1))
		learner.save_checkpoint(epoch=epoch, checkpoints_dir=dirs['log_pretrain_frozen_encoder'], is_best_epoch = is_best_epoch, save_extra_hist=False)

		# Visualize parameters with tensorboardX
		train_writer.add_scalar('lr', lr, epoch)
		train_writer.add_scalar('loss', loss_train, epoch)
		train_writer.add_scalar('diff', diff_train, epoch)
		if epoch == 1:
			train_writer.add_scalar('nparam', nparam_sum, epoch)

		val_sim_writer.add_scalar('loss', loss_val_sim, epoch)
		val_sim_writer.add_scalar('diff', diff_val_sim, epoch)

	print('\nFrozen Pre-Training finished\n')

# Test
if (args.test):
	print('Test data from ', dirs['micsig_simu_pretest'])
	assert args.simu_exp == True, 'Test mode only for simulated data'
 
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
	
	if args.test_mode == 'all':
		set_seed(args.seed)
		
		# Dataset
		dataset_pretest = at_dataset.FixMicSigDataset( 
			data_dir = dirs['micsig_simu_pretest'], 
			load_anno=False, 
			load_dp=False,
			fs=fs,
			dataset_sz=4000,
			transforms = None,
			)
		dataloader_pretest = torch.utils.data.DataLoader(dataset=dataset_pretest, batch_size=args.bs[2], shuffle=False, **kwargs)

		# Test
		loss_test, diff_test, data_vis_test = learner.pretest_epoch(dataloader_pretest, return_diff=True)

		print('Test loss: {:.4f}'.format(loss_test) )

	elif args.test_mode == 'ins':
		for dir_pretest in dirs['micsig_simu_pretest_ins']:
			set_seed(args.seed)
			
			# Dataset
			dataset_pretest = at_dataset.FixMicSigDataset( 
				data_dir = dirs['micsig_simu_pretest_ins'],
				load_anno=False, 
				load_dp=True,
				dataset_sz=None,
				fs=fs,
				transforms=None)
			dataloader_pretest = torch.utils.data.DataLoader(dataset=dataset_pretest, batch_size=len(dataset_pretest), shuffle=False, **kwargs)

			# Test
			loss_test, diff_test, data_vis_test, results_test = learner.pretest_epoch(dataloader_pretest, return_diff=True, return_eval=True)

			name_pretest = dir_pretest.split('/')[-1]
			print(name_pretest, 'Test loss: {:.4f}'.format(loss_test) )
	
			data_path = dirs['log_pretrain'] + '/test_result/'
			exist_flag = os.path.exists(data_path)
			if exist_flag==False:
				os.makedirs(data_path)
 
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
				# 'dp_tar': data_vis_test['dp_tar'].cpu().detach().numpy(),
				'pesq': results_test['pesq'].cpu().detach().numpy(),
				'pesq_mask_ch': results_test['pesq_mask_ch'].cpu().detach().numpy()})
		