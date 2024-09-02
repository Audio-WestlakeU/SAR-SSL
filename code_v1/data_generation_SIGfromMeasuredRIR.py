""" 
    Generate both real-world and simulated microphone signals from pre-generated RIRs
		source_state='static', nmic =2
	Usage: 		Need to specify stage, data-id, wnoise (, save-orisrc, data-op )
"""

import os
import argparse
import multiprocessing as mp
import tqdm
from collections import namedtuple
from data_generation_opt import opt
from common.utils import set_seed, save_file, load_file, cross_validation_datadir, one_validation_datadir_simdata
from dataset import Parameter, ArraySetup
import dataset as at_dataset
from torch.utils.data import DataLoader

cpu_num = 16
os.environ["OMP_NUM_THREADS"] = str(cpu_num) 
 
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generating multi-channel audio signals')
	parser.add_argument('--stage', type=str, default='pretrain', metavar='Stage', help='stage that generated data used for (default: Pretrain)') # ['pretrain', 'preval', 'train', 'val', 'test']
	parser.add_argument('--gpu-id', type=str, default='0', metavar='GPU', help='GPU ID (default: 7)')
	parser.add_argument('--data-id', type=int, nargs='+', default=[3], metavar='Datasets', help='dataset IDs (default: 9)')
	parser.add_argument('--wnoise', action='store_true', default=False, help='with noise (default: False)')
	parser.add_argument('--room', type=str, default='all', metavar='RoomNum', help='number of room used for training (default: all)')
	parser.add_argument('--room-trial-id', type=int, default=0, metavar='RoomTrial', help='index of room trial (default: 0)') 
	parser.add_argument('--save-orisrc', action='store_true', default=False, help='save original source signal (default: False)')
	parser.add_argument('--data-op', type=str, default='save', metavar='DataOp', help='operation for generated data (default: Save signal)') 
	parser.add_argument('--workers', type=int, default=32, metavar='Worker', help='number of workers (default: 32)')
	args = parser.parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


	opts = opt(args.wnoise)
	room_setting = opts.room_setting
	micsig_setting = opts.micsig_setting
	dirs = opts.dir()

	rir_list = 	   				['DCASE', 	'MIR', 		'Mesh',	  'ACE', 	'dEchorate',	'BUTReverb','simulate']
	sig_num_list = {'pretrain': [10240*17, 	10240*17, 	10240*17, 	10240*17,	10240*17,	10240*17,	10240*50,	], 
					'preval':	[2560*2, 		2560,		2560, 		2560,		2560,		2560*2,		2560,		], 		 
					'train':	[10240*2,	10240*2, 	10240*2,	10240*2, 	10240*2,	10240*2,	10240*2,		],	 
					'val': 		[2560, 		2560, 		2560,		2560, 		2560,		2560,		8000,		],		
					'test': 	[2560, 		2560, 		2560,		2560, 		2560,		2560,		8000,		],}	
	idx_list = args.data_id

	if args.room == 'all':
		src_num = None
		room_flag = ''
		extra_name = room_flag + ''  
	else: # use_limited_rooms
		trial_idx = args.room_trial_id
		nroom = int(args.room)
		room_range_list = { 'train':	[100+nroom*trial_idx, 100+nroom*trial_idx+int(args.room)],
							'val':		[50, 50+int(args.room)],
							'test':		[0, 0+int(args.room)],}
		nrir_eachroom = 50
		nsrcsig_echaroom = 100
		src_num = int(args.room) * nsrcsig_echaroom
		room_flag = 'R'+args.room
		if args.stage == 'train':
			extra_name = room_flag + '' + 'T' + str(trial_idx)
		else:
			extra_name = room_flag + '' 
			

	noise_type_specify = None
	if noise_type_specify is None:
		noise_flag = ''
	else:
		noise_flag = noise_type_specify
	extra_name += noise_flag + '_selectroom'

	date_flag = opts.time + ''

	print('Room condition (wnoise, SNR, noise_type_in_dataset, roomNum)', args.wnoise, room_setting['snr_range'], room_setting['noise_type'], noise_type_specify, room_flag)

	if (args.data_op == 'save'):
		for list_idx in idx_list:

			# Dataset
			rir = rir_list[list_idx]
			data_num = sig_num_list[args.stage][list_idx]
			print('Dataset:', rir, data_num/1024, 'K')
			if 'simulate' in rir:
				noise_type = room_setting['noise_type']
				load_noise = False			
			else:
				noise_type = ['']
				load_noise = args.wnoise

			if args.stage == 'pretrain':
				set_seed(100+list_idx)
			elif args.stage == 'preval':
				set_seed(200+list_idx)
			elif args.stage == 'train':
				set_seed(300+list_idx)
			elif args.stage == 'val':
				set_seed(400+list_idx)
			elif args.stage == 'test':
				set_seed(500+list_idx+1)
			else:
				raise Exception('Stage unrecognized!')
			
			# Microphone signal
			fs = micsig_setting['fs']
			T = micsig_setting['T']

			# Array
			nmic = 2

			# Noise signal
			noiseDataset = at_dataset.NoiseDataset(
				T = T, 
				fs = fs, 
				nmic = nmic, 
				noise_type = Parameter(noise_type, discrete=True), 
				noise_path = dirs['noisig_'+args.stage], 
				c = room_setting['sound_speed'])
			
			# Source signal
			# sourceDataset = at_dataset.LibriSpeechDataset(
			# 	path = dirs['sousig_'+args.stage], 
			# 	T = T, 
			# 	fs = fs, 
			# 	num_source = max(args.sources), 
			# 	return_vad = True, 
			# 	clean_silence = False)
			sourceDataset = at_dataset.WSJ0Dataset(
				path = dirs['sousig_'+args.stage],
				T = T,
				fs = fs,
				size = src_num)

			# RIR
			rir_dirs = dirs['rir']
			rir_dir = None
			for rdir in rir_dirs:
				if (rir in rdir) & ('noise' not in rdir):
					if 'simulate' in rir:
						rir_dir = rdir + date_flag 
					else:
						rir_dir = rdir + ''
					break
			rir_dir_list = [rir_dir]
			if args.room != 'all':
				if 'simulate' in rir:
					rir_dir_set = [one_validation_datadir_simdata(rir_dir, train_room_idx=room_range_list['train'], val_room_idx=room_range_list['val'], test_room_idx=room_range_list['test'])]
				else: # unused
					rir_dir_set = cross_validation_datadir(rir_dir)
				cv_idx = 0
				rir_dir_list = rir_dir_set[cv_idx][args.stage]

			##############################################
			if len(idx_list)==1 & ('DCASE' in rir_dir_list[0]):
				if 'pretrain' == args.stage:
					room_names = ['bomb_shelter', 'gym', 'pb132', 'pc226',
										'sa203', 'sc203', #'se201',
										'tc352'] # train
				elif 'preval' == args.stage:
					room_names = ['tb103', 'se203'] # val

				rir_dir_temp = rir_dir_list[0]
				rir_dir_list = []
				for room_name in room_names:
					rir_dir_list += [rir_dir_temp + '/' + room_name]
			
			if len(idx_list)==1 & ('BUTReverb' in rir_dir_list[0]):
				if 'pretrain' == args.stage:
					room_names = ['Hotel_SkalskyDvur_ConferenceRoom2', 
                          'Hotel_SkalskyDvur_Room112', 
                          'VUT_FIT_L207', 
                          'VUT_FIT_L212', 
                          'VUT_FIT_L227', 
                          'VUT_FIT_Q301', 
                          'VUT_FIT_C236', 
                          'VUT_FIT_D105'] 
				elif 'preval' == args.stage:
					room_names = ['VUT_FIT_E112'] 
					
				rir_dir_temp = rir_dir_list[0]
				rir_dir_list = []
				for room_name in room_names:
					rir_dir_list += [rir_dir_temp + '/' + room_name]
			print(rir_dir_list)
			###############################################
			rirDataset = at_dataset.RIRDataset(
				data_dir_list = rir_dir_list,
				fs = fs,
				load_noise=load_noise, 
				load_noise_duration=T,
				noise_type_specify=noise_type_specify,
				)

			# Data generation
			return_data = ['sig', 'scene']
			dataset = at_dataset.RandomMicSigDataset_FromRIR(
				sourceDataset = sourceDataset,
				noiseDataset = noiseDataset,
				SNR = Parameter(room_setting['snr_range'][0], room_setting['snr_range'][1]), 	
				rirDataset = rirDataset,
				dataset_sz = data_num, 
				transforms = None,
				return_data = return_data,
				)
			dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=args.workers) # collate_fn=at_dataset.pad_collate_fn)

			sensig_dir = dirs['sensig_'+args.stage]
			save_dir = None
			for sdir in sensig_dir:
				if (rir in sdir): 
					if 'simulate' in sdir:
						save_dir = sdir + date_flag + extra_name
					else:
						save_dir = sdir + extra_name
					
					break

			exist_temp = os.path.exists(save_dir)
			if exist_temp==False:
				os.makedirs(save_dir)
				print('make dir: ' + save_dir)
			else:
				print('existed dir: ' + save_dir)
				msg = input('Sure to regenerate signals? (Enter for yes)')
				if msg == '':
					print('Regenerating signals')

			pbar = tqdm.tqdm(range(0,data_num), desc='generating signals')
			# for idx in pbar:
				# mic_signals, acoustic_scene = dataset[idx]  
			for idx, (mic_signals, acoustic_scene) in enumerate(dataloader):
				pbar.update(1)
				if args.save_orisrc == False:
					acoustic_scene.source_signal = []
					acoustic_scene.noise_signal = []
					acoustic_scene.timestamps = []
					acoustic_scene.t = []
					acoustic_scene.trajectory = []
					# acoustic_scene.RIR = []
				save_idx = idx
				sig_path = save_dir + '/' + str(save_idx) + '.wav'
				acous_path = save_dir + '/' + str(save_idx) + '.npz'
				save_file(mic_signals, acoustic_scene, sig_path, acous_path)

	elif (args.data_op == 'read'):

		class AcousticScene:
			def __init__(self):
				pass
		acoustic_scene = AcousticScene()

		sig_path = dirs['sensig_test'][0] + '/' + '3.wav'
		acous_path = dirs['sensig_test'][0] + '/' + '3.npz'
		mic_signal, acoustic_scene = load_file(acoustic_scene, sig_path, acous_path)

