""" 
    Generate simulated microphone signals (infinite room conditions)
	Usage: 		Need to specify stage, gpu-id, wnoise(, save-orisrc, sources, source_state, data-op)
"""

import os
import argparse

cpu_num = 8
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)

parser = argparse.ArgumentParser(description='Generating multi-channel audio signals or RIRs')
parser.add_argument('--stage', type=str, default='pretrain', metavar='Stage', help='stage that generated data used for (default: Pretrain)') # ['pretrain', 'preval', 'train', 'val', 'test']
parser.add_argument('--gpu-id', type=str, default='7', metavar='GPU', help='GPU ID (default: 7)')
parser.add_argument('--data-part', type=int, default='0', metavar='PartID', help='Part ID (default: 0)')
parser.add_argument('--wnoise', action='store_true', default=False, help='with noise (default: False)')
parser.add_argument('--ins', action='store_true', default=False, help='specified instances (default: False)')
parser.add_argument('--save-orisrc', action='store_true', default=False, help='save original source signal (default: False)')
parser.add_argument('--sources', type=int, nargs='+', default=[1], metavar='Sources', help='number of sources (default: 1)')
parser.add_argument('--source-state', type=str, default='static', metavar='SourceState', help='state of sources (default: Static)')
parser.add_argument('--data-op', type=str, default='save', metavar='DataOp', help='operation for generated data (default: Save signal)')  
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
 
import numpy as np
import tqdm
import copy
import scipy.io
from data_generation_opt import opt
from common.utils import set_seed, save_file, load_file
from dataset import Parameter, AcousticScene, ArraySetup
import dataset as at_dataset
from data_generation_dataset import dualch_array_setup, RandomMicSigDatasetOri

opts = opt(args.wnoise, args.ins)
room_setting = opts.room_setting
dirs = opts.dir()

date_flag = opts.time + ''
if args.ins:
    sig_num_list = {'pretrain': [0], 
                    'preval':	[0], 		 
                    'train':	[0],	 
                    'val': 		[0],		
                    'test': 	[10],
                    }
else:
    sig_num_list = {'pretrain': [10240*50], 
                    'preval':	[5120*2], 		 
                    'train':	[10240*10],	 
                    'val': 		[5120*2],		
                    'test': 	[5120*2],
                    }	
 
print('Room condition (wnoise, SNR, RoomSize, T60, ArrayPosRatio, MinSrcArrayDist, MinSrcBoundaryDist)', args.wnoise, room_setting['snr_range'], room_setting['noise_type'], 
      room_setting['room_size_range'], room_setting['t60_range'], 
      room_setting['array_pos_ratio_range'], room_setting['min_src_array_dist'], room_setting['min_src_boundary_dist'])

if (args.data_op == 'save'):

    part = args.data_part
    data_num_list = sig_num_list[args.stage]
    data_num = data_num_list[part]
    print('Dataset(part, data number):', part, data_num)

    if args.stage == 'pretrain':
        set_seed(1000-part)
    elif args.stage == 'preval':
        set_seed(1100)
    elif args.stage == 'train':
        set_seed(100)
    elif args.stage == 'val':
        set_seed(200)
    elif args.stage == 'test':
        set_seed(300)
    else:
        raise Exception('Stage unrecognized!')

    speed = 343.0
    fs = 16000
    T = 4.112  # Trajectory length (s)  
    if args.source_state == 'static':
        traj_points = 1 # number of RIRs per trajectory
    elif args.source_state == 'mobile':
        traj_points = int(T/0.1) # number of RIRs per trajectory (one RIR per 0.1s)
    else:
        raise Exception('Source state mode unrecognized~')

    # Array
    array_setup = dualch_array_setup

    # Source signal
    # sourceDataset = at_dataset.LibriSpeechDataset(
    #     path = dirs['sousig_'+args.stage],
    #     T = T,
    #     fs = fs,
    #     num_source = max(args.sources),
    #     return_vad = False,
    #     clean_silence = False)
    sourceDataset = at_dataset.WSJ0Dataset(
        path = dirs['sousig_'+args.stage],
        T = T,
        fs = fs, )

    # Noise signal
    noiseDataset = at_dataset.NoiseDataset(
        T = T,
        fs = fs,
        nmic = array_setup.mic_pos.shape[0],
        noise_type = Parameter(room_setting['noise_type'], discrete=True),
        noise_path = dirs['noisig_'+args.stage],
        c = speed)

    # Room acoustics
    return_data = ['sig', 'scene']
    dataset = RandomMicSigDatasetOri(
        sourceDataset=sourceDataset,
        num_source=Parameter(args.sources, discrete=True),
        source_state=args.source_state,
        room_sz=Parameter(room_setting['room_size_range'][0], room_setting['room_size_range'][1]), 
        T60=Parameter(room_setting['t60_range'][0], room_setting['t60_range'][1]),
        abs_weights=Parameter([0.5] * 6, [1.0] * 6),
        array_setup=array_setup,
        array_pos=Parameter(room_setting['array_pos_ratio_range'][0], room_setting['array_pos_ratio_range'][1]),
        min_src_array_dist=room_setting['min_src_array_dist'],
        min_src_boundary_dist=room_setting['min_src_boundary_dist'],
        noiseDataset=noiseDataset,
        SNR=Parameter(room_setting['snr_range'][0], room_setting['snr_range'][1]),
        nb_points=traj_points,
        dataset_sz=data_num,
        c=speed,
        transforms=None,
        return_data=return_data,
    )

    # Data generation
    sensig_dir = dirs['sensig_'+args.stage]
    save_dir = None
    for sdir in sensig_dir:
        if 'simulate' in sdir:
            save_dir = copy.deepcopy(sdir) + date_flag
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

    start_idx = int(np.sum(data_num_list[0:part]))
    print('stage: ', args.stage)
    print('generating starts from: ', int(start_idx/1024), 'k')
    pbar = tqdm.tqdm(range(data_num), desc='generating signals')
    for idx in pbar:
        mic_signals, acoustic_scene = dataset[idx] # (nsamples, nmic)
        save_idx = start_idx+idx

        if args.save_orisrc == False:
            acoustic_scene.source_signal = []
            acoustic_scene.noise_signal = []
            acoustic_scene.timestamps = []
            acoustic_scene.t = []
            acoustic_scene.trajectory = []
        else:
            if args.ins:
                src_noi_rir_path = save_dir + '/' + str(save_idx) + '_mic_src_noi_rir.mat'
                src_signals = copy.deepcopy(acoustic_scene.source_signal) # (nsamples, nsrc)
                noi_signals = copy.deepcopy(acoustic_scene.noise_signal) # (nsamples, nmic)
                rir = copy.deepcopy(acoustic_scene.RIR[0]).transpose(2, 1, 0, 3) # (nsamples, nmic, npoints, nsrc)
                scipy.io.savemat(src_noi_rir_path, {'micsig': mic_signals, 'srcsig': src_signals, 'noisig': noi_signals, 'rir': rir})

        sig_path = save_dir + '/' + str(save_idx) + '.wav'
        acous_path = save_dir + '/' + str(save_idx) + '.npz'
        # for i in acoustic_scene.__dict__.keys():
        #     print(i, acoustic_scene.__dict__[i])
        save_file(mic_signals, acoustic_scene, sig_path, acous_path)

elif (args.data_op == 'read'):
    class AcousticScene:
        def __init__(self):
            pass
    acoustic_scene = AcousticScene()

    sig_path = dirs['sensig_test'][0] + '/' + '4.wav'
    acous_path = dirs['sensig_test'][0] + '/' + '4.npz'
    mic_signals, acoustic_scene = load_file(acoustic_scene, sig_path, acous_path)

    for i in acoustic_scene.__dict__.keys():
        print(i, acoustic_scene.__dict__[i])
    print(acoustic_scene.RIR[0].shape)
