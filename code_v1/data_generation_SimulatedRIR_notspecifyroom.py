""" 
    Generate simulated RIRs (infinite room conditions)
	Usage: 		Need to specify data-num, gpu-id(, sources, source_state, data-op)
"""

import os
import argparse

cpu_num = 8
os.environ["OMP_NUM_THREADS"] = str(cpu_num)

parser = argparse.ArgumentParser(description='Generating multi-channel audio signals or RIRs')
parser.add_argument('--data-num', type=int, default=1000, metavar='DataNum', help='Data Num (default: 1000)')
parser.add_argument('--gpu-id', type=str, default='1', metavar='GPU', help='GPU ID (default: 7)')
parser.add_argument('--sources', type=int, nargs='+', default=[1], metavar='Sources', help='number of sources (default: 1)')
parser.add_argument('--source-state', type=str, default='static', metavar='SourceState', help='state of sources (default: Static)')
parser.add_argument('--data-op', type=str, default='save', metavar='DataOp', help='operation for generated data (default: Save signal)')  
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

import tqdm
from torch.utils.data import DataLoader
from data_generation_opt import opt
from common.utils import set_seed, save_file, load_file
from dataset import Parameter, AcousticScene, dualch_array_setup, RandomMicSigDatasetOri

opts = opt()
room_setting = opts.room_setting
micsig_setting = opts.micsig_setting
dirs = opts.dir()
date_flag = opts.time + ''
data_num = args.data_num

print('Room condition (RoomSize, T60, ArrayPosRatio, MinSrcArrayDist, MinSrcBoundaryDist)', 
      room_setting['room_size_range'], room_setting['t60_range'], 
      room_setting['array_pos_ratio_range'], room_setting['min_src_array_dist'], room_setting['min_src_boundary_dist'])

set_seed(1)
if (args.data_op == 'save'):

    fs = micsig_setting['fs']
    T = micsig_setting['T']
    if args.source_state == 'static':
        traj_points = 1 # number of RIRs per trajectory
    elif args.source_state == 'mobile':
        traj_points = int(T/0.1) # number of RIRs per trajectory (one RIR per 0.1s)
    else:
        raise Exception('Source state mode unrecognized~')

    # Array
    array_setup = dualch_array_setup

    # Room acoustics
    class srcDataset:
        def __init__(self, fs):
            self.fs = fs
        def __len__(self):
            return 0
    return_data = ['rir']
    dataset = RandomMicSigDatasetOri(
        sourceDataset=srcDataset(fs),
        num_source=Parameter(args.sources, discrete=True),
        source_state=args.source_state,
        room_sz=Parameter(room_setting['room_size_range'][0], room_setting['room_size_range'][1]), 
        T60=Parameter(room_setting['t60_range'][0], room_setting['t60_range'][1]),
        abs_weights=Parameter([0.5] * 6, [1.0] * 6),
        array_setup=array_setup,
        array_pos=Parameter(room_setting['array_pos_ratio_range'][0], room_setting['array_pos_ratio_range'][1]),
        min_src_array_dist=room_setting['min_src_array_dist'],
        min_src_boundary_dist=room_setting['min_src_boundary_dist'],
        noiseDataset=[],
        SNR=[],
        nb_points=traj_points,
        dataset_sz=data_num,
        c=room_setting['sound_speed'],
        transforms=None,
        return_data=return_data,
    )

    # Data generation
    save_dirs = dirs['rir']
    for sdir in save_dirs:
        if ('simulate' in sdir):
            save_dir = sdir + date_flag
            break
    exist_temp = os.path.exists(save_dir)
    if exist_temp==False:
        os.makedirs(save_dir)
        print('make dir: ' + save_dir)
    else:
        print('existed dir: ' + save_dir)
        msg = input('Sure to regenerate RIRs? (Enter for yes)')
        if msg == '':
            print('Regenerating RIRs')
            
    pbar = tqdm.tqdm(range(data_num), desc='generating rirs')
    for idx in pbar:
        acoustic_scene = dataset[idx]

        acous_path = save_dir + '/' + str(idx) + '.npz'
        save_file(None, acoustic_scene, sig_path=None, acous_path=acous_path)

elif (args.data_op == 'read'):
    class AcousticScene:
        def __init__(self):
            pass
    acoustic_scene = AcousticScene()
    acous_path = dirs['rir_train'] + '/' + '0.npz'
    acoustic_scene = load_file(acoustic_scene, sig_path=None, acous_path=acous_path)

    for i in acoustic_scene.__dict__.keys():
        print(i, acoustic_scene.__dict__[i])
    print(acoustic_scene.RIR[0].shape)
