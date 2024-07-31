""" 
    Generate LOCATA dataset

    Example: python gen_LOCATA.py --stage train --save-to../../data/MicSig/real_ds_locata
            python gen_LOCATA.py --stage val --save-to../../data/MicSig/real_ds_locata
            python gen_LOCATA.py --stage test --save-to../../data/MicSig/real_ds_locata
"""

import os

cpu_num = 8*5
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
# os.environ['OMP_NUM_THREADS'] = str(cpu_num)
# os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
# os.environ['MKL_NUM_THREADS'] = str(cpu_num)
# os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
# os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)

import argparse
import numpy as np
import tqdm
import soundfile
from torch.utils.data import DataLoader
from utils_LOCATA import *

parser = argparse.ArgumentParser(description='Generating multi-channel audio signals or RIRs')
parser.add_argument('--stage', type=str, nargs='+', default=['train'], metavar='Stage', help='stage that generated data used for (default: train)')  
parser.add_argument('--workers', type=int, default=32, metavar='Worker', help='number of workers (default: 32)')
parser.add_argument('--fs', type=int, default=16000, metavar='SamplingRate', help='sampling rate (default: 16000)')
parser.add_argument('--save-to', type=str, default='../../data/MicSig/real_ds_locata', metavar='SaveTo', help='save directory')
args = parser.parse_args()
 
data_num = {'train': 80000, 'val': 1000, 'test': 4000}
seeds = {'train':6000, 'val':6100, 'test':6200}
fs = args.fs
save_to = args.save_to
for stage in args.stage:
    np.random.seed(seed=seeds[stage])
    realdataset = LOCATADataset(
        data_dir = '../../../data/MicSig/LOCATA',
        T = 1.04,
        fs = fs,
        stage = stage,
        tasks = [1, 3, 5],
        arrays = ['dicit', 'benchmark2', 'eigenmike'],
        mic_dist_range = [0.03, 0.20],
        nmic_selected = 2,
        prob_mode = [''],
        load_anno = True,
        dataset_sz = data_num[stage],
        sound_speed = 343.0,
        src_single_static = True,
        transforms = None
        )
    dataloader = DataLoader(realdataset, batch_size=None, shuffle=False, num_workers=args.workers)
    save_to = os.path.join(save_to, stage)
    exist_temp = os.path.exists(save_to)
    if exist_temp==False:
        os.makedirs(save_to)
        print('make dir: ' + save_to)
    else:
        print('existed dir: ' + save_to)
        msg = input('Sure to regenerate signals? (Enter for yes)')
        if msg == '':
            print('Regenerating signals')
    
    pbar = tqdm.tqdm(range(0, data_num[stage]), desc='generating signals')
    for idx, (mic_sig, anno) in enumerate(dataloader):
        pbar.update(1)
        save_to_file = os.path.join(save_to, str(idx) + f'.wav')
        soundfile.write(save_to_file, mic_sig, fs)
        save_to_file = os.path.join(save_to, str(idx) + f'_info.npz')
        np.savez(save_to_file, **anno)