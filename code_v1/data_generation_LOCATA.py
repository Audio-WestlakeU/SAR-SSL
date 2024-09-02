""" 
    Generate LOCATA dataset
    Usage: 		Need to specify stage(, data-id, save-orisrc, data-op)
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
import copy
import scipy
import scipy.io
import scipy.signal
import random
import soundfile
import pandas as pd
from torch.utils.data import Dataset
from collections import namedtuple
from common.utils import load_file, save_file, set_seed
from common.utils_room_acoustics import select_microphones, com_num_micpair
from data_generation_opt import opt
from dataset import Parameter, AcousticScene
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Generating multi-channel audio signals or RIRs')
parser.add_argument('--stage', type=str, default='train', metavar='Stage', help='stage that generated data used for (default: pretrain)') # ['pretrain', 'preval', 'train', 'val', 'test', 'test_pretrain']
parser.add_argument('--data-id', type=int, nargs='+', default=[0], metavar='Datasets', help='dataset IDs (default: 0)')
parser.add_argument('--save-orisrc', action='store_true', default=False, help='save original source signal (default: False)')
parser.add_argument('--data-op', type=str, default='save', metavar='DataOp', help='operation for generated data (default: Save)') 
parser.add_argument('--workers', type=int, default=32, metavar='Worker', help='number of workers (default: 32)')
args = parser.parse_args()

opts = opt()
dirs = opts.dir()

sig_list = ['LOCATA']
sig_num_list = {'pretrain': [10240*14, 	], 
                'preval':	[5120*2, 	], 		 
                'train':	[10240*10,	],	 
                'val': 		[5120*1, 	],		
                'test': 	[5120*1, 	],
                'test_pretrain': [5120*2, ],}		
        
idx_list = args.data_id
mic_dist_range = [0.03, 0.20]
arrays = ['dicit', 'benchmark2', 'eigenmike']

ArraySetup = namedtuple('ArraySetup', 'arrayType, orV, mic_scale, mic_rotate, mic_pos, mic_orV, mic_pattern')
# orV: put the source in oneside (indicated by orV) of the array
# mic_scale: half of the mic_distance should be smaller than the minimum separation between the array and the walls defined by array_pos
# mic_rotate: anticlockwise azimuth rotate in degrees
# mic_pos: relative normalized microphone postions, actural position is mic_scale*(mic_pos w mic_rotate)+array_pos*room_sz
# mic_orV: Invalid for omnidirectional microphones
# Named tuple with the characteristics of a microphone array and definitions of dual-channel array

dicit_array_setup = ArraySetup(arrayType='planar_linear',
    orV = np.array([0.0, 1.0, 0.0]),
    mic_scale = Parameter(1),
    mic_rotate = Parameter(0),
    mic_pos = np.array((( 0.96, 0.00, 0.00),
    ( 0.64, 0.00, 0.00),
    ( 0.32, 0.00, 0.00),
    ( 0.16, 0.00, 0.00),
    ( 0.08, 0.00, 0.00),
    ( 0.04, 0.00, 0.00),
    ( 0.00, 0.00, 0.00),
    ( 0.96, 0.00, 0.32),
    (-0.04, 0.00, 0.00),
    (-0.08, 0.00, 0.00),
    (-0.16, 0.00, 0.00),
    (-0.32, 0.00, 0.00),
    (-0.64, 0.00, 0.00),
    (-0.96, 0.00, 0.00),
    (-0.96, 0.00, 0.32))),
    mic_orV = np.tile(np.array([[0.0, 1.0, 0.0]]), (15,1)),
    mic_pattern = 'omni'
)

dummy_array_setup = ArraySetup(arrayType='planar',
    orV = np.array([0.0, 1.0, 0.0]),
    mic_scale = Parameter(1),
    mic_rotate = Parameter(0),
    mic_pos = np.array(((-0.079,  0.000, 0.000),
    (-0.079, -0.009, 0.000),
    ( 0.079,  0.000, 0.000),
    ( 0.079, -0.009, 0.000))),
    mic_orV = np.array(((-1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    ( 1.0, 0.0, 0.0),
    ( 1.0, 0.0, 0.0))),
    mic_pattern = 'omni'
)

benchmark2_array_setup = ArraySetup(arrayType='3D',
    orV = np.array([0.0, 1.0, 0.0]),
    mic_scale = Parameter(1),
    mic_rotate = Parameter(0),
    mic_pos = np.array(((-0.028,  0.030, -0.040),
    ( 0.006,  0.057,  0.000),
    ( 0.022,  0.022, -0.046),
    (-0.055, -0.024, -0.025),
    (-0.031,  0.023,  0.042),
    (-0.032,  0.011,  0.046),
    (-0.025, -0.003,  0.051),
    (-0.036, -0.027,  0.038),
    (-0.035, -0.043,  0.025),
    ( 0.029, -0.048, -0.012),
    ( 0.034, -0.030,  0.037),
    ( 0.035,  0.025,  0.039))),
    mic_orV = np.array(((-0.028,  0.030, -0.040),
    ( 0.006,  0.057,  0.000),
    ( 0.022,  0.022, -0.046),
    (-0.055, -0.024, -0.025),
    (-0.031,  0.023,  0.042),
    (-0.032,  0.011,  0.046),
    (-0.025, -0.003,  0.051),
    (-0.036, -0.027,  0.038),
    (-0.035, -0.043,  0.025),
    ( 0.029, -0.048, -0.012),
    ( 0.034, -0.030,  0.037),
    ( 0.035,  0.025,  0.039))),
    mic_pattern = 'omni'
)

eigenmike_array_setup = ArraySetup(arrayType='3D',
    orV = np.array([0.0, 1.0, 0.0]),
    mic_scale = Parameter(1),
    mic_rotate = Parameter(0),
    mic_pos = np.array((( 0.000,  0.039,  0.015),
    (-0.022,  0.036,  0.000),
    ( 0.000,  0.039, -0.015),
    ( 0.022,  0.036,  0.000),
    ( 0.000,  0.022,  0.036),
    (-0.024,  0.024,  0.024),
    (-0.039,  0.015,  0.000),
    (-0.024,  0.024,  0.024),
    ( 0.000,  0.022, -0.036),
    ( 0.024,  0.024, -0.024),
    ( 0.039,  0.015,  0.000),
    ( 0.024,  0.024,  0.024),
    (-0.015,  0.000,  0.039),
    (-0.036,  0.000,  0.022),
    (-0.036,  0.000, -0.022),
    (-0.015,  0.000, -0.039),
    ( 0.000, -0.039,  0.015),
    ( 0.022, -0.036,  0.000),
    ( 0.000, -0.039, -0.015),
    (-0.022, -0.036,  0.000),
    ( 0.000, -0.022,  0.036),
    ( 0.024, -0.024,  0.024),
    ( 0.039, -0.015,  0.000),
    ( 0.024, -0.024, -0.024),
    ( 0.000, -0.022, -0.036),
    (-0.024, -0.024, -0.024),
    (-0.039, -0.015,  0.000),
    (-0.024, -0.024,  0.024),
    ( 0.015,  0.000,  0.039),
    ( 0.036,  0.000,  0.022),
    ( 0.036,  0.000, -0.022),
    ( 0.015,  0.000, -0.039))),
    mic_orV = np.array((( 0.000,  0.039,  0.015),
    (-0.022,  0.036,  0.000),
    ( 0.000,  0.039, -0.015),
    ( 0.022,  0.036,  0.000),
    ( 0.000,  0.022,  0.036),
    (-0.024,  0.024,  0.024),
    (-0.039,  0.015,  0.000),
    (-0.024,  0.024,  0.024),
    ( 0.000,  0.022, -0.036),
    ( 0.024,  0.024, -0.024),
    ( 0.039,  0.015,  0.000),
    ( 0.024,  0.024,  0.024),
    (-0.015,  0.000,  0.039),
    (-0.036,  0.000,  0.022),
    (-0.036,  0.000, -0.022),
    (-0.015,  0.000, -0.039),
    ( 0.000, -0.039,  0.015),
    ( 0.022, -0.036,  0.000),
    ( 0.000, -0.039, -0.015),
    (-0.022, -0.036,  0.000),
    ( 0.000, -0.022,  0.036),
    ( 0.024, -0.024,  0.024),
    ( 0.039, -0.015,  0.000),
    ( 0.024, -0.024, -0.024),
    ( 0.000, -0.022, -0.036),
    (-0.024, -0.024, -0.024),
    (-0.039, -0.015,  0.000),
    (-0.024, -0.024,  0.024),
    ( 0.015,  0.000,  0.039),
    ( 0.036,  0.000,  0.022),
    ( 0.036,  0.000, -0.022),
    ( 0.015,  0.000, -0.039))),
    mic_pattern = 'omni'
)

miniDSP_array_setup = ArraySetup(arrayType='planar',
    orV = np.array([0.0, 0.0, 1.0]),
    mic_scale = Parameter(1),
    mic_rotate = Parameter(0),
    mic_pos = np.array((( 0.0000,  0.0430, 0.000),
        ( 0.0372,  0.0215, 0.000),
        ( 0.0372, -0.0215, 0.000),
        ( 0.0000, -0.0430, 0.000),
        (-0.0372, -0.0215, 0.000),
        (-0.0372,  0.0215, 0.000))),
    mic_orV = np.array(((0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0))),
    mic_pattern = 'omni'
)

class LOCATADataset(Dataset):
    """ 
    Refs: The LOCATA Challenge: Acoustic Source Localization and Tracking
    Code: https://github.com/cevers/sap_locata_io
    URL: https://www.locata.lms.tf.fau.de/datasets/, https://zenodo.org/record/3630471
    """
    def __init__(self, data_dir, T, fs, dataset=['dev', 'eval'], tasks=[1, 3, 5], arrays=['dicit',], mic_dist_range=[0.03, 0.20], nmic_selected=2, sig_range_selected=[0,1], transforms=None, dataset_sz=None, return_data = ['sig', 'gt']):
        """ 
        """
        self.room_sz = np.array([7.1, 9.8, 3])
        self.data_paths = []

        for ds in dataset:
            for task in tasks:
                task_path = data_dir + '/' + ds + '/' + 'task' + str(task)
                for recording in os.listdir( task_path ):
                    arrays_list = os.listdir( os.path.join(task_path, recording) )
                    for array in arrays:
                        if array in arrays_list:
                            data_path = os.path.join(task_path, recording, array)
                            self.data_paths.append( data_path )

        self.transforms = transforms
        self.dataset_sz = len(self.data_paths) if dataset_sz is None else dataset_sz
        self.T = T
        self.fs = fs
        self.mic_dist_range = mic_dist_range
        self.nmic_selected = nmic_selected
        self.sig_range_selected = sig_range_selected
        self.return_data = return_data

    def __len__(self):
        return self.dataset_sz
    
    def duration(self, idx):
        data_path = self.data_paths[idx]
        array = data_path.split('/')[-1]
        task = data_path.split('/')[-3]
        mic_signals, fs = soundfile.read( os.path.join(data_path, 'audio_array_' + array + '.wav') )
        print(array, task, mic_signals.shape[:, 0]/fs, 's')

    def __getitem__(self, idx):

        idx = np.random.randint(0, len(self.data_paths))
        
        data_path = self.data_paths[idx]
        array = data_path.split('/')[-1]
        task = data_path.split('/')[-3]
        if array == 'dummy':
            array_setup = dummy_array_setup
        elif array == 'eigenmike':
            array_setup = eigenmike_array_setup
        elif array == 'benchmark2':
            array_setup = benchmark2_array_setup
        elif array == 'dicit':
            array_setup = dicit_array_setup
        else:
            raise Exception('Array not exist in LOCATA dataset')

        mic_signals, fs = soundfile.read( os.path.join(data_path, 'audio_array_' + array + '.wav') )
        if self.fs != fs:
            mic_signals = scipy.signal.resample_poly(mic_signals, self.fs, fs)

        df = pd.read_csv(os.path.join(data_path, 'position_array_' + array + '.txt'), sep='\t')
        array_pos = np.stack((df['x'].values, df['y'].values, df['z'].values), axis=-1)
        array_ref_vec = np.stack((df['ref_vec_x'].values, df['ref_vec_y'].values, df['ref_vec_z'].values), axis=-1)
        array_rotation = np.zeros((array_pos.shape[0], 3, 3))
        for i in range(3):
            for j in range(3):
                array_rotation[:, i, j] = df['rotation_' + str(i + 1) + str(j + 1)]

        # Remove initial silence
        start = np.argmax(mic_signals[:,0] > mic_signals[:,0].max()*0.15)
        mic_signals = mic_signals[start:, :]
        t = (np.arange(len(mic_signals)) + start)/self.fs

        df = pd.read_csv(os.path.join(data_path, 'required_time.txt'), sep='\t')
        required_time = df['hour'].values * 3600 + df['minute'].values * 60 + df['second'].values
        timestamps = required_time - required_time[0]

        sources_name = [] # loudspeaker
        for file in os.listdir( data_path ):
            if file.startswith('audio_source') and file.endswith('.wav'):
                ls = file[13:-4]
                sources_name.append(ls)

        sources_signal = []
        for source_name in sources_name:
            file = 'audio_source_' + source_name + '.wav'
            source_signal, fs_src = soundfile.read(os.path.join(data_path, file))
            if fs_src > self.fs:
                source_signal = scipy.signal.decimate(source_signal, int(fs_src / self.fs), axis=0)
            source_signal = source_signal[start:start + len(t)]
            sources_signal.append(source_signal)
        sources_sig = np.stack(sources_signal).transpose(1,0)

        sources_pos = []
        trajectories = []
        for source_name in sources_name:
            file = 'position_source_' + source_name + '.txt'
            df = pd.read_csv(os.path.join(data_path, file), sep='\t')
            source_pos = np.stack((df['x'].values, df['y'].values, df['z'].values), axis=-1)

            sources_pos.append(source_pos)
            trajectories.append(
                np.array([np.interp(t, timestamps, source_pos[:, i]) for i in range(3)]).transpose())
            
        traj_pts = np.stack(sources_pos).transpose(1, 2, 0)
        trajectories = np.stack(trajectories).transpose(1, 2, 0)

        sensor_vads = []
        for source_name in sources_name:
            array = data_path.split('/')[-1]
            file = 'VAD_' + array + '_' + source_name + '.txt'
            df = pd.read_csv(os.path.join(data_path, file), sep='\t')
            sensor_vad_ori = df['VAD'].values

            # VAD values @48kHz matched with timestamps @16kHz
            L_audio = len(sensor_vad_ori)
            t_stamps_audio = np.linspace(0, L_audio-1, L_audio) / fs_src # 48 kHz
            t_stamps_opti = t + 0.0 # 16 kHz
            sensor_vad = np.zeros(len(t_stamps_opti))
            cnt = 0
            for i in range(1, L_audio):
                if t_stamps_audio[i] >= t_stamps_opti[cnt]:
                    sensor_vad[cnt] = sensor_vad_ori[i - 1]
                    cnt = cnt + 1
                if cnt > len(sensor_vad)-1:
                    break
            if cnt <= len(sensor_vad)-1:
                VAD[cnt: end] = sensor_vad_ori[end]
                if cnt < len(sensor_vad) - 2:
                    print('Warning: VAD values do not match~')

            sensor_vads.append(sensor_vad)
        sensor_vads = np.stack(sensor_vads)
        vad = sensor_vads.transpose(1,0)

        # Microphone positions & source positions
        mic_poss_relative2center = np.matmul( array_rotation[0,...], np.expand_dims(array_setup.mic_pos*array_setup.mic_scale.getValue(), axis=-1) ).squeeze()
        n = com_num_micpair(mic_poss_relative2center, self.mic_dist_range)
        # Select microphone pair & get microphone positions
        mic_idxes, mic_pos_relative2center = select_microphones(mic_poss_relative2center, self.nmic_selected, self.mic_dist_range)

        # Absolute microphone positions
        if (task=='task1') | (task=='task3') | (task=='task2') | (task=='task4'):
            mic_pos = mic_pos_relative2center + array_pos[0, :] # (nch, 3)
            array_pos = array_pos[0, :] #(3, )
        elif (task=='task5') | (task=='task6'):
            mic_pos = mic_pos_relative2center[np.newaxis, :, :] + array_pos[:, np.newaxis, :] # (npoint, nch, 3)
        else:
            raise Exception('task unrecognized~')

        # Microphone signals
        mic_signals = mic_signals[:, mic_idxes]

        # Guarantee time duration larger than T seconds
        nsample = t.shape[0]
        nsample_desired = int(self.T * self.fs)
        sources_sig_copy = copy.deepcopy(sources_sig)
        t_copy = copy.deepcopy(t)
        trajectories_copy = copy.deepcopy(trajectories)
        mic_signals_copy = copy.deepcopy(mic_signals)
        while nsample < nsample_desired: # for static source in task 1
            print('the number of samples is smaller than that desired')
            sources_sig_copy = np.concatenate((sources_sig_copy, sources_sig), axis=0)
            mic_signals_copy = np.concatenate((mic_signals_copy, mic_signals), axis=0)
            t_copy = np.concatenate((t_copy, t), axis=0)
            trajectories_copy = np.concatenate((trajectories_copy, trajectories), axis=0)
            nsample = sources_sig_copy.shape[0]

        # Select T-second signal
        T_all = nsample/self.fs
        if self.sig_range_selected ==[0,1]:
            range_selected = self.sig_range_selected
        else:
            if T_all < 2.2:
                print('signal length: ', T_all)
                raise Exception('Signal length is too short (LOCATA)')
            elif (T_all >= 2.2) & (T_all < 10):
                if (self.sig_range_selected[0] + self.sig_range_selected[1])/2 < 0.5:
                    range_selected = [0, 0.5]
                else:
                    range_selected = [0.5, 1]
            else:
                range_selected = self.sig_range_selected
            
        st = random.randint(round(nsample*range_selected[0]), round(nsample*range_selected[1]) - nsample_desired)
        ed = st + nsample_desired
        sources_sig = sources_sig_copy[st:ed, :]
        mic_signals = mic_signals_copy[st:ed, :]
        t = t_copy[st:ed]
        trajectories = trajectories_copy[st:ed, :, :]

        nsource = traj_pts.shape[-1]
        assert nsource==1, 'Multiple sources are not supported!'
        timestamps = timestamps - start/self.fs
        t = t - start/self.fs
        
        acoustic_scene = AcousticScene(
                room_sz = self.room_sz,
                beta = [],
                T60 = [],
                array_setup = [],
                mic_pos = mic_pos, # (nch,3)/(npoint,nch,3)
                array_pos = array_pos, # (3,)/(npoint,3)
                traj_pts = traj_pts,  # (npoint,3,nsource) original annonated 
                fs = self.fs,
                RIR = [],
                source_signal = sources_sig, # (nsample,nsource)
                noise_signal = [],
                SNR = [],
                timestamps = timestamps, # (npoint) original annonated 
                t = t,  # (nsample) remove silence
                trajectory = trajectories,  # (nsample,3,nsource) remove silence
                )

        # add attr: TDOA
        npoint = timestamps.shape[0]
        nsample = mic_signals.shape[0]
        nmic = mic_pos.shape[-2]
        nsource = traj_pts.shape[-1]

        if len(acoustic_scene.mic_pos.shape) == 2:
            mic_pos = np.tile(acoustic_scene.mic_pos[np.newaxis, :, :], (npoint, 1, 1))
        elif len(acoustic_scene.mic_pos.shape) == 3:
            mic_pos = acoustic_scene.mic_pos
        else:
            raise Exception('shape of mic_pos is out of range~')
        corr_diff = np.tile(traj_pts[:, np.newaxis, :, :], (1, nmic, 1, 1)) - np.tile(mic_pos[:, :, :, np.newaxis], (1, 1, 1, nsource))
        dist = np.sqrt(np.sum(corr_diff**2, axis=2))  # (npoint,3,nsource)-(nch,3)=(nnpoint,nch,3,nsource)
        re_dist = dist[:, 1:, :] - np.tile(dist[:, 0:1, :], (1, nmic - 1, 1))  # (npoint,nch-1,nsource)
        TDOA = re_dist / acoustic_scene.c  # (npoint,nch-1,nsource)
        acoustic_scene.TDOA = np.zeros((nsample, TDOA.shape[1], nsource))  # (nsample,nch-1,nsource)
        for source_idx in range(nsource):
            for ch_idx in range(TDOA.shape[1]):
                acoustic_scene.TDOA[:, ch_idx, source_idx] = np.interp(t, timestamps, TDOA[:, ch_idx, source_idx])
        
        # add attr: DOA
        # nsample = t.shape[0]
        # acoustic_scene.DOA = np.zeros((nsample, 2, nsource))  # (nsample,2,nsource)
        # for source_idx in range(nsource):
        #     acoustic_scene.DOA[:, :, source_idx] = cart2sph(trajectories[:, :, source_idx] - array_pos[0, :])[:, [1,0]])

        # transformation
        if self.transforms is not None:
            for t in self.transforms:
                mic_signals, acoustic_scene = t(mic_signals, acoustic_scene)

        if self.return_data == ['sig', 'scene']:
            
            return mic_signals, acoustic_scene

        elif self.return_data == ['sig', 'gt']:
            mic_signals = mic_signals.astype(np.float32)
            gts = {}
            gts['TDOA'] = acoustic_scene.TDOAw.astype(np.float32)

            return mic_signals, gts


if __name__ == '__main__':

    if (args.data_op == 'save'):
        for list_idx in idx_list:
            sig = sig_list[list_idx]
            sig_dir = dirs[sig]
            data_num = sig_num_list[args.stage][list_idx]
            print('Dataset:', sig, data_num/1024, 'K')

            extra_name = ''
            if args.stage == 'pretrain':
                ds = ['eval']
                tasks = [1]
                T = 4.112
                sig_range_selected = [0, 1]
                set_seed(5000+list_idx)
            elif args.stage == 'preval':
                ds = ['eval']
                tasks = [1]
                T = 4.112
                sig_range_selected = [0, 1]
                set_seed(5100+list_idx)
            elif args.stage == 'train':
                ds = ['eval']
                tasks = [1,3,5]
                T = 1.04
                sig_range_selected = [0, 0.8]
                set_seed(6000+list_idx)
            elif args.stage == 'val':
                ds = ['eval']
                tasks = [1,3,5]
                T = 1.04
                sig_range_selected = [0.8, 1]
                set_seed(6100+list_idx)
            elif args.stage == 'test':
                ds = ['dev']
                tasks = [1,3,5]
                T = 1.04
                sig_range_selected = [0, 1]
                set_seed(6200+list_idx)
            elif args.stage == 'test_pretrain':
                ds = ['dev']
                tasks = [1]
                T = 4.112
                sig_range_selected = [0, 1]
                set_seed(6200+list_idx)
                extra_name += '_task1_4s'
            else:
                raise Exception('Stage unrecognized!')

            # Microphone signal
            fs = opts.micsig_setting['fs']

            # Room acoustics
            if (arrays == ['dicit']) | (arrays == ['dummy']) | (arrays == ['benchmark2']) | (arrays == ['eigenmike']):
                extra_name += '_'+arrays[0]
            elif arrays == ['dicit', 'benchmark2', 'eigenmike']:
                extra_name += ''
            elif arrays == ['dicit', 'dummy', 'benchmark2', 'eigenmike']:
                extra_name += ''
                
            dataset = LOCATADataset(
                data_dir = sig_dir,
                T = T,
                fs = fs,
                dataset = ds,
                tasks = tasks,
                arrays = arrays,
                mic_dist_range = mic_dist_range,
                sig_range_selected=sig_range_selected,
                dataset_sz = data_num,
                return_data = ['sig', 'scene'],
                )
            dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=args.workers)

            # Data generation
            sensig_dir = dirs['sensig_'+args.stage.split('_')[0]] 
            save_dir = None
            for sdir in sensig_dir:
                if sig in sdir:
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
           
            pbar = tqdm.tqdm(range(0, data_num), desc='generating signals')
            for idx, (mic_signals, acoustic_scene) in enumerate(dataloader):
                pbar.update(1)
            # pbar = tqdm.tqdm(range(data_num), desc='generating signals')
            # for idx in pbar:
            #     mic_signals, acoustic_scene = dataset[idx]

                if args.save_orisrc == False:
                    acoustic_scene.source_signal = []
                    acoustic_scene.noise_signal = []
                    acoustic_scene.timestamps = []
                    acoustic_scene.t = []
                    acoustic_scene.trajectory = []
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

