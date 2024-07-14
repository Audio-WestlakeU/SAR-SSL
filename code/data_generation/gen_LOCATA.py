import os
import numpy as np
import scipy
import scipy.io
import scipy.signal
import random
import soundfile
import librosa
import itertools
import textgrid
import numpy as np
import pandas as pd
from typing import *
from pathlib import Path
from numpy.linalg import norm
from torch.utils.data import Dataset
from abc import ABC, abstractmethod

def pad_cut_sig_sameutt(sig, nsample_desired):
    """ Pad (by repeating the same utterance) and cut signal to desired length
        Args:       sig             - signal (nsample, )
                    nsample_desired - desired sample length
        Returns:    sig_pad_cut     - padded and cutted signal (nsample_desired,)
    """ 
    nsample = sig.shape[0]
    while nsample < nsample_desired:
        sig = np.concatenate((sig, sig), axis=0)
        nsample = sig.shape[0]
    st = np.random.randint(0, nsample - nsample_desired)
    ed = st + nsample_desired
    sig_pad_cut = sig[st:ed]

    return sig_pad_cut


def pad_cut_sig_samespk(utt_path_list, current_utt_idx, nsample_desired, fs_desired):
    """ Pad (by adding utterance of the same spearker) and cut signal to desired length
        Args:       utt_path_list             - 
                    current_utt_idx
                    nsample_desired - desired sample length
                    fs_desired
        Returns:    sig_pad_cut     - padded and cutted signal (nsample_desired,)
    """ 
    sig = np.array([])
    nsample = sig.shape[0]
    while nsample < nsample_desired:
        utterance, fs = soundfile.read(utt_path_list[current_utt_idx])
        if fs != fs_desired:
            utterance = scipy.signal.resample_poly(utterance, up=fs_desired, down=fs)
            raise Warning(f'Signal is downsampled from {fs} to {fs_desired}')
        sig = np.concatenate((sig, utterance), axis=0)
        nsample = sig.shape[0]
        current_utt_idx += 1
        if current_utt_idx >= len(utt_path_list): current_utt_idx=0
    st = np.random.randint(0, nsample - nsample_desired)
    ed = st + nsample_desired
    sig_pad_cut = sig[st:ed]

    return sig_pad_cut

def select_microphone_pairs(mic_poss, nmic_selected, mic_dist_range):
    """ Randomly select certain number of microphones within microphone distance range 
        Args:       mic_poss        - microphone postions (nmic, 3)
                    nmic_selected   - the number of selected microphones
                    mic_dist_range  - the range of microphone distance [lower, upper]
        Returns:    mic_idxes       - indexes of selected microphones
    """
    nmic = mic_poss.shape[0]
    mic_pair_idxes_selected = []
    mic_pos_selected = []
    mic_pair_idxes_set = itertools.permutations(range(nmic), nmic_selected)
    for mic_pair_idxes in mic_pair_idxes_set:
        mic_pos = mic_poss[mic_pair_idxes, :]
        dist = np.sqrt(np.sum((mic_pos[0, :]-mic_pos[1, :])**2))
        if ( (dist >= mic_dist_range[0]) | (dist <= mic_dist_range[1]) ):
            mic_pair_idxes_selected += [mic_pair_idxes]
            mic_pos_selected += [mic_pos]
    assert (not mic_pair_idxes_selected)==False, f'No microphone pairs satisfy the microphone distance range {mic_dist_range}'
    return mic_pair_idxes_selected, mic_pos_selected

class RealMicSigDataset(Dataset):
    def __init__(self,  
                 data_dir: str, 
                 T: float, 
                 fs: int, 
                 stage: str,  # 'train', 'val', 'test' 
                 arrays: List[str] = ['array'], 
                 mic_dist_range: List[float] = [0.03, 0.20], 
                 nmic_selected: int = 2, 
                 prob_mode:List[str] = ['duration', 'micpair'], 
                 dataset_sz = None,
                 remove_spkoverlap = False
                 ):
        super().__init__()
        
        self.mic_idxes_selected = self.select_micpairs(arrays, nmic_selected, mic_dist_range)
        self.data_items, self.data_probs_cumsum = self.get_items_probs(data_dir, arrays, self.mic_idxes_selected, T, stage, prob_mode)
        assert (self.data_probs_cumsum[-1] == 1), self.data_probs_cumsum[-1]
        assert len(self.data_items) == len(self.data_probs_cumsum), [len(self.data_items), len(self.data_probs_cumsum)]

        self.dataset_sz = len(self.data_items) if dataset_sz is None else dataset_sz
        self.T = T
        self.fs = fs
        self.mic_dist_range = mic_dist_range
        self.nmic_selected = nmic_selected
        self.remove_spkoverlap = remove_spkoverlap
        
    def __len__(self):
        return self.dataset_sz

    @abstractmethod
    def get_items_probs(self, data_dir, tasks, arrays, mic_idxes_selected, duration_min_limit, stage, prob_mode):
        pass

    @abstractmethod
    def read_micsig(self, data_path, st=None, ed=None, mic_idxes_selected=None):
        pass
    
    @abstractmethod
    def select_micpairs(self, arrays, nmic_selected, mic_dist_range):
        pass

    def duration(self):
        
        data_paths = []
        for idx in range(len(self.data_items)):
            data_path = self.data_items[idx][0]
            mic_idx = self.data_items[idx][-1]
            data_paths += [data_path]
            # data_paths += [data_path, mic_idx]
        data_paths = list(set(data_paths))
        durations = []
        for data_path in data_paths:
            duration = soundfile.info(data_path).duration
            durations += [duration]
        # print(f'Duration is {durations} s')
        print(f'Total duration is {np.sum(np.array(durations))/3600:.2f} h')

        max_size = np.minimum(1048575, len(data_paths))
        data = {
            'Data path': data_paths[:max_size],
            'Duration': durations[:max_size],
            }
        df = pd.DataFrame(data)
        df.to_excel("Duration.xlsx", index=True)

        return durations, np.sum(np.array(durations))/3600

    def __getitem__(self, idx=None):
        idx = np.searchsorted(self.data_probs_cumsum, np.random.uniform())
        assert idx < len(self.data_items), [idx, len(self.data_items),self.data_items[0],len(self.data_probs_cumsum)]
        data_return = self.data_items[idx]
        if len(data_return) == 3:
            data_path, steds, mic_idxes = data_return
    
            duration = steds[-1] # soundfile.info(data_path).duration
            fs = soundfile.info(data_path).samplerate
            nsample = int(duration * fs)
            nsample_desired = int(self.T * fs)
            if nsample>nsample_desired:
                st = np.random.randint(0, nsample - nsample_desired) + int(fs*steds[0])
                ed = st + nsample_desired
                assert ed <= fs*soundfile.info(data_path).duration, 'error'
            elif nsample==nsample_desired:
                st = int(fs*steds[0])
                ed = st + nsample_desired
            else:
                raise Exception('error')
            mic_signals = self.read_micsig(data_path, st=st, ed=ed, mic_idxes_selected=mic_idxes)
        
        elif len(data_return) == 2:
            data_path, mic_idxes = data_return
    
            duration = soundfile.info(data_path).duration
            fs = soundfile.info(data_path).samplerate
            nsample = int(duration * fs)
            nsample_desired = int(self.T * fs)
            if nsample<nsample_desired:
                mic_signals = self.read_micsig(data_path, mic_idxes_selected=mic_idxes)
                mic_signals = pad_cut_sig_sameutt(mic_signals, nsample_desired)
                print('smaller number of samples')
            elif nsample==nsample_desired:
                mic_signals = self.read_micsig(data_path, st=0, ed=nsample, mic_idxes_selected=mic_idxes)
            else:    
                st = np.random.randint(0, nsample - nsample_desired)
                ed = st + nsample_desired
                mic_signals = self.read_micsig(data_path, st=st, ed=ed, mic_idxes_selected=mic_idxes)

        if self.fs != fs:
            mic_signals = scipy.signal.resample_poly(mic_signals, self.fs, fs)

        return mic_signals


class LOCATADataset(Dataset):
    """ 
	Refs: The LOCATA Challenge: Acoustic Source Localization and Tracking
	Code: https://github.com/cevers/sap_locata_io
	URL: https://www.locata.lms.tf.fau.de/datasets/, https://zenodo.org/record/3630471
	"""
    def __init__(self, 
                 data_dir: str, 
                 T: float, 
                 fs: int, 
                 stage: str,  # 'train', 'val', 'test' 
                 tasks: List[int] = [1,3,5], 
                 arrays: List[str] = ['dicit', 'benchmark2', 'eigenmike'], 
                 mic_dist_range: List[float] = [0.03, 0.20], 
                 nmic_selected: int = 2, 
                 prob_mode:List[str] = [''], #'duration', 'micpair'
                 dataset_sz = None, 
                 sound_speed: float = 343.0):
        
        self.room_sz = np.array([7.1, 9.8, 3])

        self.mic_idxes_selected = self.select_micpairs(arrays, nmic_selected, mic_dist_range)
        self.data_items, self.data_probs_cumsum = self.get_items_probs(data_dir, tasks, arrays, self.mic_idxes_selected, T, stage, prob_mode)

        self.dataset_sz = len(self.data_items) if dataset_sz is None else dataset_sz
        self.T = T
        self.fs = fs
        self.mic_dist_range = mic_dist_range
        self.nmic_selected = nmic_selected
        self.sound_speed = sound_speed
        self.stage = stage

    def __len__(self):
        return self.dataset_sz

    def __getitem__(self, idx=None):
        idx = np.searchsorted(self.data_probs_cumsum, np.random.uniform())
        assert idx < len(self.data_items), [idx, len(self.data_items),self.data_items[0],len(self.data_probs_cumsum)]
        data_return = self.data_items[idx]
        
        data_path, mic_idxes = data_return

        duration = soundfile.info(data_path).duration
        fs = soundfile.info(data_path).samplerate
        nsample = int(duration * fs)
        nsample_desired = int(self.T * fs)
        if nsample<nsample_desired:
            mic_signals = self.read_micsig(data_path, mic_idxes_selected=mic_idxes)
            mic_signals = pad_cut_sig_sameutt(mic_signals, nsample_desired)
            print('smaller number of samples')
        elif nsample==nsample_desired:
            mic_signals = self.read_micsig(data_path, st=0, ed=nsample, mic_idxes_selected=mic_idxes)
        else:    
            st = np.random.randint(0, nsample - nsample_desired)
            ed = st + nsample_desired
            mic_signals = self.read_micsig(data_path, st=st, ed=ed, mic_idxes_selected=mic_idxes)

        if self.fs != fs:
            mic_signals = scipy.signal.resample_poly(mic_signals, self.fs, fs)

            
        df = pd.read_csv(time_path, sep='\t')
        required_time = df['hour'].values * 3600 + df['minute'].values * 60 + df['second'].values
        timestamps = required_time - required_time[0]

        
        
        t = (np.arange(len(mic_signals)))/self.fs
        sources_pos = []
        trajectories = []
        for src_idx in range(len(src_pos_path)):
            file = src_pos_path[src_idx]

            df = pd.read_csv(file, sep='\t')
            source_pos = np.stack((df['x'].values, df['y'].values, df['z'].values), axis=-1)

            sources_pos.append(source_pos)
            trajectories.append(
                np.array([np.interp(t, timestamps, source_pos[:, i]) for i in range(3)]).transpose())
            
        traj_pts = np.stack(sources_pos).transpose(1, 2, 0)
        trajectories = np.stack(trajectories).transpose(1, 2, 0)


        sensor_vads = []
        for src_idx in range(len(vad_path)):
 
            file = vad_path[src_idx] 
            df = pd.read_csv(file, sep='\t')
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

        return mic_signals

    def get_items_probs(self, data_dir, tasks, arrays, mic_idxes_selected, duration_min_limit, stage, prob_mode):
        dataset_split = {'train':['eval'],
                         'val':['eval'],
                         'test':['dev']}
        data_items = []
        data_probs = []
        for ds in dataset_split[stage]:
            for task in tasks:
                task_path = data_dir + '/' + ds + '/' + 'task' + str(task)
                for recording in os.listdir( task_path ):
                    arrays_list = os.listdir( os.path.join(task_path, recording) )
                    for array in arrays:
                        if array in arrays_list:
                            file_dir = os.path.join(task_path, recording, array)
                            wav_path = os.path.join(file_dir, 'audio_array_' + array + '.wav')
                            audio_duration = soundfile.info(wav_path).duration
                            time_path = os.path.join(file_dir, 'required_time.txt')
                            src_pos_path = [] 
                            vad_path = []
                            for file in os.listdir(file_dir):
                                if file.startswith('audio_source') and file.endswith('.wav'):
                                    src_name = file[13:-4]
                                    src_pos_path.append(os.path.join(file_dir, 'position_source_' + src_name + '.txt'))
                                    vad_path.append(os.path.join(file_dir, 'VAD_' + array + '_' + source_name + '.txt'))
                            
                            if audio_duration >= duration_min_limit:
                                # according to room
                                # according to utterance number
                                data_prob = 1

                                # according to utterance duration
                                if 'duration' in prob_mode:
                                    data_prob *= audio_duration
                                
                                # according to microphone pairs
                                nmicpair = len(mic_idxes_selected[array])
                                for micpair_idx in range(nmicpair):
                                    data_items.append((wav_path, time_path, src_pos_path,mic_idxes_selected[array][micpair_idx]))
                                    if 'micpair' in prob_mode:
                                        data_probs.append(data_prob)
                                    else:
                                        data_probs.append(data_prob/nmicpair)

                                 

        data_probs_cumsum = []
        if len(data_probs)>0:
            data_probs_sum = sum(data_probs)
            data_probs = [prob/data_probs_sum for prob in data_probs]
            data_probs_cumsum=np.cumsum(data_probs, dtype=np.float32)
            data_probs_cumsum[-1] = 1

        return data_items, data_probs_cumsum

    def read_micsig(self, data_path, st=None, ed=None, mic_idxes_selected=None):
        if (st==None) & (ed==None):
            mic_signals, _ = soundfile.read(data_path, dtype='float32')
        else:
            mic_signals, _ = soundfile.read(data_path, start=st, stop=ed, dtype='float32')
        mic_signals = mic_signals[:, mic_idxes_selected]

        return mic_signals
    
    def select_micpairs(self, arrays, nmic_selected, mic_dist_range):
        mic_idxes_selected = {}
        mic_poss = {'dummy': np.array((
                            (-0.079,  0.000, 0.000),                    
                            (-0.079, -0.009, 0.000),
                            ( 0.079,  0.000, 0.000),
                            ( 0.079, -0.009, 0.000))),
                    'eigenmike': np.array((
                            ( 0.000,  0.039,  0.015),
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
                    'benchmark2': np.array((
                            (-0.028,  0.030, -0.040),
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
                    'dicit': np.array((
                            ( 0.96, 0.00, 0.00),
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
                            (-0.96, 0.00, 0.32)))}
        for array in arrays:
            mic_idxes_selected[array], _ = select_microphone_pairs(mic_poss[array], nmic_selected, mic_dist_range)                       

        return mic_idxes_selected



if __name__ == '__main__':
    from opt import opt_pretrain
    opts = opt_pretrain()
    dirs = opts.dir()


    sig_dir = dirs['LOCATA']
    dataset = LOCATADataset(
				data_dir = sig_dir,
				T = 4.112,
				fs = 16000,
				stage = 'train',
				# tasks = [1],
				# arrays = ['benchmark2'],
				mic_dist_range = [0.05, 20],
				dataset_sz = 10
			)
    _, total_duration = dataset.duration()
    print('LOCATA: ', total_duration, 'h')
    # for i in range(10):
    #     sig = dataset[i]
    #     print(sig.shape)
