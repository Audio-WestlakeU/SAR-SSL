import os
import numpy as np
import scipy
import scipy.io
import scipy.signal
import soundfile
import itertools
import pandas as pd
from typing import *
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from functools import lru_cache, cache

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
    st = np.random.randint(0, nsample - nsample_desired+1)
    ed = st + nsample_desired
    sig_pad_cut = sig[st:ed]

    return sig_pad_cut


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
        load_anno: bool = True,
        dataset_sz = None, 
        sound_speed: float = 343.0,
        src_single_static: bool = True,
        transforms: Callable = None):
        
        self.room_sz = np.array([7.1, 9.8, 3])

        self.mic_idxes_selected, self.mic_pos_selected = self.select_micpairs(arrays, nmic_selected, mic_dist_range)
        self.data_items, self.data_probs_cumsum = self.get_items_probs(
            data_dir, 
            tasks, 
            arrays, 
            self.mic_idxes_selected, 
            self.mic_pos_selected, 
            T, 
            stage, 
            prob_mode)

        self.dataset_sz = len(self.data_items) if dataset_sz is None else dataset_sz
        self.T = T
        self.fs = fs
        self.mic_dist_range = mic_dist_range
        self.nmic_selected = nmic_selected
        self.sound_speed = sound_speed
        self.stage = stage
        self.load_anno = load_anno
        self.transforms = transforms
        self.src_single_static = src_single_static

    def __len__(self):
        return self.dataset_sz

    def __getitem__(self, idx=None, min_dura=1.1):
        idx = np.searchsorted(self.data_probs_cumsum, np.random.uniform())
        assert idx < len(self.data_items), [idx, len(self.data_items),self.data_items[0],len(self.data_probs_cumsum)]
        wav_path, time_path, array_pos_path, src_pos_path, vad_path, mic_idxes, mic_pos, st_ed_ratio, sil_duration = self.data_items[idx]

        duration = soundfile.info(wav_path).duration - sil_duration
        fs = soundfile.info(wav_path).samplerate
        nsample = int(duration * fs)
        nsample_desired = int(self.T * fs)
        assert (nsample >= nsample_desired) & (duration >= (2*min_dura)), f'Signal length is too short (LOCATA): {nsample/fs}'
        if (duration < 10):
            if (st_ed_ratio[0] + st_ed_ratio[1])/2 < 0.5:
                st_ed_ratio = [0, 0.5]
            else:
                st_ed_ratio = [0.5, 1]
        st = np.random.randint(round(nsample*st_ed_ratio[0]+fs*sil_duration), round(nsample*st_ed_ratio[1]+fs*sil_duration) - nsample_desired)
        ed = st + nsample_desired

        mic_sig = self.read_micsig(wav_path, st=st, ed=ed, mic_idxes_selected=mic_idxes)
        if self.fs != fs:
            mic_sig = scipy.signal.resample_poly(mic_sig, self.fs, fs)

        t = np.arange(mic_sig.shape[0])/self.fs + st/fs
        if self.load_anno:
            TDOA = self.load_annotation(t, fs, self.sound_speed, mic_pos, time_path, array_pos_path, src_pos_path, vad_path=None)[0]
        assert ((len(t) == len(mic_sig)) & (TDOA.shape[0]==len(mic_sig))), [len(t), len(mic_sig), TDOA.shape[0]]
        
        if self.transforms is not None:
            for trans in self.transforms:
                mic_sig = trans(mic_sig)
                if self.load_anno:
                    TDOA = trans(TDOA)

        max_value = np.max(np.abs(mic_sig))
        mic_sig = mic_sig/(max_value+1e-8)*0.9

        if self.src_single_static:
            if self.load_anno:
                TDOA = np.array(np.mean(TDOA))
 
        anno = {
            'TDOA': TDOA.astype(np.float32), 
            # 'T60': np.array(np.NAN),  
            # 'DRR': np.array(np.NAN),
            # 'C50': np.array(np.NAN),
            # 'ABS': np.array(np.NAN),
            }
        if self.load_anno:
            return mic_sig.astype(np.float32), anno
        else:
            return mic_sig.astype(np.float32)
 
    def get_items_probs(self, data_dir, tasks, arrays, mic_idxes_selected, mic_pos_selected, duration_min_limit, stage, prob_mode):
        dataset_split = {'train':['eval'],
                         'val':['eval'],
                         'test':['dev']}
        st_ed_ratio = {'train':[0, 0.8],
                       'val':[0.8, 1],
                       'test':[0, 1]}
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
                            sil_duration = self._calculate_silence_beginning(wav_path)
                            audio_duration = soundfile.info(wav_path).duration
                            time_path = os.path.join(file_dir, 'required_time.txt')
                            src_pos_path = [] 
                            vad_path = []
                            for file in os.listdir(file_dir):
                                if file.startswith('audio_source') and file.endswith('.wav'):
                                    src_name = file[13:-4]
                                    src_pos_path.append(os.path.join(file_dir, 'position_source_' + src_name + '.txt'))
                                    vad_path.append(os.path.join(file_dir, 'VAD_' + array + '_' + src_name + '.txt'))
                            
                            array_pos_path = os.path.join(file_dir, 'position_array_' + array + '.txt')

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
                                    data_items.append((wav_path, time_path, array_pos_path, src_pos_path, vad_path, 
                                        mic_idxes_selected[array][micpair_idx], mic_pos_selected[array][micpair_idx], st_ed_ratio[stage], sil_duration))
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

    def _calculate_silence_beginning(self, data_path, max_dura=4):
        fs = soundfile.info(data_path).samplerate
        mic_sig, _ = soundfile.read(data_path, start=0, stop=int(fs*max_dura), dtype='float32')
        sil_duration = np.argmax(mic_sig[:,0] > mic_sig[:,0].max()*0.15) / fs
        return sil_duration

    # @cache
    def read_micsig(self, data_path, st=None, ed=None, mic_idxes_selected=None):
        if (st==None) & (ed==None):
            mic_sig, _ = soundfile.read(data_path, dtype='float32')
        else:
            mic_sig, _ = soundfile.read(data_path, start=st, stop=ed, dtype='float32')
        if mic_idxes_selected is not None:
            mic_sig = mic_sig[:, mic_idxes_selected]

        return mic_sig

    #@cache
    def load_annotation(self, t, fs, sound_speed, mic_pos, time_path, array_pos_path, src_pos_path, vad_path=None):
        return_data = []
 
        df = pd.read_csv(time_path, sep='\t')
        required_time = df['hour'].values * 3600 + df['minute'].values * 60 + df['second'].values
        timestamps = required_time - required_time[0]
        
        df = pd.read_csv(array_pos_path, sep='\t')
        array_pos = np.stack((df['x'].values, df['y'].values, df['z'].values), axis=-1)
        array_ref_vec = np.stack((df['ref_vec_x'].values, df['ref_vec_y'].values, df['ref_vec_z'].values), axis=-1)
        array_rotation = np.zeros((array_pos.shape[0], 3, 3))
        for i in range(3):
            for j in range(3):
                array_rotation[:, i, j] = df['rotation_' + str(i + 1) + str(j + 1)]
        # Microphone positions & source positions
        mic_pos_relative2center = np.matmul( array_rotation[0,...], np.expand_dims(mic_pos, axis=-1) ).squeeze()
        # Absolute microphone positions
        if ('task1' in array_pos_path) | ('task3' in array_pos_path) | ('task2' in array_pos_path) | ('task4' in array_pos_path):
            mic_pos = mic_pos_relative2center + array_pos[0, :] # (nch, 3)
            array_pos = array_pos[0, :] #(3, )
        elif ('task5' in array_pos_path) | ('task6' in array_pos_path):
            mic_pos = mic_pos_relative2center[np.newaxis, :, :] + array_pos[:, np.newaxis, :] # (npoint, nch, 3)
 
        traj_pts = []
        traj_pts_in_sample = []
        for src_idx in range(len(src_pos_path)):
            file = src_pos_path[src_idx]
            df = pd.read_csv(file, sep='\t')
            source_pos = np.stack((df['x'].values, df['y'].values, df['z'].values), axis=-1)
            traj_pts.append(source_pos)
            traj_pts_in_sample.append(
                np.array([np.interp(t, timestamps, source_pos[:, i]) for i in range(3)]).transpose())
        traj_pts = np.stack(traj_pts).transpose(1, 2, 0)
        traj_pts_in_sample = np.stack(traj_pts_in_sample).transpose(1, 2, 0)

        if len(mic_pos.shape) == 2:
            mic_pos = np.tile(mic_pos[np.newaxis, :, :], (traj_pts.shape[0], 1, 1))
        elif len(mic_pos.shape) == 3:
            pass
        else:
            raise Exception('shape of mic_pos is out of range~')
        nsource = traj_pts.shape[-1]
        nmic = mic_pos.shape[1]
        nsample = t.shape[0]
        corr_diff = np.tile(traj_pts[:, np.newaxis, :, :], (1, nmic, 1, 1)) - np.tile(mic_pos[:, :, :, np.newaxis], (1, 1, 1, nsource))
        dist = np.sqrt(np.sum(corr_diff**2, axis=2))  # (npoint,3,nsource)-(nch,3)=(nnpoint,nch,3,nsource)
        re_dist = dist[:, 1:, :] - np.tile(dist[:, 0:1, :], (1, nmic - 1, 1))  # (npoint,nch-1,nsource)
        TDOA = re_dist / sound_speed  # (npoint,nch-1,nsource)
        TDOA_in_sample = np.zeros((nsample, TDOA.shape[1], nsource))  # (nsample,nch-1,nsource)
        for source_idx in range(nsource):
            for ch_idx in range(TDOA.shape[1]):
                TDOA_in_sample[:, ch_idx, source_idx] = np.interp(t, timestamps, TDOA[:, ch_idx, source_idx])
        return_data.append(TDOA_in_sample)

        if vad_path is not None:
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
            return_data.append(vad)

        return return_data
    
    def select_micpairs(self, arrays, nmic_selected, mic_dist_range):
        mic_idxes_selected = {}
        mic_pos_selected = {}
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
            mic_idxes_selected[array], mic_pos_selected[array] = self._select_microphone_pairs(mic_poss[array], nmic_selected, mic_dist_range)                       
        return mic_idxes_selected, mic_pos_selected

    def _select_microphone_pairs(self, mic_poss, nmic_selected, mic_dist_range):
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
            if ( (dist >= mic_dist_range[0]) & (dist <= mic_dist_range[1]) ):
                mic_pair_idxes_selected += [mic_pair_idxes]
                mic_pos_selected += [mic_pos]
        assert (not mic_pair_idxes_selected)==False, f'No microphone pairs satisfy the microphone distance range {mic_dist_range}'
        return mic_pair_idxes_selected, mic_pos_selected


if __name__ == '__main__':
    pass
