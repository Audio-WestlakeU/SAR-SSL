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


class RealMANDataset(RealMicSigDataset):
    """ Refs: RealMAN: A Real-Recorded and Annotated Microphone Array Dataset for Dynamic Speech Enhancement and Localization
	    Urls: https://github.com/Audio-WestlakeU/RealMAN
	"""
    def __init__(self, 
                 data_dir: str, 
                 T: float, 
                 fs: int, 
                 stage: str,  # 'train', 'val', 'test' 
                 tasks: List[int] = ['static'], 
                 arrays: List[str] = ['high'], 
                 mic_dist_range: List[float] = [0.03, 0.20], 
                 nmic_selected: int = 2, 
                 prob_mode:List[str] = ['duration', 'micpair'],
                 dataset_sz = None, 
                 remove_spkoverlap = False,
                 sound_speed: float = 343.0):

        self.mic_idxes_selected = self.select_micpairs(arrays, nmic_selected, mic_dist_range)
        self.data_items, self.data_probs_cumsum = self.get_items_probs(data_dir, tasks, arrays, self.mic_idxes_selected, T, stage, prob_mode)

        self.dataset_sz = len(self.data_items) if dataset_sz is None else dataset_sz
        self.T = T
        self.fs = fs
        self.mic_dist_range = mic_dist_range
        self.nmic_selected = nmic_selected
        self.remove_spkoverlap = remove_spkoverlap
        self.sound_speed = sound_speed

    def get_items_probs(self, data_dir, tasks, arrays, mic_idxes_selected, duration_min_limit, stage, prob_mode):
        dataset_split = {'train':[
                                    'LivingRoom1',  #'scene_0427_教师公寓_客厅',
                                    'LivingRoom3'   #'scene_0703_Hotel',
                                    'LivingRoom4',  #'scene_0704_Loft',
                                    'LivingRoom5',  #'scene_0706_民宿',
                                    'LivingRoom6',  #'scene_0707_民宿2',
                                    'LivingRoom7',  #'scene_0709_轰趴馆',
                                    'LivingRoom8',  #'scene_0714_学生宿舍',
                                    'Classroom1',   #'scene_0418_212',
                                    'Classroom2',   #'scene_0718_201',
                                    'Classroom3',   #'scene_0803_小学音乐教室',
                                    'OfficeRoom1',  #'scene_0305_TeacherOffice',
                                    'OfficeRoom3',  #'scene_0420_210',
                                    'OfficeRoom4',  #'scene_0726_KTV',
                                    'OfficeLobby',  #'scene_0411_1号门大厅',
                                    'Library',      #'scene_0717_Library',
                                    'Auditorium',   #'scene_0730_dalitang',
                                    'BadmintonCourt1',  #'scene_0417_羽毛球馆',
                                    'BadmintonCourt2',  #'scene_0308_badminton_court',
                                    'BasketballCourt2', #'scene_0311_basketball'
                                    'SunkenPlaza1',     #'scene_0724_C19下沉广场',
                                    'Gym',              # 'scene_0427_健身房',
                                    'Cafeteria1',       #'scene_0409_canteen',
                                    'UndergroundParking1',  #'scene_0802_车库',
                                    'UndergroundParking2',  #'scene_0306_A2park',
                                    'Car-Gasoline',     #'scene_0625_油车',
                                    'Car-Electric', #'scene_0630_电车',
                                    'Bus-Electric', #'scene_0831_公交车',                                        
                                ],
                         'val': [
                                    'LivingRoom2',      #'scene_0427_教师公寓_小房间'
                                    'OfficeRoom2',      # 'scene_0305_LabOffice'
                                    'BasketballCourt1', # 'scene_0415_操场'
                                    'Market',           # 'scene_0516_菜市场'
                                    'Cafeteria3',       # 'scene_0307_c18two',
                                ],
                         'test': []
                         }
        
        data_items = []
        data_probs = []
        for scene in dataset_split[stage]:
            for task in tasks:
                for array in arrays:
                    wavs = Path(data_dir).rglob('*/ma_speech/' + scene + '/' + task + '/*/*CH0.flac')
                                
                    for wav_path in wavs:
                        audio_duration = soundfile.info(wav_path).duration
                        if audio_duration >= duration_min_limit:
                            data_prob = 1

                            # according to utterance duration
                            if 'duration' in prob_mode:
                                data_prob *= audio_duration
                            
                            # according to microphone pairs
                            nmicpair = len(mic_idxes_selected[array])
                            for micpair_idx in range(nmicpair):
                                data_items.append((wav_path, mic_idxes_selected[array][micpair_idx]))
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
        nmic = len(mic_idxes_selected)
        mic_signals = []
        for mic_idx in range(nmic):
            data_path = data_path.parent / data_path.name.replace('.CH0.wav', f'.CH{mic_idxes_selected[mic_idx]}.wav')
            if (st==None) & (ed==None):
                mic_signal, _ = soundfile.read(data_path, dtype='float32')
            else:
                mic_signal, _ = soundfile.read(data_path, start=st, stop=ed, dtype='float32')
            mic_signals += [mic_signal]
            
        return np.array(mic_signals).transpose(1, 0)
    
    def select_micpairs(self, arrays, nmic_selected, mic_dist_range):
        mic_idxes_selected = {}

        mic_poss = {'high': self.high_resolution_array(),
                    }
        for array in arrays:
            mic_idxes_selected[array], _ = select_microphone_pairs(mic_poss[array], nmic_selected, mic_dist_range)                       

        return mic_idxes_selected

    def circular_cm_array(self, radius: float, mic_num: int) -> np.ndarray:
        # circular array with central microphone
        pos_rcv = np.zeros((mic_num, 3))
        pos_rcv_c = self.circular_array_geometry(radius=radius, mic_num=mic_num - 1)
        pos_rcv[1:, :] = pos_rcv_c

        return pos_rcv

    def high_resolution_array(self) -> np.array:
        R = 0.03
        pos_rcv = np.zeros((32, 3))
        pos_rcv[1:9, :] = self.circular_array_geometry(radius=R, mic_num=8)
        pos_rcv[9:17, :] = self.circular_array_geometry(radius=R * 2, mic_num=8)
        pos_rcv[17:25, :] = self.circular_array_geometry(radius=R * 3, mic_num=8)
        pos_rcv[25, :] = np.array([-R * 4, 0, 0])
        pos_rcv[26, :] = np.array([R * 4, 0, 0])
        pos_rcv[27, :] = np.array([R * 5, 0, 0])

        L = 0.045
        pos_rcv[28, :] = np.array([0, 0, L * 2])
        pos_rcv[29, :] = np.array([0, 0, L ])
        pos_rcv[30, :] = np.array([0, 0, -L])
        pos_rcv[31, :] = np.array([0, 0, -L * 2])

        return pos_rcv

    # def low_resolution_array() -> np.array:
    #     R = 0.03
    #     L = 0.03
    #     pos_rcv = np.zeros((16, 3))
    #     pos_rcv[:8, :] = self.circular_array_geometry(radius=R, mic_num=8)
    #     pos_rcv[9, :] = np.array([L * 2, 0, 0])
    #     pos_rcv[10, :] = np.array([L * 3, 0, 0])
    #     pos_rcv[11, :] = np.array([L * 4, 0, 0])
    #     pos_rcv[12, :] = np.array([-L * 2, 0, 0])
    #     pos_rcv[13, :] = np.array([-L * 3, 0, 0])

    #     pos_rcv[14, :] = np.array([0, L * 2, 0])
    #     pos_rcv[15, :] = np.array([0, -L * 2, 0])
    #     return pos_rcv
    
    def circular_array_geometry(self, radius: float, mic_num: int) -> np.ndarray:

        pos_rcv = np.empty((mic_num, 3))
        v1 = np.array([1, 0, 0])  
        v1 = self.normalize(v1)  
        angles = np.arange(0, 2 * np.pi, 2 * np.pi / mic_num)
        for idx, angle in enumerate(angles):
            x = v1[0] * np.cos(angle) - v1[1] * np.sin(angle)
            y = v1[0] * np.sin(angle) + v1[1] * np.cos(angle)
            pos_rcv[idx, :] = self.normalize(np.array([x, y, 0]))
        pos_rcv *= radius

        return pos_rcv
   
    def normalize(self, vec: np.ndarray) -> np.ndarray:
        # get unit vector
        vec = vec / norm(vec)
        vec = vec / norm(vec)
        assert np.isclose(norm(vec), 1), 'norm of vec is not close to 1'

        return vec


class RealMANDatasetOri(RealMicSigDataset):
    """ 
	    urls:  https://github.com/Audio-WestlakeU/RealMAN
	"""
    def __init__(self, 
                 data_dir: str, 
                 T: float, 
                 fs: int, 
                 stage: str,  # 'train', 'val', 'test' 
                 tasks: List[int] = ['static'], 
                 arrays: List[str] = ['high'], 
                 mic_dist_range: List[float] = [0.03, 0.20], 
                 nmic_selected: int = 2, 
                 prob_mode:List[str] = ['duration', 'micpair'],
                 dataset_sz = None, 
                 remove_spkoverlap = False,
                 sound_speed: float = 343.0):

        self.mic_idxes_selected = self.select_micpairs(arrays, nmic_selected, mic_dist_range)
        self.data_items, self.data_probs_cumsum = self.get_items_probs(data_dir, tasks, arrays, self.mic_idxes_selected, T, stage, prob_mode)

        self.dataset_sz = len(self.data_items) if dataset_sz is None else dataset_sz
        self.T = T
        self.fs = fs
        self.mic_dist_range = mic_dist_range
        self.nmic_selected = nmic_selected
        self.remove_spkoverlap = remove_spkoverlap
        self.sound_speed = sound_speed

    def get_items_probs(self, data_dir, tasks, arrays, mic_idxes_selected, duration_min_limit, stage, prob_mode):
        dataset_split = {'train':[ 'scene_0417_羽毛球馆',
                                'scene_0427_健身房',
                                'scene_0409_canteen',
                                'scene_0427_教师公寓_客厅',
                                'scene_0704_Loft',
                                'scene_0703_Hotel',
                                'scene_0418_212',
                                'scene_0420_210',
                                'scene_0306_A2park',
                                'scene_0411_1号门大厅',
                                'scene_0308_badminton_court',
                                'scene_0625_油车',
                                'scene_0630_电车',
                                'scene_0706_民宿',
                                'scene_0707_民宿2',
                                'scene_0709_轰趴馆',
                                'scene_0714_学生宿舍',
                                'scene_0717_Library',
                                'scene_0718_201',
                                'scene_0724_C19下沉广场',
                                'scene_0726_KTV',
                                'scene_0730_dalitang',
                                'scene_0802_车库',
                                'scene_0803_小学音乐教室',
                                'scene_0831_公交车',
                                'scene_0305_TeacherOffice',
                                'scene_0311_basketball'],
                         'val':['scene_0305_LabOffice',
                                'scene_0307_c18two',
                                'scene_0415_操场',
                                'scene_0427_教师公寓_小房间',
                                'scene_0516_菜市场'],
                        'test':[]
                         }
        
        data_items = []
        data_probs = []
        for scene in dataset_split[stage]:
            for task in tasks:
                task_dir = data_dir + '/' + scene + '/' + task
                for spk in os.listdir( task_dir ):
                    for array in arrays:
                        recoring_dir = task_dir + '/' + spk + '/' + array + '/' + 'record'
                        for uttr in os.listdir( recoring_dir ):
                            uttr_path = recoring_dir + '/' + uttr
                            audio_duration = soundfile.info(uttr_path).duration
                            if audio_duration >= duration_min_limit:
                                data_prob = 1

                                # according to utterance duration
                                if 'duration' in prob_mode:
                                    data_prob *= audio_duration
                                
                                # according to microphone pairs
                                nmicpair = len(mic_idxes_selected[array])
                                for micpair_idx in range(nmicpair):
                                    data_items.append((uttr_path, mic_idxes_selected[array][micpair_idx]))
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

        mic_poss = {'high': self.high_resolution_array(),
                    }
        for array in arrays:
            mic_idxes_selected[array], _ = select_microphone_pairs(mic_poss[array], nmic_selected, mic_dist_range)                       

        return mic_idxes_selected

    def circular_cm_array(self, radius: float, mic_num: int) -> np.ndarray:
        # circular array with central microphone
        pos_rcv = np.zeros((mic_num, 3))
        pos_rcv_c = self.circular_array_geometry(radius=radius, mic_num=mic_num - 1)
        pos_rcv[1:, :] = pos_rcv_c

        return pos_rcv

    def high_resolution_array(self) -> np.array:
        R = 0.03
        pos_rcv = np.zeros((32, 3))
        pos_rcv[:8, :] = self.circular_array_geometry(radius=R, mic_num=8)
        pos_rcv[8:16, :] = self.circular_array_geometry(radius=R * 2, mic_num=8)
        pos_rcv[16:24, :] = self.circular_array_geometry(radius=R * 3, mic_num=8)
        pos_rcv[25, :] = np.array([R * 4, 0, 0])
        pos_rcv[26, :] = np.array([R * 5, 0, 0])
        pos_rcv[27, :] = np.array([-R * 4, 0, 0])

        L = 0.045
        pos_rcv[28, :] = np.array([0, 0, L])
        pos_rcv[29, :] = np.array([0, 0, L * 2])
        pos_rcv[30, :] = np.array([0, 0, -L])
        pos_rcv[31, :] = np.array([0, 0, -L * 2])

        return pos_rcv

    def low_resolution_array() -> np.array:
        R = 0.03
        L = 0.03
        pos_rcv = np.zeros((16, 3))
        pos_rcv[:8, :] = self.circular_array_geometry(radius=R, mic_num=8)
        pos_rcv[9, :] = np.array([L * 2, 0, 0])
        pos_rcv[10, :] = np.array([L * 3, 0, 0])
        pos_rcv[11, :] = np.array([L * 4, 0, 0])
        pos_rcv[12, :] = np.array([-L * 2, 0, 0])
        pos_rcv[13, :] = np.array([-L * 3, 0, 0])

        pos_rcv[14, :] = np.array([0, L * 2, 0])
        pos_rcv[15, :] = np.array([0, -L * 2, 0])
        return pos_rcv
    
    def circular_array_geometry(self, radius: float, mic_num: int) -> np.ndarray:

        pos_rcv = np.empty((mic_num, 3))
        v1 = np.array([1, 0, 0])  
        v1 = self.normalize(v1)  
        angles = np.arange(0, 2 * np.pi, 2 * np.pi / mic_num)
        for idx, angle in enumerate(angles):
            x = v1[0] * np.cos(angle) - v1[1] * np.sin(angle)
            y = v1[0] * np.sin(angle) + v1[1] * np.cos(angle)
            pos_rcv[idx, :] = self.normalize(np.array([x, y, 0]))
        pos_rcv *= radius

        return pos_rcv
   
    def normalize(self, vec: np.ndarray) -> np.ndarray:
        # get unit vector
        vec = vec / norm(vec)
        vec = vec / norm(vec)
        assert np.isclose(norm(vec), 1), 'norm of vec is not close to 1'

        return vec


class LOCATADataset(RealMicSigDataset):
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
                 tasks: List[int] = [1,], 
                 arrays: List[str] = ['dicit', 'benchmark2', 'eigenmike'], 
                 mic_dist_range: List[float] = [0.03, 0.20], 
                 nmic_selected: int = 2, 
                 prob_mode:List[str] = ['duration', 'micpair'],
                 dataset_sz = None, 
                 remove_spkoverlap = False,
                 sound_speed: float = 343.0):
        
        self.room_sz = np.array([7.1, 9.8, 3])

        self.mic_idxes_selected = self.select_micpairs(arrays, nmic_selected, mic_dist_range)
        self.data_items, self.data_probs_cumsum = self.get_items_probs(data_dir, tasks, arrays, self.mic_idxes_selected, T, stage, prob_mode)

        self.dataset_sz = len(self.data_items) if dataset_sz is None else dataset_sz
        self.T = T
        self.fs = fs
        self.mic_dist_range = mic_dist_range
        self.nmic_selected = nmic_selected
        self.remove_spkoverlap = remove_spkoverlap
        self.sound_speed = sound_speed

    def get_items_probs(self, data_dir, tasks, arrays, mic_idxes_selected, duration_min_limit, stage, prob_mode):
        dataset_split = {'train':['eval', 'dev'],
                         'val':[],
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
                            data_path = os.path.join(task_path, recording, array, 'audio_array_' + array + '.wav')
                            audio_duration = soundfile.info(data_path).duration
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
                                    data_items.append((data_path, mic_idxes_selected[array][micpair_idx]))
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


    
class MCWSJDataset(RealMicSigDataset):
    """ 
	Refs: The multi-channel Wall Street Journal audio visual corpus (MC-WSJ-AV): specification and initial experiments
	Code: 
	URL: https://catalog.ldc.upenn.edu/LDC2014S03
    Format: ./MC-WSJ-AV/audio/<task>/T<#1>[_T<#2>]/<mic_type>/{adap|5k\20k}/*.wav
            <tasks > defining the task, i.e. stat (single static), move (single moving) or olap (two overlapping )
            T<#> defining the participant and his/her number #
            <mic_type> defining the microphone type, e.g. array1/2, headset1/2 or lapel1/2
            array1/2: 8-ch circular array with diameter of 20 cm
	"""
    def __init__(self,  
                 data_dir: str, 
                 T: float, 
                 fs: int, 
                 stage: str,  # 'train', 'val', 'test' 
                 tasks: List[int] = ['stat'], 
                 arrays: List[str] = ['array1', 'array2'], 
                 mic_dist_range: List[float] = [0.03, 0.20], 
                 nmic_selected: int = 2, 
                 prob_mode:List[str] = ['duration', 'micpair'], 
                 dataset_sz = None,
                 remove_spkoverlap = False,
                 sound_speed: float = 343.0):
        
        self.mic_idxes_selected = self.select_micpairs(arrays, nmic_selected, mic_dist_range)
        self.data_items, self.data_probs_cumsum = self.get_items_probs(data_dir, tasks, arrays, self.mic_idxes_selected, T, stage, prob_mode)

        self.dataset_sz = len(self.data_items) if dataset_sz is None else dataset_sz
        self.T = T
        self.fs = fs
        self.mic_dist_range = mic_dist_range
        self.nmic_selected = nmic_selected
        self.remove_spkoverlap = remove_spkoverlap
        
    def get_items_probs(self, data_dir, tasks, arrays, mic_idxes_selected, duration_min_limit, stage, prob_mode):

        dataset_split = {'train':['Dev', 'Eval'], 
                         'val':[], 
                         'test':[]}
        dataset_split_ratio = { 'train':    [0, 1],
                                'val':      [1, 1],
                                'test':     [1, 1]}
        data_items = []
        data_probs = []
        for ds in dataset_split[stage]:
            for task in tasks:
                task_dir = Path(data_dir) / ('MC_WSJ_AV_' + ds) / 'audio' / task 
                for spk in os.listdir(task_dir):
                    for array in arrays:
                        array_dir = task_dir / spk / array
                        for dir_sub in os.listdir(array_dir):
                            dir_sub = array_dir / dir_sub
                            uttrs = list(dir_sub.rglob('*-1_T*.wav'))
                            uttrs = uttrs[int(len(uttrs)*dataset_split_ratio[stage][0]):int(len(uttrs)*dataset_split_ratio[stage][1])]

                            if 'duration' in prob_mode:
                                probs_uttr = [soundfile.info(wp).duration for wp in uttrs] 
                            else:
                                probs_uttr = [1 for wp in uttrs]

                            nmicpair = len(mic_idxes_selected[array])
                            for idx in range(len(uttrs)):
                                if soundfile.info(uttrs[idx]).duration >= duration_min_limit:
                                    for micpair_idx in range(nmicpair):
                                        data_items.append((uttrs[idx], mic_idxes_selected[array][micpair_idx]))
                                        if 'micpair' in prob_mode:
                                            data_probs.append(probs_uttr[idx])
                                        else:
                                            data_probs.append(probs_uttr[idx]/nmicpair)
        data_probs_cumsum = []
        if len(data_probs)>0:
            data_probs_sum = sum(data_probs)
            data_probs = [prob/data_probs_sum for prob in data_probs]
            data_probs_cumsum = np.cumsum(data_probs, dtype=np.float32)
            data_probs_cumsum[-1] = 1

        return data_items, data_probs_cumsum

    def read_micsig(self, data_path, st=None, ed=None, mic_idxes_selected=None):
        nmic = len(mic_idxes_selected)
        mic_signals = []
        for mic_idx in range(nmic):
            data_path = data_path.parent / data_path.name.replace('-1_T.wav', f'-{mic_idxes_selected[mic_idx]+1}_T.wav')
            if (st==None) & (ed==None):
                mic_signal, _ = soundfile.read(data_path, dtype='float32')
            else:
                mic_signal, _ = soundfile.read(data_path, start=st, stop=ed, dtype='float32')
            mic_signals += [mic_signal]
            
        return np.array(mic_signals).transpose(1, 0)
    
    def select_micpairs(self, arrays, nmic_selected, mic_dist_range):
        mic_idxes_selected = {}
        mic_poss = {'array1': np.array((
                            ( 0.100,  0.000, 0.000),
                            ( 0.071,  0.071, 0.000),
                            ( 0.000,  0.100, 0.000),
                            (-0.071,  0.071, 0.000),
                            (-0.100,  0.000, 0.000),
                            (-0.071, -0.071, 0.000),
                            ( 0.000, -0.100, 0.000),
                            ( 0.071, -0.071, 0.000))),
                    'array2': np.array((
                            ( 0.100,  0.000, 0.000),
                            ( 0.071,  0.071, 0.000),
                            ( 0.000,  0.100, 0.000),
                            (-0.071,  0.071, 0.000),
                            (-0.100,  0.000, 0.000),
                            (-0.071, -0.071, 0.000),
                            ( 0.000, -0.100, 0.000),
                            ( 0.071, -0.071, 0.000)))
                        }
        for array in arrays:
            mic_idxes_selected[array], _ = select_microphone_pairs(mic_poss[array], nmic_selected, mic_dist_range)                       

        return mic_idxes_selected


class LibriCSSDataset(RealMicSigDataset):
    """"
        URL: https://github.com/chenzhuo1011/libri_css
    """
    def __init__(
            self,
            data_dir: str,
            T: float,
            fs: int,
            stage: str,  # 'train', 'val', 'test' 
            tasks: List[int] = ['overlap_ratio_0.0_*'],  # ,'overlap_ratio_10.0_*','overlap_ratio_20.0_*','overlap_ratio_30.0_*','overlap_ratio_40.0_*'
            arrays: List[str] = ['array'],
            mic_dist_range: List[float] = [0.03, 0.20],
            nmic_selected: int = 2,
            prob_mode: List[str] = ['duration', 'micpair'],
            dataset_sz = None,
            remove_spkoverlap = False,
            sound_speed: float = 343.0):

        self.mic_idxes_selected = self.select_micpairs(arrays, nmic_selected, mic_dist_range)
        self.data_items, self.data_probs_cumsum = self.get_items_probs(data_dir, tasks, arrays, self.mic_idxes_selected, T, stage, prob_mode)

        self.dataset_sz = len(self.data_items) if dataset_sz is None else dataset_sz
        self.T = T
        self.fs = fs
        self.mic_dist_range = mic_dist_range
        self.nmic_selected = nmic_selected
        self.remove_spkoverlap = remove_spkoverlap

    def get_items_probs(self, data_dir, tasks, arrays, mic_idxes_selected, duration_min_limit, stage, prob_mode):
        data_dir = Path(data_dir).expanduser() / 'exp' / 'data' / '7ch' / 'utterances'

        uttrs = []
        for task in tasks:
            ovlp_dirs = list(data_dir.glob(task))
            for ovlp_dir in ovlp_dirs:
                uttrs += list(ovlp_dir.rglob('*.wav'))
        uttrs.sort()

        rng = np.random.default_rng(2024)
        rng.shuffle(uttrs)
        rng.shuffle(uttrs)
        dataset_split = {
            'train': [uttrs[:int(len(uttrs) * 1)]],
            'val': [uttrs[int(len(uttrs) * 1): int(len(uttrs) * 1)]],
            'test': [uttrs[int(len(uttrs) * 1):]],
        }

        data_items = []
        data_probs = []
        for ds_uttrs in dataset_split[stage]:
            if 'duration' in prob_mode:
                probs_uttr = [soundfile.info(wp).duration for wp in ds_uttrs]  
            else:
                probs_uttr = [1 for wp in ds_uttrs]

            # if 'room' in prob_mode:
            #     sum_probs_uttr = sum(probs_uttr)
            #     probs_uttr = [prob / sum_probs_uttr for prob in probs_uttr]
            
            for idx in range(len(ds_uttrs)):
                if soundfile.info(ds_uttrs[idx]).duration >= duration_min_limit:
                    for array in arrays:
                        nmicpair = len(mic_idxes_selected[array])
                        for micpair_idx in range(nmicpair):
                            data_items.append((ds_uttrs[idx], mic_idxes_selected[array][micpair_idx]))
                            if 'micpair' in prob_mode:
                                data_probs.append(probs_uttr[idx])
                            else:
                                data_probs.append(probs_uttr[idx] / nmicpair)
        data_probs_cumsum = []
        if len(data_probs)>0:
            data_probs_sum = sum(data_probs)
            data_probs = [prob / data_probs_sum for prob in data_probs]
            data_probs_cumsum = np.cumsum(data_probs, dtype=np.float32)
            data_probs_cumsum[-1] = 1

        return data_items, data_probs_cumsum

    def read_micsig(self, data_path, st=None, ed=None, mic_idxes_selected=None):
        if (st == None) & (ed == None):
            mic_signal, _ = soundfile.read(data_path, dtype='float32')
        else:
            mic_signal, _ = soundfile.read(data_path, start=st, stop=ed, dtype='float32')

        mic_signal = mic_signal[..., mic_idxes_selected] 

        return mic_signal

    def select_micpairs(self, arrays, nmic_selected, mic_dist_range):
        pos_rcv = np.zeros((7, 3)) 
        v1 = np.array([1, 0, 0]) 
        angles = np.arange(0, 2 * np.pi, 2 * np.pi / 6)
        for idx, angle in enumerate(angles):
            x = v1[0] * np.cos(angle) - v1[1] * np.sin(angle)
            y = v1[0] * np.sin(angle) + v1[1] * np.cos(angle)
            vec = np.array([x, y, 0])
            vec = vec / np.linalg.norm(vec)
            vec = vec / np.linalg.norm(vec)
            pos_rcv[idx + 1, :] = vec
        pos_rcv *= 0.0425

        mic_idxes_selected = {}
        mic_poss = {'array': pos_rcv}
        for array in arrays:
            mic_idxes_selected[array], _ = select_microphone_pairs(mic_poss[array], nmic_selected, mic_dist_range)

        return mic_idxes_selected


class AMIDataset(RealMicSigDataset):
    """
        URL: https://groups.inf.ed.ac.uk/ami/download/; https://www.openslr.org/16/
        Scenario Meetings: Edinburgh (ES), IDIAP (IS), TNO (TS)
        Non Scenario Meetings: Edinburgh (EN), ISSCO-IDIAP (IB), IDIAP (IN)
        For scenario meetings, 1 day-recording session is divided into four [a, b, c, d] 1-hour meetings. 
        Selecting ES2008 meeting session together with 'a' below allows you to get signals for ES2008a meeting.
    """
    def __init__(self,  
                 data_dir: str, 
                 T: float, 
                 fs: int, 
                 stage: str,  # 'train', 'val', 'test' 
                 tasks: List[int] = ['ScenarioMeetings', 'NonScenarioMeetings'], 
                 arrays: List[str] = ['Array1'], 
                 mic_dist_range: List[float] = [0.03, 0.20], 
                 nmic_selected: int = 2, 
                 prob_mode:List[str] = ['duration', 'micpair'], 
                 dataset_sz = None,
                 remove_spkoverlap = False,
                 sound_speed: float = 343.0):
        
        self.remove_spkoverlap = remove_spkoverlap      
        self.mic_idxes_selected = self.select_micpairs(arrays, nmic_selected, mic_dist_range)
        self.data_items, self.data_probs_cumsum = self.get_items_probs(data_dir, tasks, arrays, self.mic_idxes_selected, T, stage, prob_mode)

        self.dataset_sz = len(self.data_items) if dataset_sz is None else dataset_sz
        self.T = T
        self.fs = fs
        self.mic_dist_range = mic_dist_range
        self.nmic_selected = nmic_selected

    def get_items_probs(self, data_dir, tasks, arrays, mic_idxes_selected, duration_min_limit, stage, prob_mode):

        dataset_split = {'train':['ES', 'IS', 'TS', 'EN', 'IB', 'IN'], 'val':[], 'test':[]}
        data_items = []
        data_probs = []
   
        for task in tasks:
            task_dir = Path(data_dir) / task 
            for session in os.listdir(task_dir):
                if (session[0:2] in  dataset_split[stage]):
                    wav_dir = task_dir / session / 'audio'
                    uttrs = []
                    for array in arrays:
                        uttrs += list(wav_dir.rglob(session[0:2]+'*.'+array+'-01.wav'))

                    if 'duration' in prob_mode:
                        probs_uttr = [soundfile.info(wp).duration for wp in uttrs] 
                    else:
                        probs_uttr = [1 for wp in uttrs]

                    nmicpair = len(mic_idxes_selected[array])
                    for idx in range(len(uttrs)):
                        if soundfile.info(uttrs[idx]).duration >= duration_min_limit:
                            for micpair_idx in range(nmicpair):
                                data_items.append((uttrs[idx], mic_idxes_selected[array][micpair_idx]))
                                if 'micpair' in prob_mode:
                                    data_probs.append(probs_uttr[idx])
                                else:
                                    data_probs.append(probs_uttr[idx]/nmicpair)
        data_probs_cumsum = []
        if len(data_probs)>0:
            data_probs_sum = sum(data_probs)
            data_probs = [prob/data_probs_sum for prob in data_probs]
            data_probs_cumsum = np.cumsum(data_probs, dtype=np.float32)
            data_probs_cumsum[-1] = 1

        return data_items, data_probs_cumsum

    def read_micsig(self, data_path, st=None, ed=None, mic_idxes_selected=None):
        nmic = len(mic_idxes_selected)
        mic_signals = []
        for mic_idx in range(nmic):
            data_path = data_path.parent / data_path.name.replace('-01.wav', f'-0{mic_idxes_selected[mic_idx]+1}.wav')
            if (st==None) & (ed==None):
                mic_signal, _ = soundfile.read(data_path, dtype='float32')
            else:
                mic_signal, _ = soundfile.read(data_path, start=st, stop=ed, dtype='float32')
            if len(mic_signal.shape) == 2:
                mic_signal = mic_signal[:, 0]
            mic_signals += [mic_signal]

        return np.array(mic_signals).transpose(1, 0)
    
    def select_micpairs(self, arrays, nmic_selected, mic_dist_range):
        nmic = 8
        mic_idxes_selected = {}
        for array in arrays:
            mic_idxes_selected[array] = []
            for mic_idxes in itertools.permutations(range(nmic), nmic_selected):
                mic_idxes_selected[array] += [mic_idxes] # no array size provided, and use all microphone pairs
        
        # mic_idxes_selected = {}
        # mic_poss = {'Array1': 1*np.array((
        #                     ( 0.100,  0.000, 0.000),
        #                     ( 0.071,  0.071, 0.000),
        #                     ( 0.000,  0.100, 0.000),
        #                     (-0.071,  0.071, 0.000),
        #                     (-0.100,  0.000, 0.000),
        #                     (-0.071, -0.071, 0.000),
        #                     ( 0.000, -0.100, 0.000),
        #                     ( 0.071, -0.071, 0.000)))}
        # for array in arrays:
        #     mic_idxes_selected[array], _ = select_microphone_pairs(mic_poss[array], nmic_selected, mic_dist_range)                       


        return mic_idxes_selected

class AISHELL4Dataset(RealMicSigDataset):
    """
        URL: https://www.aishelltech.com/aishell_4
    """

    def __init__(self, 
                 data_dir: str, 
                 T: float, 
                 fs: int, 
                 stage: str,  # 'train', 'val', 'test' 
                 tasks: List[int] = [None], 
                 arrays: List[str] = ['array1'], 
                 mic_dist_range: List[float] = [0.03, 0.20], 
                 nmic_selected: int = 2, 
                 prob_mode:List[str] = ['duration', 'micpair'],
                 dataset_sz = None, 
                 remove_spkoverlap = False,
                 sound_speed: float = 343.0):
        
        self.remove_spkoverlap = remove_spkoverlap
        self.mic_idxes_selected = self.select_micpairs(arrays, nmic_selected, mic_dist_range)
        self.data_items, self.data_probs_cumsum = self.get_items_probs(data_dir, tasks, arrays, self.mic_idxes_selected, T, stage, prob_mode)

        self.dataset_sz = len(self.data_items) if dataset_sz is None else dataset_sz
        self.T = T
        self.fs = fs
        self.mic_dist_range = mic_dist_range
        self.nmic_selected = nmic_selected

    def get_items_probs(self, data_dir, tasks, arrays, mic_idxes_selected, duration_min_limit, stage, prob_mode):

        dataset_range = {   'train':   {'train_L': ['L_R001', 'L_R002'], 
                                        'train_M':['M_R001', 'M_R002'], 
                                        'train_S':['S_R001'],
                                        'test':['S_R003', 'S_R004', 'L_R003', 'L_R004']}, 
                            'val':     {'train_L':[], 
                                        'train_M':[], 
                                        'train_S':[],
                                        'test':['M_R003']}, 
                            'test':    {'test':[]}}
        # dataset_split_ratio = { 'train':    [0, 1],
                                # 'val':      [1, 1],
                                # 'test':     [1, 1]}
        if self.remove_spkoverlap:
            data_dir = Path(data_dir).expanduser()
            session_trans_files = []
            for ds in dataset_range[stage].keys():
                for room in dataset_range[stage][ds]:
                    stfs = list((data_dir / ds).rglob('*'+room+'*.TextGrid'))
                    session_trans_files += stfs
                # stfs = list((data_dir / ds).rglob('*.TextGrid'))
                # rng = np.random.default_rng(2024)
                # rng.shuffle(stfs)
                # rng.shuffle(stfs)
                # stfs = stfs[int(len(stfs)*dataset_split_ratio[stage][0]): int(len(stfs)*dataset_split_ratio[stage][1])]
                # session_trans_files += stfs

            data_items = []
            data_probs = []

            for sess_tran_file in session_trans_files:
                if sess_tran_file.name not in ['20200622_M_R002S07C01.TextGrid', '20200710_M_R002S06C01.TextGrid']:
                    tg_spks = textgrid.TextGrid.fromFile(sess_tran_file)
                    sentence_infos = []
                    for tg_spk in tg_spks:
                        tgs = list(tg_spk)
                        for tg in tgs:
                            if tg.mark != '':
                                sentence_infos += [tg]
                    sentence_infos.sort(key=lambda x: x.minTime)
                    audio_duration = soundfile.info(list(sess_tran_file.parent.parent.rglob(sess_tran_file.name.replace('.TextGrid', '.flac')))[0]).duration

                    # the latest end time of the previous sentence
                    etbts, etbt = [], 0.0
                    for i in range(0, len(sentence_infos)):
                        si = sentence_infos[i]
                        etbts.append(etbt)
                        if si.maxTime > etbt:
                            etbt = si.maxTime

                    # the start time of current sentence - the end time of the previous sentence >= duration_min_limit
                    si_selected = []  # [(start_time, end_time, duration)]
                    for i, si in enumerate(sentence_infos):
                        if i == len(sentence_infos) - 1:
                            continue
                        if (sentence_infos[i + 1].minTime - etbts[i] >= duration_min_limit) & (sentence_infos[i + 1].minTime<audio_duration):
                            si_selected.append((etbts[i], sentence_infos[i + 1].minTime, sentence_infos[i + 1].minTime - etbts[i]))

                    if 'duration' in prob_mode:
                        probs_uttr = [x[-1] for x in si_selected] 
                    else:
                        probs_uttr = [1 for x in si_selected]

                    # 给room时，归一化不同session的概率
                    # if 'room' in prob_mode:
                    #     sum_probs_uttr = sum(probs_uttr)
                    #     probs_uttr = [prob / sum_probs_uttr for prob in probs_uttr]

                    for array in arrays:
                        nmicpair = len(mic_idxes_selected[array])
                        for idx in range(len(si_selected)):
                            wav_files = list(sess_tran_file.parent.parent.rglob(sess_tran_file.name.replace('.TextGrid', '.flac')))
                            assert len(wav_files) == 1, len(wav_files)
                            wav_file = wav_files[0]

                            for micpair_idx in range(nmicpair):
                                data_items.append((wav_file, si_selected[idx], mic_idxes_selected[array][micpair_idx]))
                                if 'micpair' in prob_mode:
                                    data_probs.append(probs_uttr[idx])
                                else:
                                    data_probs.append(probs_uttr[idx] / nmicpair)

        else:
            dataset_split = {}
            uttrs = []
            for ds in dataset_range[stage].keys():
                for room in dataset_range[stage][ds]:
                    uttr_dir = Path(data_dir) / ds / 'wav'
                    uttrs += list(uttr_dir.rglob('*'+room+'*.flac'))
            uttrs.sort()
            rng = np.random.default_rng(2024)
            rng.shuffle(uttrs)
            rng.shuffle(uttrs)
            dataset_split[stage] = uttrs #[int(len(uttrs)*dataset_split_ratio[stage][0]): int(len(uttrs)*dataset_split_ratio[stage][1])]

            data_items = []
            data_probs = []
            for wav_path in dataset_split[stage]: 
                audio_duration = soundfile.info(wav_path).duration
                if audio_duration >= duration_min_limit:
                    # according to room
                    # according to utterance number
                    data_prob = 1

                    # according to utterance duration
                    if 'duration' in prob_mode:
                        data_prob *= audio_duration
                    
                    # according to microphone pairs
                    for array in arrays:
                        nmicpair = len(mic_idxes_selected[array])
                        for micpair_idx in range(nmicpair):
                            data_items.append((wav_path, mic_idxes_selected[array][micpair_idx]))
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
            try:
                mic_signals, _ = soundfile.read(data_path, start=st, stop=ed, dtype='float32')
            except:
                # mic_signals, _ = torchaudio.load(data_path, frame_offset=st, num_frames=ed-st, channels_first=False)
                # mic_signals = mic_signals.numpy()
                fs = soundfile.info(data_path).samplerate
                mic_signals, _ = librosa.load(data_path, sr=fs, offset=st/fs, duration=(ed-st)/fs, mono=False)
                mic_signals = mic_signals.transpose(1, 0)
        mic_signals = mic_signals[:, mic_idxes_selected]
        
        return mic_signals


    def select_micpairs(self, arrays, nmic_selected, mic_dist_range):
        mic_idxes_selected = {}
        mic_poss = {'array1': 0.5*np.array((
                            ( 0.100,  0.000, 0.000),
                            ( 0.071,  0.071, 0.000),
                            ( 0.000,  0.100, 0.000),
                            (-0.071,  0.071, 0.000),
                            (-0.100,  0.000, 0.000),
                            (-0.071, -0.071, 0.000),
                            ( 0.000, -0.100, 0.000),
                            ( 0.071, -0.071, 0.000)))}
        for array in arrays:
            mic_idxes_selected[array], _ = select_microphone_pairs(mic_poss[array], nmic_selected, mic_dist_range)                       

        return mic_idxes_selected
    
 
class M2MeTDataset(RealMicSigDataset):
    """"
        URL: https://www.openslr.org/119/
    """
    def __init__(
            self,
            data_dir: str,
            T: float,
            fs: int,
            stage: str,  # 'train', 'val', 'test' 
            tasks: List[int] = ['task'],
            arrays: List[str] = ['array'],
            mic_dist_range: List[float] = [0.03, 0.20],
            nmic_selected: int = 2,
            prob_mode: List[str] = ['duration', 'micpair'],
            dataset_sz = None, 
            remove_spkoverlap = False,
            sound_speed: float = 343.0):

        self.remove_spkoverlap = remove_spkoverlap
        self.mic_idxes_selected = self.select_micpairs(arrays, nmic_selected, mic_dist_range)
        self.data_items, self.data_probs_cumsum = self.get_items_probs(data_dir, tasks, arrays, self.mic_idxes_selected, T, stage, prob_mode)

        self.dataset_sz = len(self.data_items) if dataset_sz is None else dataset_sz
        self.T = T
        self.fs = fs
        self.mic_dist_range = mic_dist_range
        self.nmic_selsected = nmic_selected

    def get_items_probs(self, data_dir, tasks, arrays, mic_idxes_selected, duration_min_limit, stage, prob_mode):
        data_dir = Path(data_dir).expanduser()

        dataset_split = {   'train':    {'Train_Ali/Train_Ali_far': ['R0003', 'R0004', 'R0005', 'R0008', 'R0014', 'R0015', 'R0020', 
                                                                     'R1019', 'R1021', 'R2001', 'R2105', 'R2108'],
                                         'Eval_Ali/Eval_Ali_far': ['R8001', 'R8003', 'R8007', 'R8008', 'R8009'], 
                                         'Test_Ali/Test_Ali_far': ['R8004','R8005','R8008','R8009'],
                                         }, 
                            'val':      {'Train_Ali/Train_Ali_far': [],
                                         'Eval_Ali/Eval_Ali_far': [], 
                                         'Test_Ali/Test_Ali_far': ['R8002','R8006'],
                                         },  
                            'test':     {'Train_Ali/Train_Ali_far': [],
                                         'Eval_Ali/Eval_Ali_far': [], 
                                         'Test_Ali/Test_Ali_far': [],
                                         }}
        ds_dir_str = dataset_split[stage]

       
        
        data_items = []
        data_probs = []
        if self.remove_spkoverlap:
            session_trans_files = []
            for ds in ds_dir_str.keys():
                for room in ds_dir_str[ds]:
                    session_trans_files +=list((data_dir / f'{ds}/textgrid_dir').glob(room+'*.TextGrid'))
            
            for sess_tran_file in session_trans_files:
                tg_spks = textgrid.TextGrid.fromFile(sess_tran_file)
                sentence_infos = []
                for tg in tg_spks:
                    sentence_infos += list(tg)
                sentence_infos.sort(key=lambda x: x.maxTime)
                wav_dir = sess_tran_file.parent.parent / 'audio_dir'
                audio_duration = soundfile.info(list(wav_dir.glob(sess_tran_file.name.replace('.TextGrid', '*.wav')))[0]).duration

                # print(sentence_infos[-1], audio_duration)
                # assert sentence_infos[-1].maxTime <= audio_duration, 'error'
                
                sentence_infos.sort(key=lambda x: x.minTime)
                # the latest end time of the previous sentence
                etbts, etbt = [], 0.0
                for i in range(0, len(sentence_infos)):
                    si = sentence_infos[i]
                    etbts.append(etbt)
                    if si.maxTime > etbt:
                        etbt = si.maxTime

                # the start time of current sentence - the end time of the previous sentence >= duration_min_limit
                si_selected = []  # [(si, start_time, end_time, duration)]
                for i, si in enumerate(sentence_infos):
                    if i == len(sentence_infos) - 1:
                        continue
                    if (sentence_infos[i + 1].minTime - etbts[i] >= duration_min_limit) & (sentence_infos[i + 1].minTime<=audio_duration):
                        si_selected.append((etbts[i], sentence_infos[i + 1].minTime, sentence_infos[i + 1].minTime - etbts[i]))

                if 'duration' in prob_mode:
                    probs_uttr = [x[-1] for x in si_selected] 
                else:
                    probs_uttr = [1 for x in si_selected]

                # 给room时，归一化不同session的概率
                # if 'room' in prob_mode:
                #     sum_probs_uttr = sum(probs_uttr)
                #     probs_uttr = [prob / sum_probs_uttr for prob in probs_uttr]

                for array in arrays:
                    nmicpair = len(mic_idxes_selected[array])
                    for idx in range(len(si_selected)):
                        wav_files = list(wav_dir.glob(sess_tran_file.name.replace('.TextGrid', '*.wav')))
                        assert len(wav_files) == 1, len(wav_files)
                        wav_file = wav_files[0]

                        for micpair_idx in range(nmicpair):
                            # e.g. "/*/data/SenSig/AliMeeting/Eval_Ali/Eval_Ali_far/audio_dir/R8001_M8004_MS801.wav", (start_time, end_time, duration), (0, 1)
                            data_items.append((wav_file, si_selected[idx], mic_idxes_selected[array][micpair_idx]))
                            if 'micpair' in prob_mode:
                                data_probs.append(probs_uttr[idx])
                            else:
                                data_probs.append(probs_uttr[idx] / nmicpair)
        else:
            data_items = []
            data_probs = []
            wav_paths = []
            for ds in ds_dir_str.keys():
                for room in ds_dir_str[ds]:
                    wav_dir = data_dir / f'{ds}/audio_dir'
                    wav_paths += list(wav_dir.glob(room+'*.wav'))
                # wav_dir = Path(data_dir) / dataset_split[stage] / 'audio_dir'
                # wav_paths = wav_dir.glob('*.wav')
            # for wav_name in os.listdir(wav_dir): 
            #     wav_path = wav_dir / wav_name
            for wav_path in wav_paths:
                audio_duration = soundfile.info(wav_path).duration
                if audio_duration >= duration_min_limit:
                    # according to room
                    # according to utterance number
                    data_prob = 1

                    # according to utterance duration
                    if 'duration' in prob_mode:
                        data_prob *= audio_duration
                    
                    # according to microphone pairs
                    for array in arrays:
                        nmicpair = len(mic_idxes_selected[array])
                        for micpair_idx in range(nmicpair):
                            data_items.append((wav_path, mic_idxes_selected[array][micpair_idx]))
                            if 'micpair' in prob_mode:
                                data_probs.append(data_prob)
                            else:
                                data_probs.append(data_prob/nmicpair)
        data_probs_cumsum = []
        if len(data_probs)>0:
            data_probs_sum = sum(data_probs)
            data_probs = [prob / data_probs_sum for prob in data_probs]
            data_probs_cumsum = np.cumsum(data_probs, dtype=np.float32)
            data_probs_cumsum[-1] = 1

        return data_items, data_probs_cumsum

    def read_micsig(self, data_path, st=None, ed=None, mic_idxes_selected=None):
        if (st==None) & (ed==None):
            mic_signals, _ = soundfile.read(data_path, dtype='float32')
        else:
            try:
                mic_signals, _ = soundfile.read(data_path, start=st, stop=ed, dtype='float32')
            except:
                fs = soundfile.info(data_path).samplerate
                mic_signals, _ = librosa.load(data_path, sr=fs, offset=st/fs, duration=(ed-st)/fs, mono=False)
                mic_signals = mic_signals.transpose(1, 0)
        mic_signals = mic_signals[:, mic_idxes_selected]

        return mic_signals

    def select_micpairs(self, arrays, nmic_selected, mic_dist_range):
        mic_idxes_selected = {}
        mic_poss = {'array': 0.51*np.array((
                            ( 0.100,  0.000, 0.000),
                            ( 0.071,  0.071, 0.000),
                            ( 0.000,  0.100, 0.000),
                            (-0.071,  0.071, 0.000),
                            (-0.100,  0.000, 0.000),
                            (-0.071, -0.071, 0.000),
                            ( 0.000, -0.100, 0.000),
                            ( 0.071, -0.071, 0.000)))}
        for array in arrays:
            mic_idxes_selected[array], _ = select_microphone_pairs(mic_poss[array], nmic_selected, mic_dist_range)

        return mic_idxes_selected

class CHiME3Dataset(RealMicSigDataset):
    def __init__(self,  
                 data_dir: str, 
                 T: float, 
                 fs: int, 
                 stage: str,  # 'train', 'val', 'test' 
                 tasks: List[int] = [None], 
                 arrays: List[str] = ['array'], 
                 mic_dist_range: List[float] = [0.03, 0.20], 
                 nmic_selected: int = 2, 
                 prob_mode:List[str] = ['duration', 'micpair'], 
                 dataset_sz = None,
                 remove_spkoverlap = False,
                 sound_speed: float = 343.0):
        
        self.mic_idxes_selected = self.select_micpairs(arrays, nmic_selected, mic_dist_range)
        self.data_items, self.data_probs_cumsum = self.get_items_probs(data_dir, tasks, arrays, self.mic_idxes_selected, T, stage, prob_mode)

        self.dataset_sz = len(self.data_items) if dataset_sz is None else dataset_sz
        self.T = T
        self.fs = fs
        self.mic_dist_range = mic_dist_range
        self.nmic_selected = nmic_selected
        self.remove_spkoverlap = remove_spkoverlap
        
    def get_items_probs(self, data_dir, tasks, arrays, mic_idxes_selected, duration_min_limit, stage, prob_mode):
        data_dir = Path(data_dir) / 'data' / 'audio' / '16kHz' / 'isolated'
        envirs = ['bth', 'bus_real', 'caf_real', 'ped_real', 'str_real']
        dataset_split = {'train':['tr05'], 'val':['dt05'], 'test':['et05']}
        data_items = []
        data_probs = []
        for ds in dataset_split[stage]:
            for dir_sub in envirs:
                dir_sub = data_dir / (ds + '_' + dir_sub)

                uttrs = list(dir_sub.rglob('*.CH0.wav'))
                if 'duration' in prob_mode:
                    probs_uttr = [soundfile.info(wp).duration for wp in uttrs]  # 每个wav的权重为其duration
                else:
                    probs_uttr = [1 for wp in uttrs]

                if 'room' in prob_mode:
                    sum_probs_uttr = sum(probs_uttr)
                    probs_uttr = [prob/sum_probs_uttr for prob in probs_uttr]

                for array in arrays:
                    nmicpair = len(mic_idxes_selected[array])
                    for idx in range(len(uttrs)):
                        if soundfile.info(uttrs[idx]).duration >= duration_min_limit:
                            for micpair_idx in range(nmicpair):
                                data_items.append((uttrs[idx], mic_idxes_selected[array][micpair_idx]))
                                if 'micpair' in prob_mode:
                                    data_probs.append(probs_uttr[idx])
                                else:
                                    data_probs.append(probs_uttr[idx]/nmicpair)
        data_probs_cumsum = []
        if len(data_probs)>0:  
            data_probs_sum = sum(data_probs)
            data_probs = [prob/data_probs_sum for prob in data_probs]
            data_probs_cumsum = np.cumsum(data_probs, dtype=np.float32)
            data_probs_cumsum[-1] = 1

        return data_items, data_probs_cumsum

    def read_micsig(self, data_path, st=None, ed=None, mic_idxes_selected=None):
        nmic = len(mic_idxes_selected)
        mic_signals = []
        for mic_idx in range(nmic):
            data_path = data_path.parent / data_path.name.replace('.CH0.wav', f'.CH{mic_idxes_selected[mic_idx]}.wav')
            if (st==None) & (ed==None):
                mic_signal, _ = soundfile.read(data_path, dtype='float32')
            else:
                mic_signal, _ = soundfile.read(data_path, start=st, stop=ed, dtype='float32')
            mic_signals += [mic_signal]
            
        return np.array(mic_signals).transpose(1, 0)
    
    def select_micpairs(self, arrays, nmic_selected, mic_dist_range):
        mic_idxes_selected = {}
        mic_poss = {'array': np.array((
                            (-0.100,  0.950, 0.000),                    
                            ( 0.000,  0.950, 0.000),
                            ( 0.100,  0.950, 0.000),
                            (-0.100, -0.950, 0.000),                    
                            ( 0.000, -0.950, 0.000),
                            ( 0.100, -0.950, 0.000)))
                        }
        for array in arrays:
            mic_idxes_selected[array], _ = select_microphone_pairs(mic_poss[array], nmic_selected, mic_dist_range)                       

        return mic_idxes_selected

# class CHiME56Dataset(RealMicSigDataset):
#     '''
#     只保留不重叠部分，还没调试
#     '''
#     def __init__(
#             self,
#             data_dir: str,
#             T: float, 
#             fs: int,
#             stage: str,  # 'train', 'val', 'test' 
#             tasks: List[int] = ['task'],
#             arrays: List[str] = ['U01', 'U02', 'U03', 'U04', 'U05', 'U06'],
#             mic_dist_range: List[float] = [0.03, 0.20],
#             nmic_selected: int = 2,
#             prob_mode: List[str] = ['duration', 'micpair'],
#             dataset_sz=None,
#             remove_spkoverlap = False,
#           sound_speed: float = 343.0):

#         self.remove_spkoverlap = remove_spkoverlap
#         self.mic_idxes_selected = self.select_micpairs(arrays, nmic_selected, mic_dist_range)
#         self.data_items, self.data_probs_cumsum = self.get_items_probs(data_dir, tasks, arrays, self.mic_idxes_selected, T, stage, prob_mode)

#         self.dataset_sz = len(self.data_items) if dataset_sz is None else dataset_sz
#         self.T = T
#         self.fs = fs
#         self.mic_dist_range = mic_dist_range
#         self.nmic_selected = nmic_selected

#     def datatime2float(self, a: datetime):
#         t = a.time()
#         return t.hour * 3600 + t.minute * 60 + t.second + (t.microsecond / 1.0e6)

#     def get_items_probs(self, data_dir, tasks, arrays, mic_idxes_selected, duration_min_limit, stage, prob_mode):
#         data_dir = Path(data_dir).expanduser()

#         if self.remove_spkoverlap:
#             ds_dir_str = {'train': 'train', 'val': 'dev', 'test': 'eval'}[stage]
#             wav_dir = data_dir / f'audio/{ds_dir_str}'
#             session_trans_files = list((data_dir / 'transcriptions' / ds_dir_str).glob('*.json'))

#             uttrs = list(data_dir.rglob('S*_U*.CH1.wav'))
#             uttrs.sort()

#             data_items = []
#             data_probs = []
#             for sess_tran_file in session_trans_files:
#                 with open(sess_tran_file, 'r') as f:
#                     sentence_infos = json.load(f)
#                 sentence_infos.sort(key=lambda x: x['start_time'])

#                 for i, si in enumerate(sentence_infos):
#                     si['start_time'] = self.datatime2float(datetime.strptime(si['start_time'], "%H:%M:%S.%f"))
#                     si['end_time'] = self.datatime2float(datetime.strptime(si['end_time'], "%H:%M:%S.%f"))
#                     si['dura'] = si['end_time'] - si['start_time']

#                 # the latest end time of the previous sentence
#                 etbt = 0.0
#                 for i in range(0, len(sentence_infos)):
#                     si = sentence_infos[i]
#                     si['end_time_prev_elems'] = etbt
#                     if si['end_time'] > etbt:
#                         etbt = si['end_time']

#                 # the start time of current sentence - the end time of the previous sentence >= duration_min_limit
#                 si_selected = []
#                 for i, si in enumerate(sentence_infos):
#                     if i == len(sentence_infos) - 1:
#                         continue
#                     if sentence_infos[i + 1]['start_time'] - si['end_time_prev_elems'] >= duration_min_limit:
#                         si['start_time_next'] = sentence_infos[i + 1]['start_time']
#                         si['dura_expanded'] = sentence_infos[i + 1]['start_time'] - si['end_time_prev_elems']
#                         si_selected.append(si)

#                 if 'duration' in prob_mode:
#                     probs_uttr = [si['dura_expanded'] for si in si_selected]  # 每个wav的权重为其duration
#                 else:
#                     probs_uttr = [1 for si in si_selected]

#                 # # 给room时，归一化不同session的概率
#                 # if 'room' in prob_mode:
#                 #     sum_probs_uttr = sum(probs_uttr)
#                 #     probs_uttr = [prob / sum_probs_uttr for prob in probs_uttr]

#                 for array in arrays:
#                     nmicpair = len(mic_idxes_selected[array])
#                     for idx in range(len(si_selected)):
#                         wav_file = wav_dir / (sess_tran_file.name.replace('.json', '') + '_' + array + '.CH1.wav')
#                         for micpair_idx in range(nmicpair):
#                             # e.g. "/*/data/SenSig/CHiME5_6/train/audio/train/S24_U06.CH4.wav", (start, end), (0, 1)
#                             data_items.append((wav_file, si_selected[idx], mic_idxes_selected[array][micpair_idx]))
#                             if 'micpair' in prob_mode:
#                                 data_probs.append(probs_uttr[idx])
#                             else:
#                                 data_probs.append(probs_uttr[idx] / nmicpair)
#         else:
#             pass
#         data_probs_cumsum = []
#         if len(data_probs)>0:
#             data_probs_sum = sum(data_probs)
#             data_probs = [prob / data_probs_sum for prob in data_probs]
#             data_probs_cumsum = np.cumsum(data_probs, dtype=np.float32)
#             data_probs_cumsum[-1] = 1

#         return data_items, data_probs_cumsum

#     # def duration(self):
#     #     durations = []
#     #     data_paths = []
#     #     for idx in range(len(self.data_items)):
#     #         data_path = self.data_items[idx][0]
#     #         duration = self.data_items[idx][1]['dura_expanded']
#     #         data_paths += [data_path]
#     #         durations += [duration]
#     #     print(f'Duration is {durations} s')
#     #     print(f'Total duration is {np.sum(np.array(durations))/3600:.2f} h')

#     #     data = {'Data path': data_paths, 'Duration': durations}
#     #     df = pd.DataFrame(data)
#     #     df.to_excel("Duration.xlsx", index=True)

#         # return duration

#     def read_micsig(self, data_path, st, ed, mic_idxes_selected):
#         mic_signals = []
#         for mi in mic_idxes_selected:
#             wav_path = data_path.parent / data_path.name.replace('CH1', f'CH{mi+1}')
#             mic_signal, _ = soundfile.read(wav_path, start=st, stop=ed, dtype='float32')
#             mic_signals.append(mic_signal)
#         mic_signal = np.stack(mic_signals).T
#         return mic_signal

#     def __getitem__(self, idx):
#         idx = np.searchsorted(self.data_probs_cumsum, np.random.uniform())
#         data_path, si, mic_idxes = self.data_items[idx]

#         fs = soundfile.info(data_path).samplerate
#         nsample = int(si['dura_expanded'] * fs)
#         nsample_desired = int(self.T * fs)
#         if nsample < nsample_desired:
#             mic_signals = self.read_micsig(data_path, st=int(fs * si['end_time_prev_elems']), ed=int(fs * si['start_time_next']), mic_idxes_selected=mic_idxes)
#             mic_signals = pad_cut_sig_sameutt(mic_signals, nsample_desired)
#             print('smaller number of samples')
#         else:
#             st = random.randint(0, nsample - nsample_desired) + int(fs * si['end_time_prev_elems'])
#             ed = st + nsample_desired
#             mic_signals = self.read_micsig(data_path, st=st, ed=ed, mic_idxes_selected=mic_idxes)

#         if self.fs != fs:
#             mic_signals = scipy.signal.resample_poly(mic_signals, self.fs, fs)

#         return mic_signals

#     def select_micpairs(self, arrays, nmic_selected, mic_dist_range):
#         pos_rcv = np.zeros((4, 3))
#         pos_rcv[0, :] = np.array([-0.113, 0, 0])
#         pos_rcv[1, :] = np.array([-0.076, 0, 0])
#         pos_rcv[2, :] = np.array([-0.036, 0, 0])
#         pos_rcv[3, :] = np.array([0.113, 0, 0])

#         mic_idxes_selected = {}
#         mic_poss = {f'U0{i+1}': pos_rcv for i in range(6)}
#         for array in arrays:
#             mic_idxes_selected[array], _ = select_microphone_pairs(mic_poss[array], nmic_selected, mic_dist_range)

#         return mic_idxes_selected


if __name__ == '__main__':
    from opt import opt_pretrain
    opts = opt_pretrain()
    dirs = opts.dir()

    # sig_dir = dirs['LOCATA']
    # dataset = LOCATADataset(
	# 			data_dir = sig_dir,
	# 			T = 4.112,
	# 			fs = 16000,
	# 			stage = 'train',
	# 			# tasks = [1],
	# 			# arrays = ['benchmark2'],
	# 			mic_dist_range = [0.05, 20],
	# 			dataset_sz = 10
	# 		)
    # _, total_duration = dataset.duration()
    # print('LOCATA: ', total_duration, 'h')
    # # for i in range(10):
    # #     sig = dataset[i]
    # #     print(sig.shape)

    # sig_dir = dirs['MCWSJ']
    # dataset = MCWSJDataset(
	# 			data_dir = sig_dir,
	# 			T = 4.112,
	# 			fs = 16000,
	# 			stage = 'train',
	# 			# tasks = ['stat'],
	# 			# arrays = ['array1', 'array2'],
	# 			mic_dist_range = [0.05, 20],
	# 			dataset_sz = 10
	# 		)
    # _, total_duration = dataset.duration()
    # print('MCWSJ: ', total_duration, 'h')
    # # for i in range(10):
    # #     sig = dataset[i]
    # #     print(sig.shape)

    # sig_dir = dirs['LibriCSS']
    # dataset = LibriCSSDataset(
	# 			data_dir = sig_dir,
	# 			T = 4.112,
	# 			fs = 16000,
	# 			stage = 'train',
	# 			# tasks = ['stat'],
	# 			# arrays = ['array1', 'array2'],
	# 			mic_dist_range = [0.05, 20],
	# 			dataset_sz = 10
	# 		)
    # _, total_duration = dataset.duration()
    # print('LibriCSS: ', total_duration, 'h')
    # # for i in range(10):
    # #     sig = dataset[i]
    # #     print(sig.shape)

    # sig_dir = dirs['CHiME3']
    # dataset = CHiME3Dataset(
	# 			data_dir = sig_dir,
	# 			T = 4.112,
	# 			fs = 16000,
	# 			stage = 'train',
	# 			# tasks = [None],
	# 			# arrays = ['array'],
	# 			mic_dist_range = [0.05, 20],
	# 			dataset_sz = 10
	# 		)
    # _, total_duration = dataset.duration()
    # print('CHiME3: ', total_duration, 'h')
    # # for i in range(10):
    # #     sig = dataset[i]
    # #     print(sig.shape)

    # sig_dir = dirs['AMI']
    # dataset = AMIDataset(
	# 			data_dir = sig_dir,
	# 			T = 4.112,
	# 			fs = 16000,
	# 			stage = 'train',
	# 			# tasks = ['stat'],
	# 			# arrays = ['array1', 'array2'],
	# 			mic_dist_range = [0.05, 20],
	# 			dataset_sz = 10
	# 		)
    # _, total_duration = dataset.duration()
    # print('AMI: ', total_duration, 'h')
    # # for i in range(100):
    # #     sig = dataset[i]
    # #     print(sig.shape)
    
    # sig_dir = dirs['AISHELL4']
    # dataset = AISHELL4Dataset(
	# 			data_dir = sig_dir,
	# 			T = 4.112,
	# 			fs = 16000,
	# 			stage = 'train',
	# 			# tasks = ['stat'],
	# 			# arrays = ['array1', 'array2'],
	# 			mic_dist_range = [0.05, 20],
	# 			dataset_sz = 10,
    #           remove_spkoverlap = True,
	# 		)
    # _, total_duration = dataset.duration()
    # print('AISHELL4: ', total_duration, 'h')
    # # for i in range(10):
    # #     sig = dataset[i]
    # #     print(sig.shape)

    # sig_dir = dirs['M2MeT']
    # dataset = M2MeTDataset(
	# 			data_dir = sig_dir,
	# 			T = 4.112,
	# 			fs = 16000,
	# 			stage = 'train',
	# 			# tasks = ['stat'],
	# 			# arrays = ['array1', 'array2'],
	# 			mic_dist_range = [0.05, 20],
	# 			dataset_sz = 100,
    #             remove_spkoverlap = False,
	# 		)
    # _, total_duration = dataset.duration()
    # print('M2Met: ', total_duration, 'h')
    # for i in range(100000):
    #     sig = dataset[i]
    #     print(sig.shape)

    # sig_dir = dirs['CHiME56']
    # dataset = CHiME56Dataset(
	# 			data_dir = sig_dir,
	# 			T = 4.112,
	# 			fs = 16000,
	# 			stage = 'train',
	# 			# tasks = ['stat'],
	# 			# arrays = ['array1', 'array2'],
	# 			mic_dist_range = [0.05, 20],
	# 			dataset_sz = 10,
    #             remove_spkoverlap = True,
	# 		)
    # dataset.duration()
    # for i in range(10):
    #     sig = dataset[i]
    #     print(sig.shape)

    sig_dir = dirs['RealMAN']
    dataset = RealMANDataset(
				data_dir = sig_dir,
				T = 4.112,
				fs = 16000,
				stage = 'train',
				mic_dist_range = [0.05, 20],
				dataset_sz = 100,
                remove_spkoverlap = False,
			)
    _, total_duration = dataset.duration()
    print('RealMAN: ', total_duration, 'h')
    for i in range(10):
        sig = dataset[i]
        print(sig.shape) 