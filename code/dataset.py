import numpy as np
import scipy.signal
# import librosa # cause CPU overload, for data generation (scipy.signal.resample, librosa.resample) 
import soundfile
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from data_generation.utils_real_micsig import *
from data_generation.utils_src import *
from data_generation.utils_noise import *
import data_generation.gen_sig_from_real_rir as real_dataset
import data_generation.utils_simu_rir_sig as simu_dataset 


class RandomRealDataset(Dataset):
    """ Load ramdom real-world microphone signals (RealSig) or microphone signals generated with real RIRs (RealRIR)
    """
    def __init__(
        self, 
        data_dirs, 
        T, 
        fs, 
        mic_dist_range, 
        nmic_selected, 
        stage = 'train',
        seed = 1,
        dataset_sz = None, 
        transforms = None, 
        prob_mode = ['duration', 'micpair'],
        remove_spkoverlap = True,
        dataset_list = ['LOCATA', 'MCWSJ', 'LibriCSS', 'AMI', 'AISHELL4', 'M2MeT', 'RealMAN', # RealSig
					    'DCASE', 'MIR', 'Mesh', 'ACE', 'dEchorate', 'BUTReverb'], # RealRIR
        dataset_probs = [1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1], 
        sound_speed = 343.0):
        # RealSig
        # train duration (hours): [33, 74, 83, 101, 4124, 835979, 514811], 
        # train duration (hours, utterance level) [0.07, 1.3, 2.0, 3.4, 74, 95, 111]
        # room condition: [1 3 1 5 3 10 13]
        # [1, 5, 5, 5, 8, 8, 8]

        print(f'dataset ratio: {dataset_list}')
        print(f'dataset ratio: {dataset_probs}')
 
        self.dataset_list = []
        for dataset_name in dataset_list:
            data_dir = data_dirs[dataset_name]
            if dataset_name in ['LOCATA', 'MCWSJ', 'LibriCSS', 'AMI', 'AISHELL4', 'M2MeT', 'RealMAN']:
                if (dataset_name == 'AISHELL4') or (dataset_name == 'M2MeT'):
                    remove_spkoverlap = True
                else:
                    remove_spkoverlap = False
                dataset_name_str = dataset_name+'Dataset'
                dataset = globals()[dataset_name_str](
                    data_dir = data_dir,
                    T = T,
                    fs = fs,
                    stage = stage, 
                    mic_dist_range = mic_dist_range,
                    nmic_selected = nmic_selected,
                    prob_mode = prob_mode,
                    dataset_sz = None, 
                    remove_spkoverlap = remove_spkoverlap,
                    sound_speed = sound_speed)
                self.dataset_list += [dataset]
            elif dataset_name in ['DCASE', 'MIR', 'Mesh', 'ACE', 'dEchorate', 'BUTReverb']:
                ds_sz = {'train': 102400, 'val':2560, 'test':2560}
                dataset_rir = FixMicSigDataset(
                    data_dir = data_dir,
                    load_anno = False,
                    dataset_sz = ds_sz[stage],
                    transforms = None,
                )
                self.dataset_list += [dataset_rir]

        self.dataset_sz = dataset_sz
        self.transforms = transforms
        self.seed = seed
      
        assert len(self.dataset_list) == len(dataset_probs), [len(self.dataset_list), len(dataset_probs)]
            
        dataset_probs_sum = sum(dataset_probs)
        dataset_probs = [prob/dataset_probs_sum for prob in dataset_probs]
        self.ds_probs_cumsum = np.cumsum(dataset_probs, dtype=np.float32)
        self.ds_probs_cumsum[-1] = 1
 
    def __len__(self):
        return self.dataset_sz

    def __getitem__(self, idx):

        dataset_idx = np.searchsorted(self.ds_probs_cumsum, np.random.uniform())
        ins_idx = np.random.randint(0, len(self.dataset_list[dataset_idx]))
        np.random.seed(seed=self.seed+ins_idx)
        mic_sig = self.dataset_list[dataset_idx].__getitem__(ins_idx)
 
        if self.transforms is not None:
            for t in self.transforms:
                mic_sig = t(mic_sig)

        return mic_sig 


class FixMicSigDataset(Dataset):
    """ Load fixed presaved microphone signals (*wav) (simulated data or microphones signal generated with real RIRs)
        1. for pretraining with simulated data, microphone signals generated with real RIR
        2. for fine-tuning with simulated data
	"""
    def __init__(self, data_dir, load_anno, dataset_sz, transforms=None):

        self.data_paths = []
        
        if isinstance(data_dir, list):
            self.files = []
            for d in data_dir:
                self.files += list(Path(d).rglob('*.wav'))
            np.random.shuffle(self.files)
        else:
            self.files = list(Path(data_dir).rglob('*.wav'))
        if dataset_sz is None:
            self.dataset_sz = np.min([len(self.files), dataset_sz])
        else:
            self.dataset_sz = dataset_sz
        self.load_anno = load_anno
        self.transforms = transforms

    def __len__(self):
        return self.dataset_sz

    def __getitem__(self, idx):

        file_name = str(self.files[idx])
        mic_sig, fs = soundfile.read(file_name)
        
        if self.transforms is not None:
            for t in self.transforms:
                mic_sig = t(mic_sig)

        if self.load_anno:
            info_file_name = file_name.replace('.wav', '_info.npz')
            info = np.load(info_file_name)
            vol = info['room_sz'][0] * info['room_sz'][1] * info['room_sz'][2]
            sur = info['room_sz'][0] * info['room_sz'][1] + info['room_sz'][0] * info['room_sz'][2] + info['room_sz'][1] * info['room_sz'][2]
            annos = {
                'TDOA': info['TDOA'].astype(np.float32), 
                'T60': info['T60_edc'].astype(np.float32),  
                'DRR': info['DRR'].astype(np.float32),
                'C50': info['C50'].astype(np.float32),
                'ABS': (0.161*vol/sur/info['T60_edc']).astype(np.float32),
                }
            return mic_sig.astype(np.float32), annos

        else:
            return mic_sig.astype(np.float32)
 
class RandomMicSigDataset(Dataset):
    def __init__(self, 
        real_sig_dir, 
        sim_sig_dir, 
        real_sim_ratio, 
        load_anno, 
        dataset_sz, 
        transforms=None):

        realdataset = FixMicSigDataset(
            data_dir = real_sig_dir,
            load_anno = load_anno,
            dataset_sz = None,
            transforms = transforms
            )

        simdataset = FixMicSigDataset(
            data_dir = sim_sig_dir,
            load_anno = load_anno,
            dataset_sz = None,
            transforms = transforms
            )
        
        assert real_sim_ratio in [[0,1], [1,0], [1,1]], real_sim_ratio
        if real_sim_ratio == [0,1]:
            self.dataset_list = [simdataset]
        elif real_sim_ratio == [1,0]:
            self.dataset_list = [realdataset]
        elif real_sim_ratio == [1,1]:
            self.dataset_list = [simdataset, realdataset]
        self.dataset_sz = dataset_sz
        self.load_anno = load_anno
    
    def __len__(self):
        return self.dataset_sz

    def __getitem__(self, idx=None):

        dataset = np.random.choice(self.dataset_list, 1)[0]
        idx = np.random.randint(0, len(dataset))
        mic_sig, annos = dataset.__getitem__(idx)
        TDOA = annos['TDOA']
        print(TDOA.shape)
        print(a)

        return mic_sig.astype(np.float32), TDOA.astype(np.float32)


class RandomMicSigFromRIRDataset(Dataset):
    """ Generate microphone signals from real RIRs
        1. for fine-tuning with microphone signals generated with real RIR+ simulated RIR
	"""
    def __init__(
        self, 
        real_rir_dir_list, 
        sim_rir_dir_list, 
        src_dir,
        dataset_sz, 
        T, 
        fs, 
        c,
        nmic,
        snr_range, 
        real_sim_ratio,
        transforms=None,
        seed=1
        ):
        
        srcdataset = WSJ0Dataset(
            path = src_dir,
            T = T,
            fs = fs
            )
        noidataset = NoiseSignal(
            T = T,
            fs = fs,
            nmic = nmic,
            noise_type = 'diffuse_white',
            noise_path = '',
            c = c
            )
        realrirdataset = real_dataset.RIRDataset(
            fs=fs, 
            rir_dir_list=real_rir_dir_list, 
            dataset_sz=None, 
            load_info=True, 
            load_noise=True, 
            load_noise_duration=T
            ) 
        realdataset = real_dataset.MicSigFromRIRDataset(
            rirnoidataset=realrirdataset,
            srcdataset=srcdataset,
            snr_range=snr_range,
            fs=fs,
            dataset_sz=None,
            seed=seed,
            load_info=True,
            save_anno=False,
            save_to=None,
            )
        simrirdataset = simu_dataset.RIRDataset(
            fs=fs,
            rir_dir_list=sim_rir_dir_list,
            dataset_sz=None,
            load_dp=True,
            load_info=True
            )
        simdataset = simu_dataset.MicSigFromRIRDataset(
            rirdataset=simrirdataset,
            srcdataset=srcdataset,
            noidataset=noidataset,
            snr_range=snr_range,
            fs=fs,
            dataset_sz=None,
            seed=seed,
            load_info=True,
            save_anno=False,
            save_to=None, 
            )
        assert real_sim_ratio in [[0,1], [1,0], [1,1]], real_sim_ratio
        if real_sim_ratio == [0,1]:
            self.dataset_list = [simdataset]
        elif real_sim_ratio == [1,0]:
            self.dataset_list = [realdataset]
        elif real_sim_ratio == [1,1]:
            self.dataset_list = [realdataset, simdataset]
        self.seed = seed
        self.dataset_sz = dataset_sz
        self.transforms = transforms

    def __len__(self):
        return self.dataset_sz

    def __getitem__(self, idx=None):
        
        dataset_idx = np.random.randint(0, len(self.dataset_list))
        dataset = self.dataset_list[dataset_idx]
        idx = np.random.randint(0, len(dataset))
        mic_sig, annos = dataset.__getitem__(idx)
        if self.transforms is not None:
            for t in self.transforms:
                mic_sig = t(mic_sig)

        return mic_sig.astype(np.float32), annos
 

## Transform classes
# class Segmenting(object):
#     """ Segmenting transform
# 	"""
#     def __init__(self, K, step, window=None):
#         self.K = K
#         self.step = step
#         if window is None:
#             self.w = np.ones(K)
#         elif callable(window):
#             try: self.w = window(K)
#             except: raise Exception('window must be a NumPy window function or a Numpy vector with length K')
#         elif len(window) == K:
#             self.w = window
#         else:
#             raise Exception('window must be a NumPy window function or a Numpy vector with length K')

#     def __call__(self, x, acoustic_scene):

        # L = x.shape[0]
        # N_w = np.floor(L/self.step - self.K/self.step + 1).astype(int)

        # if self.K > L:
        #     raise Exception('The window size can not be larger than the signal length ({})'.format(L))
        # elif self.step > L:
        #     raise Exception('The window step can not be larger than the signal length ({})'.format(L))

        # # Pad and window the signal
        # # x = np.append(x, np.zeros((N_w * self.step + self.K - L, N_mics)), axis=0)
        # # shape_Xw = (N_w, self.K, N_mics)
        # # strides_Xw = [self.step * N_mics, N_mics, 1]
        # # strides_Xw = [strides_Xw[i] * x.itemsize for i in range(3)]
        # # Xw = np.lib.stride_tricks.as_strided(x, shape=shape_Xw, strides=strides_Xw)
        # # Xw = Xw.transpose((0, 2, 1)) * self.w

        # if acoustic_scene is not None:
        #     # Pad and window the DOA if it exists
        #     if hasattr(acoustic_scene, 'DOA'): # (nsample,naziele,nsource)
        #         N_dims = acoustic_scene.DOA.shape[1]
        #         num_source = acoustic_scene.DOA.shape[-1]
        #         DOA = []
        #         for source_idx in range(num_source):
        #             DOA += [np.append(acoustic_scene.DOA[:,:,source_idx], np.tile(acoustic_scene.DOA[-1,:,source_idx].reshape((1,-1)),
        #             [N_w*self.step+self.K-L, 1]), axis=0)] # Replicate the last known DOA
        #         DOA = np.array(DOA).transpose(1,2,0)

        #         shape_DOAw = (N_w, self.K, N_dims) # (nwindow, win_len, naziele)
        #         strides_DOAw = [self.step*N_dims, N_dims, 1]
        #         strides_DOAw = [strides_DOAw[i] * DOA.itemsize for i in range(3)]
        #         DOAw_sources = []
        #         for source_idx in range(num_source):
        #             DOAw = np.lib.stride_tricks.as_strided(DOA[:,:,source_idx], shape=shape_DOAw, strides=strides_DOAw)
        #             DOAw = np.ascontiguousarray(DOAw)
        #             for i in np.flatnonzero(np.abs(np.diff(DOAw[..., 1], axis=1)).max(axis=1) > np.pi):
        #                 DOAw[i, DOAw[i,:,1]<0, 1] += 2*np.pi  # Avoid jumping from -pi to pi in a window
        #             DOAw = np.mean(DOAw, axis=1)
        #             DOAw[DOAw[:,1]>np.pi, 1] -= 2*np.pi
        #             DOAw_sources += [DOAw]
        #         acoustic_scene.DOAw = np.array(DOAw_sources).transpose(1, 2, 0) # (nsegment,naziele,nsource)

        #     # Pad and window the VAD if it exists
        #     if hasattr(acoustic_scene, 'mic_vad'): # (nsample,1)
        #         vad = acoustic_scene.mic_vad[:, np.newaxis]
        #         vad = np.append(vad, np.zeros((L - vad.shape[0], 1)), axis=0)

        #         shape_vadw = (N_w, self.K, 1)
        #         strides_vadw = [self.step * 1, 1, 1]
        #         strides_vadw = [strides_vadw[i] * vad.itemsize for i in range(3)]

        #         acoustic_scene.mic_vad = np.lib.stride_tricks.as_strided(vad, shape=shape_vadw, strides=strides_vadw)[..., 0] # (nsegment, nsample)

        #     # Pad and window the VAD if it exists
        #     if hasattr(acoustic_scene, 'mic_vad_sources'): # (nsample,nsource)
        #         shape_vadw = (N_w, self.K, 1)
        #         strides_vadw = [self.step * 1, 1, 1]
        #         strides_vadw = [strides_vadw[i] * vad.itemsize for i in range(3)]
        #         num_source = acoustic_scene.mic_vad_sources.shape[1]
        #         vad_sources = []
        #         for source_idx in range(num_source):
        #             vad = acoustic_scene.mic_vad_sources[:, source_idx:source_idx+1]
        #             vad = np.append(vad, np.zeros((L - vad.shape[0], 1)), axis=0)

        #             vad_sources += [np.lib.stride_tricks.as_strided(vad, shape=shape_vadw, strides=strides_vadw)[..., 0]]

        #         acoustic_scene.mic_vad_sources = np.array(vad_sources).transpose(1,2,0) # (nsegment, nsample, nsource)

        #     # Pad and window the TDOA if it exists
        #     if hasattr(acoustic_scene, 'TDOA'): # (nsample,nch-1,nsource)
        #         num_source = acoustic_scene.TDOA.shape[-1]
        #         TDOA = []
        #         for source_idx in range(num_source):
        #             TDOA += [np.append(acoustic_scene.TDOA[:,:,source_idx], np.tile(acoustic_scene.TDOA[-1,:,source_idx].reshape((1,-1)),
        #             [N_w*self.step+self.K-L, 1]), axis=0)] # Replicate the last known TDOA
        #         TDOA = np.array(TDOA).transpose(1,2,0)

        #         nch = TDOA.shape[1]
        #         shape_TDOAw = (N_w, self.K, nch)
        #         strides_TDOAw = [self.step * nch, nch, 1]
        #         strides_TDOAw = [strides_TDOAw[i] * TDOA.itemsize for i in range(3)]

        #         TDOAw_sources = []
        #         for source_idx in range(num_source):
        #             TDOAw = np.lib.stride_tricks.as_strided(TDOA[:,:,source_idx], shape=shape_TDOAw, strides=strides_TDOAw)
        #             TDOAw = np.mean(TDOAw, axis=1)
        #             TDOAw_sources += [TDOAw]
        #         acoustic_scene.TDOAw = np.array(TDOAw_sources).transpose(1,2,0) # (nsegment,nch-1,nsource)

        #     # Pad and window the DRR if it exists
        #     if hasattr(acoustic_scene, 'DRR'): # (nsample,nsource)
        #         num_source = acoustic_scene.DRR.shape[-1]
        #         DRR = []
        #         for source_idx in range(num_source):
        #             DRR += [np.append(acoustic_scene.DRR[:,source_idx], np.tile(acoustic_scene.DRR[-1:,source_idx],
        #             [N_w*self.step+self.K-L]), axis=0)] # Replicate the last known DRR
        #         DRR = np.array(DRR).transpose(1,0)

        #         nch = DRR.shape[1]
        #         shape_DRRw = (N_w, self.K, 1)
        #         strides_DRRw = [self.step * 1, 1, 1]
        #         strides_DRRw = [strides_DRRw[i] * DRR.itemsize for i in range(3)]

        #         DRRw_sources = []
        #         for source_idx in range(num_source):
        #             DRRw = np.lib.stride_tricks.as_strided(DRR[:,source_idx], shape=shape_DRRw, strides=strides_DRRw)
        #             DRRw = np.mean(DRRw, axis=1)
        #             DRRw_sources += [DRRw[..., 0]]
        #         acoustic_scene.DRRw = np.array(DRRw_sources).transpose(1,0) # (nsegment,nsource)

        #     # Pad and window the C50 if it exists
        #     if hasattr(acoustic_scene, 'C50'): # (nsample,nsource)
        #         num_source = acoustic_scene.C50.shape[-1]
        #         C50 = []
        #         for source_idx in range(num_source):
        #             C50 += [np.append(acoustic_scene.C50[:,source_idx], np.tile(acoustic_scene.C50[-1:,source_idx],
        #             [N_w*self.step+self.K-L]), axis=0)] # Replicate the last known C50
        #         C50 = np.array(C50).transpose(1,0)

        #         nch = C50.shape[1]
        #         shape_C50w = (N_w, self.K, 1)
        #         strides_C50w = [self.step * 1, 1, 1]
        #         strides_C50w = [strides_C50w[i] * C50.itemsize for i in range(3)]

        #         C50w_sources = []
        #         for source_idx in range(num_source):
        #             C50w = np.lib.stride_tricks.as_strided(C50[:,source_idx], shape=shape_C50w, strides=strides_C50w)
        #             C50w = np.mean(C50w, axis=1)
        #             C50w_sources += [C50w[..., 0]]
        #         acoustic_scene.C50w = np.array(C50w_sources).transpose(1,0) # (nsegment,nsource)

        #     # Pad and window the C80 if it exists
        #     if hasattr(acoustic_scene, 'C80'): # (nsample,nsource)
        #         num_source = acoustic_scene.C80.shape[-1]
        #         C80 = []
        #         for source_idx in range(num_source):
        #             C80 += [np.append(acoustic_scene.C80[:,source_idx], np.tile(acoustic_scene.C80[-1:,source_idx],
        #             [N_w*self.step+self.K-L]), axis=0)] # Replicate the last known C80
        #         C80 = np.array(C80).transpose(1,0)

        #         nch = C80.shape[1]
        #         shape_C80w = (N_w, self.K, 1)
        #         strides_C80w = [self.step * 1, 1, 1]
        #         strides_C80w = [strides_C80w[i] * C80.itemsize for i in range(3)]

        #         C80w_sources = []
        #         for source_idx in range(num_source):
        #             C80w = np.lib.stride_tricks.as_strided(C80[:,source_idx], shape=shape_C80w, strides=strides_C80w)
        #             C80w = np.mean(C80w, axis=1)
        #             C80w_sources += [C80w[..., 0]]
        #         acoustic_scene.C80w = np.array(C80w_sources).transpose(1,0) # (nsegment,nsource)

        #     # Timestamp for each window
        #     acoustic_scene.tw = np.arange(0, (L-self.K), self.step) / acoustic_scene.fs

        # return x, acoustic_scene

class Selecting(object):
    def __init__(self, select_range):
        self.select_range = select_range

    def __call__(self, mic_sig):
        nsample = mic_sig.shape[0]
        assert self.select_range[-1]<=nsample, f'Selecting range ({self.select_range[-1]}) is larger than signal length ({nsample})~'
        mic_sig = mic_sig[self.select_range[0]:self.select_range[1], ...]
        
        return mic_sig


if __name__ == "__main__":
    from opt import opt_downstream
    dirs = opt_downstream().dir()

    ## Noise
    # T = 20
    # RIRdataset = RIRDataset(fs=16000, data_dir='SAR-SSL/data/RIR-pretrain5-2/', dataset_sz=4)
    # acoustic_scene = RIRdataset[1]

    # souDataset = WSJ0Dataset(path=dirs['sousig_pretrain'], T=T, fs=16000, num_source=1, size=50)

    # mic_pos = np.array(((-0.05, 0.0, 0.0), (0.05, 0.0, 0.0)))
    # noise_type = 'diffuse_fromRIR'
    # a = NoiseDataset(T=T, 
    #                  fs=16000, 
    #                  nmic=2, 
    #                  noise_type=Parameter([noise_type], discrete=True), 
    #                  noise_path=dirs['sousig_pretrain'], 
    #                  c=343.0, 
    #                  size=1)
    # noise_signal = a.get_random_noise(mic_pos=mic_pos, acoustic_scene=acoustic_scene, source_dataset=souDataset, eps=1e-5)
    # soundfile.write(noise_type+'.wav', noise_signal, 16000)

    import torch
    import seaborn as sns
    # data = np.random.randn(1000)
    # sns.histplot(data, bins=100, kde=True)
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.savefig('tdoas_histogram_random.png')
    # plt.close()
    # print(a)
    rirDataset = RIRDataset( 
        data_dir_list = ['/data/home/yangbing/SAR-SSL/data/RIR/ACE/'],
        data_prob_ratio_list=[1],
        load_noise=True,
        load_noise_duration = 4.112,
        noise_type_specify=None,  
        fs=16000, 
        dataset_sz=None)
    kwargs = {'num_workers': 0, 'pin_memory': True}
    rirDataset = torch.utils.data.DataLoader(rirDataset, batch_size=None, shuffle=False, **kwargs )
    print(len(rirDataset))
    tdoas = []
    for idx, acoustic_scene in enumerate(rirDataset):
        print(idx)
        tdoas += [acoustic_scene.TDOA[0,0,0]*16000]
        # tdoas += [acoustic_scene.TDOA[0,0,0]*16000*(-1)]
    print(len(tdoas), set(tdoas))
    sns.boxplot(tdoas)
    plt.savefig('tdoas_boxplot_switch.png')
    plt.close()
        
    sns.histplot(tdoas, bins=100, kde=True)
    plt.xlabel('Sample')
    plt.ylabel('Counts')
    plt.savefig('tdoas_histogram_switch.png')
    plt.close()