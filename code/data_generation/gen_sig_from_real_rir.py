"""
    Generate microphone signals from real-world room impulse responses (RIRs)

    Examples:
        python gen_sig_from_real_rir.py --stage pretrain --dataset Mesh MIR DCASE dEchorate BUTReverb ACE --src_dir ../../../data/SrcSig/wsj0 --rir_dir ../../../data/RIR/real --save_dir ../../data/MicSig/real 
        python gen_sig_from_real_rir.py --stage preval --dataset DCASE BUTReverb --src_dir ../../../data/SrcSig/wsj0 --rir_dir ../../../data/RIR/real --save_dir ../../data/MicSig/real  
"""
   
import os
cpu_num = 8*5
os.environ["OMP_NUM_THREADS"] = str(cpu_num) 
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)

import argparse
import numpy as np
import scipy
import scipy.io
import scipy.signal
import soundfile
import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from functools import lru_cache, cache
from torch.utils.data import Dataset
try:
    from utils_src import *
except:
    from data_generation.utils_src import *


# @cache
# def _search_files(dir, file_extension):
#     # paths = list(Path(dir).rglob(f"*.{file_extension}"))
#     # return [str(p) for p in paths]

#     paths = []
#     for item in os.listdir(dir):
#         if os.path.isdir( os.path.join(dir, item) ):
#             paths += self._search_files( os.path.join(dir, item), file_extension )
#         elif item.split(".")[-1] == file_extension:
#             paths += [os.path.join(dir, item)]
#     return paths 

# @cache
# def _match(noise_dir, mic_attr_match,noise_type_specify=None):

#     noise_paths = _search_files(dir=noise_dir, file_extension='wav')
    
#     match_paths = []
#     for noise_path in noise_paths:
#         wav_name = noise_path.split('/')[-1]
#         noise_mic = wav_name.split('_')[1]
#         noise_type = wav_name.split('_')[-1].split('.')[0]
#         if noise_mic == mic_attr_match:
#             if noise_type_specify is None:
#                 match_paths += [noise_path]
#             else:
#                 if noise_type in noise_type_specify:
#                     match_paths += [noise_path] 
#     return match_paths     

# @cache
# def _match(noise_dir, mic_attr_match):
#     noise_files = list(Path(noise_dir).rglob(f"*_{mic_attr_match}*.wav"))
#     return noise_files 

class RIRDataset(Dataset):
    def __init__(
        self, 
        fs, 
        rir_dir_list, 
        dataset_sz=None, 
        load_info=False, 
        load_noise=True, 
        load_noise_duration=None):

        self.fs = fs
        if isinstance(rir_dir_list, list):
            self.rir_files = []
            for rir_dir in rir_dir_list:
                self.rir_files += list(Path(rir_dir).rglob('*.npy'))
        else:
            self.rir_files = list(Path(rir_dir_list).rglob('*.npy'))
        self.dataset_sz = len(self.rir_files) if dataset_sz is None else dataset_sz
        self.load_info = load_info
        self.load_noise = load_noise
        self.load_noise_duration = load_noise_duration

    def __len__(self):
        return self.dataset_sz
    
    def __getitem__(self, idx):
        # idx = np.random.randint(0, self.dataset_sz)
        rir_file = self.rir_files[idx] 
        rir = np.load(rir_file)
        rir = rir.astype(np.float32)
        info_file = str(rir_file).replace('.npy', '_info.npz')
        info = np.load(info_file)
        if self.fs!= info['fs']:
            rir = scipy.signal.resample_poly(rir, self.fs, info['fs'])
        return_data = [rir]
        
        if self.load_noise:
            noise_file = str(rir_file).replace('.npy', '_noise.wav')
            rir_attrs = str(rir_file).split('/')
            mic_attr_match = rir_attrs[-1].split('_')[1].split('.')[0]
            noise_dir = str(rir_file.parent).replace(rir_attrs[-4], rir_attrs[-4]+'_noise')
            # noise_files = _match(noise_dir, mic_attr_match)
            noise_files = list(Path(noise_dir).rglob(f"*_{mic_attr_match}*.wav"))
            if noise_files == []:
                nmic = rir.shape[1]
                nsample = int(self.load_noise_duration * self.fs)
                noise_signal = np.zeros((nsample, nmic))
            else:
                noise_file = np.random.choice(noise_files, 1, replace=False)[0]
                noise_fs = soundfile.info(noise_file).samplerate
                noise_duration = soundfile.info(noise_file).duration
                nsample_noise = int(noise_duration * noise_fs)
                nsample_desired = int(self.load_noise_duration * noise_fs)
                assert nsample_noise>=nsample_desired, 'the sample number of noise signal is smaller than desired duration~'
                st = np.random.randint(0, nsample_noise - nsample_desired+1)
                ed = st + nsample_desired
                noise_signal = soundfile.read(noise_file, start=st, stop=ed, dtype='float32')[0]

                if self.fs != noise_fs:
                    noise_signal = scipy.signal.resample_poly(noise_signal, self.fs, noise_fs)
            return_data.append(noise_signal)

        if self.load_info:
            return_data.append(info)

        return return_data

    def rir_conv_src(self, rir, src_signal, gpuConv=False):
        ''' rir : (npoint,nch,nsam,nsource)
        '''
        if gpuConv:
            import gpuRIR

        # Source conv. RIR
        mic_signal_srcs = []
        num_source = rir.shape[-1]
        nsample = src_signal.shape[0]
        for source_idx in range(num_source):
            rir_per_src = rir[:, :, :, source_idx]  # (npoint,nch,nsample）

            if gpuConv:
                mic_sig_per_src = gpuRIR.simulateTrajectory(src_signal[:, source_idx], rir_per_src, timestamps=self.timestamps, fs=self.fs)
                mic_sig_per_src = mic_sig_per_src[0:nsample, :]

            else:
                if rir_per_src.shape[0] == 1:
                    mic_sig_per_src = self._conv(sou_sig=src_signal[:, source_idx], rir=rir_per_src[0, :, :].transpose(1, 0))
                
                else: # to be written
                    mic_sig_per_src = 0  
                    raise Exception('Uncomplete code for RIR-Source-Conv for moving source')
                    # mixeventsig = 481.6989*ctf_ltv_direct(src_signal[:, source_idx], RIRs[:, :, riridx], ir_times, self._fs_mix, self._stft_winsize_moving) / float(len(eventsig))

            mic_signal_srcs += [mic_sig_per_src]

        mic_signal_srcs = np.array(mic_signal_srcs).transpose(1, 2, 0)  # (nsamples,nch,nsources)
        mic_signal = np.sum(mic_signal_srcs, axis=2)  # (nsamples, nch) 
 
        return mic_signal, mic_signal_srcs    

    def _conv(self, sou_sig, rir):
        """ Perform convolution between source signal and room impulse reponses (RIRs)
            Args:       sou_sig   - source signal (nsample, )
                        rir       - multi-channel RIR (nrirsample, nch)
            Returns:    mic_sig   - multi-channel microphone signals (nsample, nch)
        """ 
        nsample = sou_sig.shape[0]

        # mic_sig_temp = scipy.signal.convolve(sou_sig[:, np.newaxis], rir, mode='full', method='fft')
        mic_sig_temp = scipy.signal.fftconvolve(sou_sig[:, np.newaxis], rir, mode='full', axes=0)
        mic_sig = mic_sig_temp[0:nsample, :]
        
        return mic_sig 


class MicSigFromRIRDataset(Dataset):
    def __init__(
        self,
        rirnoidataset,
        srcdataset,
        snr_range,
        fs,
        dataset_sz,
        seed,
        load_info,
        save_anno,
        save_to=None,
        ):

        self.rirdataset = rirnoidataset
        self.srcdataset = srcdataset
        self.snr_range = snr_range
        self.fs = fs
        self.seed = seed
        self.load_info = load_info
        self.save_anno = save_anno
        self.save_to = save_to
        if dataset_sz is None:
            self.dataset_sz = int(1e8)
        else:
            self.dataset_sz = dataset_sz

    def __len__(self):
        return self.dataset_sz

    def __getitem__(self, idx):
        np.random.seed(seed=self.seed+idx)

        # Get ramdom RIR
        rir_idx = np.random.randint(0, len(self.rirdataset))
        if self.load_info:
            rir, noi_sig, annos = self.rirdataset[rir_idx]
        else:
            rir, noi_sig = self.rirdataset[rir_idx]

        # Get random source signal
        src_idx = np.random.randint(0, len(self.srcdataset))
        src_sig = self.srcdataset[src_idx]

        # Generate clean or direct-path microphone signal
        mic_sig_clean, mic_sig_srcs_clean = self.rirdataset.rir_conv_src(rir, src_sig)
        rir_dp = self._find_dpmax_from_RIR(rir, dp_time=2.5, fs=self.fs)
        mic_sig_dp, mic_sig_srcs_dp = self.rirdataset.rir_conv_src(rir_dp, src_sig)

        # Generate noisy microphone signal
        snr = np.random.uniform(*self.snr_range)
        mic_sig = self.add_noise(mic_sig_clean, noi_sig, snr, mic_sig_dp=mic_sig_dp) 

        # Check whether the values of microphone signals is in the range of [-1, 1] for wav saving (soundfile.write)
        max_value = np.max(mic_sig)
        min_value = np.min(mic_sig)
        max_value_dp = np.max(mic_sig_dp)
        min_value_dp = np.min(mic_sig_dp)
        value = np.max([np.abs(max_value), np.abs(min_value), np.abs(max_value_dp), np.abs(min_value_dp)])
        mic_sig = mic_sig / value *0.9
        mic_sig_dp = mic_sig_dp / value *0.9
        mic_sig_srcs_clean = mic_sig_srcs_clean / value *0.9
        mic_sig_srcs_dp = mic_sig_srcs_dp / value *0.9

        # Save data
        if self.save_to:
            Path(self.save_to).mkdir(parents=True, exist_ok=True)
            save_to_file = os.path.join(self.save_to, str(idx) + f'.wav')
            soundfile.write(save_to_file, mic_sig, self.fs)
            if self.save_anno:
                annos['SNR'] = snr
                save_to_file = os.path.join(self.save_to, str(idx) + f'_info.npz')
                np.savez(save_to_file, **annos)

        if self.load_info:
            annos = {
                'T60': annos['T60fromDataset'].astype(np.float32), 
                'DRR': annos['DRR'].astype(np.float32),
                'C50': annos['C50'].astype(np.float32),
                'ABS': annos['ABS'].astype(np.float32),}
            return mic_sig, annos
        else:
            return mic_sig

    def _find_dpmax_from_RIR(self, rir, dp_time, fs):
        """ Function: find direct-path RIR by finding the maximum value of RIR (the accuray is not good)
            Args: rir (npoints, nmic, nsample, nsources)
        """
        nsamp = rir.shape[2]
        nd = np.argmax(rir, axis=2) # (npoint, nmic, nsources)
        nd = np.tile(nd[:,:,np.newaxis,:], (1,1,nsamp,1)) # (npoints,nch,nsamples,nsources)
        n0 = int(fs*dp_time/1000)*np.ones_like(rir)
        whole_range = np.array(range(0, nsamp))
        whole_range = np.tile(whole_range[np.newaxis,np.newaxis,:,np.newaxis], (rir.shape[0], rir.shape[1], 1, rir.shape[3]))
        dp_range = (whole_range>=(nd-n0)) & (whole_range<=(nd+n0)) 
        dp_range = dp_range.astype('float')
        dp_rir = rir*dp_range 

        return dp_rir

    def add_noise(self, mic_sig_clean, noi_sig, snr, mic_sig_dp, eps=1e-10) :
        """ Add noise to clean microphone signals with a given signal-to-noise ratio (SNR)
            Args:       mic_sig_clean - clean microphone signals without any noise (nsample, nch)
                        noi_sig       - noise signals (nsample, nch)
                        snr           - specific SNR
                        mic_sig_dp    - clean microphone signals without any noise and reverberation (nsample, nch)
            Returns:    mic_sig       - microphone signals with noise (nsample, nch)
        """ 
        nsample, _ = mic_sig_clean.shape
        if mic_sig_dp is None: # signal power includes reverberation
            av_pow = np.mean(np.sum(mic_sig_clean**2, axis=0)/nsample, axis=0) 	# average mic power across all received signals
            av_pow_noise = np.mean(np.sum(noi_sig**2, axis=0)/nsample, axis=0)
            noi_sig_snr = np.sqrt(av_pow / (10 ** (snr / 10)))/ (np.sqrt(av_pow_noise)+eps) * noi_sig
        else: # signal power do not include reverberation
            av_pow = np.mean(np.sum(mic_sig_dp**2, axis=0)/nsample, axis=0) 	# average mic power across all received signals
            av_pow_noise = np.mean(np.sum(noi_sig**2, axis=0)/nsample, axis=0)
            noi_sig_snr = np.sqrt(av_pow / (10 ** (snr / 10)))/ (np.sqrt(av_pow_noise)+eps) * noi_sig
        mic_sig = mic_sig_clean + noi_sig_snr
        
        return mic_sig

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generating multi-channel audio signals')
    parser.add_argument('--stage', type=str, default='pretrain', metavar='Stage', help='stage that generated data used for (default: Pretrain)') # ['pretrain', 'preval', 'train', 'val', 'test']
    parser.add_argument('--dataset', type=str, nargs='+', default=['DCASE'], metavar='Datasets', help='dataset names (default: DCASE)')
    parser.add_argument('--fs', type=int, default=16000, metavar='SamplingRate', help='sampling rate (default: 16000)')
    parser.add_argument('--T', type=float, default=4.112, metavar='TimeDuration', help='time duration (default: 4.112)')
    parser.add_argument('--snr_range', type=list, default=[15, 30], metavar='SNRRange', help='range of SNR (default: [15, 30])')
    parser.add_argument('--src_dir', type=str, default='', metavar='SrcDir', help='source directory')
    parser.add_argument('--rir_dir', type=str, default='', metavar='RIRDir', help='RIR directory')
    parser.add_argument('--save_dir', type=str, default='', metavar='SaveDir', help='save directory')
    parser.add_argument('--workers', type=int, default=32, metavar='Worker', help='number of workers (default: 32)')
    args = parser.parse_args()

    if args.stage == 'pretrain':
        seed = 1
    elif args.stage == 'preval':
        seed = 2e6
    elif args.stage == 'pretest':
        seed = 3e6

    dataset_list = 	   			['DCASE', 	'MIR', 		'Mesh', 	'dEchorate','BUTReverb','ACE'       ]
    sig_num_list = {'pretrain': [10240*10, 	10240*10, 	10240*10, 	10240*10,	10240*10,	10240*10,   ], 
                    'preval':	[2560, 	    2560,		2560, 		2560,		2560,		2560,		], 		 
                    }	

    for dataset_name in args.dataset:
        assert dataset_name in dataset_list, 'Dataset not found'
        dataset_idx = dataset_list.index(dataset_name)
        sig_num = sig_num_list[args.stage][dataset_idx]

        print('Dataset=', dataset_name)
        read_dir = os.path.join(args.rir_dir, dataset_name)
        save_dir = os.path.join(args.save_dir, args.stage,dataset_name)
        exist_temp = os.path.exists(save_dir)
        if exist_temp==False:
            os.makedirs(save_dir)
            print('make dir: ' + save_dir)
        else:
            print('existed dir: ' + save_dir)
            msg = input('Sure to regenerate microphone signals? (Enter for yes)')
            if msg == '':
                print('Regenerating microphone signals')

        if dataset_name == 'DCASE':
            if args.stage == 'pretrain':
                room_names = ['bomb_shelter', 'gym', 'pb132', 'pc226',
                            'sa203', 'sc203', #'se201',
                            'tc352'] 
            elif args.stage == 'preval':
                room_names = ['tb103', 'se203'] 
            rir_dir_list = []
            for room_name in room_names:
                rir_dir_list += [read_dir + '/' + room_name]
        elif dataset_name == 'Mesh':
            assert args.stage == 'pretrain', [dataset_name, args.stage]
            rir_dir_list = [read_dir]
        elif dataset_name == 'MIR':
            assert args.stage == 'pretrain', [dataset_name, args.stage]
            rir_dir_list = [read_dir]
        elif dataset_name == 'dEchorate':
            assert args.stage == 'pretrain', [dataset_name, args.stage]
            rir_dir_list = [read_dir]
        elif dataset_name == 'BUTReverb':
            if args.stage == 'pretrain':
                room_names = [
                    'Hotel_SkalskyDvur_ConferenceRoom2', 
                    'Hotel_SkalskyDvur_Room112', 
                    'VUT_FIT_L207', 
                    'VUT_FIT_L212', 
                    'VUT_FIT_L227', 
                    'VUT_FIT_Q301', 
                    'VUT_FIT_C236', 
                    'VUT_FIT_D105'] 
            elif args.stage == 'preval':
                room_names = ['VUT_FIT_E112']  
            rir_dir_list = []
            for room_name in room_names:
                rir_dir_list += [read_dir + '/' + room_name]
        elif dataset_name == 'ACE':
            assert args.stage == 'pretrain', [dataset_name, args.stage]
            rir_dir_list = [read_dir]
        print(rir_dir_list)

        # Source signal dataset 
        if args.stage == 'pretrain':
            src_dir = args.src_dir + '/tr'
        elif args.stage == 'preval':
            src_dir = args.src_dir + '/dt'
        elif args.stage == 'pretest':
            src_dir = args.src_dir + '/et'
        srcdataset = WSJ0Dataset(
            path = src_dir,
            T = args.T,
            fs = args.fs,
            num_source = 1
        )

        # RIR dataset
        rirnoidataset = RIRDataset(
            fs=args.fs, 
            rir_dir_list=rir_dir_list, 
            dataset_sz=None, 
            load_info=False,
            load_noise=True, 
            load_noise_duration=args.T
            )
        
        # Microphone signal dataset
        micdataset = MicSigFromRIRDataset(
            rirnoidataset,
            srcdataset,
            snr_range=args.snr_range,
            fs=args.fs,
            dataset_sz=sig_num,
            seed=int(seed + dataset_idx*10e6),
            load_info=False,
            save_anno=False,
            save_to=save_dir,
            )
        
        dataloader = DataLoader(micdataset, batch_size=None, shuffle=False, num_workers=args.workers) # collate_fn=at_dataset.pad_collate_fn)
        pbar = tqdm.tqdm(range(0, sig_num), desc='generating signals')
        for mic_sig in (dataloader):
            pbar.update(1)
