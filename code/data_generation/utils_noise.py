import os
import numpy as np
import math
import scipy
import scipy.io
import scipy.signal
import random 
import soundfile
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

def explore_corpus(path, file_extension):
        directory_tree = {}
        path_set = []
        for item in os.listdir(path):   
            if os.path.isdir( os.path.join(path, item) ):
                directory_tree[item], path_set_temp = explore_corpus( os.path.join(path, item), file_extension )
                path_set += path_set_temp
            elif item.split(".")[-1] == file_extension:
                directory_tree[ item.split(".")[0] ] = os.path.join(path, item)
                path_set += [os.path.join(path, item)]
        return directory_tree, path_set

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

class NoiseSignal(Dataset):
    def __init__(self, T, fs, nmic, noise_type, noise_path=None, c=343.0, size=None):
        self.T = T
        self.fs= fs
        self.nmic = nmic
        self.noise_type = noise_type 
        
        if (noise_path != None) & ((noise_type=='diffuse_xsrc') | (noise_type=='real-world')):
            _, self.path_set = explore_corpus(noise_path, 'wav')
            # self.path_set.sort()
        if (noise_type=='diffuse_xsrc') | (noise_type=='real-world'):
            self.sz = len(self.path_set) if size is None else size
        else:
            self.sz = 1 if size is None else size
        self.c = c 

    def __len__(self):
        return self.sz

    def __getitem__(self, idx):
        pass

    def generate_random_noise(self, mic_pos=None, eps=1e-8):
        # mic_pos is valid for 'diffuse'

        if self.noise_type == 'spatial_white':
            noise_signal = self.generate_Gaussian_noise(self.T, self.fs, self.nmic)

        elif self.noise_type == 'diffuse_white':
            noise = np.random.standard_normal((int(self.T * self.fs), self.nmic))
            noise_signal = self.generate_diffuse_noise(noise, mic_pos, c=self.c)
            noise_signal = noise_signal/(np.max(noise_signal)+eps)

        elif self.noise_type == 'diffuse_babble': # from single-speaker speech dataset
            # Generate M mutually 'independent' input signals
            M = mic_pos.shape[0]
            nsample_desired = int(self.T*self.fs)
            nspeech_babble = 10
            noise_M = np.zeros([nsample_desired, M])
            for m in range(0, M):
                noise = np.zeros((nsample_desired))
                for idx in range(nspeech_babble):
                    idx = np.random.randint(0, len(self.path_set))
                    speech, fs = soundfile.read(self.path_set[idx], dtype='float32')
                    if fs != self.fs:
                        speech = scipy.signal.resample_poly(speech, up=self.fs, down=fs)
                    speech = pad_cut_sig_sameutt(speech, nsample_desired)
                    speech = speech - np.mean(speech)
                    noise += speech
                noise_M[:, m] = noise 
            noise_signal = self.generate_diffuse_noise(noise_M, mic_pos, c=self.c)
            noise_signal = noise_signal/(np.max(noise_signal)+eps)
            # soundfile.write('noise.wav',noise_signal, self.fs)
            # print('save')

        elif self.noise_type == 'diffuse_xsrc':
            idx = np.random.randint(0, len(self.path_set))
            noise, fs = soundfile.read(self.path_set[idx], dtype='float32')

            nsample_desired = int(self.T * fs * self.nmic)
            noise = pad_cut_sig_sameutt(noise, nsample_desired)

            if fs != self.fs:
                 noise = scipy.signal.resample_poly(noise, up=self.fs, down=fs)

            # Generate M mutually 'independent' input signals
            M = mic_pos.shape[0]
            L = int(self.T*self.fs)
            noise = noise - np.mean(noise)
            noise_M = np.zeros([L, M])
            for m in range(0, M):
                noise_M[:, m] = noise[m*L:(m+1)*L]
            noise_signal = self.generate_diffuse_noise(noise_M, mic_pos, c=self.c)
            noise_signal = noise_signal/(np.max(noise_signal)+eps)


        elif self.noise_type == 'real_world': # The array topology should be consistent
            idx = np.random.randint(0, len(self.path_set))
            noise, fs = soundfile.read(self.path_set[idx], dtype='float32')
            nmic = noise.shape[-1]
            if nmic != self.nmic:
                raise Exception('Unexpected number of microphone channels')

            nsample_desired = int(self.T * fs)
            noise = pad_cut_sig_sameutt(noise, nsample_desired)
            if fs != self.fs:
                noise_signal = scipy.signal.resample_poly(noise, up=self.fs, down=fs)
            noise_signal = noise_signal/(np.max(noise_signal)+eps)
                
        else:
            nsample_desired = int(self.T * self.fs)
            noise_signal = np.zeros((nsample_desired, self.nmic))

        return noise_signal

    def generate_Gaussian_noise(self, T, fs, nmic):

        noise = np.random.standard_normal((int(T*fs), nmic))

        return noise

    def generate_diffuse_noise(self, noise_M, mic_pos, nfft=256, c=343.0, type_nf='spherical'):
        """ Reference:  E. A. P. Habets, “Arbitrary noise field generator.” https://github.com/ehabets/ANF-Generator
            Args: noise_M - M mutually 'independent' input signals (sample, M)
        """
        # Generate matrix with desired spatial coherence
        w_rad = 2*math.pi*self.fs*np.array([i for i in range(nfft//2+1)])/nfft
        DC = self._desired_spatial_coherence(mic_pos, type_nf, c, w_rad)

        C = self._mix_matrix(DC)
        noise_signal = self._diffuse_noise(noise_M, C)

        # Plot desired and generated spatial coherence
        # self._sc_test(DC, noise_signal, mic_idxes=[0, 1], save_name='diffuse')

        return noise_signal
    
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

    def _desired_spatial_coherence(self, mic_pos, type_nf, c, w_rad):
        """
			mic_pos: relative positions of mirophones  (nmic, 3)
			type_nf: type of noise field, 'spherical' or 'cylindrical'
			c: speed of sound
			w_rad: angular frequency in radians
		"""
        M = mic_pos.shape[0]

        # Generate matrix with desired spatial coherence
        nf = len(w_rad)
        DC = np.zeros([M, M, nf])

        for p in range(0,M):
            for q in range(0,M):
                if p == q:
                    DC[p, q, :] = np.ones([1, 1, nf])
                else:
                    dist = np.linalg.norm(mic_pos[p, :] - mic_pos[q, :])
                    if type_nf == 'spherical':
                        DC[p, q, :] = np.sinc(w_rad*dist/(c*math.pi))
                    elif type_nf == 'cylindrical':
                        DC[p, q, :] = scipy.special.jn(0, w_rad*dist/c)
                    else:
                        raise Exception('Unknown noise field')
        # Alternative
        # dist = np.linalg.norm(mic_pos[:, np.newaxis, :] - mic_pos[np.newaxis, :, :], axis=-1, keepdims=True)
        # if type_nf == 'spherical':
        # 	DC = np.sinc(w_rad * dist / (c * math.pi))
        # elif type_nf == 'cylindrical':
        # 	DC = scipy.special.jn(0, w_rad * dist / c)
        # else:
        # 	raise Exception('Unknown noise field')

        return DC

    def _mix_matrix(self, DC, method='cholesky'):
        """ 
			DC: desired spatial coherence (nch, nch, nf)
			C:mix matrix (nf, nch, nch)
		"""

        M = DC.shape[0]
        num_freqs = DC.shape[2] 
        C = np.zeros((num_freqs, M, M), dtype=complex)
        for k in range(1, num_freqs):
            if method == 'cholesky':
                C[k, ...] = scipy.linalg.cholesky(DC[:,:,k])
            elif method == 'eigen': # Generated cohernce and noise signal are slightly different from MATLAB version
                D, V = np.linalg.eig(DC[:,:,k])
                C[k, ...] = V.T * np.sqrt(D)[:, np .newaxis]
            else:
                raise Exception('Unknown method specified')

        return C
    
    def _diffuse_noise(self, noise, C):
        """ 
			C: mix matrix (nf, nch, nch)
			x: diffuse noise (nsample, nch)
		"""

        K = (C.shape[0]-1)*2 # Number of frequency bins

        # Compute short-time Fourier transform (STFT) of all input signals
        noise = noise.transpose()
        f, t, N = scipy.signal.stft(noise, window='hann', nperseg=K, noverlap=0.75*K, nfft=K)

        # Generate output in the STFT domain for each frequency bin k
        X = np.einsum('fmn,mft->nft', np.conj(C), N)

        # Compute inverse STFT
        F, df_noise = scipy.signal.istft(X,window='hann', nperseg=K, noverlap=0.75*K, nfft=K)
        df_noise = df_noise.transpose()

        return df_noise

    def _sc_test(self, DC, df_noise, mic_idxes, save_name='SC'):
        dc = DC[mic_idxes[0], mic_idxes[1], :]
        noise = df_noise[:, mic_idxes].transpose()

        nfft = (DC.shape[2]-1)*2
        f, t, X = scipy.signal.stft(noise, window='hann', nperseg=nfft, noverlap=0.75*nfft, nfft=nfft)  # X: [M,F,T]
        phi_x = np.mean(np.abs(X)**2, axis=2)
        psi_x = np.mean(X[0, :, :] * np.conj(X[1, :, :]), axis=-1)
        sc_generated = np.real(psi_x / np.sqrt(phi_x[[0], :] * phi_x[[1], :]))

        sc_theory = dc[np.newaxis, :]

        plt.switch_backend('agg')
        plt.plot(list(range(nfft // 2 + 1)), sc_theory[0, :])
        plt.plot(list(range(nfft // 2 + 1)), sc_generated[0, :])
        plt.title(str(1))
        # plt.show()
        plt.savefig(save_name)

    



 