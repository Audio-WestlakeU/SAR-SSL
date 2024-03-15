""" 
    Define some basic modules
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def complex_multiplication(x, y):
    return torch.stack([ x[...,0]*y[...,0] - x[...,1]*y[...,1],   x[...,0]*y[...,1] + x[...,1]*y[...,0]  ], dim=-1)


def complex_conjugate_multiplication(x, y):
    return torch.stack([ x[...,0]*y[...,0] + x[...,1]*y[...,1],   x[...,1]*y[...,0] - x[...,0]*y[...,1]  ], dim=-1)


# def complex_cart2polar(x):
#     mod = torch.sqrt( complex_conjugate_multiplication(x, x)[..., 0] )
#     phase = torch.atan2(x[..., 1], x[..., 0])
#     return torch.stack((mod, phase), dim=-1)


class STFT(nn.Module):
    """ Get STFT coefficients of microphone signals (batch processing by pytorch)
        Args:       win_len         - the length of frame / window
                    win_shift_ratio - the ratio between frame shift and frame length
                    nfft            - the number of fft points
                    win             - window type 
                                      'boxcar': a rectangular window (equivalent to no window at all)
                                      'hann': a Hann window
					signal          - the microphone signals in time domain (nbatch, nsample, nch)
        Returns:    stft            - STFT coefficients (nbatch, nf, nt, nch)
    """

    def __init__(self, win_len, win_shift_ratio, nfft, win='hann', inv=False):
        super(STFT, self).__init__()

        self.win_len = win_len
        self.win_shift_ratio = win_shift_ratio
        self.nfft = nfft
        self.win = win
        self.inv = inv

    def forward(self, signal):

        nsample = signal.shape[-2]
        nch = signal.shape[-1]
        win_shift = int(self.win_len * self.win_shift_ratio)
        nf = int(self.nfft / 2) + 1

        nb = signal.shape[0]
        if self.inv:
            nt = int((nsample) / win_shift) + 1  # for iSTFT
        else:
            nt = np.floor((nsample - self.win_len) / win_shift + 1).astype(int)
        stft = torch.zeros((nb, nf, nt, nch), dtype=torch.complex64).to(signal.device)

        if self.win == 'hann':
            window = torch.hann_window(window_length=self.win_len, device=signal.device)
        for ch_idx in range(0, nch, 1):
            if self.inv:
                stft[:, :, :, ch_idx] = torch.stft(signal[:, :, ch_idx], n_fft = self.nfft, hop_length = win_shift, win_length = self.win_len,
                window = window, center = True, normalized = False, return_complex = True)  # for iSTFT
            else:
                stft[:, :, :, ch_idx] = torch.stft(signal[:, :, ch_idx], n_fft=self.nfft, hop_length=win_shift, win_length=self.win_len,
                    window=window, center=False, normalized=False, return_complex=True)
        return stft


class ISTFT(nn.Module):
    """ Get inverse STFT (batch processing by pytorch) 
		Args:		stft            - STFT coefficients (nbatch, nf, nt, nch)
					win_len         - the length of frame / window
					win_shift_ratio - the ratio between frame shift and frame length
					nfft            - the number of fft points
		Returns:	signal          - time-domain microphone signals (nbatch, nsample, nch)
	"""
    def __init__(self, win_len, win_shift_ratio, nfft, inv=False):
        super(ISTFT, self).__init__()

        self.win_len = win_len
        self.win_shift_ratio = win_shift_ratio
        self.nfft = nfft
        self.inv = inv

    def forward(self, stft):
        # stft: nb, nf, nt, nch
        nf = stft.shape[-3]
        nt = stft.shape[-2]
        nch = stft.shape[-1]
        nb = stft.shape[0]
        win_shift = int(self.win_len * self.win_shift_ratio)
        if self.inv:
            nsample = (nt - 1) * win_shift
        else:
            nsample = (nt + 1) * win_shift #-1
        signal = torch.zeros((nb, nsample, nch)).to(stft.device)
        win_shift = int(self.win_len * self.win_shift_ratio)
        for ch_idx in range(0, nch, 1):
            if self.inv:
                signal_temp = torch.istft(stft[:, :, :, ch_idx], n_fft=self.nfft, hop_length=win_shift, win_length=self.win_len,
                                                    center=True, normalized=False, return_complex=False)
                signal[:, :, ch_idx] = signal_temp[:, 0:nsample] # for STFT
            else:
                signal_temp = torch.istft(stft[:, :, :, ch_idx], n_fft=self.nfft, hop_length=win_shift, win_length=self.win_len,
                                                    center=False, normalized=False, return_complex=False)
                signal[:, :, ch_idx] = signal_temp
        return signal


class AddChToBatch(nn.Module):
    """ Change dimension from  (nb, nch, ...) to (nb*(nch-1), ...) 
	"""
    def __init__(self, ch_mode):
        super(AddChToBatch, self).__init__()
        self.ch_mode = ch_mode

    def forward(self, data):
        nb = data.shape[0]
        nch = data.shape[1]

        if self.ch_mode == 'M':
            data_adjust = torch.zeros((nb*(nch-1),2)+data.shape[2:], dtype=torch.complex64).to(data.device) # (nb*(nch-1),2,nf,nt)
            for b_idx in range(nb):
                st = b_idx*(nch-1)
                ed = (b_idx+1)*(nch-1)
                data_adjust[st:ed, 0, ...] = data[b_idx, 0 : 1, ...].expand((nch-1,)+data.shape[2:])
                data_adjust[st:ed, 1, ...] = data[b_idx, 1 : nch, ...]

        elif self.ch_mode == 'MM':
            data_adjust = torch.zeros((nb*int((nch-1)*nch/2),2)+data.shape[2:], dtype=torch.complex64).to(data.device) # (nb*(nch-1)*nch/2,2,nf,nt)
            for b_idx in range(nb):
                for ch_idx in range(nch-1):
                    st = b_idx*int((nch-1)*nch/2) + int((2*nch-2-ch_idx+1)*ch_idx/2)
                    ed = b_idx*int((nch-1)*nch/2) + int((2*nch-2-ch_idx)*(ch_idx+1)/2)
                    data_adjust[st:ed, 0, ...] = data[b_idx, ch_idx:ch_idx+1, ...].expand((nch-ch_idx-1,)+data.shape[2:])
                    data_adjust[st:ed, 1, ...] = data[b_idx, ch_idx+1:, ...]

        return data_adjust.contiguous()

class RemoveChFromBatch(nn.Module):
    """ Change dimension from (nb*nmic, nt, nf) to (nb, nmic, nt, nf)
	"""
    def __init__(self, ch_mode):
        super(RemoveChFromBatch, self).__init__()
        self.ch_mode = ch_mode

    def forward(self, data, nb):
        nmic = int(data.shape[0]/nb)
        data_adjust = torch.zeros((nb, nmic)+data.shape[1:], dtype=torch.float32).to(data.device)
        for b_idx in range(nb):
            st = b_idx * nmic
            ed = (b_idx + 1) * nmic
            data_adjust[b_idx, ...] = data[st:ed, ...]

        return data_adjust.contiguous()



# %% Patch Split and Patch Mask

class PatchSplit(nn.Module):
    def __init__(self, patch_shape, f_first=False):
        super(PatchSplit, self).__init__()

        self.patch_shape = patch_shape
        self.f_first = f_first

    def forward(self, data):
		# data: (nbatch, nf, nt, nmic)/ (nbatch, nf, nt, nreim, nmic)

        if len(data.shape) == 4:
            nbatch, _, _, nmic = data.shape
            data = data.permute(0, 3, 1, 2) # (nbatch, nmic, nf, nt)
            if self.f_first:
                data = data.permute(0, 1, 3, 2) # (nbatch, nmic, nt, nf)
                vec = F.unfold(data, kernel_size=[self.patch_shape[-1],self.patch_shape[0]], stride=[self.patch_shape[-1],self.patch_shape[0]])
            else:
                vec = F.unfold(data, kernel_size=self.patch_shape, stride=self.patch_shape)  # (nbatch, nmic*dpatch, npatch) unfold first the last dim, then the second last, like conv layers
            vec = vec.reshape(nbatch, nmic, self.patch_shape[0] * self.patch_shape[1], vec.shape[-1])  # (nbatch, nmic, dpatch, npatch)
            vec = vec.permute(0, 3, 1, 2) # (nbatch, npatch, dpatch, nmic)

        elif len(data.shape) == 5:
            nbatch, nf, nt, nreim, nmic = data.shape
            data = data.permute(0, 3, 4, 1, 2).reshape(nbatch, nreim*nmic, nf, nt) # (nbatch, nreim*nmic, nf, nt)
            if self.f_first:
                data = data.permute(0, 1, 3, 2) # (nbatch, nreim*nmic, nt, nf)
                vec = F.unfold(data, kernel_size=[self.patch_shape[-1],self.patch_shape[0]], stride=[self.patch_shape[-1],self.patch_shape[0]])  
            else:
                vec = F.unfold(data, kernel_size=self.patch_shape, stride=self.patch_shape)  # (nbatch, nreim*nmic*dpatch, npatch)
            vec = vec.reshape(nbatch, nreim, nmic, self.patch_shape[0] * self.patch_shape[1], vec.shape[-1])  # (nbatch, nreim, nmic, dpatch, npatch)
            vec = vec.permute(0, 4, 3, 1, 2) # (nbatch, npatch, dpatch, nreim, nmic)

        return vec  # (nbatch, npatch, dpatch, nreim, nmic)


class PatchRecover(nn.Module):
    def __init__(self, output_shape, patch_shape, f_first=False):
        super(PatchRecover, self).__init__()

        self.output_shape = output_shape
        self.patch_shape = patch_shape
        self.f_first = f_first

    def forward(self, data):
        # data: (nbatch, npatch, dpatch, nmic) / (nbatch, npatch, dpatch, nreim, nmic)
        nf = self.output_shape[0]
        nt = self.output_shape[1]
        if len(data.shape) == 4:
            nbatch, npatch, _, nmic = data.shape
            vec = data.permute(0, 3, 2, 1).reshape(nbatch, -1, npatch)  # (nbatch, nmic, dpatch, npatch)
            if self.f_first:
                vec = F.fold(vec, kernel_size=[self.patch_shape[-1],self.patch_shape[0]], stride=[self.patch_shape[-1],self.patch_shape[0]], output_size=(nt, nf))  # (nbatch, nmic, nt, nf)
                vec = vec.permute(0, 1, 3, 2) # (nbatch, nmic, nf, nt)
            else:
                vec = F.fold(vec, kernel_size=self.patch_shape, stride=self.patch_shape, output_size=(nf, nt))  # (nbatch, nmic, nf, nt)
            vec = vec.reshape(nbatch, nmic, nf, nt)  # (nbatch, nmic, nf, nt)
            vec = vec.permute(0, 2, 3, 1) # (nbatch, nf, nt, nmic)

        elif len(data.shape) == 5: 
            nbatch, npatch, _, nreim, nmic = data.shape
            vec = data.permute(0, 3, 4, 2, 1).reshape(nbatch, -1, npatch)  # (nbatch, nreim, nmic, dpatch, npatch)
            if self.f_first:
                vec = F.fold(vec, kernel_size=[self.patch_shape[-1],self.patch_shape[0]], stride=[self.patch_shape[-1],self.patch_shape[0]], output_size=(nt, nf))  # (nbatch, nreim*nmic, nt, nf)
                vec = vec.permute(0, 1, 3, 2) # (nbatch, nreim*nmic, nf, nt)
            else:
                vec = F.fold(vec, kernel_size=self.patch_shape, stride=self.patch_shape, output_size=(nf, nt))  # (nbatch, nreim*nmic, nf, nt)
            vec = vec.reshape(nbatch, nreim, nmic, nf, nt)  # (nbatch, nreim, nmic, nf, nt)
            vec = vec.permute(0, 3, 4, 1, 2) # (nbatch, nf, nt, nreim, nmic)

        return vec  # (nbatch, nf, nt, nmic) / (nbatch, nf, nt, nreim, nmic)


class PatchMask(nn.Module):
    def __init__(self, patch_mode, nmasked_patch, npatch_shape, device):
        super(PatchMask, self).__init__()
        self.patch_mode = patch_mode
        self.nmasked_patch = nmasked_patch
        self.npatch_shape = npatch_shape
        self.device = device

    def forward(self, data_shape):
        nbatch, npatch, dpatch, _, nmic = data_shape

        mask_dense = torch.ones((nbatch, npatch, dpatch, nmic), device=self.device)  # (nbatch, npatch, dpatch, nmic)
        mask_patch_dense = torch.ones((nbatch, npatch, dpatch, nmic), device=self.device)  # (nbatch, npatch, dpatch, nmic)
        mask_ch_dense = torch.ones((nbatch, npatch, dpatch, nmic), device=self.device)  # (nbatch, npatch, dpatch, nmic)
        mask_patch_idx = torch.empty((nbatch, self.nmasked_patch), device=self.device, requires_grad=False).long()  # (nbatch, nmasked_patch)
        mask_ch_idx = torch.empty((nbatch, 1), device=self.device, requires_grad=False).long() # (nbatch, 1)
        for b_idx in range(nbatch):
            # randomly generate # nmasked_patch mask indexes without duplicate
            mask_patch_idx[b_idx, :] = self.gen_mask_idx(npatch_shape=self.npatch_shape, nmasked_patch=self.nmasked_patch, patch_mode=self.patch_mode)
            mask_ch_idx[b_idx, :] = random.randrange(nmic)
            mask_dense[b_idx, mask_patch_idx[b_idx], :, mask_ch_idx[b_idx, :]] = 0
            mask_patch_dense[b_idx, mask_patch_idx[b_idx], :, :] = 0
            mask_ch_dense[b_idx, :, :, mask_ch_idx[b_idx, :]] = 0

        return mask_dense, mask_patch_dense, mask_ch_dense, mask_patch_idx, mask_ch_idx
        # mask_dense/mask_patch_dense/mask_ch_dense: (nbatch, npatch, dpatch, nmic), mask_patch_idx: (nbatch, nmasked_patch), mask_ch_idx: (nbatch, 1)

    def gen_mask_idx(self, npatch_shape=[16, 16], nmasked_patch=10, cluster=1, patch_mode='TF'):
        npatch = npatch_shape[0] * npatch_shape[1]

        if nmasked_patch > npatch:
            raise Exception('Number of masked patches is out of range')

        if patch_mode == 'TF':
            """ 
				generate mask for patches with a dimension of 16*16 (time frames*frequency bins)
				Process: find one block of patches, and then another .... until the number of patches reaches nmasked_patch 

			"""
            mask_id = []

            # randomize clutering factor in [3,6)
            cur_clus = random.randrange(cluster) + 3

            while len(list(set(mask_id))) <= nmasked_patch:
                start_id = random.randrange(npatch)

                cur_mask = []
                for i in range(0, cur_clus):
                    for j in range(0, cur_clus):
                        mask_cand = start_id + npatch_shape[1] * i + j
                        if mask_cand > 0 and mask_cand < npatch:
                            cur_mask.append(mask_cand)
                mask_id = mask_id + cur_mask
            mask_id = list(set(mask_id))[:nmasked_patch]
            return torch.tensor(mask_id)

        elif patch_mode == 'T':

            mask_id = random.sample(range(0, npatch), nmasked_patch)
            return torch.tensor(mask_id)
        
        elif patch_mode == 'T_cluster':
            mask_id = []

            # randomize clutering factor in [0,5)
            cur_clus = random.randrange(cluster) + 5

            while len(list(set(mask_id))) <= nmasked_patch:
                start_id = random.randrange(npatch)

                cur_mask = []
                for i in range(0, cur_clus):
                        mask_cand = start_id + i
                        if mask_cand > 0 and mask_cand < npatch:
                            cur_mask.append(mask_cand)
                mask_id = mask_id + cur_mask
            mask_id = list(set(mask_id))[:nmasked_patch]

            return torch.tensor(mask_id)
        
        elif patch_mode == 'T_cluster_inverse':
            mask_id = []
            mask_id_set = list(range(npatch))

            # randomize clutering factor in [0,5)
            cur_clus = random.randrange(cluster) + 5

            while len(list(set(mask_id))) <= nmasked_patch:
                start_id = random.randrange(npatch)

                cur_mask = []
                for i in range(0, cur_clus):
                        mask_cand = start_id + i
                        if mask_cand > 0 and mask_cand < npatch:
                            cur_mask.append(mask_cand)
                mask_id = mask_id + cur_mask
            mask_id = list(set(mask_id))[:nmasked_patch]

            for id in mask_id:
                mask_id_set.remove(id)

            return torch.tensor(mask_id_set)
        
        elif patch_mode == 'T_cluster2':
            clu_size = 5
            clu_start_id = random.sample(range(0, npatch, clu_size), math.ceil(nmasked_patch/clu_size)+1)
            mask_id = []
            for clu_idx in range(len(clu_start_id)):
                for idx in range(clu_size):
                    mask_cand = clu_start_id[clu_idx] + idx
                    if mask_cand > 0 and mask_cand < npatch:
                        mask_id = mask_id + [mask_cand]
            mask_id = list(set(mask_id))[:nmasked_patch]

            return torch.tensor(mask_id)
        
        elif patch_mode == 'T_1s':

            mask_id = list(range(192, 256)) # 1s masked
            return torch.tensor(mask_id)

        else:
            raise Exception('Patch mode is unrecognized')


# class DPIPD(nn.Module):
#     """ Complex-valued Direct-path inter-channel phase difference	
# 	"""

#     def __init__(self, ndoa_candidate, mic_location, nf=257, fre_max=8000, ch_mode='M', speed=343.0):
#         super(DPIPD, self).__init__()

#         self.ndoa_candidate = ndoa_candidate
#         self.mic_location = mic_location
#         self.nf = nf
#         self.fre_max = fre_max
#         self.speed = speed
#         self.ch_mode = ch_mode

#         nmic = mic_location.shape[-2]
#         nele = ndoa_candidate[0]
#         nazi = ndoa_candidate[1]
#         ele_candidate = np.linspace(0, np.pi, nele)
#         azi_candidate = np.linspace(-np.pi, np.pi, nazi)
#         ITD = np.empty((nele, nazi, nmic, nmic))  # Time differences, floats
#         IPD = np.empty((nele, nazi, nf, nmic, nmic))  # Phase differences
#         fre_range = np.linspace(0.0, fre_max, nf)
#         for m1 in range(nmic):
#             for m2 in range(nmic):
#                 r = np.stack([np.outer(np.sin(ele_candidate), np.cos(azi_candidate)),
#                      np.outer(np.sin(ele_candidate), np.sin(azi_candidate)),
#                      np.tile(np.cos(ele_candidate), [nazi, 1]).transpose()], axis=2)
#                 ITD[:, :, m1, m2] = np.dot(r, mic_location[m2, :] - mic_location[m1, :]) / speed
#                 IPD[:, :, :, m1, m2] = -2 * np.pi * np.tile(fre_range[np.newaxis, np.newaxis, :], [nele, nazi, 1]) * \
#                         np.tile(ITD[:, :, np.newaxis, m1, m2], [1, 1, nf])
#         dpipd_template_ori = np.exp(1j * IPD)
#         self.dpipd_template = self.data_adjust(dpipd_template_ori) # (nele, nazi, nf, nmic-1) / (nele, nazi, nf, nmic*(nmic-1)/2)

#         # 	# import scipy.io
#         # 	# scipy.io.savemat('dpipd_template_nele_nazi_2nf_nmic-1.mat',{'dpipd_template': self.dpipd_template})
#         # 	# print(a)

#         del ITD, IPD

#     def forward(self, source_doa=None):
#         # source_doa: (nb, ntimestep, 2, nsource)
#         mic_location = self.mic_location
#         nf = self.nf
#         fre_max = self.fre_max
#         speed = self.speed

#         if source_doa is not None:
#             source_doa = source_doa.transpose(0, 1, 3, 2) # (nb, ntimestep, nsource, 2)
#             nmic = mic_location.shape[-2]
#             nb = source_doa.shape[0]
#             nsource = source_doa.shape[-2]
#             ntime = source_doa.shape[-3]
#             ITD = np.empty((nb, ntime, nsource, nmic, nmic))  # Time differences, floats
#             IPD = np.empty((nb, ntime, nsource, nf, nmic, nmic))  # Phase differences
#             fre_range = np.linspace(0.0, fre_max, nf)

#             for m1 in range(nmic):
#                 for m2 in range(nmic):
#                     r = np.stack([np.sin(source_doa[:, :, :, 0]) * np.cos(source_doa[:, :, :, 1]),
#                          np.sin(source_doa[:, :, :, 0]) * np.sin(source_doa[:, :, :, 1]),
#                          np.cos(source_doa[:, :, :, 0])], axis=3)
#                     ITD[:, :, :, m1, m2] = np.dot(r, mic_location[m1, :] - mic_location[m2, :]) / speed # t2- t1
#                     IPD[:, :, :, :, m1, m2] = -2 * np.pi * np.tile(fre_range[np.newaxis, np.newaxis, np.newaxis, :],
#                          [nb, ntime, nsource, 1]) * np.tile(ITD[:, :, :, np.newaxis, m1, m2], [1, 1, 1, nf])*(-1)  # !!!! delete -1

#             dpipd_ori = np.exp(1j * IPD)
#             dpipd = self.data_adjust(dpipd_ori) # (nb, ntime, nsource, nf, nmic-1) / (nb, ntime, nsource, nf, nmic*(nmic-1)/2)

#             dpipd = dpipd.transpose(0, 1, 3, 4, 2) # (nb, ntime, nf, nmic-1, nsource)

#         else:
#             dpipd = None

#         return self.dpipd_template, dpipd

#     def data_adjust(self, data):
#         # change dimension from (..., nmic-1) to (..., nmic*(nmic-1)/2)
#         if self.ch_mode == 'M':
#             data_adjust = data[..., 0, 1:] # (..., nmic-1)
#         elif self.ch_mode == 'MM':
#             nmic = data.shape[-1]
#             data_adjust = np.empty(data.shape[:-2] + (int(nmic*(nmic-1)/2),), dtype=np.complex64)
#             for mic_idx in range(nmic - 1):
#                 st = int((2 * nmic - 2 - mic_idx + 1) * mic_idx / 2)
#                 ed = int((2 * nmic - 2 - mic_idx) * (mic_idx + 1) / 2)
#                 data_adjust[..., st:ed] = data[..., mic_idx, (mic_idx+1):] # (..., nmic*(nmic-1)/2)
#         else:
#             raise Exception('Microphone channel mode unrecognised')

#         return data_adjust


# class GCC(nn.Module):
# 	""" Compute the Generalized Cross Correlation of the inputs.
# 	In the constructor of the layer, you need to indicate the number of signals (N) and the window length (K).
# 	You can use tau_max to output only the central part of the GCCs and transform='PHAT' to use the PHAT transform.
# 	"""

# 	def __init__(self, N, K, tau_max=None, transform=None):
# 		assert transform is None or transform == 'PHAT', 'Only the \'PHAT\' transform is implemented'
# 		assert tau_max is None or tau_max <= K // 2
# 		super(GCC, self).__init__()

# 		self.K = K
# 		self.N = N
# 		self.tau_max = tau_max if tau_max is not None else K // 2
# 		self.transform = transform

# 	def forward(self, x):
# 		x_fft_c = torch.fft.rfft(x)
# 		x_fft = torch.stack((x_fft_c.real, x_fft_c.imag), -1)  

# 		if self.transform == 'PHAT':
# 			mod = torch.sqrt(complex_conjugate_multiplication(x_fft, x_fft))[..., 0]
# 			mod += 1e-12  # To avoid numerical issues
# 			x_fft /= mod.reshape(tuple(x_fft.shape[:-1]) + (1,))

# 		gcc = torch.empty(list(x_fft.shape[0:-3]) + [self.N, self.N, 2 * self.tau_max + 1], device=x.device)
# 		for n in range(self.N):
# 			gcc_fft_batch = complex_conjugate_multiplication(x_fft[..., n, :, :].unsqueeze(-3), x_fft)
# 			gcc_fft_batch_c = torch.complex(gcc_fft_batch[..., 0], gcc_fft_batch[..., 1])
# 			gcc_batch = torch.fft.irfft(gcc_fft_batch_c)    

# 			gcc[..., n, :, 0:self.tau_max + 1] = gcc_batch[..., 0:self.tau_max + 1]
# 			gcc[..., n, :, -self.tau_max:] = gcc_batch[..., -self.tau_max:]

# 		return gcc


# class SRP_map(nn.Module):
#     """ Compute the SRP-PHAT maps from the GCCs taken as input.
# 	In the constructor of the layer, you need to indicate the number of signals (N) and the window length (K), the
# 	desired resolution of the maps (resTheta and resPhi), the microphone positions relative to the center of the
# 	array (rn) and the sampling frequency (fs).
# 	With normalize=True (default) each map is normalized to ethe range [-1,1] approximately
# 	"""

#     def __init__(self, N, K, resTheta, resPhi, rn, fs, c=343.0, normalize=True, thetaMax=np.pi / 2):
#         super(SRP_map, self).__init__()

#         self.N = N
#         self.K = K
#         self.resTheta = resTheta
#         self.resPhi = resPhi
#         self.fs = float(fs)
#         self.normalize = normalize

#         self.cross_idx = np.stack([np.kron(np.arange(N, dtype='int16'), np.ones((N), dtype='int16')),
#                  np.kron(np.ones((N), dtype='int16'), np.arange(N, dtype='int16'))])

#         self.theta = np.linspace(0, thetaMax, resTheta)
#         self.phi = np.linspace(-np.pi, np.pi, resPhi + 1)
#         self.phi = self.phi[0:-1]

#         self.IMTDF = np.empty((resTheta, resPhi, self.N, self.N))  # Time differences, floats
#         for k in range(self.N):
#             for l in range(self.N):
#                 r = np.stack(
#                  [np.outer(np.sin(self.theta), np.cos(self.phi)), np.outer(np.sin(self.theta), np.sin(self.phi)),
#                   np.tile(np.cos(self.theta), [resPhi, 1]).transpose()], axis=2)
#                 self.IMTDF[:, :, k, l] = np.dot(r, rn[l, :] - rn[k, :]) / c

#         tau = np.concatenate([range(0, K // 2 + 1), range(-K // 2 + 1, 0)]) / float(fs)  # Valid discrete values
#         self.tau0 = np.zeros_like(self.IMTDF, dtype=np.int)
#         for k in range(self.N):
#             for l in range(self.N):
#                 for i in range(resTheta):
#                     for j in range(resPhi):
#                         self.tau0[i, j, k, l] = int(np.argmin(np.abs(self.IMTDF[i, j, k, l] - tau)))
#         self.tau0[self.tau0 > K // 2] -= K
#         self.tau0 = self.tau0.transpose([2, 3, 0, 1])

#     def forward(self, x):
#         tau0 = self.tau0
#         tau0[tau0 < 0] += x.shape[-1]
#         maps = torch.zeros(list(x.shape[0:-3]) + [self.resTheta, self.resPhi], device=x.device).float()
#         for n in range(self.N):
#             for m in range(self.N):
#                 maps += x[..., n, m, tau0[n, m, :, :]]

#         if self.normalize:
#             maps -= torch.mean(torch.mean(maps, -1, keepdim=True), -2, keepdim=True)
#             maps += 1e-12  # To avoid numerical issues
#             maps /= torch.max(torch.max(maps, -1, keepdim=True)[0], -2, keepdim=True)[0]

#         return maps


if __name__ == "__main__":
    import torch
    win_len = 512
    win_shift_ratio = 0.5
    nfft = 512
    a = torch.randn((100, 512, 1))
    inv = False
    dostft = STFT(
        win_len=win_len, 
		win_shift_ratio=win_shift_ratio, 
		nfft=nfft,
        inv=inv)
    doistft = ISTFT(
        win_len=win_len, 
		win_shift_ratio=win_shift_ratio, 
		nfft=nfft,
        inv=inv)
    stft = dostft(a)
    aa = doistft(stft)
    print(a.shape, stft.shape, aa.shape)  

