import numpy as np
import os
import math
import copy
import csv
import scipy
import scipy.io
import scipy.signal
import random
import warnings
import pandas as pd
# import librosa # cause CPU overload, for data generation (scipy.signal.resample, librosa.resample) 
import soundfile
import webrtcvad
import gpuRIR
from collections import namedtuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from torch.utils.data import Dataset
from common.utils import load_file
from common.utils_room_acoustics import add_noise, sou_conv_rir, cart2sph, sph2cart, rt60_from_rirs, dpRIR_from_RIR


ArraySetup = namedtuple('ArraySetup', 'arrayType, orV, mic_scale, mic_rotate, mic_pos, mic_orV, mic_pattern')
# orV: put the source in oneside (indicated by orV) of the array
# mic_scale: half of the mic_distance should be smaller than the minimum separation between the array and the walls defined by array_pos
# mic_rotate: anticlockwise azimuth rotate in degrees
# mic_pos: relative normalized microphone postions, actural position is mic_scale*(mic_pos w mic_rotate)+array_pos*room_sz
# mic_orV: Invalid for omnidirectional microphones
# Named tuple with the characteristics of a microphone array and definitions of dual-channel array

class Parameter:
    """ Random parammeter class
	"""
    def __init__(self, *args, discrete=False):
        self.discrete = discrete
        if discrete:
            self.value_range = args[0]
        else:
            if len(args) == 1:
                self.min_value = np.array(args[0])
                self.max_value = np.array(args[0])
            elif len(args) == 2:
                self.min_value = np.array(args[0])
                self.max_value = np.array(args[1])
            else:
                raise Exception('Parammeter must be called with one (value) or two (min and max value) array_like parammeters')

    def getValue(self):
        if self.discrete:
            idx = np.random.randint(0, len(self.value_range))
            return self.value_range[idx]
        else:
            return self.min_value + np.random.random(self.min_value.shape) * (self.max_value - self.min_value)

dualch_array_setup = ArraySetup(arrayType='planar_linear',
    orV = np.array([0.0, 1.0, 0.0]), 
    mic_scale = Parameter(0.3, 2), 
    mic_rotate = Parameter(0, 360), 
    mic_pos = np.array(((-0.05, 0.0, 0.0),
                        (0.05, 0.0, 0.0))), 
    mic_orV = np.array(((-1.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0))), 
    mic_pattern = 'omni'
)

class AcousticScene:
    """ Acoustic scene class
	"""
    def __init__(self, room_sz, beta, T60, array_setup, mic_pos, array_pos, traj_pts, fs, RIR, source_signal=[], noise_signal=[], SNR=[], timestamps=[], t=[], trajectory=[], c=343.0):
        self.room_sz = room_sz  # Room size
        self.T60 = T60  # Reverberation time of the simulated room
        self.beta = beta  # Reflection coefficients of the walls of the room (make sure it corresponds with T60)
        self.array_setup = array_setup  # Named tuple with the characteristics of the array
        self.mic_pos = mic_pos  # Position of microphones (nch,3)/(npoint,nch,3) moving array
        self.array_pos = array_pos  # Position of array center (3,)/(npoint,3) moving array
        self.traj_pts = traj_pts  # Trajectory points to simulate (npoint,3,nsource)
        self.fs = fs  # Sampling frequency of the source signal and the simulations
        self.RIR = RIR  # Room impulse responses [RIR, dp_RIR] (npoint,nch,nsam,nsource)
        if len(source_signal)==0 & len(noise_signal)==0 & bool(SNR==[]) & len(timestamps)==0 & len(t)==0 & len(trajectory)==0:
            pass
        else:
            self.source_signal = source_signal  # Source signal (nsample,nsource)
            self.noise_signal = noise_signal  # Noise signal (nsample',nch)
            self.SNR = SNR  # Signal to (omnidirectional) noise ratio to simulate
            self.timestamps = timestamps  # Time of each simulation (it does not need to correspond with the DOA estimations) (npoint)
            self.t = t  # Continuous time (nsample)
            self.trajectory = trajectory  # Continuous trajectory (nsample,3,nsource)
        self.c = c  # Speed of sound

    def simulate(self, gpuConv=False, eps=1e-8): 
        RIRs_sources = self.RIR[0]  # (npoint,nch,nsam,nsource)
        dp_RIRs_sources = self.RIR[-1] # M1
        num_source = self.traj_pts.shape[-1]

        # Source conv. RIR
        mic_signals_sources = []
        dp_mic_signals_sources = []
        nsample = len(self.t)
        for source_idx in range(num_source):
            RIRs = RIRs_sources[:, :, :, source_idx]  # (npoint,nch,nsample）
            dp_RIRs = dp_RIRs_sources[:, :, :, source_idx]
            if gpuConv:
                mic_sig = gpuRIR.simulateTrajectory(self.source_signal[:, source_idx], RIRs, timestamps=self.timestamps, fs=self.fs)
                dp_mic_sig = gpuRIR.simulateTrajectory(self.source_signal[:, source_idx], dp_RIRs, timestamps=self.timestamps, fs=self.fs)
                mic_sig = mic_sig[0:nsample, :]
                dp_mic_sig = dp_mic_sig[0:nsample, :]
            else:
                if RIRs.shape[0] == 1:
                    mic_sig = sou_conv_rir(sou_sig=self.source_signal[:, source_idx], rir=RIRs[0, :, :].transpose(1, 0))
                    dp_mic_sig = sou_conv_rir(sou_sig=self.source_signal[:, source_idx], rir=dp_RIRs[0, :, :].transpose(1, 0))
                else: # to be written
                    mic_sig = 0  
                    dp_mic_sig = 0
                    raise Exception('Uncomplete code for RIR-Source-Conv for moving source')
                    # mixeventsig = 481.6989*ctf_ltv_direct(self.source_signal[:, source_idx], RIRs[:, :, riridx], ir_times, self._fs_mix, self._stft_winsize_moving) / float(len(eventsig))

            mic_signals_sources += [mic_sig]
            dp_mic_signals_sources += [dp_mic_sig]

        mic_signals_sources = np.array(mic_signals_sources).transpose(1, 2, 0)  # (nsamples,nch,nsources)
        dp_mic_signals_sources = np.array(dp_mic_signals_sources).transpose(1, 2, 0)

        # Add Noise
        mic_signals = np.sum(mic_signals_sources, axis=2)  # (nsamples, nch) 
        dp_mic_signals = np.sum(dp_mic_signals_sources, axis=2)
        mic_signals = add_noise(mic_signals, self.noise_signal, self.SNR, mic_sig_wonr=dp_mic_signals)

        # Check whether the values of microphone signals is in the range of [-1, 1] for wav saving (soundfile.write)
        max_value = np.max(mic_signals)
        min_value = np.min(mic_signals)
        max_value_dp = np.max(dp_mic_signals)
        min_value_dp = np.min(dp_mic_signals)
        value = np.max([np.abs(max_value), np.abs(min_value), np.abs(max_value_dp), np.abs(min_value_dp)])
        mic_signals = mic_signals / value
        dp_mic_signals = dp_mic_signals / value
        mic_signals_sources = mic_signals_sources / value
        dp_mic_signals_sources = dp_mic_signals_sources / value

        # Use the signal-to-noise ratio (dp_mic_signals/mic_signals) to compute the VAD
        # must combine with clean silence of source signals, to avoid the random results when a smaller value divided by a smaller value
        # the denominator is the mixed signals of multiple sources, which may be problematic when the number of sources is larger
        # segment-level results approximate to webrtc
        if hasattr(self, 'mic_vad'):
            if hasattr(self, 'mic_vad'):
                sig_len = mic_signals.shape[0]
                win_len = int(self.fs * 0.032) # 32ms 
                win_shift_ratio = 1
                nt = int((sig_len - win_len*(1-win_shift_ratio)) / (win_len*win_shift_ratio))
                self.mic_vad_sources = np.zeros((nsample, num_source))
                th = 0.001**2
                for t_idx in range(nt):
                    st = int(t_idx * win_len * win_shift_ratio)
                    ed = st + win_len 
                    dp_mic_signal_sources_sch = dp_mic_signals_sources[st:ed, 0, :]
                    mic_signal_sources_sch = mic_signals[st:ed, 0]
                    win_engergy_ratio = np.sum(dp_mic_signal_sources_sch**2, axis=0) / (np.sum(mic_signal_sources_sch**2, axis=0) + eps) 
                    self.mic_vad_sources[st:ed, :] = win_engergy_ratio[np.newaxis, :].repeat(win_len, axis=0) 
                self.mic_vad = np.sum(self.mic_vad_sources, axis=1) #>= th

        # Apply the propagation delay to the VAD information if it exists
        elif hasattr(self, 'source_vad'): 
            self.mic_vad_sources = []  # binary value, for vad of separate sensor signals of sources
            for source_idx in range(num_source):
                if gpuConv:
                    vad = gpuRIR.simulateTrajectory(self.source_vad[:, source_idx], dp_RIRs_sources[:, :, :, source_idx], timestamps=self.timestamps, fs=self.fs)
                vad_sources = vad[0:nsample, :].mean(axis=1) > vad[0:nsample, :].max() * 1e-3
                self.mic_vad_sources += [vad_sources] 
            self.mic_vad_sources = np.array(self.mic_vad_sources).transpose(1, 0)
            self.mic_vad = np.sum(self.mic_vad_sources, axis=1) > 0.5  # binary value, for vad of mixed sensor signals of sources

        if hasattr(self, 'DOA'):  # [ele, azi]
            self.DOA = np.zeros((nsample, 2, num_source))  # (nsample, 2, nsource)
            for source_idx in range(num_source):
                self.DOA[:, :, source_idx] = cart2sph(self.trajectory[:, :, source_idx] - self.array_pos)[:, [1,0]]

        if hasattr(self, 'TDOA'): 
            npoint = self.traj_pts.shape[0]
            nmic = self.mic_pos.shape[-2]
            if len(self.mic_pos.shape) == 2:
                mic_pos = np.tile(self.mic_pos[np.newaxis, :, :], (npoint, 1, 1))
            elif len(self.mic_pos.shape) == 3:
                mic_pos = self.mic_pos
            else:
                raise Exception('shape of mic_pos is out of range~')
            corr_diff = np.tile(self.traj_pts[:, np.newaxis, :, :], (1, nmic, 1, 1)) - np.tile(mic_pos[:, :, :, np.newaxis], (1, 1, 1, num_source))
            dist = np.sqrt(np.sum(corr_diff**2, axis=2))  # (npoint,3,nsource)-(nch,3)=(nnpoint,nch,3,nsource)
            re_dist = dist[:, 1:, :] - np.tile(dist[:, 0:1, :], (1, nmic - 1, 1))  # (npoint,nch-1,nsource)
            TDOA = re_dist / self.c  # (npoint,nch-1,nsource)

            # interpolate 
            self.TDOA = np.zeros((nsample, TDOA.shape[1], num_source))  # (nsample,nch-1,nsource)
            for source_idx in range(num_source):
                for ch_idx in range(TDOA.shape[1]):
                    self.TDOA[:, ch_idx, source_idx] = np.interp(self.t, self.timestamps, TDOA[:, ch_idx, source_idx])

        if hasattr(self, 'DRR') | hasattr(self, 'C50') | hasattr(self, 'C80'):
            RIR_len = RIRs_sources.shape[2]
            dp_RIR_len = dp_RIRs_sources.shape[2]
            nmic = self.mic_pos.shape[-2]
            nb_traj_pts = self.traj_pts.shape[0]
            zeros_pad = np.zeros((nb_traj_pts, nmic, abs(RIR_len - dp_RIR_len), num_source))
            if RIR_len >= dp_RIR_len:  # When RT60=0.15s, RIR_len = dp_RIR_len
                dp_RIRs_sources_pad = np.concatenate((dp_RIRs_sources, zeros_pad), axis=2)  # (npoints,nch,nsamples,nsources)
                RIRs_sources_pad = RIRs_sources
            else:
                dp_RIRs_sources_pad = dp_RIRs_sources
                RIRs_sources_pad = np.concatenate((RIRs_sources, zeros_pad), axis=2)  # (npoints,nch,nsamples,nsources)

            if hasattr(self, 'DRR'):
                if hasattr(self, 'DRRfromDataset'):
                    DRR = self.DRRfromDataset[np.newaxis, 0, np.newaxis] # for real-world dataset with anotations
                else:
                    # Calculate DRR according to RIR and dp_RIR
                    # dp_pow = np.sum(dp_RIRs_sources_pad**2, axis=2) # (npoints,nch,nsources)
                    # rev_pow = np.sum((RIRs_sources_pad-dp_RIRs_sources_pad)**2, axis=2) # (npoints,nch,nsources)
                    # DRR = 10*np.log10(dp_pow/rev_pow) # (npoints,nch,nsources)
                    # DRR = np.mean(DRR, axis=1) # (npoints,nsources)

                    # Calculate DRR according to RIR
                    nsamp = np.max([dp_RIR_len, RIR_len])
                    nd = np.argmax(dp_RIRs_sources_pad, axis=2)  # (npoints,nch,nsources)
                    nd = np.tile(nd[:, :, np.newaxis, :], (1, 1, nsamp, 1))  # (npoints,nch,nsamples,nsources)
                    n0 = int(self.fs * 2.5 / 1000) * np.ones_like(RIRs_sources_pad)
                    whole_range = np.array(range(0, nsamp))
                    whole_range = np.tile(whole_range[np.newaxis, np.newaxis, :, np.newaxis], (RIRs_sources_pad.shape[0], RIRs_sources_pad.shape[1], 1, RIRs_sources_pad.shape[3]))
                    dp_range = (whole_range >= (nd - n0)) & (whole_range <= (nd + n0))
                    dp_range = dp_range.astype('float')
                    rev_range = np.ones_like(dp_range) - dp_range
                    dp_pow = np.sum(RIRs_sources_pad**2 * dp_range, axis=2)
                    rev_pow = np.sum(RIRs_sources_pad**2 * rev_range, axis=2)
                    DRR = 10 * np.log10(dp_pow / (rev_pow+eps)+eps)  # (npoints,nch,nsources)
                    DRR = DRR[:, 0, :]  # reference channel (npoints,nsources)

                # Interpolate 
                self.DRR = np.zeros((nsample, num_source))
                for source_idx in range(num_source):
                    self.DRR[:, source_idx] = np.interp(self.t, self.timestamps, DRR[:, source_idx])  # (nsample,nsource)
                    # np.array([np.interp(self.t, self.timestamps, DRR[:,i,source_idx]) for i in range(nch)]).transpose() # (nsample,nch,nsource)

            if hasattr(self, 'C50'):
                nsamp = np.max([dp_RIR_len, RIR_len])
                nd = np.argmax(dp_RIRs_sources_pad, axis=2)  # (npoints,nch,nsources)
                nd = np.tile(nd[:, :, np.newaxis, :], (1, 1, nsamp, 1))  # (npoints,nch,nsamples,nsources)
                n0 = int(self.fs * 50 / 1000) * np.ones_like(RIRs_sources_pad)
                whole_range = np.array(range(0, nsamp))
                whole_range = np.tile(whole_range[np.newaxis, np.newaxis, :, np.newaxis], (RIRs_sources_pad.shape[0], RIRs_sources_pad.shape[1], 1, RIRs_sources_pad.shape[3]))
                early_range = (whole_range <= (nd + n0))
                early_range = early_range.astype('float')
                late_range = np.ones_like(early_range) - early_range
                early_pow = np.sum(RIRs_sources_pad**2 * early_range, axis=2)
                late_pow = np.sum(RIRs_sources_pad**2 * late_range, axis=2)
                C50 = 10 * np.log10(early_pow / (late_pow + eps)+eps)  # (npoints,nch,nsources)
                C50 = C50[:, 0, :]  # reference channel, (npoints,nsources)

                # Interpolate C50 
                self.C50 = np.zeros((nsample, num_source))
                for source_idx in range(num_source):
                    self.C50[:, source_idx] = np.interp(self.t, self.timestamps, C50[:, source_idx])  # (nsample,nsource)

            if hasattr(self, 'C80'):
                nsamp = np.max([dp_RIR_len, RIR_len])
                nd = np.argmax(dp_RIRs_sources_pad, axis=2)  # (npoints,nch,nsources)
                nd = np.tile(nd[:, :, np.newaxis, :], (1, 1, nsamp, 1))  # (npoints,nch,nsamples,nsources)
                n0 = int(self.fs * 80 / 1000) * np.ones_like(RIRs_sources_pad)
                whole_range = np.array(range(0, nsamp))
                whole_range = np.tile(whole_range[np.newaxis, np.newaxis, :, np.newaxis], (RIRs_sources_pad.shape[0], RIRs_sources_pad.shape[1], 1, RIRs_sources_pad.shape[3]))
                # early_range = (whole_range >= (nd - n0)) & (whole_range <= (nd + n0))
                early_range = (whole_range <= (nd + n0))
                early_range = early_range.astype('float')
                late_range = np.ones_like(early_range) - early_range
                early_pow = np.sum(RIRs_sources_pad**2 * early_range, axis=2)
                late_pow = np.sum(RIRs_sources_pad**2 * late_range, axis=2)
                C80 = 10 * np.log10(early_pow / (late_pow + eps)+eps)  # (npoints,nch,nsources)
                C80 = C80[:, 0, :]  # reference channel, (npoints,nsources)

                # Interpolate 
                self.C80 = np.zeros((nsample, num_source))
                for source_idx in range(num_source):
                    self.C80[:, source_idx] = np.interp(self.t, self.timestamps, C80[:, source_idx])  # (nsample,nsource)

        if hasattr(self, 'dp_mic_signal'):
            self.dp_mic_signal = dp_mic_signals_sources  # (nsamples,nch,nsources)

        if hasattr(self, 'speakerID'):
            pass  # to be written

        return mic_signals

    def plotScene(self, view='3D'):
        """ Plots the source trajectory and the microphones within the room
		"""
        assert view in ['3D', 'XYZ', 'XY', 'XZ', 'YZ']

        plt.switch_backend('agg')
        fig = plt.figure()
        src_idx = 0
        pt_idx = 0 
        # self.mic_pos = self.mic_pos[pt_idx,...]
        if view == '3D' or view == 'XYZ':
            ax = Axes3D(fig)
            ax.set_xlim3d(0, self.room_sz[0])
            ax.set_ylim3d(0, self.room_sz[1])
            ax.set_zlim3d(0, self.room_sz[2])

            ax.scatter(self.traj_pts[:,0,src_idx], self.traj_pts[:,1,src_idx], self.traj_pts[:,2,src_idx])
            ax.scatter(self.mic_pos[:,0], self.mic_pos[:,1], self.mic_pos[:,2])
            ax.text(self.traj_pts[0,0], self.traj_pts[0,1], self.traj_pts[0,2], 'start')

            ax.set_title('$T_{60}$' + ' = {:.3f}s, SNR = {:.1f}dB'.format(self.T60, self.SNR))
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_zlabel('z [m]')

        else:
            ax = fig.add_subplot(111)
            plt.gca().set_aspect('equal', adjustable='box')

            if view == 'XY':
                ax.set_xlim(0, self.room_sz[0])
                ax.set_ylim(0, self.room_sz[1])
                ax.scatter(self.traj_pts[:,0], self.traj_pts[:,1])
                ax.scatter(self.mic_pos[:,0], self.mic_pos[:,1])
                ax.text(self.traj_pts[0,0], self.traj_pts[0,1], 'start')
                ax.legend(['Source trajectory', 'Microphone array'])
                ax.set_xlabel('x [m]')
                ax.set_ylabel('y [m]')
            elif view == 'XZ':
                ax.set_xlim(0, self.room_sz[0])
                ax.set_ylim(0, self.room_sz[2])
                ax.scatter(self.traj_pts[:,0], self.traj_pts[:,2])
                ax.scatter(self.mic_pos[:,0], self.mic_pos[:,2])
                ax.text(self.traj_pts[0,0], self.traj_pts[0,2], 'start')
                ax.legend(['Source trajectory', 'Microphone array'])
                ax.set_xlabel('x [m]')
                ax.set_ylabel('z [m]')
            elif view == 'YZ':
                ax.set_xlim(0, self.room_sz[1])
                ax.set_ylim(0, self.room_sz[2])
                ax.scatter(self.traj_pts[:,1], self.traj_pts[:,2])
                ax.scatter(self.mic_pos[:,1], self.mic_pos[:,2])
                ax.text(self.traj_pts[0,1], self.traj_pts[0,2], 'start')
                ax.legend(['Source trajectory', 'Microphone array'])
                ax.set_xlabel('y [m]')
                ax.set_ylabel('z [m]')

        # plt.show()
        plt.savefig('scene')

    def plotDOA(self):
        """ Plots the groundtruth DOA
		"""
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.t, self.DOA * 180/np.pi)

        ax.legend(['Elevation', 'Azimuth'])
        ax.set_xlabel('time [s]')
        ax.set_ylabel('DOA [º]')

        plt.show()

    def plotEstimation(self, legned_loc='best'):
        """ Plots the DOA groundtruth and its estimation.
		"""
        fig = plt.figure()
        gs = fig.add_gridspec(7, 1)
        ax = fig.add_subplot(gs[0,0])
        ax.plot(self.t, self.source_signal)
        plt.xlim(self.tw[0], self.tw[-1])
        plt.tick_params(axis='both', which='both', bottom=False, labelbottom=False, left=False, labelleft=False)

        ax = fig.add_subplot(gs[1:,0])
        ax.plot(self.tw, self.DOAw * 180/np.pi)
        plt.gca().set_prop_cycle(None)
        ax.plot(self.tw, self.DOAw_pred * 180/np.pi, '--')
        if hasattr(self, 'DOAw_srpMax'):
            plt.gca().set_prop_cycle(None)
            ax.plot(self.tw, self.DOAw_srpMax * 180 / np.pi, 'x', markersize=4)

        plt.legend(['Elevation', 'Azimuth'], loc=legned_loc)
        plt.xlabel('time [s]')
        plt.ylabel('DOA [º]')

        silences = self.mic_vad.mean(axis=1) < 2/3
        silences_idx = silences.nonzero()[0]
        start, end = [], []
        for i in silences_idx:
            if not i - 1 in silences_idx:
                start.append(i)
            if not i + 1 in silences_idx:
                end.append(i)
        for s, e in zip(start, end):
            plt.axvspan((s-0.5)*self.tw[1], (e+0.5)*self.tw[1], facecolor='0.5', alpha=0.5)

        plt.xlim(self.tw[0], self.tw[-1])
        plt.show()

## Source signal Datasets

class WSJ0Dataset(Dataset):
    """ WSJ0Dataset
        train: /tr 81h (both speaker independent and dependent)
        val: /dt 5h
        test: /et 5h
    """
    def _exploreCorpus(self, path, file_extension):
        directory_tree = {}
        for item in os.listdir(path):
            if os.path.isdir( os.path.join(path, item) ):
                directory_tree[item] = self._exploreCorpus( os.path.join(path, item), file_extension )
            elif item.split(".")[-1] == file_extension:
                directory_tree[ item.split(".")[0] ] = os.path.join(path, item)
        return directory_tree
    
    def __init__(self, path, T, fs, num_source=1, size=None):
        self.corpus = self._exploreCorpus(path, 'wav')
        self.paths = []
        file_names = self.corpus.keys()
        for file_name in file_names:
            spk_names = self.corpus[file_name].keys()
            for spk_name in spk_names:
                wav_names = self.corpus[file_name][spk_name].keys()
                for wav_name in wav_names:
                    path = self.corpus[file_name][spk_name][wav_name]
                    self.paths += [path]
        # self.paths.sort()
        random.shuffle(self.paths)
        self.fs = fs
        self.T = T
        self.sum_source = num_source
        self.sz = len(self.paths) if size is None else min([len(self.paths), size])

    def __len__(self):
        return self.sz

    def __getitem__(self, idx):
        if idx < 0: idx = len(self) + idx

        s_shape_desired = int(self.T * self.fs)
        s_sources = []
        assert self.sum_source == 1 # for num_source>1, to be written
        for s_idx in range(self.sum_source):
            s = np.array([])
            while s.shape[0] < s_shape_desired:
                utterance, fs = soundfile.read(self.paths[idx])
                if fs != self.fs:
                    utterance = scipy.signal.resample_poly(utterance, up=self.fs, down=fs)
                    raise Warning('WSJ0 is downsampled to requrired frequency~')
                s = np.concatenate([s, utterance])
            st = random.randint(0, s.shape[0] - s_shape_desired)
            ed = st + s_shape_desired
            s = s[st: ed]
            s -= s.mean()
            s_sources += [s]
        s_sources = np.array(s_sources).transpose(1,0)

        return s_sources

class LibriSpeechDataset(Dataset):
    """ LibriSpeechDataset (about 1000h)
        https://www.openslr.org/12
	"""

    def _exploreCorpus(self, path, file_extension):
        directory_tree = {}
        for item in os.listdir(path):
            if os.path.isdir( os.path.join(path, item) ):
                directory_tree[item] = self._exploreCorpus( os.path.join(path, item), file_extension )
            elif item.split(".")[-1] == file_extension:
                directory_tree[ item.split(".")[0] ] = os.path.join(path, item)
        return directory_tree

    def _cleanSilences(self, s, aggressiveness, return_vad=False):
        self.vad.set_mode(aggressiveness)

        vad_out = np.zeros_like(s)
        vad_frame_len = int(10e-3 * self.fs)  # 0.001s,16samples gives one same vad results
        n_vad_frames = len(s) // vad_frame_len # 1/0.001s
        for frame_idx in range(n_vad_frames):
            frame = s[frame_idx * vad_frame_len: (frame_idx + 1) * vad_frame_len]
            frame_bytes = (frame * 32767).astype('int16').tobytes()
            vad_out[frame_idx*vad_frame_len: (frame_idx+1)*vad_frame_len] = self.vad.is_speech(frame_bytes, self.fs)
        s_clean = s * vad_out

        return (s_clean, vad_out) if return_vad else s_clean

    def __init__(self, path, T, fs, num_source, size=None, return_vad=False, readers_range=None, clean_silence=True):
        self.corpus = self._exploreCorpus(path, 'flac')
        if readers_range is not None:
            for key in list(map(int, self.nChapters.keys())):
                if int(key) < readers_range[0] or int(key) > readers_range[1]:
                    del self.corpus[key]

        self.nReaders = len(self.corpus)
        self.nChapters = {reader: len(self.corpus[reader]) for reader in self.corpus.keys()}
        self.nUtterances = {reader: {
          chapter: len(self.corpus[reader][chapter]) for chapter in self.corpus[reader].keys()
         } for reader in self.corpus.keys()}

        self.chapterList = []
        for chapters in list(self.corpus.values()):
            self.chapterList += list(chapters.values())
        # self.chapterList.sort()

        self.fs = fs
        self.T = T
        self.num_source = num_source

        self.clean_silence = clean_silence
        self.return_vad = return_vad
        self.vad = webrtcvad.Vad()

        self.sz = len(self.chapterList) if size is None else size

    def __len__(self):
        return self.sz

    def __getitem__(self, idx):
        if idx < 0: idx = len(self) + idx
        while idx >= len(self.chapterList): idx -= len(self.chapterList)

        s_sources = []
        s_clean_sources = []
        vad_out_sources = []
        speakerID_list = []

        for source_idx in range(self.num_source):
            if source_idx==0:
                chapter = self.chapterList[idx]
                utts = list(chapter.keys())
                spakerID = utts[0].split('-')[0]
            else:
                idx_othersources = np.random.randint(0, len(self.chapterList))
                chapter = self.chapterList[idx_othersources]
                utts = list(chapter.keys())
                spakerID = utts[0].split('-')[0]
                while spakerID in speakerID_list:
                    idx_othersources = np.random.randint(0, len(self.chapterList))
                    chapter = self.chapterList[idx_othersources]
                    utts = list(chapter.keys())
                    spakerID = utts[0].split('-')[0]

            speakerID_list += [spakerID]

            utt_paths = list(chapter.values())
            s_shape_desired = int(self.T * self.fs)
            s_clean = np.zeros((s_shape_desired, 1)) # random initialization
            while np.sum(s_clean) == 0: # avoid full-zero s_clean
                # Get a random speech segment from the selected chapter
                n = np.random.randint(0, len(chapter))
                s = np.array([])
                while s.shape[0] < s_shape_desired:
                    utterance, fs = soundfile.read(utt_paths[n])
                    assert fs == self.fs
                    s = np.concatenate([s, utterance])
                    n += 1
                    if n >= len(chapter): n=0
                st = random.randint(0, s.shape[0] - s_shape_desired)
                ed = st + s_shape_desired
                s = s[st: ed]
                s -= s.mean()

                # Clean silences, it starts with the highest aggressiveness of webrtcvad,
                # but it reduces it if it removes more than the 66% of the samples
                s_clean, vad_out = self._cleanSilences(s, 3, return_vad=True)
                if np.count_nonzero(s_clean) < len(s_clean) * 0.66:
                    s_clean, vad_out = self._cleanSilences(s, 2, return_vad=True)
                if np.count_nonzero(s_clean) < len(s_clean) * 0.66:
                    s_clean, vad_out = self._cleanSilences(s, 1, return_vad=True)

            s_sources += [s]
            s_clean_sources += [s_clean]
            vad_out_sources += [vad_out]

        s_sources = np.array(s_sources).transpose(1,0)
        s_clean_sources = np.array(s_clean_sources).transpose(1,0)
        vad_out_sources = np.array(vad_out_sources).transpose(1,0)

        # scipy.io.savemat('source_data.mat',{'s':s_sources, 's_clean':s_clean_sources})

        if self.clean_silence:
            return (s_clean_sources, vad_out_sources) if self.return_vad else s_clean_sources
        else:
            return (s_sources, vad_out_sources) if self.return_vad else s_sources

# Noise signal Dataset
class NoiseDataset(Dataset):
    def __init__(self, T, fs, nmic, noise_type, noise_path=None, c=343.0, size=None):
        self.T = T
        self.fs= fs
        self.nmic = nmic
        self.noise_type = noise_type # 'diffuse' and 'real_world' cannot exist at the same time
        
        if (noise_path != None) & (('diffuse_xsrc' in noise_type.value_range) | ('real-world' in noise_type.value_range)):
            _, self.path_set = self._exploreCorpus(noise_path, 'wav')
            # self.path_set.sort()
        if ('diffuse_xsrc' in noise_type.value_range) | ('real-world' in noise_type.value_range):
            self.sz = len(self.path_set) if size is None else size
        else:
            self.sz = 1 if size is None else size
        self.c = c 

    def __len__(self):
        return self.sz

    def __getitem__(self, idx):
        pass

    def get_random_noise(self, mic_pos=None, acoustic_scene=None, source_dataset=None, eps=1e-8):
        # mic_pos is valid for 'diffuse'
        noise_type = self.noise_type.getValue()

        if noise_type == 'spatial_white':
            noise_signal = self.gen_Gaussian_noise(self.T, self.fs, self.nmic)

        elif noise_type == 'diffuse_white':
            noise = np.random.standard_normal((int(self.T * self.fs), self.nmic))
            noise_signal = self.gen_diffuse_noise(noise, mic_pos, c=self.c)
            noise_signal = noise_signal/(np.max(noise_signal)+eps)

        elif noise_type == 'diffuse_babble':
            # babbles (number of channels) are independent 
            pass

        elif noise_type == 'diffuse_xsrc':
            idx = random.randint(0, len(self.path_set)-1)
            noise, fs = soundfile.read(self.path_set[idx])

            nsample_desired = int(self.T * fs * self.nmic)
            nsample = noise.shape[0]
            if nsample < nsample_desired:
                noise_copy = copy.deepcopy(noise)
                while nsample < nsample_desired:
                    noise_copy = np.concatenate((noise_copy, noise), axis=0)
                    nsample = noise_copy.shape[0]
                st = random.randint(0, nsample - nsample_desired)
                ed = st + nsample_desired
                noise_copy = noise_copy[st:ed]
            else:
                st = random.randint(0, nsample - nsample_desired)
                ed = st + nsample_desired
                noise_copy = noise[st:ed]

            if fs != self.fs:
                noise_copy = scipy.signal.resample_poly(noise_copy, up=self.fs, down=fs)

            # Generate M mutually 'independent' input signals
            M = mic_pos.shape[0]
            L = int(self.T*self.fs)
            noise = noise - np.mean(noise)
            noise_M = np.zeros([L, M])
            for m in range(0, M):
                noise_M[:, m] = noise[m*L:(m+1)*L]
            noise_signal = self.gen_diffuse_noise(noise_M, mic_pos, c=self.c)
            noise_signal = noise_signal/(np.max(noise_signal)+eps)

        elif noise_type == 'diffuse_babble_fromRIR':
            num_source = 20

            RIR_is_ok = False
            while(RIR_is_ok==False):
                # Trajectory points for noise
                traj_pts_noise = self.genTrajectory(acoustic_scene.array_setup, acoustic_scene.room_sz, acoustic_scene.array_pos, acoustic_scene.traj_pts.shape[0], num_source) #  (npoint,3,nsource)

                # RIRs for noise
                RIRs_noise = self.genRIR(acoustic_scene.array_setup, acoustic_scene.room_sz, acoustic_scene.beta, acoustic_scene.T60, acoustic_scene.mic_pos, traj_pts_noise) # (npoint, nch, nsample, nsource）
                late_rev_time = 0/1000
                RIRs_noise[:, :, :int(late_rev_time*self.fs), :] = 0

                RIR_is_ok = self.checkRIR(RIRs_noise)
            
            # RIR conv noise source
            nsample_desired = int(self.T * self.fs)
            noise_signal = np.zeros((nsample_desired, self.nmic))
            idxes = random.sample([i for i in range(len(source_dataset))], num_source)
            for source_idx in range(num_source):
                idx = idxes[source_idx]
                sou_sig = source_dataset[idx][:, 0]
                noi_sig = sou_conv_rir(sou_sig=sou_sig, rir=RIRs_noise[0, :, :, source_idx].transpose(1, 0))
                noi_pow = np.mean(np.sum(noi_sig ** 2, axis=0))
                noise_signal += noi_sig / (noi_pow + eps)

            # plot generated spatial coherence
            # nfft = 25
            # w_rad = 2*math.pi*self.fs*np.array([i for i in range(nfft//2+1)])/nfft
            # DC = self.gen_desired_spatial_coherence(mic_pos, type_nf='spherical', c=343, w_rad=w_rad)
            # self.sc_test(DC, noise_signal, mic_idxes=[0, 1], save_name='diffuse_fromRIR')

        elif noise_type == 'diffuse_white_fromRIR':
            num_source = 20

            RIR_is_ok = False
            while(RIR_is_ok==False):
                # Trajectory points for noise
                traj_pts_noise = self.genTrajectory(acoustic_scene.array_setup, acoustic_scene.room_sz, acoustic_scene.array_pos, acoustic_scene.traj_pts.shape[0], num_source) #  (npoint,3,nsource)

                # RIRs for noise
                RIRs_noise = self.genRIR(acoustic_scene.array_setup, acoustic_scene.room_sz, acoustic_scene.beta, acoustic_scene.T60, acoustic_scene.mic_pos, traj_pts_noise) # (npoint, nch, nsample, nsource）
                late_rev_time = 0/1000
                RIRs_noise[:, :, :int(late_rev_time*self.fs), :] = 0

                RIR_is_ok = self.checkRIR(RIRs_noise)
            
            # RIR conv noise source
            nsample_desired = int(self.T * self.fs)
            noise_signal = np.zeros((nsample_desired, self.nmic))
            noise_source = np.random.standard_normal((nsample_desired, num_source))
            for source_idx in range(num_source):
                sou_sig = noise_source[:, source_idx]
                noi_sig = sou_conv_rir(sou_sig=sou_sig, rir=RIRs_noise[0, :, :, source_idx].transpose(1, 0))
                noi_pow = np.mean(np.sum(noi_sig ** 2, axis=0))
                noise_signal += noi_sig / (noi_pow + eps)

        elif noise_type == 'real_world': # The array topology should be consistent
            idx = random.randint(0, len(self.path_set)-1)
            noise, fs = soundfile.read(self.path_set[idx])
            nmic = noise.shape[-1]
            if nmic != self.nmic:
                raise Exception('Unexpected number of microphone channels')

            nsample_desired = int(self.T * fs)
            noise_copy = copy.deepcopy(noise)
            nsample = noise.shape[0]
            while nsample < nsample_desired:
                noise_copy = np.concatenate((noise_copy, noise), axis=0)
                nsample = noise_copy.shape[0]

            st = random.randint(0, nsample - nsample_desired)
            ed = st + nsample_desired
            noise_copy = noise_copy[st:ed, :]

            if fs != self.fs:
                noise_signal = scipy.signal.resample_poly(noise_copy, up=self.fs, down=fs)
                noise_signal = noise_signal/(np.max(noise_signal)+eps)
                
        else:
            nsample_desired = int(self.T * self.fs)
            noise_signal = np.zeros((nsample_desired, self.nmic))

        return noise_signal

    def _exploreCorpus(self, path, file_extension):
        directory_tree = {}
        directory_path = []
        for item in os.listdir(path):
            if os.path.isdir( os.path.join(path, item) ):
                directory_tree[item], directory_path = self._exploreCorpus( os.path.join(path, item), file_extension )
            elif item.split(".")[-1] == file_extension:
                directory_tree[ item.split(".")[0] ] = os.path.join(path, item)
                directory_path += [os.path.join(path, item)]
        return directory_tree, directory_path

    def gen_Gaussian_noise(self, T, fs, nmic):

        noise = np.random.standard_normal((int(T*fs), nmic))

        return noise

    def gen_diffuse_noise(self, noise_M, mic_pos, nfft=256, c=343.0, type_nf='spherical'):
        """ Reference:  E. A. P. Habets, “Arbitrary noise field generator.” https://github.com/ehabets/ANF-Generator
            Args: noise_M - M mutually 'independent' input signals (sample, M)
        """
        # Generate matrix with desired spatial coherence
        w_rad = 2*math.pi*self.fs*np.array([i for i in range(nfft//2+1)])/nfft
        DC = self.gen_desired_spatial_coherence(mic_pos, type_nf, c, w_rad)

        C = self.mix_matrix(DC)
        noise_signal = self.diffuse_noise(noise_M, C)

        # Plot desired and generated spatial coherence
        # self.sc_test(DC, noise_signal, mic_idxes=[0, 1], save_name='diffuse')

        return noise_signal

    def gen_desired_spatial_coherence(self, mic_pos, type_nf, c, w_rad):
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
                        DC[p, q, :] = scipy.special(0, w_rad*dist/c)
                    else:
                        raise Exception('Unknown noise field')
        # Alternative
        # dist = np.linalg.norm(mic_pos[:, np.newaxis, :] - mic_pos[np.newaxis, :, :], axis=-1, keepdims=True)
        # if type_nf == 'spherical':
        # 	DC = np.sinc(w_rad * dist / (c * math.pi))
        # elif type_nf == 'cylindrical':
        # 	DC = scipy.special(0, w_rad * dist / c)
        # else:
        # 	raise Exception('Unknown noise field')

        return DC

    def mix_matrix(self, DC, method='cholesky'):
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
    
    def diffuse_noise(self, noise, C):
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


    def sc_test(self, DC, df_noise, mic_idxes, save_name='SC'):
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

    def genTrajectory(self, array_setup, room_sz, array_pos, nb_points, num_source, source_state='static'):
        src_pos_min = np.array([0.0, 0.0, 0.0])
        src_pos_max = room_sz * 1

        traj_pts = np.zeros((nb_points, 3, num_source))
        for source_idx in range(num_source):
            if source_state == 'static':
                src_pos = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)
                traj_pts[:, :, source_idx] = np.ones((nb_points, 1)) * src_pos

            elif source_state == 'mobile':
                src_pos_ini = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)
                src_pos_end = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)

                Amax = np.min(np.stack((src_pos_ini - src_pos_min,
                                        src_pos_max - src_pos_ini,
                                        src_pos_end - src_pos_min,
                                        src_pos_max - src_pos_end)),
                                        axis=0)

                A = np.random.random(3) * np.minimum(Amax, 1) # Oscilations with 1m as maximum in each axis
                w = 2*np.pi / nb_points * np.random.random(3) * 2  # Between 0 and 2 oscilations in each axis

                traj_pts[:, :, source_idx] = np.array([np.linspace(i,j,nb_points) for i,j in zip(src_pos_ini, src_pos_end)]).transpose()
                traj_pts[:, :, source_idx] += A * np.sin(w * np.arange(nb_points)[:, np.newaxis])

                if np.random.random(1) < 0.25:
                    traj_pts[:, :, source_idx] = np.ones((nb_points, 1)) * src_pos_ini 

        return traj_pts

    def genRIR(self, array_setup, room_sz, beta, T60, mic_pos, traj_pts):
        if T60 == 0:
            Tdiff = 0.1
            Tmax = 0.1
            nb_img = [1,1,1]

        else:
            Tdiff = gpuRIR.att2t_SabineEstimator(12, T60) # Use ISM until the RIRs decay 12dB
            Tmax = gpuRIR.att2t_SabineEstimator(40, T60)  # Use diffuse model until the RIRs decay 40dB
            if T60 < 0.15: Tdiff = Tmax # Avoid issues with too short RIRs
            nb_img = gpuRIR.t2n(Tdiff, room_sz)

        RIRs_sources = []
        num_source = traj_pts.shape[-1]
        for source_idx in range(num_source):
            RIRs = gpuRIR.simulateRIR(room_sz=room_sz, beta=beta, pos_src=traj_pts[:,:,source_idx], pos_rcv=mic_pos,
               nb_img=nb_img, Tmax=Tmax, fs=self.fs, Tdiff=Tdiff, orV_rcv=array_setup.mic_orV,
               mic_pattern=array_setup.mic_pattern, c=self.c)
            RIRs_sources += [RIRs]
        RIRs_sources = np.array(RIRs_sources).transpose(1,2,3,0) # (npoints,nch,nsamples,nsources)
        
        return RIRs_sources
    
    def checkRIR(self, RIRs):
        ok_flag = True
        nan_flag = np.isnan(RIRs)
        inf_flag = np.isinf(RIRs)
        if (True in nan_flag):
            warnings.warn('NAN exists in noise RIR~')
            ok_flag = False
        if (True in inf_flag):
            warnings.warn('INF exists in RIR~')
            ok_flag = False
        zero_flag = (np.sum(RIRs) == 0)
        if zero_flag:
            warnings.warn('Noise RIR is all zeros~')
            ok_flag = False
        return ok_flag

#  RIR Datasets
class RIRDataset(Dataset):
    """ Load RIR data as class, and use all RIR data in the list of data dirs and subdirs
	"""
    def search_files(self, dir, file_extension):
        paths = []
        for item in os.listdir(dir):
            if os.path.isdir( os.path.join(dir, item) ):
                paths += self.search_files( os.path.join(dir, item), file_extension )
            elif item.split(".")[-1] == file_extension:
                paths += [os.path.join(dir, item)]
        return paths

    def __init__(self, data_dir_list, fs, data_prob_ratio_list=None, dataset_sz=None, load_noise=False, noise_type_specify=None):

        self.fs = fs
        self.data_paths = []
        self.data_paths_prob_ratio = []
        self.prob_ratio_list = data_prob_ratio_list

        for data_dir_idx in range(len(data_dir_list)):
            
            data_dir = data_dir_list[data_dir_idx]
            
            data_paths = self.search_files(dir=data_dir, file_extension='npz')
            self.data_paths += data_paths
            npath = len(data_paths)
            if self.prob_ratio_list is not None:
                prob_ratio = self.prob_ratio_list[data_dir_idx]
                self.data_paths_prob_ratio += npath * [prob_ratio/npath]

        if self.prob_ratio_list is not None:
            self.cumfunc = np.cumsum(np.array(self.data_paths_prob_ratio))

        self.dataset_sz = len(self.data_paths) if dataset_sz is None else dataset_sz
        self.load_noise = load_noise
        self.noise_type_specify = noise_type_specify

    def __len__(self):
        return self.dataset_sz

    def __getitem__(self, idx):
        if self.prob_ratio_list is None: 
            idx = np.random.randint(0, self.dataset_sz)
        else:
            cum_value = np.random.uniform(0, self.cumfunc[-1])
            idx = (np.abs(self.cumfunc - cum_value)).argmin()

        acous_path = self.data_paths[idx]

        acoustic_scene = AcousticScene(
           room_sz = [],
           beta = [],
           T60 = [],
           array_setup = [],
           mic_pos = [],
           array_pos = [],
           traj_pts = [],
           fs = [],
           RIR = [],  # [RIRs, dp_RIRs, dp_RIRs_gen] (npoints,nch,nsamples,nsources)
           c = []
          )

        acoustic_scene = load_file(acoustic_scene, sig_path=None, acous_path=acous_path)

        if self.fs != acoustic_scene.fs:
            RIRs = acoustic_scene.RIR[0].transpose(2, 0, 1, 3)
            dp_RIRs = acoustic_scene.RIR[1].transpose(2, 0, 1, 3)
            acoustic_scene.RIR[0] = scipy.signal.resample_poly(RIRs, self.fs, acoustic_scene.fs)
            acoustic_scene.RIR[1] = scipy.signal.resample_poly(dp_RIRs, self.fs, acoustic_scene.fs)

        if self.load_noise:
            
            attrs = acous_path.split('/')
            npz_name = attrs[-1]
            dataset_name = attrs[-4]
            if (len(npz_name.split('_'))>1):
                mic_attrs_match = npz_name.split('_')[1].split('.')[0]
                noise_dir = acous_path.replace(npz_name,'').replace(dataset_name, dataset_name+'_noise')
                noise_paths = self.search_files(dir=noise_dir, file_extension='wav')
                match_paths = []
                for noise_path in noise_paths:
                    wav_name = noise_path.split('/')[-1]
                    noise_mic = wav_name.split('_')[1]
                    noise_type = wav_name.split('_')[-1].split('.')[0]
                    if noise_mic == mic_attrs_match:
                        if self.noise_type_specify is None:
                            match_paths += [noise_path]
                        else:
                            if noise_type in self.noise_type_specify:
                                match_paths += [noise_path]
                noise_path = random.sample(match_paths, 1)[0]
                noise_signal, noise_fs = soundfile.read(noise_path)

                if self.fs != noise_fs:
                    noise_signal = scipy.signal.resample_poly(noise_signal, self.fs, noise_fs)
                acoustic_scene.noise_signal = noise_signal 

        return acoustic_scene

# Microphone Signal Datasets

class FixMicSigDataset(Dataset):
    """ Get microphone signals from pre-saved dataset
	"""
    def __init__(self, data_dir_list, dataset_sz_ratio_list=None, dataset_sz=None, transforms=None, return_data = ['sig', 'gt']):
        self.transforms = transforms
        self.data_paths = []
        if dataset_sz_ratio_list is None:
            dataset_sz_ratio_list = [1] * len(data_dir_list)
        dataset_sz_ratio = np.array(dataset_sz_ratio_list) / np.sum(np.array(dataset_sz_ratio_list)) 

        ndataset = len(data_dir_list)
        for data_idx in range(ndataset):
            data_dir = data_dir_list[data_idx]
            data_names = os.listdir(data_dir)
            # data_names.sort()
            if dataset_sz is None:
                neach = len(data_names) // 2
            else:
                neach = math.ceil(dataset_sz * dataset_sz_ratio[data_idx])
            print(data_dir.split('/')[-1], dataset_sz_ratio_list[data_idx], neach,'<=',len(data_names)/2, 'noise:', 'Noi' in data_dir, 'dir', data_dir)
            for idx in range(neach):
                fname = str(idx) + '.wav'
                self.data_paths.append((os.path.join(data_dir, fname)))
  
        # self.data_paths.sort()
        self.dataset_sz = len(self.data_paths) if dataset_sz is None else dataset_sz
        assert len(self.data_paths)>=self.dataset_sz, 'dataset size {} is out of range {}'.format(self.dataset_sz, len(self.data_paths))
        self.return_data = return_data

    def __len__(self):
        return self.dataset_sz

    def __getitem__(self, idx):
        sig_path = self.data_paths[idx]
        acous_path = sig_path.replace('wav','npz')

        acoustic_scene = AcousticScene(
           room_sz = [],
           beta = [],
           T60 = [],
           array_setup = [],
           mic_pos = [],
           array_pos = [],
           traj_pts = [],
           fs = [],
           RIR = [],
           source_signal = [],
           noise_signal = [],
           SNR = [],
           timestamps = [],
           t = [],
           trajectory = [],
           # DOA = [],
           c = []
          )

        if self.return_data == ['sig', '']:
            mic_signals = load_file(acoustic_scene, sig_path, acous_path=None)
            if self.transforms is not None:
                for t in self.transforms:
                    mic_signals, _ = t(mic_signals, acoustic_scene=None)
            mic_signals = mic_signals.astype(np.float32)
            gts = {}

            return mic_signals, gts

        else:
            mic_signals, acoustic_scene = load_file(acoustic_scene, sig_path, acous_path)

            if self.transforms is not None:
                for t in self.transforms:
                    mic_signals, acoustic_scene = t(mic_signals, acoustic_scene)

            if self.return_data == ['sig', 'scene']:
                mic_signals = mic_signals.astype(np.float32)

                return mic_signals, acoustic_scene
            
            if self.return_data == ['sig', 'dp_mic_signal']:
                mic_signals = mic_signals.astype(np.float32)
                dp_mic_signals = acoustic_scene.dp_mic_signal.astype(np.float32)

                return mic_signals, dp_mic_signals
            
            else: # self.return_data == ['sig', task]
                mic_signals = mic_signals.astype(np.float32)
                # gts = [] # both list and dictionary are okay
                gts = {}
                task = self.return_data[-1].split('-')
                if 'simulate' in sig_path:
                    acoustic_scene.T60 = acoustic_scene.T60_sabine # M1
                if 'TDOA' in task:
                    gts['TDOA'] = acoustic_scene.TDOAw.astype(np.float32) 
                if 'T60' in task:
                    gts['T60'] = np.array(acoustic_scene.T60).astype(np.float32) 
                if 'DRR' in task:
                    gts['DRR'] = acoustic_scene.DRRw.astype(np.float32)
                if 'C50' in task:
                    gts['C50'] = acoustic_scene.C50w.astype(np.float32) 
                if 'SNR' in task:
                    gts['SNR'] = np.array(acoustic_scene.SNR).astype(np.float32) 
                if 'SUR' in task:
                    gts['SUR'] = ((acoustic_scene.room_sz[0]*acoustic_scene.room_sz[1] + acoustic_scene.room_sz[1]*acoustic_scene.room_sz[2] + acoustic_scene.room_sz[0]*acoustic_scene.room_sz[2])*2).astype(np.float32)
                if 'VOL' in task:
                    gts['VOL'] = (acoustic_scene.room_sz[0]*acoustic_scene.room_sz[1]*acoustic_scene.room_sz[2]).astype(np.float32)
                if 'ABS' in task:
                    gts_t60 = np.array(acoustic_scene.T60).astype(np.float32) 
                    gts_vol = (acoustic_scene.room_sz[0]*acoustic_scene.room_sz[1]*acoustic_scene.room_sz[2]).astype(np.float32)
                    gts_sur = ((acoustic_scene.room_sz[0]*acoustic_scene.room_sz[1] + acoustic_scene.room_sz[1]*acoustic_scene.room_sz[2] + acoustic_scene.room_sz[0]*acoustic_scene.room_sz[2])*2).astype(np.float32)
                    gts['ABS'] = (0.161*gts_vol/gts_t60/gts_sur).astype(np.float32) 

                return mic_signals, gts

class RandomMicSigDataset_FromRIR(Dataset):
    """  generate microphone signals on-the-fly from pre-stored RIRs
	"""
    def __init__(self, rirDataset, sourceDataset, noiseDataset, SNR, dataset_sz, transforms=None, return_data = ['sig', 'gt']):
 
        self.rirDataset = rirDataset
        self.rirDataset_idx = Parameter(range(len(self.rirDataset)), discrete=True)

        self.sourceDataset = sourceDataset
        self.sourceDataset_idx = Parameter(range(len(self.sourceDataset)), discrete=True)

        self.noiseDataset = noiseDataset
        self.SNR = SNR if type(SNR) is Parameter else Parameter(SNR)

        self.dataset_sz = dataset_sz
        self.transforms = transforms
        self.return_data = return_data

    def __len__(self):
        return self.dataset_sz 

    def __getitem__(self, idx):

        if idx < 0: idx = len(self) + idx

        acoustic_scene = self.getRandomScene(idx)
        mic_signals = acoustic_scene.simulate()

        if self.transforms is not None:
            for t in self.transforms:
                mic_signals, acoustic_scene = t(mic_signals, acoustic_scene)

        if self.return_data == ['sig', 'scene']:
            return mic_signals, acoustic_scene

        elif self.return_data == ['sig', '']:
            mic_signals = mic_signals.astype(np.float32)
            gts = {}
            return mic_signals, gts
        
        else: # ['sig', task]
            mic_signals = mic_signals.astype(np.float32)
            gts = {}
            task = self.return_data[-1].split('-')
            if 'TDOA' in task:
                gts['TDOA'] = acoustic_scene.TDOAw.astype(np.float32) 
            if 'T60' in task:
                gts['T60'] = np.array(acoustic_scene.T60).astype(np.float32) 
            if 'DRR' in task:
                gts['DRR'] = acoustic_scene.DRRw.astype(np.float32)
            if 'C50' in task:
                gts['C50'] = acoustic_scene.C50w.astype(np.float32) 
            if 'SNR' in task:
                gts['SNR'] = np.array(acoustic_scene.SNR).astype(np.float32) 
            if 'SUR' in task:
                gts['SUR'] = ((acoustic_scene.room_sz[0]*acoustic_scene.room_sz[1] + acoustic_scene.room_sz[1]*acoustic_scene.room_sz[2] + acoustic_scene.room_sz[0]*acoustic_scene.room_sz[2])*2).astype(np.float32)
            if 'VOL' in task:
                gts['VOL'] = (acoustic_scene.room_sz[0]*acoustic_scene.room_sz[1]*acoustic_scene.room_sz[2]).astype(np.float32)
            if 'ABS' in task:
                gts_t60 = np.array(acoustic_scene.T60).astype(np.float32) 
                gts_vol = (acoustic_scene.room_sz[0]*acoustic_scene.room_sz[1]*acoustic_scene.room_sz[2]).astype(np.float32)
                gts_sur = ((acoustic_scene.room_sz[0]*acoustic_scene.room_sz[1] + acoustic_scene.room_sz[1]*acoustic_scene.room_sz[2] + acoustic_scene.room_sz[0]*acoustic_scene.room_sz[2])*2).astype(np.float32)
                # gts['abs'] = 1.0-acoustic_scene.beta.astype(np.float32)**2
                gts['ABS'] = (0.161*gts_vol/gts_t60/gts_sur).astype(np.float32) 
            if 'DOA' in task:
                gts['DOA'] = acoustic_scene.DOAw.astype(np.float32) 

            return mic_signals, gts

    def getRandomScene(self, idx):
        # Room acoustics & RIRs
        nrir = len(self.rirDataset)
        rir_idx = idx % nrir
        acoustic_scene = self.rirDataset[rir_idx]
        fs = acoustic_scene.fs
        assert fs == self.sourceDataset.fs, "Frequency sampling rate is not matched"

        # Sources
        src_idx = self.sourceDataset_idx.getValue()
        source_signal = self.sourceDataset[src_idx]

        # Noises
        if not hasattr(acoustic_scene, 'noise_signal'):
            noise_signal = self.noiseDataset.get_random_noise(acoustic_scene.mic_pos, acoustic_scene, self.sourceDataset)
        else: # get noise from RIR database
            nsample = source_signal.shape[0]
            nsample_noise = acoustic_scene.noise_signal.shape[0]
            assert nsample_noise>=nsample, 'the sample number of noise signal is smaller than source signal~'
            st = random.randint(0, nsample_noise - nsample)
            ed = st + nsample
            noise_signal = acoustic_scene.noise_signal[st:ed, :]

        SNR = float(self.SNR.getValue())      

        # Interpolate trajectory points
        nb_points = acoustic_scene.traj_pts.shape[0]
        timestamps = np.arange(nb_points) * len(source_signal) / fs / nb_points
        t = np.arange(len(source_signal)) / fs
        num_source = acoustic_scene.traj_pts.shape[-1]
        trajectory = np.zeros((len(t), 3, num_source))
        for source_idx in range(num_source):
            trajectory[:, :, source_idx] = np.array([np.interp(t, timestamps, acoustic_scene.traj_pts[:, i, source_idx]) for i in range(3)]).transpose()

        # Add attributes
        acoustic_scene.source_signal = source_signal[:, 0:num_source]
        acoustic_scene.noise_signal = noise_signal
        acoustic_scene.SNR = SNR
        acoustic_scene.timestamps = timestamps
        acoustic_scene.t = t
        acoustic_scene.trajectory = trajectory

        # acoustic_scene.source_vad = vad[:,0:num_source] # a mask
        # acoustic_scene.DOA = []
        acoustic_scene.TDOA = []
        acoustic_scene.DRR = []
        acoustic_scene.C50 = []
        # acoustic_scene.C80 = []
        # acoustic_scene.dp_mic_signal = []
        # acoustic_scene.spakerID = []

        return acoustic_scene


class RandomMicSigDataset(Dataset):
    """ generate microphone signals & RIRs (specify fixed number of rooms)
	"""
    def __init__(self, room_sz, abs_weights, T60, array_setup, array_pos, num_source, source_state,min_src_array_dist, min_src_boundary_dist, nb_points,
            sourceDataset, noiseDataset, SNR, room_num=np.inf, rir_num_eachroom=50, c=343.0,
            transforms=None, return_data = ['sig', 'gt'], ):
        """
		sourceDataset: dataset with the source signals 
		num_source: Number of sources
		source_state: Static or mobile sources
		room_sz: Size of the rooms in meters
		T60: Reverberation time of the room in seconds
		abs_weights: Absorption coefficients ratios of the walls
		array_setup: Named tuple with the characteristics of the array
		array_pos: Position of the center of the array as a fraction of the room size
		SNR: Signal to (omnidirectional) noise ratio
		nb_points: Number of points to simulate along the trajectory
		c: Speed of sound 
		transforms: Transform to perform to the simulated microphone signals and the Acoustic Scene
		return_data: ['rir'], ['sig', 'scene'] or ['sig', 'gt']

		Any parameter (except from nb_points and transforms) can be Parameter object to make it random.
		"""
        self.room_sz = room_sz if type(room_sz) is Parameter else Parameter(room_sz)
        self.T60 = T60 if type(T60) is Parameter else Parameter(T60)
        self.abs_weights = abs_weights if type(abs_weights) is Parameter else Parameter(abs_weights)

        assert np.count_nonzero(array_setup.orV) == 1, "array_setup.orV must be parallel to an axis"
        self.array_setup = array_setup
        self.N = array_setup.mic_pos.shape[0]  # the number of microphones
        self.array_pos = array_pos if type(array_pos) is Parameter else Parameter(array_pos)
        self.mic_scale = array_setup.mic_scale if type(array_setup.mic_scale) is Parameter else Parameter(array_setup.mic_scale)
        self.mic_rotate = array_setup.mic_rotate if type(array_setup.mic_rotate) is Parameter else Parameter(array_setup.mic_rotate)
        self.min_src_array_dist = min_src_array_dist # control the distance between sources and array (in cm)
        self.min_src_boundary_dist = min_src_boundary_dist
        
        self.num_source = num_source if type(num_source) is Parameter else Parameter(num_source, discrete=True)
        self.source_state = source_state
        self.nb_points = nb_points

        self.sourceDataset = sourceDataset
        self.sourceDataset_idx = Parameter(range(len(self.sourceDataset)), discrete=True)

        self.noiseDataset = noiseDataset
        self.SNR = SNR if type(SNR) is Parameter else Parameter(SNR)

        self.fs = sourceDataset.fs
        self.c = c   # Speed of sound

        self.transforms = transforms
        self.return_data = return_data
        self.room_num = room_num
        self.rir_num_eachroom = rir_num_eachroom
        self.rooms = []  
        if 'rir' in self.return_data:
            gen_mode = 'rir'
        elif 'sig' in self.return_data:
            gen_mode = 'sig'
        while (len(self.rooms)<self.room_num) & (self.room_num is not np.inf):
            configs = self.getRandomScene(gen_mode=gen_mode)
            self.rooms += [configs]

    def __len__(self):
        return self.room_num 

    def __getitem__(self, idx):

        if 'rir' in self.return_data:
            acoustic_scene = self.getRandomScene(gen_mode='rir', room_config=self.rooms[idx])
            return acoustic_scene
        
        if 'sig' in self.return_data:
            acoustic_scene = self.getRandomScene(gen_mode='sig', room_config=self.rooms[idx])
            mic_signals = acoustic_scene.simulate()

            if self.transforms is not None:
                for t in self.transforms:
                    mic_signals, acoustic_scene = t(mic_signals, acoustic_scene)

            mic_signals = mic_signals.astype(np.float32)

            if 'scene' in self.return_data:
                return mic_signals, acoustic_scene

            if 'gt' in self.return_data:
                gts = {}
                gts['TDOA'] = acoustic_scene.TDOAw.astype(np.float32)
                gts['T60'] = acoustic_scene.T60.astype(np.float32)
                gts['DRR'] = acoustic_scene.DRRw.astype(np.float32)
                gts['C50'] = acoustic_scene.C50w.astype(np.float32)
                gts['SNR'] = acoustic_scene.SNR.astype(np.float32)
                gts['SUR'] = (acoustic_scene.room_sz[0]*acoustic_scene.room_sz[1] + acoustic_scene.room_sz[1]*acoustic_scene.room_sz[2] + acoustic_scene.room_sz[0]*acoustic_scene.room_sz[2])*2
                gts['VOL'] = (acoustic_scene.room_sz[0]*acoustic_scene.room_sz[1]*acoustic_scene.room_sz[2]).astype(np.float32)
                gts['ABS'] = (0.161*gts['VOL']/gts['T60']/gts['SUR']).astype(np.float32)
                return mic_signals, gts
                
    def saveRoomConfigurations(self, save_dir, room_range):
        if (self.rooms != []):
            row_name = ['RoomSize', 'RT60', 'AbsorptionCoefficient',]
            csv_name = save_dir + '/Room' + str(room_range[0]) + '-' + str(room_range[1]-1) + '_sz_t60_abs' + '.csv'
            with open(csv_name, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row_name)
                for csv_row in self.rooms:
                    writer.writerow(csv_row)

    def genTrajectory(self, room_sz, array_pos, array_setup, min_src_array_dist, min_src_boundary_dist, num_source, traj_pt_mode='time'):
        src_pos_min = np.array([0.0, 0.0, 0.0]) + np.array([min_src_boundary_dist, min_src_boundary_dist, min_src_boundary_dist])
        src_pos_max = room_sz - np.array([min_src_boundary_dist, min_src_boundary_dist, min_src_boundary_dist])
        if array_setup.arrayType == 'planar_linear':   
            # suited cases: annotation is symetric about end-fire direction (like TDOA, not DOA), front-back confusion not exists, half-plane is set
            # src can be at any 3D point in half-plane space
            if np.sum(array_setup.orV) > 0:
                src_pos_min[np.nonzero(array_setup.orV)] = array_pos[np.nonzero(array_setup.orV)] 
                src_pos_min += min_src_array_dist * np.abs(array_setup.orV)
            else:
                src_pos_max[np.nonzero(array_setup.orV)] = array_pos[np.nonzero(array_setup.orV)] 
                src_pos_max -= min_src_array_dist * np.abs(array_setup.orV)

        # if array_setup.arrayType == 'planar_linear':   
        #     # suited cases: annotation is symetric about end-fire direction (like DOA, not TDOA), front-back confusion exists, half-plane is set
        #     # src can be at any planar point in half-plane space
        #     assert (array_setup.orV==[0,1,0]) | (array_setup.orV==[0,-1,0]) | (array_setup.orV==[1,0,0]) | (array_setup.orV==[-1,0,0]), 'array orientation must along x or y axis'
        #     if array_setup.orV[0] == 1:
        #         src_pos_min[0] = array_pos[0] + min_src_array_dist 
        #     elif array_setup.orV[0] == -1:
        #         src_pos_max[0] = array_pos[0] - min_src_array_dist 
        #     elif array_setup.orV[1] == 1:
        #         src_pos_min[1] = array_pos[1] + min_src_array_dist 
        #     elif array_setup.orV[1] == -1:
        #         src_pos_max[1] = array_pos[1] - min_src_array_dist 
        #     src_pos_min[2] = array_pos[2] - 0.0
        #     src_pos_max[2] = array_pos[2] + 0.0

        # elif array_setup.arrayType == 'planar': 
        #     # suited cases: annotation is not symetric about end-fire direction (like DOA), front-back confusion not exists, all plane is set
        #     # src can be at any planar point in the all-plane space
        #     assert array_setup.mic_rotate == 0, 'array rotate must be 0'
        #     direction_candidates = ['x', 'y', '-x', '-y']
        #     direction = random.sample(direction_candidates, 1)[0]
        #     if direction == 'x':
        #         src_pos_min[0] = array_pos[0] + min_src_array_dist 
        #     elif direction == '-x':
        #         src_pos_max[0] = array_pos[0] - min_src_array_dist 
        #     elif direction == 'y':
        #         src_pos_min[1] = array_pos[1] + min_src_array_dist 
        #     elif direction == '-y':
        #         src_pos_max[1] = array_pos[1] - min_src_array_dist 
        #     else:
        #         raise Exception('Unrecognized direction~')
        #     src_pos_min[2] = array_pos[2] - 0.0
        #     src_pos_max[2] = array_pos[2] + 0.0
        #     # src_pos = np.concatenate((src_pos_min[np.newaxis, :], src_pos_max[np.newaxis, :]), axis=0)
            
        # elif array_setup.arrayType == '3D': 
        #     # suited cases: annotation is not symetric about end-fire direction (like DOA), front-back confusion not exists, all plane is set
        #     # src can be at some 3D point in the all-plane space
        #     assert array_setup.rotate == 0, 'array rotate must be 0'
        #     direction_candidates = ['x', 'y', '-x', '-y']
        #     direction = random.sample(direction_candidates, 1)[0]
        #     if direction == 'x':
        #         src_pos_min[0] = array_pos[0] + min_src_array_dist 
        #     elif direction == '-x':
        #         src_pos_max[0] = array_pos[0] - min_src_array_dist 
        #     elif direction == 'y':
        #         src_pos_min[1] = array_pos[1] + min_src_array_dist 
        #     elif direction == '-y':
        #         src_pos_max[1] = array_pos[1] - min_src_array_dist 
        #     else:
        #         raise Exception('Unrecognized direction~')
        #     src_array_relative_height = 0.3
        #     src_pos_min[2] = array_pos[2] - src_array_relative_height
        #     src_pos_max[2] = array_pos[2] + src_array_relative_height
        
        else:
            raise Exception('Undefined array type~')

        for i in range(3):
            assert src_pos_min[i]<=src_pos_max[i], 'Src postion range error: '+str(src_pos_min[i])+ '>' + str(src_pos_max[i]) + '(array boundary dist >= src boundary dist + src array dist)'


        traj_pts = np.zeros((self.nb_points, 3, num_source))
        for source_idx in range(num_source):
            if self.source_state == 'static':
                src_pos = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)
                traj_pts[:, :, source_idx] = np.ones((self.nb_points, 1)) * src_pos

            elif self.source_state == 'mobile': # 3D sinusoidal trjactories
                src_pos_ini = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)
                src_pos_end = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)
                
                Amax = np.min(np.stack((src_pos_ini - src_pos_min, src_pos_max - src_pos_ini,
                                        src_pos_end - src_pos_min, src_pos_max - src_pos_end)), axis=0)
                A = np.random.random(3) * np.minimum(Amax, 1)    # Oscilations with 1m as maximum in each axis
                if traj_pt_mode == 'time': # Specify nb_points according to time 
                    w = 2*np.pi / self.nb_points * np.random.random(3) * 2  # Between 0 and 2 oscilations in each axis
                    traj_pts[:,:,source_idx] = np.array([np.linspace(i,j,self.nb_points) for i,j in zip(src_pos_ini, src_pos_end)]).transpose()
                    traj_pts[:,:,source_idx] += A * np.sin(w * np.arange(self.nb_points)[:, np.newaxis])
                
                elif traj_pt_mode == 'distance_line': # Specify nb_points according to line distance (pointing src_pos_end from src_pos_ini) 
                    assert num_source == 1, 'number of source larger than one is not supported at this mode'
                    nb_points = int(np.sqrt(np.sum(src_pos_end-src_pos_ini)**2)//0.1 + 1) # adaptive number of points, namely one point per liner 10cm
                    w = 2*np.pi / nb_points * np.random.random(3) * 2  # Between 0 and 2 oscilations in each axis
                    traj_pts = np.array([np.linspace(i,j,nb_points) for i,j in zip(src_pos_ini, src_pos_end)]).transpose()
                    traj_pts += A * np.sin(w * np.arange(nb_points)[:, np.newaxis])
                
                elif traj_pt_mode == 'distance_sin': # Specify nb_points according to direct sin distance
                    desired_dist = 0.1 # between ajacent points
                    src_ini_end_dist = np.sqrt(np.sum(src_pos_end-src_pos_ini)**2)
                    src_ini_end_dirc_vec = (src_pos_end - src_pos_ini)/src_ini_end_dist
                    w = 2*np.pi / src_ini_end_dist * np.random.random(3) * 2  # Between 0 and 2 oscilations in each axis
                    assert num_source == 1, 'number of source larger than one is not supported at this mode'
                    traj_pts = []
                    line_pts = []
                    current_dist_along_dirc_vec = 0
                    while current_dist_along_dirc_vec < src_ini_end_dist:
                        # store current point
                        osc = A * np.sin(w * current_dist_along_dirc_vec)
                        line = src_pos_ini + src_ini_end_dirc_vec * current_dist_along_dirc_vec
                        pos0 = line + osc
                        traj_pts.append(pos0) 
                        line_pts.append(line)

                        # find next point
                        for factor in [1.0, 1.5, 3]:
                            res = minimize(self.dist_err_func, x0=[desired_dist / 10], bounds=[(0, desired_dist * factor)], 
                                           tol=desired_dist / 100, args=(current_dist_along_dirc_vec, src_ini_end_dirc_vec, src_pos_ini, pos0, desired_dist, A, w))
                            if res.fun < desired_dist / 100:
                                break
                        current_dist_along_dirc_vec = current_dist_along_dirc_vec + res.x[0]
                    traj_pts = np.array(traj_pts)
                    line_pts = np.array(line_pts)

                # if np.random.random(1) < 0.25:
                #     traj_pts[:,:,source_idx] = np.ones((self.nb_points,1)) * src_pos_ini
                # traj_pts[:,2,source_idx] = array_pos[2] # if sources and array are in the same horizontal plane
                    
            # self.plotScene(room_sz=room_sz, traj_pts=traj_pts , mic_pos=array_setup.mic_pos, view='XY', save_path='./')

        if traj_pt_mode != 'distance_sin':
            return traj_pts 
        else:
            return traj_pts, line_pts

    def dist_err_func(self, delta_dist_along_dirc_vec, current_dist_along_dirc_vec, ini_end_dirc_vec, pos_ini, pos0, desired_dist, A, w):
        osc = A * np.sin(w * (current_dist_along_dirc_vec + delta_dist_along_dirc_vec))
        line = pos_ini + ini_end_dirc_vec * (current_dist_along_dirc_vec + delta_dist_along_dirc_vec) 
        pos_current = line + osc
        dist = np.sqrt(np.sum((pos_current - pos0)**2))
        return np.abs(dist - desired_dist)

    def genRIR(self, room_sz, beta, T60, array_setup, traj_pts):
        if T60 == 0:
            Tdiff = 0.1
            Tmax = 0.1
            nb_img = [1,1,1]

        else:
            Tdiff = gpuRIR.att2t_SabineEstimator(12, T60) # Use ISM until the RIRs decay 12dB
            Tmax = gpuRIR.att2t_SabineEstimator(40, T60)  # Use diffuse model until the RIRs decay 40dB
            if T60 < 0.15: Tdiff = Tmax # Avoid issues with too short RIRs
            nb_img = gpuRIR.t2n( Tdiff, room_sz )

        RIRs_sources = []
        dp_RIRs_sources = []
        num_source = traj_pts.shape[-1]
        for source_idx in range(num_source):
            RIRs = gpuRIR.simulateRIR(room_sz=room_sz, beta=beta, pos_src=traj_pts[:,:,source_idx], pos_rcv=array_setup.mic_pos,
               nb_img=nb_img, Tmax=Tmax, fs=self.fs, Tdiff=Tdiff, orV_rcv=array_setup.mic_orV,
               mic_pattern=array_setup.mic_pattern, c=self.c)

            dp_RIRs = gpuRIR.simulateRIR(room_sz=room_sz, beta=beta, pos_src=traj_pts[:,:,source_idx], pos_rcv=array_setup.mic_pos,
               nb_img=[1,1,1], Tmax=0.1, fs=self.fs, orV_rcv=array_setup.mic_orV,
               mic_pattern=array_setup.mic_pattern, c=self.c)

            RIRs_sources += [RIRs]
            dp_RIRs_sources += [dp_RIRs]

        RIRs_sources = np.array(RIRs_sources).transpose(1,2,3,0) # (npoints,nch,nsamples,nsources)
        dp_RIRs_sources = np.array(dp_RIRs_sources).transpose(1,2,3,0) # (npoints,nch,nsamples,nsources)

        return RIRs_sources, dp_RIRs_sources
    
    def checkRIR(self, RIRs):
        ok_flag = True
        nan_flag = np.isnan(RIRs)
        inf_flag = np.isinf(RIRs)
        if (True in nan_flag):
            warnings.warn('NAN exists in RIR~')
            ok_flag = False
        if (True in inf_flag):
            warnings.warn('INF exists in RIR~')
            ok_flag = False
        zero_flag = (np.sum(RIRs**2) == 0)
        if zero_flag:
            warnings.warn('RIR is all zeros~')
            ok_flag = False
        return ok_flag

    def T60isValid(self, room_sz, T60, alpha, th=0.005, eps=1e-4):
        Sa = (alpha[0]+alpha[1]) * room_sz[1]*room_sz[2] + \
            (alpha[2]+alpha[3]) * room_sz[0]*room_sz[2] + \
            (alpha[4]+alpha[5]) * room_sz[0]*room_sz[1]
        V = np.prod(room_sz)
        if Sa == 0: # avoid cases when all walls are reflective and T60 (from sabine eq) is extremely large
            valid_flag = False 
        else:
            T60_sabine = 0.161 * V / (Sa+eps) 
            valid_flag = bool(abs(T60-T60_sabine)<th)  # avoid cases when T60<T60_min, or the abs_weights is not suited
        beta_prod = np.prod(1-alpha) # avoid sparse reflections for room size estimation
        return valid_flag & bool(beta_prod!=0), T60_sabine

    def getRandomScene(self, gen_mode='sig', room_config=None):
        RIR_is_ok = False
        nan_zero_correct_flag = False
        while(RIR_is_ok==False):
            # Room
            T60_is_valid = False
            while(T60_is_valid==False):
                if room_config is None:
                    room_sz = self.room_sz.getValue()
                    T60_specify = float(self.T60.getValue())
                    abs_weights = self.abs_weights.getValue()
                else:
                    room_sz, T60_specify, abs_weights = room_config
                beta = gpuRIR.beta_SabineEstimation(room_sz, T60_specify, abs_weights)
                T60_is_valid, T60_sabine = self.T60isValid(room_sz, T60_specify, alpha=1-beta**2)

            # Microphones
            array_pos = self.array_pos.getValue() * room_sz
            mic_scale = self.mic_scale.getValue()
            mic_rotate = self.mic_rotate.getValue()
            rotate_matrix = np.array([[np.cos(mic_rotate/180*np.pi), -np.sin(mic_rotate/180*np.pi), 0], 
                                    [np.sin(mic_rotate/180*np.pi), np.cos(mic_rotate/180*np.pi), 0], 
                                    [0, 0, 1]]) # (3, 3)
            mic_pos_rotate = np.dot(rotate_matrix, self.array_setup.mic_pos.transpose(1, 0)).transpose(1, 0)
            mic_pos = array_pos + mic_pos_rotate * mic_scale # (nch,3)
            mic_orV = np.dot(rotate_matrix, self.array_setup.mic_orV.transpose(1, 0)).transpose(1, 0)
            orV = np.dot(rotate_matrix, self.array_setup.orV)

            array_setup = ArraySetup( arrayType=self.array_setup.arrayType,
                orV = orV,
                mic_scale = mic_scale,
                mic_rotate = mic_rotate,
                mic_pos = mic_pos, 
                mic_orV = mic_orV,
                mic_pattern = self.array_setup.mic_pattern
                ) # updated with random mic_scale, mic_rotate

            # Sources
            num_source = self.num_source.getValue()

            # Trajectory points for sources
            traj_pts = self.genTrajectory(room_sz, array_pos, array_setup, self.min_src_array_dist, self.min_src_boundary_dist, num_source) #  (npoint,3,nsource)

            # RIRs for sources
            RIRs_sources, dp_RIRs_sources = self.genRIR(room_sz, beta, T60_specify, array_setup, traj_pts)
            dp_RIRs_com = dpRIR_from_RIR(rir=RIRs_sources, dp_time=2.5, fs=self.fs)
            RIR = [RIRs_sources, dp_RIRs_com, dp_RIRs_sources]
            RIR_is_ok = self.checkRIR(RIRs_sources) & self.checkRIR(dp_RIRs_sources)

            # T60 from EDC
            if RIR_is_ok:
                T60_edc = []
                R = []
                for mic_idx in range(RIRs_sources.shape[1]):
                    rt60, r = rt60_from_rirs(RIRs_sources[0, mic_idx, :, 0], self.fs)
                    T60_edc += [rt60]
                    R += [r]
                T60_edc = np.mean(T60_edc)
                R = np.mean(r)
                # if (abs(R)<0.98):
                #     print('Correlation coefficient (<0.98):', R, 'T60 calculated (EDC):', T60, 'T60 specified:', T60_gt)
                RIR_is_ok = RIR_is_ok & bool(abs(T60_edc-T60_specify)<0.05) & bool(abs(R)>0.5)
            else:
                nan_zero_correct_flag = True
            if nan_zero_correct_flag & RIR_is_ok:
                print('RIR is corrected~')

        if gen_mode == 'rir':
            acoustic_scene = AcousticScene(
                room_sz = room_sz,
                beta = beta,
                T60 = T60_edc,
                array_setup = array_setup,
                mic_pos = mic_pos,
                array_pos = array_pos,
                traj_pts = traj_pts,
                fs = self.fs,
                RIR = RIR,
                c = self.c
            )
            acoustic_scene.T60_specify = T60_specify
            acoustic_scene.T60_sabine = T60_sabine

        elif gen_mode == 'sig':
            acoustic_scene = AcousticScene(
                room_sz = room_sz,
                beta = beta,
                T60 = T60_edc,
                array_setup = array_setup,
                mic_pos = mic_pos,
                array_pos = array_pos,
                traj_pts = traj_pts,
                fs = self.fs,
                RIR = RIR,
                c = self.c
            )
            
            # Source signals
            source_signal = self.sourceDataset[self.sourceDataset_idx.getValue()]
            source_signal = source_signal[:,0:num_source]

            # Noise signals
            noise_signal = self.noiseDataset.get_random_noise(mic_pos, acoustic_scene, self.sourceDataset)
            SNR = float(self.SNR.getValue())
          
            # Interpolate trajectory points
            timestamps = np.arange(self.nb_points) * len(source_signal) / self.fs / self.nb_points
            t = np.arange(len(source_signal)) / self.fs
            trajectory = np.zeros((len(t), 3, num_source))
            for source_idx in range(num_source):
                trajectory[:,:,source_idx] = np.array([np.interp(t, timestamps, traj_pts[:,i,source_idx]) for i in range(3)]).transpose()

            acoustic_scene = AcousticScene(
                room_sz = room_sz,
                beta = beta,
                T60 = T60_edc,
                array_setup = array_setup,
                mic_pos = mic_pos,
                array_pos = array_pos,
                traj_pts = traj_pts,
                fs = self.fs,
                RIR = RIR,
                source_signal = source_signal,
                noise_signal = noise_signal,
                SNR = SNR,
                timestamps = timestamps,
                t = t,
                trajectory = trajectory,
                c = self.c
                )
            acoustic_scene.T60_specify = T60_specify
            acoustic_scene.T60_sabine = T60_sabine

            # acoustic_scene.source_vad = vad[:,0:num_source] # use webrtcvad
            # acoustic_scene.mic_vad = [] # use snr
            # acoustic_scene.DOA = []
            acoustic_scene.TDOA = []
            acoustic_scene.DRR = []
            acoustic_scene.C50 = []
            # acoustic_scene.C80 = []
            # acoustic_scene.dp_mic_signal = []
            # acoustic_scene.spakerID = []
        else:
            raise Exception('Unrecognized data generation mode')

        if room_config is None:
            return [room_sz, T60_sabine, abs_weights]
        else:
            return acoustic_scene
        
class RandomMicSigDatasetOri(Dataset):
    """ generating microphone signals & RIRs (not specify fix number of rooms)
	"""
    def __init__(self, room_sz, abs_weights, T60, array_setup, array_pos, num_source, source_state, min_src_array_dist, min_src_boundary_dist, nb_points,
            sourceDataset, noiseDataset, SNR, dataset_sz, c=343.0,
            transforms=None, return_data = ['sig', 'gt']):
        """
		sourceDataset: dataset with the source signals (such as LibriSpeechDataset)
		num_source: Number of sources
		source_state: Static or mobile sources
		room_sz: Size of the rooms in meters
		T60: Reverberation time of the room in seconds
		abs_weights: Absorption coefficients ratios of the walls
		array_setup: Named tuple with the characteristics of the array
		array_pos: Position of the center of the array as a fraction of the room size
		SNR: Signal to (omnidirectional) noise ratio
		nb_points: Number of points to simulate along the trajectory
		c: Speed of sound 
		transforms: Transform to perform to the simulated microphone signals and the Acoustic Scene
		return_data: ['rir'], ['sig', 'scene'] or ['sig', 'gt']

		Any parameter (except from nb_points and transforms) can be Parameter object to make it random.
		"""
        self.room_sz = room_sz if type(room_sz) is Parameter else Parameter(room_sz)
        self.T60 = T60 if type(T60) is Parameter else Parameter(T60)
        self.abs_weights = abs_weights if type(abs_weights) is Parameter else Parameter(abs_weights)

        assert np.count_nonzero(array_setup.orV) == 1, "array_setup.orV must be parallel to an axis"
        self.array_setup = array_setup
        self.N = array_setup.mic_pos.shape[0]  # the number of microphones
        self.array_pos = array_pos if type(array_pos) is Parameter else Parameter(array_pos)
        self.mic_scale = array_setup.mic_scale if type(array_setup.mic_scale) is Parameter else Parameter(array_setup.mic_scale)
        self.mic_rotate = array_setup.mic_rotate if type(array_setup.mic_rotate) is Parameter else Parameter(array_setup.mic_rotate)
        self.min_src_array_dist = min_src_array_dist # control the distance between sources and array (in cm)
        self.min_src_boundary_dist = min_src_boundary_dist

        self.num_source = num_source if type(num_source) is Parameter else Parameter(num_source, discrete=True)
        self.source_state = source_state
        self.nb_points = nb_points

        self.sourceDataset = sourceDataset
        self.sourceDataset_idx = Parameter(range(len(self.sourceDataset)), discrete=True)

        self.noiseDataset = noiseDataset
        self.SNR = SNR if type(SNR) is Parameter else Parameter(SNR)

        self.fs = sourceDataset.fs
        self.c = c   # Speed of sound

        self.dataset_sz = dataset_sz
        self.transforms = transforms
        self.return_data = return_data

    def __len__(self):
        return self.dataset_sz  

    def __getitem__(self, idx):

        if 'rir' in self.return_data:
            acoustic_scene = self.getRandomScene(gen_mode='rir')
            return acoustic_scene
        if 'sig' in self.return_data:
            acoustic_scene = self.getRandomScene(gen_mode='sig')
            mic_signals = acoustic_scene.simulate()
            # acoustic_scene.plotScene( view='XY', save_path='./'+str(idx)+'_')

            if self.transforms is not None:
                for t in self.transforms:
                    mic_signals, acoustic_scene = t(mic_signals, acoustic_scene)

            mic_signals = mic_signals.astype(np.float32)

            if 'scene' in self.return_data:
                return mic_signals, acoustic_scene

            if 'gt' in self.return_data:
                gts = {}
                gts['TDOA'] = acoustic_scene.TDOAw.astype(np.float32)
                gts['T60'] = acoustic_scene.T60.astype(np.float32)
                gts['DRR'] = acoustic_scene.DRRw.astype(np.float32)
                gts['C50'] = acoustic_scene.C50w.astype(np.float32)
                gts['SNR'] = acoustic_scene.SNR.astype(np.float32)
                gts['SUR'] = (acoustic_scene.room_sz[0]*acoustic_scene.room_sz[1] + acoustic_scene.room_sz[1]*acoustic_scene.room_sz[2] + acoustic_scene.room_sz[0]*acoustic_scene.room_sz[2])*2
                gts['VOL'] = (acoustic_scene.room_sz[0]*acoustic_scene.room_sz[1]*acoustic_scene.room_sz[2]).astype(np.float32)
                gts['ABS'] = (0.161*gts['VOL']/gts['T60']/gts['SUR']).astype(np.float32)
                return mic_signals, gts

    def genTrajectory(self, room_sz, array_pos, array_setup, min_src_array_dist, min_src_boundary_dist, num_source, traj_pt_mode='time'):
        traj_pts = np.zeros((self.nb_points, 3, num_source))
        for source_idx in range(num_source):
            src_pos_min = np.array([0.0, 0.0, 0.0]) + np.array([min_src_boundary_dist, min_src_boundary_dist, min_src_boundary_dist])
            src_pos_max = room_sz - np.array([min_src_boundary_dist, min_src_boundary_dist, min_src_boundary_dist])
            if array_setup.arrayType == 'planar_linear':   
                # suited cases: annotation is symetric about end-fire direction (like TDOA, not DOA), front-back confusion not exists, half-plane is set
                # src can be at any 3D point in half-plane space
                if np.sum(array_setup.orV) > 0:
                    src_pos_min[np.nonzero(array_setup.orV)] = array_pos[np.nonzero(array_setup.orV)] 
                    src_pos_min += min_src_array_dist * np.abs(array_setup.orV)
                else:
                    src_pos_max[np.nonzero(array_setup.orV)] = array_pos[np.nonzero(array_setup.orV)] 
                    src_pos_max -= min_src_array_dist * np.abs(array_setup.orV)

            # if array_setup.arrayType == 'planar_linear':   
            #     # suited cases: annotation is symetric about end-fire direction (like DOA, not TDOA), front-back confusion exists, half-plane is set
            #     # src can be at any planar point in half-plane space
            #     assert (array_setup.orV==[0,1,0]) | (array_setup.orV==[0,-1,0]) | (array_setup.orV==[1,0,0]) | (array_setup.orV==[-1,0,0]), 'array orientation must along x or y axis'
            #     if array_setup.orV[0] == 1:
            #         src_pos_min[0] = np.maximum(array_pos[0] + min_src_array_dist, src_pos_min[0])
            #     elif array_setup.orV[0] == -1:
            #         src_pos_max[0] = np.minimum(array_pos[0] - min_src_array_dist, src_pos_max[0])
            #     elif array_setup.orV[1] == 1:
            #         src_pos_min[1] = np.maximum(array_pos[1] + min_src_array_dist, src_pos_min[1])
            #     elif array_setup.orV[1] == -1:
            #         src_pos_max[1] = np.minimum(array_pos[1] - min_src_array_dist, src_pos_max[1])
            #     src_pos_min[2] = np.maximum(array_pos[2] - 0.0, src_pos_min[2])
            #     src_pos_max[2] = np.minimum(array_pos[2] + 0.0, src_pos_max[2])
 
            # elif array_setup.arrayType == 'planar': 
            #     # suited cases: annotation is not symetric about end-fire direction (like DOA), front-back confusion not exists, all plane is set
            #     # src can be at any planar point in the all-plane space
            #     assert array_setup.mic_rotate == 0, 'array rotate must be 0'
            #     direction_candidates = ['x', 'y', '-x', '-y']
            #     direction = random.sample(direction_candidates, 1)[0]
            #     if direction == 'x':
            #         src_pos_min[0] = np.maximum(array_pos[0] + min_src_array_dist, src_pos_min[0])
            #     elif direction == '-x':
            #         src_pos_max[0] = np.minimum(array_pos[0] - min_src_array_dist, src_pos_max[0])
            #     elif direction == 'y':
            #         src_pos_min[1] = np.maximum(array_pos[1] + min_src_array_dist, src_pos_min[1])
            #     elif direction == '-y':
            #         src_pos_max[1] = np.minimum(array_pos[1] - min_src_array_dist, src_pos_max[1]) 
            #     else:
            #         raise Exception('Unrecognized direction~')
            #     src_pos_min[2] = np.maximum(array_pos[2] - 0.0, src_pos_min[2])
            #     src_pos_max[2] = np.minimum(array_pos[2] + 0.0, src_pos_max[2])
            #     # src_pos = np.concatenate((src_pos_min[np.newaxis, :], src_pos_max[np.newaxis, :]), axis=0)
                
            # elif array_setup.arrayType == '3D': 
            #     # suited cases: annotation is not symetric about end-fire direction (like DOA), front-back confusion not exists, all plane is set
            #     # src can be at some 3D point in the all-plane space
            #     assert array_setup.mic_rotate == 0, 'array rotate must be 0'
            #     direction_candidates = ['x', 'y', '-x', '-y']
            #     direction = random.sample(direction_candidates, 1)[0]
            #     if direction == 'x':
            #         src_pos_min[0] = np.maximum(array_pos[0] + min_src_array_dist, src_pos_min[0])
            #     elif direction == '-x':
            #         src_pos_max[0] = np.minimum(array_pos[0] - min_src_array_dist, src_pos_max[0]) 
            #     elif direction == 'y':
            #         src_pos_min[1] = np.maximum(array_pos[1] + min_src_array_dist, src_pos_min[1])
            #     elif direction == '-y':
            #         src_pos_max[1] = np.minimum(array_pos[1] - min_src_array_dist, src_pos_max[1])  
            #     else:
            #         raise Exception('Unrecognized direction~')
            #     src_array_relative_height = 0.5
            #     src_pos_min[2] = np.maximum(array_pos[2] - src_array_relative_height, src_pos_min[2]) 
            #     src_pos_max[2] = np.minimum(array_pos[2] + src_array_relative_height, src_pos_max[2])

            else:
                raise Exception('Undefined array type~')

            for i in range(3):
                assert src_pos_min[i]<=src_pos_max[i], 'Src postion range error: '+str(src_pos_min[i])+ '>' + str(src_pos_max[i]) + '(array boundary dist >= src boundary dist + src array dist)'
                
            if self.source_state == 'static':
                src_pos = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)
                traj_pts[:, :, source_idx] = np.ones((self.nb_points, 1)) * src_pos

            elif self.source_state == 'mobile': # 3D sinusoidal trjactories
                src_pos_ini = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)
                src_pos_end = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)
                # self.nb_points = np.sqrt(np.sum(src_pos_end-src_pos_ini)**2)//0.01 + 1 # adaptive number of points, namely one point per liner 10cm

                Amax = np.min(np.stack((src_pos_ini - src_pos_min, src_pos_max - src_pos_ini,
                                        src_pos_end - src_pos_min, src_pos_max - src_pos_end)), axis=0)
                A = np.random.random(3) * np.minimum(Amax, 1)    # Oscilations with 1m as maximum in each axis
                if traj_pt_mode == 'time': # Specify nb_points according to time 
                    w = 2*np.pi / self.nb_points * np.random.random(3) * 2  # Between 0 and 2 oscilations in each axis
                    traj_pts[:,:,source_idx] = np.array([np.linspace(i,j,self.nb_points) for i,j in zip(src_pos_ini, src_pos_end)]).transpose()
                    traj_pts[:,:,source_idx] += A * np.sin(w * np.arange(self.nb_points)[:, np.newaxis])
                
                elif traj_pt_mode == 'distance_line': # Specify nb_points according to line distance (pointing src_pos_end from src_pos_ini) 
                    assert num_source == 1, 'number of source larger than one is not supported at this mode'
                    nb_points = int(np.sqrt(np.sum(src_pos_end-src_pos_ini)**2)//0.1 + 1) # adaptive number of points, namely one point per liner 10cm
                    w = 2*np.pi / nb_points * np.random.random(3) * 2  # Between 0 and 2 oscilations in each axis
                    traj_pts = np.array([np.linspace(i,j,nb_points) for i,j in zip(src_pos_ini, src_pos_end)]).transpose()
                    traj_pts += A * np.sin(w * np.arange(nb_points)[:, np.newaxis])
                
                elif traj_pt_mode == 'distance_sin': # Specify nb_points according to direct sin distance
                    desired_dist = 0.1 # between ajacent points
                    src_ini_end_dist = np.sqrt(np.sum(src_pos_end-src_pos_ini)**2)
                    src_ini_end_dirc_vec = (src_pos_end - src_pos_ini)/src_ini_end_dist
                    w = 2*np.pi / src_ini_end_dist * np.random.random(3) * 2  # Between 0 and 2 oscilations in each axis
                    assert num_source == 1, 'number of source larger than one is not supported at this mode'
                    traj_pts = []
                    line_pts = []
                    current_dist_along_dirc_vec = 0
                    while current_dist_along_dirc_vec < src_ini_end_dist:
                        # store current point
                        osc = A * np.sin(w * current_dist_along_dirc_vec)
                        line = src_pos_ini + src_ini_end_dirc_vec * current_dist_along_dirc_vec
                        pos0 = line + osc
                        traj_pts.append(pos0) 
                        line_pts.append(line)

                        # find next point
                        for factor in [1.0, 1.5, 3]:
                            res = minimize(self.dist_err_func, x0=[desired_dist / 10], bounds=[(0, desired_dist * factor)], 
                                           tol=desired_dist / 100, args=(current_dist_along_dirc_vec, src_ini_end_dirc_vec, src_pos_ini, pos0, desired_dist, A, w))
                            if res.fun < desired_dist / 100:
                                break
                        current_dist_along_dirc_vec = current_dist_along_dirc_vec + res.x[0]
                    traj_pts = np.array(traj_pts)
                    line_pts = np.array(line_pts)

                # if np.random.random(1) < 0.25:
                #     traj_pts[:,:,source_idx] = np.ones((self.nb_points,1)) * src_pos_ini
                # traj_pts[:,2,source_idx] = array_pos[2] # if sources and array are in the same horizontal plane
            
        if traj_pt_mode != 'distance_sin':
            return traj_pts 
        else:
            return traj_pts, line_pts

    def dist_err_func(self, delta_dist_along_dirc_vec, current_dist_along_dirc_vec, ini_end_dirc_vec, pos_ini, pos0, desired_dist, A, w):
        osc = A * np.sin(w * (current_dist_along_dirc_vec + delta_dist_along_dirc_vec))
        line = pos_ini + ini_end_dirc_vec * (current_dist_along_dirc_vec + delta_dist_along_dirc_vec) 
        pos_current = line + osc
        dist = np.sqrt(np.sum((pos_current - pos0)**2))
        return np.abs(dist - desired_dist)

    def genRIR(self, room_sz, beta, T60, array_setup, traj_pts):
        if T60 == 0:
            Tdiff = 0.1
            Tmax = 0.1
            nb_img = [1,1,1]

        else:
            Tdiff = gpuRIR.att2t_SabineEstimator(12, T60) # Use ISM until the RIRs decay 12dB
            Tmax = gpuRIR.att2t_SabineEstimator(40, T60)  # Use diffuse model until the RIRs decay 40dB
            if T60 < 0.15: Tdiff = Tmax # Avoid issues with too short RIRs
            nb_img = gpuRIR.t2n( Tdiff, room_sz )

        RIRs_sources = []
        dp_RIRs_sources = []
        num_source = traj_pts.shape[-1]
 
        for source_idx in range(num_source):
            RIRs = gpuRIR.simulateRIR(room_sz=room_sz, beta=beta, pos_src=traj_pts[:,:,source_idx], pos_rcv=array_setup.mic_pos,
               nb_img=nb_img, Tmax=Tmax, fs=self.fs, Tdiff=Tdiff, orV_rcv=array_setup.mic_orV,
               mic_pattern=array_setup.mic_pattern, c=self.c)

            dp_RIRs = gpuRIR.simulateRIR(room_sz=room_sz, beta=beta, pos_src=traj_pts[:,:,source_idx], pos_rcv=array_setup.mic_pos,
               nb_img=[1,1,1], Tmax=0.1, fs=self.fs, orV_rcv=array_setup.mic_orV,
               mic_pattern=array_setup.mic_pattern, c=self.c)

            RIRs_sources += [RIRs]
            dp_RIRs_sources += [dp_RIRs]

        RIRs_sources = np.array(RIRs_sources).transpose(1,2,3,0) # (npoints,nch,nsamples,nsources)
        dp_RIRs_sources = np.array(dp_RIRs_sources).transpose(1,2,3,0) # (npoints,nch,nsamples,nsources)

        return RIRs_sources, dp_RIRs_sources
    
    def checkRIR(self, RIRs):
        ok_flag = True
        nan_flag = np.isnan(RIRs)
        inf_flag = np.isinf(RIRs)
        if (True in nan_flag):
            warnings.warn('NAN exists in RIR~')
            ok_flag = False
        if (True in inf_flag):
            warnings.warn('INF exists in RIR~')
            ok_flag = False
        zero_flag = (np.sum(RIRs**2) == 0)
        if zero_flag:
            warnings.warn('RIR is all zeros~')
            ok_flag = False
        return ok_flag

    def T60isValid(self, room_sz, T60, alpha, th=0.005, eps=1e-4):
        Sa = (alpha[0]+alpha[1]) * room_sz[1]*room_sz[2] + \
            (alpha[2]+alpha[3]) * room_sz[0]*room_sz[2] + \
            (alpha[4]+alpha[5]) * room_sz[0]*room_sz[1]
        V = np.prod(room_sz)
        if Sa == 0: # avoid cases when all walls are reflective and T60 (from sabine eq) is extremely large
            valid_flag = False 
            T60_sabine = None
        else:
            T60_sabine = 0.161 * V / (Sa+eps) 
            valid_flag = bool(abs(T60-T60_sabine)<th)  # avoid cases when T60<T60_min, or the abs_weights is not suited
        beta_prod = np.prod(1-alpha) # avoid sparse reflections for room size estimation
        return valid_flag & bool(beta_prod!=0), T60_sabine

    def getRandomScene(self, gen_mode='sig'):
        RIR_is_ok = False
        while(RIR_is_ok==False):
            # Room
            T60_is_valid = False
            while(T60_is_valid==False):
                room_sz = self.room_sz.getValue()
                T60_specify = float(self.T60.getValue())
                abs_weights = self.abs_weights.getValue()
                beta = gpuRIR.beta_SabineEstimation(room_sz, T60_specify, abs_weights)
                T60_is_valid, T60_sabine = self.T60isValid(room_sz, T60_specify, alpha=1-beta**2)

            # Microphones
            array_pos = self.array_pos.getValue() * room_sz
            mic_scale = self.mic_scale.getValue()
            mic_rotate = self.mic_rotate.getValue()
            rotate_matrix = np.array([[np.cos(mic_rotate/180*np.pi), -np.sin(mic_rotate/180*np.pi), 0], 
                                    [np.sin(mic_rotate/180*np.pi), np.cos(mic_rotate/180*np.pi), 0], 
                                    [0, 0, 1]]) # (3, 3)
            mic_pos_rotate = np.dot(rotate_matrix, self.array_setup.mic_pos.transpose(1, 0)).transpose(1, 0)
            mic_pos = array_pos + mic_pos_rotate * mic_scale # (nch,3)
            mic_orV = np.dot(rotate_matrix, self.array_setup.mic_orV.transpose(1, 0)).transpose(1, 0)
            orV = np.dot(rotate_matrix, self.array_setup.orV)

            array_setup = ArraySetup( arrayType=self.array_setup.arrayType,
                orV = orV,
                mic_scale = mic_scale,
                mic_rotate = mic_rotate,
                mic_pos = mic_pos, 
                mic_orV = mic_orV,
                mic_pattern = self.array_setup.mic_pattern
                ) # updated with random mic_scale and mic_rotate

            # Sources
            num_source = self.num_source.getValue()

            # Trajectory points for sources
            traj_pts = self.genTrajectory(room_sz, array_pos, array_setup, self.min_src_array_dist, self.min_src_boundary_dist, num_source) #  (npoint,3,nsource)

            # RIRs for sources
            RIRs_sources, dp_RIRs_sources = self.genRIR(room_sz, beta, T60_specify, array_setup, traj_pts)
            dp_RIRs_com = dpRIR_from_RIR(rir=RIRs_sources, dp_time=2.5, fs=self.fs)
            RIR = [RIRs_sources, dp_RIRs_com, dp_RIRs_sources]
            RIR_is_ok = self.checkRIR(RIRs_sources) & self.checkRIR(dp_RIRs_sources)
            # RIR_is_ok = True

            # T60 from EDC
            if RIR_is_ok:
                T60_edc = []
                R = []
                for mic_idx in range(RIRs_sources.shape[1]):
                    rt60, r = rt60_from_rirs(RIRs_sources[0, mic_idx, :, 0], self.fs)
                    T60_edc += [rt60]
                    R += [r]
                T60_edc = np.mean(T60_edc)
                R = np.mean(r)
                # if (abs(R)<0.98):
                #     print('Correlation coefficient (<0.98):', R, 'T60 calculated (EDC):', T60, 'T60 specified:', T60_gt)
                RIR_is_ok = RIR_is_ok & bool(abs(T60_edc-T60_specify)<0.05) & bool(abs(R)>0.5)

        if gen_mode == 'rir':
            acoustic_scene = AcousticScene(
                room_sz = room_sz,
                beta = beta,
                T60 = T60_edc,
                array_setup = array_setup,
                mic_pos = mic_pos,
                array_pos = array_pos,
                traj_pts = traj_pts,
                fs = self.fs,
                RIR = RIR,
                c = self.c
            )
            acoustic_scene.T60_specify = T60_specify
            acoustic_scene.T60_sabine = T60_sabine

        elif gen_mode == 'sig':
            acoustic_scene = AcousticScene(
                room_sz = room_sz,
                beta = beta,
                T60 = T60_edc,
                array_setup = array_setup,
                mic_pos = mic_pos,
                array_pos = array_pos,
                traj_pts = traj_pts,
                fs = self.fs,
                RIR = RIR,
                c = self.c
            )
            
            # Source signals
            source_signal = self.sourceDataset[self.sourceDataset_idx.getValue()]
            source_signal = source_signal[:,0:num_source]

            # Noise signals
            noise_signal = self.noiseDataset.get_random_noise(mic_pos, acoustic_scene, self.sourceDataset)
            SNR = float(self.SNR.getValue())
          
            # Interpolate trajectory points
            timestamps = np.arange(self.nb_points) * len(source_signal) / self.fs / self.nb_points
            t = np.arange(len(source_signal)) / self.fs
            trajectory = np.zeros((len(t), 3, num_source))
            for source_idx in range(num_source):
                trajectory[:,:,source_idx] = np.array([np.interp(t, timestamps, traj_pts[:,i,source_idx]) for i in range(3)]).transpose()

            acoustic_scene = AcousticScene(
                room_sz = room_sz,
                beta = beta,
                T60 = T60_edc,
                array_setup = array_setup,
                mic_pos = mic_pos,
                array_pos = array_pos,
                traj_pts = traj_pts,
                fs = self.fs,
                RIR = RIR,
                source_signal = source_signal,
                noise_signal = noise_signal,
                SNR = SNR,
                timestamps = timestamps,
                t = t,
                trajectory = trajectory,
                c = self.c
                )
            acoustic_scene.T60_specify = T60_specify
            acoustic_scene.T60_sabine = T60_sabine

            # acoustic_scene.source_vad = vad[:,0:num_source] # use webrtcvad
            # acoustic_scene.mic_vad = [] # use snr
            # acoustic_scene.DOA = []
            acoustic_scene.TDOA = []
            acoustic_scene.DRR = []
            acoustic_scene.C50 = []
            # acoustic_scene.C80 = []
            # acoustic_scene.dp_mic_signal = [] # align
            # acoustic_scene.spakerID = []
        else:
            raise Exception('Unrecognized data generation mode')

        return acoustic_scene


## Transform classes
class Segmenting(object):
    """ Segmenting transform
	"""
    def __init__(self, K, step, window=None):
        self.K = K
        self.step = step
        if window is None:
            self.w = np.ones(K)
        elif callable(window):
            try: self.w = window(K)
            except: raise Exception('window must be a NumPy window function or a Numpy vector with length K')
        elif len(window) == K:
            self.w = window
        else:
            raise Exception('window must be a NumPy window function or a Numpy vector with length K')

    def __call__(self, x, acoustic_scene):

        L = x.shape[0]
        N_w = np.floor(L/self.step - self.K/self.step + 1).astype(int)

        if self.K > L:
            raise Exception('The window size can not be larger than the signal length ({})'.format(L))
        elif self.step > L:
            raise Exception('The window step can not be larger than the signal length ({})'.format(L))

        # Pad and window the signal
        # x = np.append(x, np.zeros((N_w * self.step + self.K - L, N_mics)), axis=0)
        # shape_Xw = (N_w, self.K, N_mics)
        # strides_Xw = [self.step * N_mics, N_mics, 1]
        # strides_Xw = [strides_Xw[i] * x.itemsize for i in range(3)]
        # Xw = np.lib.stride_tricks.as_strided(x, shape=shape_Xw, strides=strides_Xw)
        # Xw = Xw.transpose((0, 2, 1)) * self.w

        if acoustic_scene is not None:
            # Pad and window the DOA if it exists
            if hasattr(acoustic_scene, 'DOA'): # (nsample,naziele,nsource)
                N_dims = acoustic_scene.DOA.shape[1]
                num_source = acoustic_scene.DOA.shape[-1]
                DOA = []
                for source_idx in range(num_source):
                    DOA += [np.append(acoustic_scene.DOA[:,:,source_idx], np.tile(acoustic_scene.DOA[-1,:,source_idx].reshape((1,-1)),
                    [N_w*self.step+self.K-L, 1]), axis=0)] # Replicate the last known DOA
                DOA = np.array(DOA).transpose(1,2,0)

                shape_DOAw = (N_w, self.K, N_dims) # (nwindow, win_len, naziele)
                strides_DOAw = [self.step*N_dims, N_dims, 1]
                strides_DOAw = [strides_DOAw[i] * DOA.itemsize for i in range(3)]
                DOAw_sources = []
                for source_idx in range(num_source):
                    DOAw = np.lib.stride_tricks.as_strided(DOA[:,:,source_idx], shape=shape_DOAw, strides=strides_DOAw)
                    DOAw = np.ascontiguousarray(DOAw)
                    for i in np.flatnonzero(np.abs(np.diff(DOAw[..., 1], axis=1)).max(axis=1) > np.pi):
                        DOAw[i, DOAw[i,:,1]<0, 1] += 2*np.pi  # Avoid jumping from -pi to pi in a window
                    DOAw = np.mean(DOAw, axis=1)
                    DOAw[DOAw[:,1]>np.pi, 1] -= 2*np.pi
                    DOAw_sources += [DOAw]
                acoustic_scene.DOAw = np.array(DOAw_sources).transpose(1, 2, 0) # (nsegment,naziele,nsource)

            # Pad and window the VAD if it exists
            if hasattr(acoustic_scene, 'mic_vad'): # (nsample,1)
                vad = acoustic_scene.mic_vad[:, np.newaxis]
                vad = np.append(vad, np.zeros((L - vad.shape[0], 1)), axis=0)

                shape_vadw = (N_w, self.K, 1)
                strides_vadw = [self.step * 1, 1, 1]
                strides_vadw = [strides_vadw[i] * vad.itemsize for i in range(3)]

                acoustic_scene.mic_vad = np.lib.stride_tricks.as_strided(vad, shape=shape_vadw, strides=strides_vadw)[..., 0] # (nsegment, nsample)

            # Pad and window the VAD if it exists
            if hasattr(acoustic_scene, 'mic_vad_sources'): # (nsample,nsource)
                shape_vadw = (N_w, self.K, 1)
                strides_vadw = [self.step * 1, 1, 1]
                strides_vadw = [strides_vadw[i] * vad.itemsize for i in range(3)]
                num_source = acoustic_scene.mic_vad_sources.shape[1]
                vad_sources = []
                for source_idx in range(num_source):
                    vad = acoustic_scene.mic_vad_sources[:, source_idx:source_idx+1]
                    vad = np.append(vad, np.zeros((L - vad.shape[0], 1)), axis=0)

                    vad_sources += [np.lib.stride_tricks.as_strided(vad, shape=shape_vadw, strides=strides_vadw)[..., 0]]

                acoustic_scene.mic_vad_sources = np.array(vad_sources).transpose(1,2,0) # (nsegment, nsample, nsource)

            # Pad and window the TDOA if it exists
            if hasattr(acoustic_scene, 'TDOA'): # (nsample,nch-1,nsource)
                num_source = acoustic_scene.TDOA.shape[-1]
                TDOA = []
                for source_idx in range(num_source):
                    TDOA += [np.append(acoustic_scene.TDOA[:,:,source_idx], np.tile(acoustic_scene.TDOA[-1,:,source_idx].reshape((1,-1)),
                    [N_w*self.step+self.K-L, 1]), axis=0)] # Replicate the last known TDOA
                TDOA = np.array(TDOA).transpose(1,2,0)

                nch = TDOA.shape[1]
                shape_TDOAw = (N_w, self.K, nch)
                strides_TDOAw = [self.step * nch, nch, 1]
                strides_TDOAw = [strides_TDOAw[i] * TDOA.itemsize for i in range(3)]

                TDOAw_sources = []
                for source_idx in range(num_source):
                    TDOAw = np.lib.stride_tricks.as_strided(TDOA[:,:,source_idx], shape=shape_TDOAw, strides=strides_TDOAw)
                    TDOAw = np.mean(TDOAw, axis=1)
                    TDOAw_sources += [TDOAw]
                acoustic_scene.TDOAw = np.array(TDOAw_sources).transpose(1,2,0) # (nsegment,nch-1,nsource)

            # Pad and window the DRR if it exists
            if hasattr(acoustic_scene, 'DRR'): # (nsample,nsource)
                num_source = acoustic_scene.DRR.shape[-1]
                DRR = []
                for source_idx in range(num_source):
                    DRR += [np.append(acoustic_scene.DRR[:,source_idx], np.tile(acoustic_scene.DRR[-1:,source_idx],
                    [N_w*self.step+self.K-L]), axis=0)] # Replicate the last known DRR
                DRR = np.array(DRR).transpose(1,0)

                nch = DRR.shape[1]
                shape_DRRw = (N_w, self.K, 1)
                strides_DRRw = [self.step * 1, 1, 1]
                strides_DRRw = [strides_DRRw[i] * DRR.itemsize for i in range(3)]

                DRRw_sources = []
                for source_idx in range(num_source):
                    DRRw = np.lib.stride_tricks.as_strided(DRR[:,source_idx], shape=shape_DRRw, strides=strides_DRRw)
                    DRRw = np.mean(DRRw, axis=1)
                    DRRw_sources += [DRRw[..., 0]]
                acoustic_scene.DRRw = np.array(DRRw_sources).transpose(1,0) # (nsegment,nsource)

            # Pad and window the C50 if it exists
            if hasattr(acoustic_scene, 'C50'): # (nsample,nsource)
                num_source = acoustic_scene.C50.shape[-1]
                C50 = []
                for source_idx in range(num_source):
                    C50 += [np.append(acoustic_scene.C50[:,source_idx], np.tile(acoustic_scene.C50[-1:,source_idx],
                    [N_w*self.step+self.K-L]), axis=0)] # Replicate the last known C50
                C50 = np.array(C50).transpose(1,0)

                nch = C50.shape[1]
                shape_C50w = (N_w, self.K, 1)
                strides_C50w = [self.step * 1, 1, 1]
                strides_C50w = [strides_C50w[i] * C50.itemsize for i in range(3)]

                C50w_sources = []
                for source_idx in range(num_source):
                    C50w = np.lib.stride_tricks.as_strided(C50[:,source_idx], shape=shape_C50w, strides=strides_C50w)
                    C50w = np.mean(C50w, axis=1)
                    C50w_sources += [C50w[..., 0]]
                acoustic_scene.C50w = np.array(C50w_sources).transpose(1,0) # (nsegment,nsource)

            # Pad and window the C80 if it exists
            if hasattr(acoustic_scene, 'C80'): # (nsample,nsource)
                num_source = acoustic_scene.C80.shape[-1]
                C80 = []
                for source_idx in range(num_source):
                    C80 += [np.append(acoustic_scene.C80[:,source_idx], np.tile(acoustic_scene.C80[-1:,source_idx],
                    [N_w*self.step+self.K-L]), axis=0)] # Replicate the last known C80
                C80 = np.array(C80).transpose(1,0)

                nch = C80.shape[1]
                shape_C80w = (N_w, self.K, 1)
                strides_C80w = [self.step * 1, 1, 1]
                strides_C80w = [strides_C80w[i] * C80.itemsize for i in range(3)]

                C80w_sources = []
                for source_idx in range(num_source):
                    C80w = np.lib.stride_tricks.as_strided(C80[:,source_idx], shape=shape_C80w, strides=strides_C80w)
                    C80w = np.mean(C80w, axis=1)
                    C80w_sources += [C80w[..., 0]]
                acoustic_scene.C80w = np.array(C80w_sources).transpose(1,0) # (nsegment,nsource)

            # Timestamp for each window
            acoustic_scene.tw = np.arange(0, (L-self.K), self.step) / acoustic_scene.fs

        return x, acoustic_scene

class Selectting(object):
    """ Selecting transform
	"""
    def __init__(self, select_range):
        self.select_range = select_range

    def __call__(self, mic_sig, acoustic_scene):
        sig_len = mic_sig.shape[0]
        assert self.select_range[-1]<=sig_len, 'Selecting range is larger than signal length~'

        if acoustic_scene is not None:
            for key in acoustic_scene.__dict__.keys():
                if (type(acoustic_scene.__dict__[key]) is np.ndarray):
                    if len(acoustic_scene.__dict__[key].shape)!=0:
                        if acoustic_scene.__dict__[key].shape[0]==sig_len:
                            acoustic_scene.__dict__[key] = acoustic_scene.__dict__[key][self.select_range[0]:self.select_range[1], ...]
        
        mic_sig = mic_sig[self.select_range[0]:self.select_range[1], ...]
        
        return mic_sig, acoustic_scene


if __name__ == "__main__":
    from opt import opt
    dirs = opt().dir()

    ## Noise
    T = 20
    RIRdataset = RIRDataset(fs=16000, data_dir='SAR-SSL/data/RIR-pretrain5-2/', dataset_sz=4)
    acoustic_scene = RIRdataset[1]

    souDataset = WSJ0Dataset(path=dirs['sousig_pretrain'], T=T, fs=16000, num_source=1, size=50)

    mic_pos = np.array(((-0.05, 0.0, 0.0), (0.05, 0.0, 0.0)))
    noise_type = 'diffuse_fromRIR'
    a = NoiseDataset(T=T, 
                     fs=16000, 
                     nmic=2, 
                     noise_type=Parameter([noise_type], discrete=True), 
                     noise_path=dirs['sousig_pretrain'], 
                     c=343.0, 
                     size=1)
    noise_signal = a.get_random_noise(mic_pos=mic_pos, acoustic_scene=acoustic_scene, source_dataset=souDataset, eps=1e-5)
    soundfile.write(noise_type+'.wav', noise_signal, 16000)
