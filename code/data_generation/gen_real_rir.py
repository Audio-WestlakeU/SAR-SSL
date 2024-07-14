""" Generate real-world RIRs
        source_state='static', num_source=1, nmic =2
        data dir: dataset_name/room_name/array_name/
        data name: 	rir- 	SP*-*_MP*-a-b.npz
                    noise-	(SP*-*)_MP*-a-b_type.wav (a,b denotes the indexes of microphones)
                    rir and noise are matched according to MP*-a-b
    Usage: 	Need to specify data-id, data-type(, data-op)
"""

import os
import argparse
import numpy as np
import scipy
import scipy.io
import scipy.signal
import soundfile
import mat73
import h5py
import json
import pandas as pd
from pathlib import Path
from functools import lru_cache, cache
from torch.utils.data import Dataset
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from itertools import combinations

def micpair_dist_in_range(mic_pos, mic_dist_range):
    """ Check whether the distance between microhphone pairs is the predefined range 
        Args:   mic_pos - (2, 3)
                mic_dist_range - the range of microhphone distance
        Return: True or False
    """
    dist = np.sqrt(np.sum((mic_pos[0, :]-mic_pos[1, :])**2))
    return (dist >= mic_dist_range[0]) & (dist <= mic_dist_range[1])
 
class DCASERIRDataset():
    """ DCASE2020
    Refs: A Dataset of Reverberant Spatial Sound Scenes with Moving Sources for Sound Event Localization and Detection
    Code: https://github.com/danielkrause/DCASE2022-data-generator
    URL: https://zenodo.org/record/6408611
    """
    def __init__(self, data_dir, fs, mic_dist_range=[0.03, 0.20], nmic_selected=2, save_dir=None):

        self.room_names_all = ['bomb_shelter', 'gym', 'pb132', 'pc226',
                                'sa203', 'sc203', 'se201',
                                'se203', 'tb103', 'tc352']
        self.room_names = ['bomb_shelter', 'gym', 'pb132', 'pc226',
                                'sa203', 'sc203', #'se201',
                                'se203', 'tb103', 'tc352']

        #  ['bomb_shelter', 'gym', 'pb132_paatalo_classroom2', 'pc226_paatalo_office',
        #   'sa203_sahkotalo_lecturehall', 'sc203_sahkotalo_classroom2', 'se201_sahkotalo_classroom',
        #   'se203_sahkotalo_classroom', 'tb103_tietotalo_lecturehall', 'tc352_tietotalo_meetingroom']

        data_dir = data_dir + '/TAU-SRIR_DB'
        matdata = scipy.io.loadmat(data_dir + '/rirdata.mat')
        self.rirdata = matdata['rirdata']['room'][0][0]
        self.rir_fs = matdata['rirdata']['fs'][0][0]
        mic_radius = matdata['rirdata']['tetra_mic_radius_m'][0][0]
        mic_doa = matdata['rirdata']['tetra_mic_azel_deg'][0][0]
        sph = np.concatenate([mic_doa, mic_radius*np.ones((mic_doa.shape[0],1))], axis=1) # (nmic, 3)
        self.mic_pos_tetra = self._sph2cart(sph)

        matdata = scipy.io.loadmat(data_dir + '/measinfo.mat')
        self.room_szs = matdata['measinfo']['dimensions']
        self.array_poss = matdata['measinfo']['micPosition']

        self.array_names = ['tetra'] 
        self.audio_format = {'tetra': 'mic', 'foa_sn3d': 'foa'}
        self.fs = fs
        self.mic_dist_range = mic_dist_range
        self.nmic_selected = nmic_selected
        self.save_dir = save_dir
        self.data_dir = data_dir
        self.dataset_sz = 1
            
    def __len__(self):
        return self.dataset_sz

    def gen_rir(self):
        rir_num = 0
        for room_name in self.room_names:
            room_idx0 = self.room_names_all.index(room_name)
            rank = '0'+str(room_idx0+1)
            rank = rank[-2:]
            struct_name = 'rirs_{}_{}.mat'.format(rank, room_name)
            data_path = self.data_dir + '/' + struct_name
            rirs = mat73.loadmat(data_path)
            print('rir data loaded:', room_name)

            room_idx = self.room_names.index(room_name)
            for array_name in self.array_names:
                print('data generation: ', room_name, array_name)

                # Room size & array postion 
                room_sz = self.room_szs[room_idx][0]
                array_pos = self.array_poss[room_idx][0]
                mic_poss = array_pos + self.mic_pos_tetra # (nmic, 3)
                mic_combins = [list(c) for c in combinations(range(mic_poss.shape[0]), self.nmic_selected)]
                for mic_idxes in mic_combins:   
                    if micpair_dist_in_range(mic_poss[mic_idxes, :], self.mic_dist_range):
                        # Microphone position
                        mic_pos = mic_poss[mic_idxes, :]

                        # Random trajactory & height
                        n_traj = np.shape(self.rirdata[room_idx][0][2])[0]
                        n_rirs_max = np.max(np.sum(self.rirdata[room_idx][0][3], axis=1))
                        for traj_idx in range(n_traj):
                            nHeights = np.sum(self.rirdata[room_idx][0][3][traj_idx, :]>0)
                            for hei_idx in range(nHeights):

                                # Trajactory points
                                sph = self.rirdata[room_idx]['rirs'][0][traj_idx][hei_idx][0] # (npoint, 3)
                                traj_ptss = self._sph2cart(sph)
                                n_point = traj_ptss.shape[0]
                                for pt_idx in range(n_point):
                                    traj_pts = traj_ptss[pt_idx:pt_idx+1, :, np.newaxis] # (npoint=1, 3, nsources)

                                    # RIR
                                    rir = rirs['rirs'][self.audio_format[array_name]]
                                    lrir = len(rir[0][0])
                                    nmic = len(rir[0][0][0])
                                    RIRs = rir[traj_idx][hei_idx] # (nsample, nmic, npoint)
                                    
                                    RIRs = RIRs[:, :, pt_idx:pt_idx+1] # (nsample, nmic, npoint=1)
                                    if self.fs != self.rir_fs:
                                        RIRs = scipy.signal.resample_poly(RIRs, self.fs, self.rir_fs)
                                    RIRs = RIRs[:, mic_idxes, :, np.newaxis].transpose(2, 1, 0, 3) # (npoints=1, nmic, nsample, nsources=1)

                                    rir_num += 1

                                    if self.save_dir is not None:
                                        save_dir = self.save_dir + '/' + room_name+'/'+array_name+'/'
                                        Path(save_dir).mkdir(parents=True, exist_ok=True)
                                        file_name = 'SP'+ str(traj_idx+1)+ '-' + str(hei_idx+1)+ '-' + str(pt_idx+1) + '_MP' + '-' + str(mic_idxes[0]+1) + '-' + str(mic_idxes[1]+1) 
                                        save_file = os.path.join(save_dir, file_name + '.npy')
                                        np.save(save_file, arr=RIRs.astype(np.float16))
                                        save_file = os.path.join(save_dir, file_name + '_info.npz')
                                        data = {
                                            'room_sz': room_sz,
                                            'mic_pos': mic_pos,
                                            'array_pos': array_pos,
                                            'traj_pts': traj_pts,
                                            'fs': self.fs,
                                        }
                                        np.savez(save_file, **data)

        return rir_num
    
    def gen_noise(self):  
        num = 0
        noise_type = 'silence'
        noise_dir = self.data_dir.replace('SRIR', 'SNoise')
        for room_name in self.room_names:
            for array_name in self.array_names:
                room_idx0 = self.room_names_all.index(room_name)
                rank = '0'+str(room_idx0+1)
                rank = rank[-2:]
                struct_name = '{}_{}/ambience_{}_24k_edited.wav'.format(rank, room_name, array_name)
                
                data_path = noise_dir + '/' + struct_name
                noise, noise_fs = soundfile.read(data_path)
                
                print('noise data loaded!', room_name, array_name)

                print('data generation: ', room_name, array_name)
                room_idx = self.room_names.index(room_name) 
                array_pos = self.array_poss[room_idx][0]
                mic_poss = array_pos + self.mic_pos_tetra # (nmic, 3)
                mic_combins = [list(c) for c in combinations(range(mic_poss.shape[0]), self.nmic_selected)]
                for mic_idxes in mic_combins:   
                    if micpair_dist_in_range(mic_poss[mic_idxes, :], self.mic_dist_range):
                        noise_signal = noise[:, mic_idxes]
                        if self.fs != noise_fs:
                            noise_signal = scipy.signal.resample_poly(noise_signal, self.fs, noise_fs)
                        
                        num += 1

                        if self.save_dir is not None:
                            save_dir = self.save_dir + '/' + room_name+'/'+array_name+'/'
                            Path(save_dir).mkdir(parents=True, exist_ok=True)
                            file_name = '_MP' + '-' + str(mic_idxes[0]+1) + '-' + str(mic_idxes[1]+1) + '_' + noise_type
                            save_file = save_dir + file_name + '.wav'
                            soundfile.write(save_file, noise_signal, self.fs) 

        return num
    

    def _cart2sph(self, cart):
        """ cart [x,y,z] → sph [azi,ele,r]
        """
        xy2 = cart[:,0]**2 + cart[:,1]**2
        sph = np.zeros_like(cart)
        sph[:,0] = np.arctan2(cart[:,1], cart[:,0])
        sph[:,1] = np.arctan2(np.sqrt(xy2), cart[:,2]) # Elevation angle defined from Z-axis down
        sph[:,2] = np.sqrt(xy2 + cart[:,2]**2)

        return sph


    def _sph2cart(self, sph):
        """ sph [azi,ele,r] → cart [x,y,z]
        """
        if sph.shape[-1] == 2: sph = np.concatenate((sph, np.ones_like(sph[..., 0]).unsqueeze(-1)), dim=-1)
        x = sph[..., 2] * np.sin(sph[..., 1]) * np.cos(sph[..., 0])
        y = sph[..., 2] * np.sin(sph[..., 1]) * np.sin(sph[..., 0])
        z = sph[..., 2] * np.cos(sph[..., 1])

        return np.stack((x, y, z)).transpose(1, 0)
    

class MIRRIRDataset():
    """ 
        Refs: Multichannel audio database in various acoustic environments, 2014
        URL: https://www.eng.biu.ac.il/gannot/downloads/ 
    """
    def __init__(self, data_dir, fs, mic_dist_range=[0.03, 0.20], nmic_selected=2, save_dir=None):

        data_dir = data_dir + '/Impulse_response_Acoustic_Lab_Bar-Ilan_University'
        self.room_sz = np.array([6, 6, 2.4])
        self.data_paths = os.listdir(data_dir)
        self.t60_set = ['0.160', '0.360', '0.610']
        self.room_names = ['R1', 'R2', 'R3']
        self.mic_array_set = ['3-3-3-8-3-3-3', '4-4-4-8-4-4-4', '8-8-8-8-8-8-8']
        self.dist_set = ['1m', '2m']
        self.angle_set = ['270', '285', '300', '315', '330', '345', '000', '015', '030', '045', '060', '075', '090']

        self.fs = fs
        self.mic_dist_range = mic_dist_range
        self.nmic_selected = nmic_selected
        self.save_dir = save_dir
        self.data_dir = data_dir
        self.dataset_sz = 1

    def __len__(self):

        return self.dataset_sz

    def gen_rir(self):
        rir_num = 0
        for room_idx in range(len(self.room_names)):
            room_name = self.room_names[room_idx]
            for array_name in self.mic_array_set:
                print('data generation: ', room_name, array_name)
                for src in self.angle_set:
                    for dist in self.dist_set:
                        mat_name = 'Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_' + self.t60_set[room_idx] +'s)_' + array_name + '_' + dist + '_' + src + '.mat'
                        data = scipy.io.loadmat(self.data_dir+'/'+mat_name)

                        rirs = data['impulse_response'] # (nsample, nch)
                        sim = data['simpar'][0, 0]
                        rir_fs = sim['fs'][0, 0]
                        meta = data['metapar'][0, 0]
                        t60 = meta['reverberation'][0, 0]

                        mic_spacing = meta['mic_spacing'][0]
                        nmic = len(mic_spacing)+1
                        mic_poss = np.zeros((nmic, 1))
                        for mic_idx in range(nmic):
                            mic_poss[mic_idx] = np.sum(mic_spacing[:mic_idx])
                        mic_poss = abs(mic_poss - (mic_poss[0]+mic_poss[-1])/2)/100
                        mic_angle1 = float(meta['mic_position'][0].split(', ')[0][5:-3])*np.ones((int(nmic/2),1))
                        mic_angle2 = float(meta['mic_position'][0].split(', ')[1][5:-3])*np.ones((int(nmic/2),1))
                        mic_angle = np.concatenate((mic_angle1, mic_angle2), axis=0)/180*np.pi
                        mic_poss = np.concatenate((mic_poss*np.cos(mic_angle), mic_poss*np.sin(mic_angle), np.zeros((nmic,1))), axis=1) # relative positions (nch, 3)
                        
                        azi = meta['azimuth'][0]/180*np.pi
                        distance = meta['distance'][0][0]
                        src_poss = np.concatenate((distance*np.cos(azi), distance*np.sin(azi), np.zeros((1,))), axis=0) # (3, )

                        mic_combins = [list(c) for c in combinations(range(mic_poss.shape[0]), self.nmic_selected)]
                        for mic_idxes in mic_combins:   
                            if micpair_dist_in_range(mic_poss[mic_idxes, :], self.mic_dist_range):
                                # Microphone position
                                mic_pos = mic_poss[mic_idxes, :]

                                # RIR
                                sample_max = int(t60*2*rir_fs)
                                rir = rirs[0:sample_max, mic_idxes]
                                if self.fs != rir_fs:
                                    rir = scipy.signal.resample_poly(rir, self.fs, rir_fs) # (nsample, nmic)
                                RIRs = rir[np.newaxis, :, :, np.newaxis].transpose(0, 2, 1, 3) # (npoints=1, nmic, nsample, nsources=1)
                                
                                rir_num += 1

                                if self.save_dir is not None:
                                    save_dir = self.save_dir + '/' + room_name+'/'+array_name+'/'
                                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                                    file_name = 'SP'+ dist + '-' +  src + '_MP' + '-' + str(mic_idxes[0]+1) + '-' + str(mic_idxes[1]+1)
                                    save_file = os.path.join(save_dir, file_name + '.npy')
                                    np.save(save_file, arr=RIRs.astype(np.float16)) # (npoints,nch,nsamples,nsources)
                                    save_file = os.path.join(save_dir, file_name + '_info.npz')
                                    data = {
                                        'room_sz': self.room_sz,
                                        'mic_pos': mic_pos,
                                        'T60': t60,
                                        'fs': self.fs,
                                    }
                                    np.savez(save_file, **data)

        return rir_num


class MeshRIRDataset():
    """ 
        Refs: MeshRIR: A Dataset of Room Impulse Responses on Meshed Grid Points For Evaluating Sound Field Analysis and Synthesis Methods, 2021
        URL: https://doi.org/10.5281/zenodo.5500451, https://www.sh01.org/MeshRIR/
        Code: https://github.com/sh01k/MeshRIR
    """
    def __init__(self, data_dir, fs, mic_dist_range=[0.03, 0.20], nmic_selected=2, save_dir=None):

        S32_dir = Path(data_dir).joinpath('S32-M441_npy')
        S1_dir = Path(data_dir).joinpath('S1-M3969_npy')

        posMic_s32, posSrc_s32, ir_s32 = self.loadIR(S32_dir) # posMic (nch, 3), posSrc (nsou, 3), ir_s32 (nsou, nch, nsample)
        # posMic_s1, posSrc_s1, ir_s1 = self.loadIR(S1_dir)
        self.mic_poss = posMic_s32 # (nmic, 3)
        self.src_poss = posSrc_s32 # (nsrc, 3)
        self.rirs = ir_s32 # (nsrc, nmic, sample)

        S32_json_dir = Path(S32_dir).joinpath('data.json')
        with open(S32_json_dir, encoding='utf-8') as a:
            result = json.load(a)
            self.rir_fs = result.get('samplerate')
        self.fs = fs

        S32_room_sz = [7.0, 6.4, 2.7]
        S32_T60 = 0.19
        self.room_sz = S32_room_sz
        self.T60 = S32_T60

        self.mic_dist_range = mic_dist_range
        self.nmic_selected = nmic_selected
        self.save_dir = save_dir
        self.dataset_sz = 1

    def __len__(self):

        return self.dataset_sz

    def gen_rir(self ):
        rir_num = 0
        room_name = 'R1'
        array_name = 'A1'
        for src_idx in range(self.src_poss.shape[0]):
            print('data generation: ', src_idx)
            mic_combins = [list(c) for c in combinations(range(self.mic_poss.shape[0]), self.nmic_selected)]
            for mic_idxes in mic_combins:  
                if micpair_dist_in_range(self.mic_poss[mic_idxes, :], self.mic_dist_range):
                    # Microphone position
                    mic_pos = self.mic_poss[mic_idxes, :]

                    # RIRs
                    RIRs = self.rirs[src_idx, :, :].transpose(1, 0) #（nsample, nch）
                    RIRs = RIRs[:, mic_idxes]
                    if self.fs != self.rir_fs:
                        RIRs = scipy.signal.resample_poly(RIRs, self.fs, self.rir_fs)
                    RIRs = RIRs[np.newaxis, :, :, np.newaxis].transpose(0, 2, 1, 3) 

                    # Source positions
                    src_pos = self.src_poss[src_idx, :]
                    traj_pts = src_pos[np.newaxis, :, np.newaxis]

                    rir_num += 1
                    if self.save_dir is not None:
                        save_dir = self.save_dir + '/' + room_name+'/'+array_name+'/'
                        Path(save_dir).mkdir(parents=True, exist_ok=True)
                        file_name = 'SP'+ str(src_idx+1) + '_MP' + '-' + str(mic_idxes[0]+1) + '-' + str(mic_idxes[1]+1)
                        save_file = os.path.join(save_dir, file_name + '.npy')
                        np.save(save_file, arr=RIRs.astype(np.float16)) # (npoints,nch,nsamples,nsources)
                        save_file = os.path.join(save_dir, file_name + '_info.npz')
                        data = {
                            'room_sz': self.room_sz,
                            'mic_pos': mic_pos, # (nch, 3)
                            'T60': self.T60,
                            'traj_pts': traj_pts, # (npoints,3,nsources)
                            'fs': self.fs,
                        }
                        np.savez(save_file, **data)

        return rir_num


    def loadIR(self, sessionPath):
        """ Load impulse response data
            Args:
                sessionPath: Path to IR folder
            Returns:
                pos_mic: Microphone positions of shape (numMic, 3)
                pos_src: Source positions of shape (numSrc, 3)
                fullIR: IR data of shape (numSrc, numMic, irLen)
        """
        pos_mic = np.load(sessionPath.joinpath("pos_mic.npy"))
        pos_src = np.load(sessionPath.joinpath("pos_src.npy"))

        numMic = pos_mic.shape[0]

        allIR = []
        irIndices = []
        for f in sessionPath.iterdir():
            if not f.is_dir():
                if f.stem.startswith("ir_"):
                    allIR.append(np.load(f))
                    irIndices.append(int(f.stem.split("_")[-1]))

        assert(len(allIR) == numMic)
        numSrc = allIR[0].shape[0]
        irLen = allIR[0].shape[-1]
        fullIR = np.zeros((numSrc, numMic, irLen))
        for i, ir in enumerate(allIR):
            assert(ir.shape[0] == numSrc)
            assert(ir.shape[-1] == irLen)
            fullIR[:, irIndices[i], :] = ir

        return pos_mic, pos_src, fullIR


class dEchorateRIRDataset():
    """ 
        Refs: dEchorate: a calibrated room impulse response dataset for echo-aware signal processing, EURASIP Journal on Audio, Speech, and Music Processing, 2021
        Code: https://github.com/Chutlhu/dEchorate
        URL: https://drive.google.com/drive/folders/1yGTh_BjnVNwDgBsn5mkuW3i4rJIgZwlS
                https://zenodo.org/record/6576203 (sth wrong with RIRs)
    """
    def __init__(self, data_dir, fs, mic_dist_range=[0.03, 0.20], nmic_selected=2, c=346.98, save_dir=None):

        self.room_envs = ['000000', '000001', '000010', '000100', '001000', '010000', '011000', '011100', '011110', '011111', '020002'] # floor, ceiling, west, south, east, north
        
        data_path = data_dir + '/' + 'dEchorate_rir.h5'
        data = h5py.File(data_path, mode='r')
        self.rirs = data['rir'] 
        self.rir_fs = data.attrs['sampling_rate']

        note_path = data_dir + '/' + 'dEchorate_database.csv'
        self.note = pd.read_csv(note_path)
        mic_src_echo_note_path = data_dir + '/' + 'dEchorate_annotations.h5'
        mic_src_echo_note = h5py.File(mic_src_echo_note_path, mode='r')
        self.room_sz = mic_src_echo_note['room_size'][:]
        self.mics = mic_src_echo_note['microphones'][:] # (3, 30)
        self.arrs = mic_src_echo_note['arrays_position'][:] # (3, 6)
        self.srcs_dir = mic_src_echo_note['sources_directional_position'][:]  # (3, 6) src:'0'-'5'
        self.srcs_dir_view = mic_src_echo_note['sources_directional_direction'][:]
        self.srcs_omn = mic_src_echo_note['sources_omnidirection_position'][:]  # (3, 3) src:'6'-'8'
        self.srcs_nse = mic_src_echo_note['sources_noise_position'][:]  # (3, 4) src:'0'-'4'
        self.srcs_nse_view = mic_src_echo_note['sources_noise_direction'][:]

        self.noise_types = ['noisrc', 'babsrc', 'sil']
        self.noise_paths = {
            'noisrc': data_dir + '/' + 'dEchorate_noise_gzip7.hdf5',
            'babsrc': data_dir + '/' + 'dEchorate_babble_gzip7.hdf5',
            'sil': data_dir + '/' + 'dEchorate_silence_gzip7.hdf5'
        }

        print('data loaded!')

        self.fs = fs
        self.mic_dist_range = mic_dist_range
        self.nmic_selected = nmic_selected
        self.array_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
        self.nmic_each_array = 5

        self.dataset_sz = 1 
        self.data_dir = data_dir
        self.save_dir = save_dir

        # self.T60s, abss = self.com_gtT60_gtabs()
        
    def __len__(self):
        return self.dataset_sz

    def gen_rir(self):

        rir_num = 0
        mic_poss = self.mics.transpose(1, 0)
        for room_idx in range(len(self.room_envs)):
            room_env = self.room_envs[room_idx]
            source_set = self.rirs[room_env]
            nsource = self.srcs_omn.shape[-1]
            for s_idx in range(nsource):
                # RIR 
                source = list(source_set.keys())[s_idx+self.srcs_dir.shape[-1]] 
                rir = source_set[source] # (nsample, nmic) '31 mics = 30 mics + 1 control signals!'
                rir = rir[:, :-1]
                if self.fs != self.rir_fs:
                    rir = scipy.signal.resample_poly(rir, self.fs, self.rir_fs) # (nsample, nmic)

                for array_name in self.array_names:
                    print('data generation: ', room_env, s_idx, array_name)

                    nmic = self.nmic_each_array
                    mic_combins = [list(c) for c in combinations(range(nmic), self.nmic_selected)]
                    for mic_idxes in mic_combins:
                        if micpair_dist_in_range(mic_poss[mic_idxes, :], self.mic_dist_range): 

                            idx_offset = (int(array_name[-1])-1) * self.nmic_each_array
                            mic_idxes_offset = [i+idx_offset for i in mic_idxes] 
                            RIRs = rir[np.newaxis, :, mic_idxes_offset, np.newaxis].transpose(0, 2, 1, 3) # (npoints=1, nmic, nsample, nsources=1)

                            # Source postion
                            traj_pts = self.srcs_omn[np.newaxis, :, s_idx:s_idx+1]

                            # Microphone positions
                            mic_pos = mic_poss[mic_idxes_offset, :]

                            rir_num += 1
                            if self.save_dir is not None:
                                save_dir = self.save_dir + '/' + room_env+'/'+array_name+'/'
                                Path(save_dir).mkdir(parents=True, exist_ok=True)
                                file_name = 'SP'+ str(s_idx+1) + '_MP' + '-' + str(mic_idxes[0]+1) + '-' + str(mic_idxes[1]+1)
                                save_file = os.path.join(save_dir, file_name + '.npy')
                                np.save(save_file, arr=RIRs.astype(np.float16)) # (npoints,nch,nsamples,nsources)
                                save_file = os.path.join(save_dir, file_name + '_info.npz')
                                data = {
                                    'room_sz': self.room_sz,
                                    'mic_pos': mic_pos,
                                    'traj_pts': traj_pts,
                                    'fs': self.fs,
                                }
                                np.savez(save_file, **data)

        return rir_num

    def gen_noise(self):
        num = 0
        mic_poss = self.mics.transpose(1, 0)

        for noise_type in self.noise_types:
            
            if noise_type == 'noisrc': # white noise src
                noise_path = self.noise_paths[noise_type]
                noise_data = h5py.File(noise_path, mode='r')['noise'] # 9 dir+omni srcs
                nsource = self.srcs_dir.shape[-1] # 6 dir
                rm_silence = True
                not_rm_silence_room = ''

            elif noise_type == 'babsrc':
                noise_path = self.noise_paths[noise_type]
                noise_data = h5py.File(noise_path, mode='r')['babble'] # 4 nse srcs
                nsource = self.srcs_nse.shape[-1]
                rm_silence = True
                not_rm_silence_room = '011111'

            elif noise_type == 'sil':
                noise_path = self.noise_paths[noise_type]
                noise_data = h5py.File(noise_path, mode='r')['silence'] # without src
                nsource = 1
                rm_silence = False
                not_rm_silence_room = ''

            else:
                raise Exception('dEchorate noise type unrecognized~')
            
            for room_idx in range(len(self.room_envs)):
                room_env = self.room_envs[room_idx]
                for s_idx in range(nsource):
                    noises = noise_data[room_env]
                    source = list(noises.keys())[s_idx] 
                    if rm_silence & (room_env != not_rm_silence_room):
                        noise_signals = self.rm_silence_from_noise(noises[source][:, :-1], self.rir_fs)
                    else:
                        noise_signals = noises[source][:, :-1] # (nsample, nmic) '31 mics = 30 mics + 1 control signals!'

                    for array_name in self.array_names:
                        print('data generation: ', room_env, s_idx, array_name)
                        nmic = self.nmic_each_array
                        mic_combins = [list(c) for c in combinations(range(nmic), self.nmic_selected)]
                        for mic_idxes in mic_combins:
                            if micpair_dist_in_range(mic_poss[mic_idxes, :], self.mic_dist_range): 

                                idx_offset = (int(array_name[-1])-1) * self.nmic_each_array
                                mic_idxes_offset = [i+idx_offset for i in mic_idxes] 

                                noise_signal = noise_signals[:, mic_idxes_offset]

                                noise_fs = self.rir_fs 
                                # self.noises.attrs['sampling_rate']
                                if self.fs != noise_fs:
                                    noise_signal = scipy.signal.resample_poly(noise_signal, self.fs, noise_fs)
                                
                                num += 1
                                
                                if self.save_dir is not None:
                                    save_dir = self.save_dir + '/' + room_env+'/'+array_name+'/'
                                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                                    file_name = '_MP' + '-' + str(mic_idxes[0]+1) + '-' + str(mic_idxes[1]+1) + '_' + noise_type + '_' + str(s_idx+1)
                                    sig_path = save_dir + file_name + '.wav'
                                    soundfile.write(sig_path, noise_signal, self.fs) 

        return num
        
    def rm_silence_from_noise(self, noise_wsilence, fs, boundary_time=3, filt_time=0.4, silence_time=1.5, noise_silence_ratio=2):       
        # Remove silence part from noise signals 
        energy =np.abs(noise_wsilence)**2 
        filt_len = int(fs*filt_time)
        filt = np.ones(filt_len)/filt_len
        energy_filt = scipy.signal.convolve(energy, filt[:, np.newaxis], mode ='full')
        energy_filt_meanch = np.mean(energy_filt, axis=1)
        silence_mean = np.mean(energy_filt_meanch[filt_len:int(fs*silence_time)])
        silence_max = np.max(energy_filt_meanch[filt_len:int(fs*silence_time)])
        noise_mean = np.mean(energy_filt_meanch[int(fs*boundary_time):int(len(energy_filt_meanch)-fs*boundary_time)])
        noise_min = np.min(energy_filt_meanch[int(fs*boundary_time):int(len(energy_filt_meanch)-fs*boundary_time)])
        silence_th = (silence_mean+noise_mean)/4 + (silence_max+noise_min)/4 
        binary_energy = energy_filt_meanch>silence_th
        st = np.argmax(binary_energy[:int(fs*boundary_time)])
        ed = np.argmin(binary_energy[int(fs*boundary_time):])+int(fs*boundary_time)-filt_len
        assert ed>st, 'a problem exists in noise selection range'
        assert (ed-st)/fs>3,''
        noise_wosilence = noise_wsilence[st:ed, :]  
        
        return noise_wosilence  

    # def set_abs(self, dset_code, absb=0.1, refl=0.9):
    #     f, c, w, s, e, n = [int(i) for i in list(dset_code)]
    #     absorption = {
    #         'north': refl if n else absb,
    #         'south': refl if s else absb,
    #         'east': refl if e else absb,
    #         'west': refl if w else absb,
    #         'floor': refl if f else absb,
    #         'ceiling': refl if c else absb,
    #     }
    #     return absorption

    # def com_gtT60_gtabs(self):

    #     room_sz = self.room_sz
    #     mics = self.mics
    #     srcs_dir = self.srcs_dir
    #     srcs_omn = self.srcs_omn

    #     # Micrphone positions
    #     mic_poss = mics.transpose(1, 0)
    #     nmic = mic_poss.shape[0]

    #     # RIR
    #     nroom_env = len(self.room_envs)
    #     nsource = srcs_omn.shape[-1]
    #     T60 = np.zeros((nroom_env, nmic, nsource))
    #     R = np.zeros((nroom_env, nmic, nsource))
    #     for room_idx in range(nroom_env):
    #         room_env = self.room_envs[room_idx]  
    #         source_set = self.rirs[room_env]

    #         for s_idx in range(nsource):
    #             source = list(source_set.keys())[s_idx+srcs_dir.shape[-1]]  
    #             rir = source_set[source] # (nsample, nmic) '31 mics = 30 mics + 1 control signals!'
    #             rir = rir[:, :-1]
                
    #             if self.fs != self.rir_fs:
    #                 rir = scipy.signal.resample_poly(rir, self.fs, self.rir_fs) # (nsample, nmic)

    #             RIRs = rir[np.newaxis, :, :, np.newaxis].transpose(0, 2, 1, 3) # (npoints=1, nmic, nsample, nsources=1)

    #             # T60
    #             for mic_idx in range(RIRs.shape[1]):
    #                 T60[room_idx, mic_idx, s_idx], R[room_idx, mic_idx, s_idx] = rt60_from_rirs(RIRs[0, mic_idx, :, 0], self.fs)

    #     t60 = np.median(T60.reshape(nroom_env,-1), axis=1)
    #     S = (room_sz[0] * room_sz[1] + room_sz[0] * room_sz[2] + room_sz[1] * room_sz[2]) * 2
    #     V = np.prod(room_sz)
    #     absor = 0.161 * V / (S * t60) 

    #     return t60, absor


class BUTReverbRIRDataset():
    """ 
        Refs: Building and evaluation of a real room impulse response dataset, 2019
        URL: https://speech.fit.vutbr.cz/software/but-speech-fit-reverb-database
        Code: 
    """
    def __init__(self, data_dir, fs, mic_dist_range=[0.03, 0.20], nmic_selected=2, save_dir=None):

        self.room_names = ['Hotel_SkalskyDvur_ConferenceRoom2', 
                        'Hotel_SkalskyDvur_Room112', 
                        'VUT_FIT_E112',
                        'VUT_FIT_L207', 
                        'VUT_FIT_L212', 
                        'VUT_FIT_L227', 
                        'VUT_FIT_Q301', 
                        'VUT_FIT_C236', 
                        'VUT_FIT_D105']

        data_dir = data_dir + '/RIRs'
        self.array_names = ['spherical']
        self.nmic = 8 # 8-channel array
        self.fs = fs
        self.mic_dist_range = mic_dist_range
        self.nmic_selected = nmic_selected
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.dataset_sz = 1 

    def __len__(self):
        return self.dataset_sz
    
    def gen_rir(self):
        rir_num = 0
        for room_name in self.room_names:
            spk_dir = self.data_dir + '/' + room_name + '/MicID01'
            speaker_names = os.listdir(spk_dir)
            for array_name in self.array_names:
                print('data generation: ', room_name, array_name)
                for spk in speaker_names:
                    spk_short = spk.split('_')[0]
                    mic_dir = self.data_dir + '/' + room_name + '/MicID01/' + spk
                    mic_names = os.listdir(mic_dir)

                    attrs = {}
                    attrs['room_sz'] = np.zeros((3, self.nmic))
                    attrs['t60'] = np.zeros((1, self.nmic))
                    attrs['t30'] = np.zeros((1, self.nmic))
                    attrs['t20'] = np.zeros((1, self.nmic))
                    attrs['mic_pos'] = np.zeros((3, self.nmic))
                    attrs['sou_pos'] = np.zeros((3, self.nmic))
                    rir = []
                    
                    for mic in mic_names:
                        ir_path = self.data_dir + '/' + room_name + '/MicID01/' + spk + '/' + mic
                        if os.path.isdir(ir_path):
                            mic_idx = int(mic)-1
                            txt_path = ir_path + '/' + 'mic_meta.txt'
                            attr = {}
                            with open(txt_path, 'r', encoding="UTF-8") as infile:
                                for line in infile:
                                    data_line = line.strip("\n").split()
                                    if len(data_line) == 2:
                                        attr[data_line[0][1:]] = data_line[1]
                            
                            micID = attr['EnvMicID']
                            mictypeID = attr[ 'EnvMic' + micID + 'TypeID' ] 
                            if '01-'+micID == mictypeID: # 8-channel array
                                attrs['t60'][:, mic_idx] = float(attr['EnvMic' + micID + 'RelRT60'])
                                attrs['t30'][:, mic_idx] = float(attr['EnvMic' + micID + 'RelRT30'])
                                attrs['t20'][:, mic_idx] = float(attr['EnvMic' + micID + 'RelRT20'])
                                attrs['mic_pos'][:, mic_idx] = [max(0, float(attr['EnvMic' + micID + 'Depth'])),
                                                                max(0, float(attr['EnvMic' + micID + 'Width'])),
                                                                max(0, float(attr['EnvMic' + micID + 'Height']))]
                                attrs['sou_pos'][:, mic_idx] = [float(attr['EnvSpk1Depth']),
                                                                float(attr['EnvSpk1Width']),
                                                                float(attr['EnvSpk1Height'])]
                                attrs['room_sz'][:, mic_idx] = [float(attr['EnvDepth']), 
                                                                float(attr['EnvWidth']), 
                                                                float(attr['EnvHeight'])]

                                wav_path = ir_path + '/RIR/'
                                if os.path.isdir(wav_path):
                                    wav_names = os.listdir(wav_path)
                                    wav_name = wav_names[0]
                                    rir_one_mic, rir_fs = soundfile.read(wav_path + wav_name)
                                    rir.append(rir_one_mic)

                    rir = np.array(rir).transpose(1, 0) # (nsample, nmic)
                    mic_poss = attrs['mic_pos'].transpose(1, 0)

                    mic_combins = [list(c) for c in combinations(range(self.nmic), self.nmic_selected)]
                    for mic_idxes in mic_combins:   
                        if micpair_dist_in_range(mic_poss[mic_idxes, :], self.mic_dist_range):
                            # Microphone position
                            mic_pos = mic_poss[mic_idxes, :]

                            # Source postion 
                            sou_pos = attrs['sou_pos'][:, 0]
                            traj_pts = sou_pos[np.newaxis, : , np.newaxis] # (npoints=1,3,nsources=1)

                            # RIR
                            RIRs = rir[:, mic_idxes]
                            if self.fs != rir_fs:
                                RIRs = scipy.signal.resample_poly(RIRs, self.fs, rir_fs)
                            RIRs = RIRs[np.newaxis, :, :, np.newaxis].transpose(0, 2, 1, 3)  # (npoints=1, nmic, nsample, nsources=1)

                            room_sz = attrs['room_sz'][:, 0]

                            T60_gt = attrs['t60']
                            T60_gt = np.mean(T60_gt)
                            T30_gt = attrs['t30']
                            T30_gt = np.mean(T30_gt)
                            T20_gt = attrs['t20']
                            T20_gt = np.mean(T20_gt)
                            
                            rir_num += 1
                            if self.save_dir is not None:
                                save_dir = self.save_dir + '/' + room_name+'/'+array_name+'/'
                                Path(save_dir).mkdir(parents=True, exist_ok=True)
                                file_name = 'SP'+ spk_short + '_MP' + '-' + str(mic_idxes[0]+1) + '-' + str(mic_idxes[1]+1)
                                save_file = os.path.join(save_dir, file_name + '.npy')
                                np.save(save_file, arr=RIRs.astype(np.float16))
                                save_file = os.path.join(save_dir, file_name + '_info.npz')
                                data = {
                                    'room_sz': room_sz,
                                    'mic_pos': mic_pos,
                                    'T60': T60_gt,
                                    'traj_pts': traj_pts,
                                    'fs': self.fs,
                                }
                                np.savez(save_file, **data)

        return rir_num

    def gen_noise(self):
        num = 0
        noise_type = 'silence'
        for room_name in self.room_names:
            spk_dir = self.data_dir + '/' + room_name + '/MicID01'
            speaker_names = os.listdir(spk_dir)
            for array_name in self.array_names:
                print('data generation: ', room_name, array_name)
                for spk in speaker_names:
                    spk_short = spk.split('_')[0]
                    mic_dir = self.data_dir + '/' + room_name + '/MicID01/' + spk
                    mic_names = os.listdir(mic_dir)

                    attrs = {}
                    attrs['mic_pos'] = np.zeros((3, self.nmic))
                    noise = []
                    for mic in mic_names:
                        ir_path = self.data_dir + '/' + room_name + '/MicID01/' + spk + '/' + mic
                        if os.path.isdir(ir_path):
                            mic_idx = int(mic)-1
                            txt_path = ir_path + '/' + 'mic_meta.txt'
                            attr = {}
                            with open(txt_path, 'r', encoding="UTF-8") as infile:
                                for line in infile:
                                    data_line = line.strip("\n").split()
                                    if len(data_line) == 2:
                                        attr[data_line[0][1:]] = data_line[1]
                            
                            micID = attr['EnvMicID']
                            mictypeID = attr[ 'EnvMic' + micID + 'TypeID' ] 
                            if '01-'+micID == mictypeID: # 8-channel array
                                attrs['mic_pos'][:, mic_idx] = [max(0, float(attr['EnvMic' + micID + 'Depth'])),
                                                                max(0, float(attr['EnvMic' + micID + 'Width'])),
                                                                max(0, float(attr['EnvMic' + micID + 'Height']))]

                                noise_wav_path = ir_path + '/' + noise_type + '/'
                                if os.path.isdir(noise_wav_path):
                                    noise_wav_names = os.listdir(noise_wav_path)
                                    for noise_wav_name in noise_wav_names:
                                        if noise_wav_names.index(noise_wav_name) == 0:
                                            noise_one_mic, noise_fs = soundfile.read(noise_wav_path + noise_wav_name)
                                        else:
                                            noise_one_mic_add, noise_fs = soundfile.read(noise_wav_path + noise_wav_name)
                                            noise_one_mic = np.concatenate((noise_one_mic, noise_one_mic_add))
                                        
                                    noise.append(noise_one_mic)

                    noise = np.array(noise).transpose(1, 0) # (nsample', nmic)
                    mic_poss = attrs['mic_pos'].transpose(1, 0)

                    mic_combins = [list(c) for c in combinations(range(self.nmic), self.nmic_selected)] 
                    for mic_idxes in mic_combins:   
                        if micpair_dist_in_range(mic_poss[mic_idxes, :], self.mic_dist_range):
                            noise_signal = noise[:, mic_idxes]
                            if self.fs != noise_fs:
                                noise_signal = scipy.signal.resample_poly(noise_signal, self.fs, noise_fs)
                            num += 1

                            if self.save_dir is not None:
                                save_dir = self.save_dir + '/' + room_name+'/'+array_name+'/'
                                Path(save_dir).mkdir(parents=True, exist_ok=True)
                                file_name = 'SP'+ spk_short + '_MP' + '-' + str(mic_idxes[0]+1) + '-' + str(mic_idxes[1]+1) + '_' + noise_type
                                sig_path = save_dir + file_name + '.wav'
                                soundfile.write(sig_path, noise_signal, self.fs) 

        return num


class ACERIRDataset():
    """ 
        Refs: The ACE challenge — Corpus description and performance evaluation, 2015
        URL: https://zenodo.org/record/6257551
        Code: 
    """
    def __init__(self, data_dir, fs, mic_dist_range=[0.03, 0.20], nmic_selected=2, c=340, save_dir=None):

        self.data_dirs = {'rir': data_dir + '/RIRN/',
                            'noise': data_dir + '/RIRN/',
                            'anno': data_dir + '/Data/',}
        self.array_names = ['Chromebook', 'Mobile', 'Lin8Ch','EM32'] # 'Single'  'Crucif', 
        self.room_names = [ 'Building_Lobby', 'Lecture_Room_1', 'Lecture_Room_2', 
                            'Meeting_Room_1', 'Meeting_Room_2', 'Office_1', 'Office_2']
        self.array_pos_names = ['1', '2']
        self.mic_poss = {   'Chromebook': np.array([[0, 0, 0], [0, 0.062, 0]]), 
                            'Mobile': np.array([[0.045, 0, 0], [0, 0, 0], [0, 0.0893029, 0]]), 
                            'Crucif': np.array([[0, 0, 0], [0.25, 0, 0], [0, 0.25, 0], [-0.25, 0, 0], [0, -0.25, 0]]), 
                            'Lin8Ch': np.array([[0, 0, 0], [0.06, 0, 0], [0.12, 0, 0], [0.18, 0, 0],
                                            [0.24, 0, 0], [0.3, 0, 0], [0.36, 0, 0], [0.42, 0, 0]]), 
                            'EM32': np.array((
                                ( 0.000,  0.039,  0.015), (-0.022,  0.036,  0.000), ( 0.000,  0.039, -0.015), ( 0.022,  0.036,  0.000),
                                ( 0.000,  0.022,  0.036), (-0.024,  0.024,  0.024), (-0.039,  0.015,  0.000), (-0.024,  0.024,  0.024),
                                ( 0.000,  0.022, -0.036), ( 0.024,  0.024, -0.024), ( 0.039,  0.015,  0.000), ( 0.024,  0.024,  0.024),
                                (-0.015,  0.000,  0.039), (-0.036,  0.000,  0.022), (-0.036,  0.000, -0.022), (-0.015,  0.000, -0.039),
                                ( 0.000, -0.039,  0.015), ( 0.022, -0.036,  0.000), ( 0.000, -0.039, -0.015), (-0.022, -0.036,  0.000),
                                ( 0.000, -0.022,  0.036), ( 0.024, -0.024,  0.024), ( 0.039, -0.015,  0.000), ( 0.024, -0.024, -0.024),
                                ( 0.000, -0.022, -0.036), (-0.024, -0.024, -0.024), (-0.039, -0.015,  0.000), (-0.024, -0.024,  0.024),
                                ( 0.015,  0.000,  0.039), ( 0.036,  0.000,  0.022), ( 0.036,  0.000, -0.022), ( 0.015,  0.000, -0.039)))
                }

        self.room_szs = {'Building_Lobby': np.array([4.47, 5.13, 3.18]), 
                        'Lecture_Room_1': np.array([6.93, 9.73, 3]), 
                        'Lecture_Room_2': np.array([13.6, 9.29, 2.94]), 
                        'Meeting_Room_1': np.array([6.61, 5.11, 2.95]), 
                        'Meeting_Room_2': np.array([10.3, 9.07, 2.63]), 
                        'Office_1': np.array([3.32, 4.83, 2.95]), 
                        'Office_2': np.array([3.22, 5.1, 2.94])}

        self.rirs = {}
        self.rir_fss = {}
        
        dataset_sz = 0
        for room_name in self.room_names:
            for array_name in self.array_names:
                for array_pos_name in self.array_pos_names:
                    dataset_sz += self.mic_poss[array_name].shape[0]    

        self.fs = fs
        self.mic_dist_range = mic_dist_range
        self.nmic_selected = nmic_selected
        self.dataset_sz = dataset_sz
        self.data_dir = data_dir
        self.save_dir = save_dir

    def __len__(self):
        return self.dataset_sz
    
    def _find_dp_from_rir(self, rir, th_ratio=0.5, num_largests = 5):

        peaks, _ = find_peaks(rir)
        peak_heights = rir[peaks]

        # find a number of largest peaks
        largest_peak_indices = np.argsort(peak_heights)[-num_largests:]

        # obtrain index and value of largest peaks
        largest_peaks = peaks[largest_peak_indices]
        largest_peak_values = rir[largest_peaks]

        # find maximum value of RIR
        max_value = np.max(rir)

        # select the peaks that are larger than 0.5 times the maximum value
        threshold = th_ratio * max_value
        filtered_peaks = largest_peaks[largest_peak_values >= threshold]
        filtered_peak_values = largest_peak_values[largest_peak_values >= threshold]

        # find the peak with the smallest index
        if len(filtered_peaks) > 0:
            dp_index = filtered_peaks[np.argmin(filtered_peaks)]
            dp_value = rir[dp_index]
        else:
            dp_index = None
            dp_value = None

        return dp_index, dp_value

    def gen_rir_plot(self):
        # Select microphone pairs within mic_dist_range, calculate corresponding annotation, and resample RIRs
        
        annos = {}
        csv_name = '20150814T154139_Corpus_Mean_DRRs_and_T60s.csv'
        anno = pd.read_csv(self.data_dirs['anno'] + csv_name, sep=', ', engine='python')
        nanno = len(anno['Mic config:'])
        for idx in range(nanno):
            array_name = anno['Mic config:'][idx]
            room_name = anno['Room decode:'][idx]
            array_pos_name = str(anno['Room config:'][idx])
            ch_idx = int(anno['Chan:'][idx])-1
            FB_T60 = anno['FB T60:'][idx]
            FB_DRR = anno['FB DRR:'][idx]
            key = room_name+'/'+array_name+'/'+array_pos_name
            if ch_idx == 0:
                annos[key] = np.zeros((2, self.mic_poss[array_name].shape[0]))
            annos[key][:, ch_idx] = [FB_T60, FB_DRR]
        print('annotation data loaded!')

        rir_num = 0
        for room_name in self.room_names:
            for array_name in self.array_names:
                for array_pos_name in self.array_pos_names:
                    print('data generation: ', room_name, array_name, array_pos_name)

                    data_path = self.data_dirs['rir'] + array_name + '/' + room_name + '/' + array_pos_name
                    wavnames = os.listdir(data_path)
                    for wav_name in wavnames:
                        if 'RIR' in wav_name:
                            rirs, rir_fss = soundfile.read( os.path.join(data_path, wav_name) ) 
                                        
                    nsample = 200
                    plt.figure(figsize=(10, 5))
                    nmic = rirs.shape[1]
                    for mic_idx in range(nmic):
                        plt.subplot(nmic, 1, mic_idx+1)
                        plt.plot(range(len(rirs[0:nsample, mic_idx])), rirs[0:nsample, mic_idx])
                        dp_idx, dp_value = self._find_dp_from_rir(rirs[0:nsample, mic_idx])
                        plt.plot([dp_idx], [dp_value], 'ro')
                    ddir = self.save_dir + '/' 
                    exist_temp = os.path.exists(ddir)
                    if exist_temp==False:
                        os.makedirs(ddir)
                        print('make dir: ' + ddir)
                    plt.savefig(ddir + room_name+'-'+array_name+'-'+ 'SP1_MP' + array_pos_name + '_RIR.png')
                    plt.close()

        return rir_num


    def gen_rir(self):
        # Select microphone pairs within mic_dist_range, calculate corresponding annotation, and resample RIRs
        
        annos = {}
        csv_name = '20150814T154139_Corpus_Mean_DRRs_and_T60s.csv'
        anno = pd.read_csv(self.data_dirs['anno'] + csv_name, sep=', ', engine='python')
        nanno = len(anno['Mic config:'])
        for idx in range(nanno):
            array_name = anno['Mic config:'][idx]
            room_name = anno['Room decode:'][idx]
            array_pos_name = str(anno['Room config:'][idx])
            ch_idx = int(anno['Chan:'][idx])-1
            FB_T60 = anno['FB T60:'][idx]
            FB_DRR = anno['FB DRR:'][idx]
            key = room_name+'/'+array_name+'/'+array_pos_name
            if ch_idx == 0:
                annos[key] = np.zeros((2, self.mic_poss[array_name].shape[0]))
            annos[key][:, ch_idx] = [FB_T60, FB_DRR]
        print('annotation data loaded!')

        rir_num = 0
        for room_name in self.room_names:
            for array_name in self.array_names:
                for array_pos_name in self.array_pos_names:
                    print('data generation: ', room_name, array_name, array_pos_name)

                    data_path = self.data_dirs['rir'] + array_name + '/' + room_name + '/' + array_pos_name
                    wavnames = os.listdir(data_path)
                    for wav_name in wavnames:
                        if 'RIR' in wav_name:
                            rirs, rir_fss = soundfile.read( os.path.join(data_path, wav_name) ) 

                    key = room_name+'/'+array_name+'/'+array_pos_name
                    nmic = self.mic_poss[array_name].shape[0]
                    mic_combins = [list(c) for c in combinations(range(nmic), self.nmic_selected)]
                    for mic_idxes in mic_combins:
                        if micpair_dist_in_range(self.mic_poss[array_name][mic_idxes, :], self.mic_dist_range):
                            assert rirs.shape[1]==self.mic_poss[array_name].shape[0], 'Mic number of RIR and mic_pos is unmatched~'

                            # RIR
                            if self.fs != rir_fss:
                                rir = scipy.signal.resample_poly(rirs, self.fs, rir_fss) # (nsample, nmic)
                            RIRs = rir[np.newaxis, :, mic_idxes, np.newaxis].transpose(0, 2, 1, 3) # (npoints=1, nmic, nsample, nsources=1)
 
                            # T60, DRR
                            T60_gt = annos[key][0][mic_idxes]
                            DRR_gt = annos[key][1][mic_idxes]
                            T60_gt = np.mean(T60_gt)
                            DRR_gt = DRR_gt[0] # reference channel

                            # find direct path     
                            eps=1e-8                
                            nsample_search = int(self.fs/160)
                            npt, nmic, nsample, nsrc = RIRs.shape 
                            assert (npt==1) & (nsrc==1), [npt, nsrc]
                            dp_idx = np.zeros((nmic, nsample))
                            for mic_idx in range(nmic):
                                dp_idx[mic_idx, :], dp_value = self._find_dp_from_rir(RIRs[0, mic_idx, 0:nsample_search, 0])
                            
                            # DRR
                            dp_extra = int(self.fs * 2.5 / 1000) * np.ones((nmic, nsample))
                            whole_range = np.array(range(0, nsample))
                            whole_range = np.tile(whole_range[np.newaxis, :], (nmic, 1))
                            dp_range = (whole_range >= (dp_idx - dp_extra)) & (whole_range <= (dp_idx + dp_extra))
                            dp_range = dp_range.astype('float')
                            rev_range = np.ones_like(dp_range) - dp_range
                            dp_pow = np.sum(RIRs[0,:,:,0]**2 * dp_range, axis=1)
                            rev_pow = np.sum(RIRs[0,:,:,0]**2 * rev_range, axis=1)
                            DRR = 10 * np.log10(dp_pow / (rev_pow+eps)+eps)  # (nch,)
                            DRR = DRR[0] # reference channel

                            # C50  
                            early_extra = int(self.fs * 50 / 1000) * np.ones((nmic, nsample))
                            whole_range = np.array(range(0, nsample))
                            whole_range = np.tile(whole_range[np.newaxis, :], (nmic, 1))
                            early_range = (whole_range <= (dp_idx + early_extra))
                            early_range = early_range.astype('float')
                            late_range = np.ones_like(early_range) - early_range
                            early_pow = np.sum(RIRs[0,:,:,0]**2 * early_range, axis=1)
                            late_pow = np.sum(RIRs[0,:,:,0]**2 * late_range, axis=1)
                            C50 = 10 * np.log10(early_pow / (late_pow + eps)+eps)  # (nch,)
                            C50 = C50[0] # reference channel

                            # Room size
                            room_sz = self.room_szs[room_name]

                            # ABS
                            vol = room_sz[0]*room_sz[1]*room_sz[2] 
                            sur = (room_sz[0]*room_sz[1]+room_sz[1]*room_sz[2]+room_sz[0]*room_sz[2])*2 
                            ABS = 0.161*vol/T60_gt/sur

                            # Mic position
                            mic_pos = self.mic_poss[array_name][mic_idxes, :]

                            rir_num += 1
                            if self.save_dir is not None:
                                save_dir = self.save_dir + '/' + room_name+'/'+array_name+'/'
                                Path(save_dir).mkdir(parents=True, exist_ok=True)
                                file_name = 'SP1_MP' + array_pos_name + '-' + str(mic_idxes[0]+1) + '-' + str(mic_idxes[1]+1)
                                save_file = os.path.join(save_dir, file_name + '.npy')
                                np.save(save_file, arr=RIRs.astype(np.float16))
                                save_file = os.path.join(save_dir, file_name + '_info.npz')
                                data = {
                                    'room_sz': room_sz,
                                    'mic_pos': mic_pos,
                                    'T60fromDataset': T60_gt, #
                                    'DRRfromDataset': DRR_gt,
                                    'DRR': DRR, #
                                    'C50': C50, #
                                    'ABS': ABS, #
                                    'fs': self.fs,
                                }
                                np.savez(save_file, **data)
        return rir_num

    def gen_noise(self):
        # Select microphone pairs within mic_dist_range, and resample noise signals
        num = 0
        for room_name in self.room_names:
            for array_name in self.array_names:
                for array_pos_name in self.array_pos_names:
                    print('data generation: ', room_name, array_name, array_pos_name)

                    data_path = self.data_dirs['noise'] + array_name + '/' + room_name + '/' + array_pos_name
                    wavnames = os.listdir(data_path)
                    noises = {}
                    noise_fss = {}
                    for wav_name in wavnames:
                        if 'Noise' in wav_name:
                            noise_type = wav_name.split('_')[-1].split('.')[0]
                            noises[noise_type], noise_fss[noise_type] = soundfile.read( os.path.join(data_path, wav_name) ) 

                    nmic = self.mic_poss[array_name].shape[0]
                    mic_combins = [list(c) for c in combinations(range(nmic), self.nmic_selected)]
                    for mic_idxes in mic_combins:
                        if micpair_dist_in_range(self.mic_poss[array_name][mic_idxes, :], self.mic_dist_range):
                            noise_types = noises.keys()
                            for noise_type in noise_types:
                                num += 1
                                noise_signal = noises[noise_type]
                                noise_ch = noise_signal.shape[-1]
                                if noise_ch != self.mic_poss[array_name].shape[0]:
                                    print('Mic number of noise and rir is unmatched:', room_name+'/'+array_name+'/'+array_pos_name,noise_type,noise_signal.shape)
                                    duration = 5
                                    noise_signal = np.zeros((int(duration*self.fs), self.nmic_selected))
                                else:
                                    noise_fs = noise_fss[noise_type]
                                    noise_signal = noise_signal[:, mic_idxes]
                                    if self.fs != noise_fs:
                                        noise_signal = scipy.signal.resample_poly(noise_signal, self.fs, noise_fs) 
                                if self.save_dir is not None:
                                    save_dir = self.save_dir + '/' + room_name+'/'+array_name+'/'
                                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                                    file_name = '_MP' + array_pos_name + '-' + str(mic_idxes[0]+1) + '-' + str(mic_idxes[1]+1) + '_' + noise_type
                                    sig_path = save_dir + file_name + '.wav'
                                    soundfile.write(sig_path, noise_signal, self.fs) 

        return num
    

if __name__ == '__main__':
    
    cpu_num = 8*5
    os.environ["OMP_NUM_THREADS"] = str(cpu_num) 
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    
    parser = argparse.ArgumentParser(description='Generating multi-channel RIRs')
    # parser.add_argument('--data_id', type=int, nargs='+', default=[0], metavar='Datasets', help='dataset IDs (default: 0)')
    parser.add_argument('--dataset', type=str, nargs='+', default=['DCASE'], metavar='Datasets', help='dataset names (default: DCASE)')
    parser.add_argument('--data_type', type=str, nargs='+', default=['rir', 'noise'], metavar='DataType', help='data types (default: rir, noise)')
    parser.add_argument('--fs', type=int, default=16000, metavar='SamplingRate', help='sampling rate (default: 16000)')
    parser.add_argument('--nmic', type=int, default=2, metavar='NumMic', help='number of microphones (default: 2)')
    parser.add_argument('--mic_dist_range', type=list, default=[0.03, 0.20], metavar='MicDistRange', help='range of microphone distances (default: [0.03, 0.20])')
    parser.add_argument('--read_dir', type=str, default='', metavar='ReadDir', help='read directory')
    parser.add_argument('--save_dir', type=str, default='', metavar='SaveDir', help='save directory')
    args = parser.parse_args()

    # opts = opt()
    # dirs = opts.dir()

    fs = args.fs
    mic_dist_range = args.mic_dist_range
    nmic = args.nmic
    data_type = args.data_type
    dataset_list = args.dataset
    ori_read_dir = args.read_dir
    ori_save_dir = args.save_dir

    for dataset_name in dataset_list:
        assert dataset_name in ['DCASE', 'Mesh', 'MIR', 'dEchorate', 'BUTReverb', 'ACE'], 'Dataset not found'
        for data in data_type:

            print('Dataset=', dataset_name, 'Data=',data)
            read_dir = os.path.join(ori_read_dir, dataset_name)
            if data == 'rir':
                save_dir = os.path.join(ori_save_dir, dataset_name)
            elif data == 'noise':
                save_dir = os.path.join(ori_save_dir, dataset_name + '_noise')
 
            exist_temp = os.path.exists(save_dir)
            if exist_temp==False:
                os.makedirs(save_dir)
                print('make dir: ' + save_dir)
            else:
                print('existed dir: ' + save_dir)
                msg = input('Sure to regenerate ' +data+ '? (Enter for yes)')
                if msg == '':
                    print('Regenerating ' +data)

            # RIR dataset
            if dataset_name == 'DCASE':
                rirDataset = DCASERIRDataset(
                    fs = fs,
                    data_dir = read_dir,
                    nmic_selected=nmic,
                    mic_dist_range=mic_dist_range,
                    save_dir = save_dir,
                )
            elif dataset_name == 'Mesh':
                assert 'noise' not in data_type, 'MeshRIR dataset do not contain noise'
                rirDataset = MeshRIRDataset(
                    fs = fs,
                    data_dir = read_dir,
                    nmic_selected=nmic,
                    mic_dist_range=mic_dist_range,
                    save_dir = save_dir,
                )
            elif dataset_name == 'MIR':
                assert 'noise' not in data_type, 'MIRRIR dataset do not contain noise'
                rirDataset = MIRRIRDataset(
                    fs = fs,
                    data_dir = read_dir,
                    nmic_selected=nmic,
                    mic_dist_range=mic_dist_range,
                    save_dir = save_dir,
                )
            elif dataset_name == 'dEchorate':
                rirDataset = dEchorateRIRDataset(
                    fs = fs,
                    data_dir = read_dir,
                    nmic_selected=nmic,
                    mic_dist_range=mic_dist_range,
                    save_dir = save_dir,
                )
            elif dataset_name == 'BUTReverb':
                rirDataset = BUTReverbRIRDataset(
                    fs = fs,
                    data_dir = read_dir,
                    nmic_selected=nmic,
                    mic_dist_range=mic_dist_range,
                    save_dir = save_dir,
                )
            elif dataset_name == 'ACE':
                rirDataset = ACERIRDataset(
                    fs = fs,
                    data_dir = read_dir,
                    nmic_selected=nmic,
                    mic_dist_range=mic_dist_range,
                    save_dir = save_dir,
                )
     
            if data=='rir':
                data_num = rirDataset.gen_rir()
                # data_num = rirDataset.gen_rir_plot()
            elif data=='noise':
                data_num = rirDataset.gen_noise()
            print(data_num)


    # python gen_real_rir.py --dataset DCASE dEchorate BUTReverb ACE --data_type rir noise --read_dir /data/home/yangbing/data/RIR/ --save_dir /data/home/yangbing/SAR-SSL/data/RIR/real/
    # python gen_real_rir.py --dataset Mesh MIR --data_type rir --read_dir /data/home/yangbing/data/RIR/ --save_dir /data/home/yangbing/SAR-SSL/data/RIR/real/
