import time
import os

class opt(): 
    def __init__(self, wnoise=False, ins_enable=False):
        time_stamp = time.time()
        local_time = time.localtime(time_stamp)
        self.time = time.strftime('%m%d', local_time)
        self.work_dir = r'~'
        self.work_dir = os.path.abspath(os.path.expanduser(self.work_dir))

        # acoustic setting
        if wnoise:
            snr_range = [15,30]
            noise_type = ['diffuse_white']
        else:
            snr_range = [100, 100]
            noise_type = ['']
        room_size_range = [[3, 3, 2.5], [15, 10, 6]]
        T60_range = [0.2, 1.3]
        array_pos_ratio_range = [[0.2, 0.2, 0.1], [0.8, 0.8, 0.5]]
        min_src_array_dist = 0.3
        min_src_boundary_dist = 0.3 
        sound_speed = 343.0
        self.room_setting = {
            'snr_range': snr_range,
            'noise_type': noise_type,
            'room_size_range': room_size_range,
            't60_range': T60_range,
            'array_pos_ratio_range': array_pos_ratio_range,
            'min_src_array_dist': min_src_array_dist,
            'min_src_boundary_dist': min_src_boundary_dist,
            'sound_speed': sound_speed, 
        }
        self.micsig_setting = {
            'T': 4.112,  # time length (s) 
            'fs': 16000  # sampling rate (Hz)
        }

        # visualization instance
        if ins_enable:
            if wnoise:
                snr_range = [20,20]
            room_size_range = [[5, 3, 2.5], [10, 6, 3]]
            T60_range = [1.0, 1.0] # [0.3, 1.0]
            ins_flag = 'ins'
        else:
            ins_flag = ''

        # version setting
        self.rir_ver = ''
        self.sig_ver = ''
        if wnoise:
            self.noise_flag = '-Noi'+str(snr_range[0])+'_'+str(snr_range[1])+'dB'
        else:
            self.noise_flag = ''
        self.sig_ver = self.sig_ver + self.noise_flag
        if ins_enable:
            self.sig_ver = '-' + ins_flag

    def dir(self):
        """ Get directories of code, data and experimental results
        """ 
        work_dir = self.work_dir
        dirs = {}

        dirs['data'] = work_dir + '/data'
        dirs['gerdata'] = work_dir + '/SAR-SSL/data'
 
        dirs['sousig_pretrain'] = dirs['data'] + '/SouSig/wsj0/tr'
        dirs['sousig_preval'] = dirs['data'] + '/SouSig/wsj0/dt'
        dirs['sousig_train'] = dirs['data'] + '/SouSig/wsj0/tr'
        dirs['sousig_val'] = dirs['data'] + '/SouSig/wsj0/dt'
        dirs['sousig_test'] = dirs['data'] + '/SouSig/wsj0/et'
        
        dirs['noisig_pretrain'] = dirs['data'] + '/NoiSig/5th-DNS-Challenge/noise_fullband'
        dirs['noisig_preval'] = dirs['data'] + '/NoiSig/5th-DNS-Challenge/noise_fullband'
        dirs['noisig_train'] = dirs['data'] + '/NoiSig/NOISEX-92'
        dirs['noisig_val'] = dirs['data'] + '/NoiSig/NOISEX-92'
        dirs['noisig_test'] = dirs['data'] + '/NoiSig/NOISEX-92'

        # RIR (& noise) & signal
        # dirs['simulate'] = dirs['gerdata'] + '/RIR-simulate' 
        dirs['DCASE'] = dirs['data'] + '/RIR/DCASE/TAU-SRIR_DB'
        dirs['MIR'] = dirs['data'] + '/RIR/MIR/Impulse_response_Acoustic_Lab_Bar-Ilan_University'
        dirs['Mesh'] = dirs['data'] + '/RIR/Mesh'
        dirs['BUTReverb'] = dirs['data'] + '/RIR/BUTReverb/RIRs'
        dirs['IRArni'] = dirs['data'] + '/RIR/IR_Arni/IR'
        dirs['dEchorate'] = dirs['data'] + '/RIR/dEchorate'
        dirs['ACE'] = dirs['data'] + '/RIR/ACE'
        dirs['LOCATA'] = dirs['data'] + '/SenSig/LOCATA'

        # generated RIRs/signals
        dirs_pretrain = dirs['gerdata'] + '/SenSig-pretrain' + self.sig_ver
        dirs_preval = dirs['gerdata'] + '/SenSig-preval' + self.sig_ver 

        dirs['sensig_pretrain'] = [      
                            dirs_pretrain + '/DCASE',         
                            dirs_pretrain + '/MIR',
                            dirs_pretrain + '/Mesh',
                            dirs_pretrain + '/ACE',
                            dirs_pretrain + '/dEchorate',
                            dirs_pretrain + '/BUTReverb',
                            dirs_pretrain + '/LOCATA',
                            dirs_pretrain + '/simulate',
                        ]

        dirs['sensig_preval'] = [    
                            dirs_preval + '/DCASE',           
                            dirs_preval + '/MIR',
                            dirs_preval + '/Mesh',
                            dirs_preval + '/ACE',
                            dirs_preval + '/dEchorate',
                            dirs_preval + '/BUTReverb',
                            dirs_preval + '/LOCATA',
                            dirs_preval + '/simulate',
                        ]

        dirs_train = dirs['gerdata'] + '/SenSig-train' + self.sig_ver
        dirs_val = dirs['gerdata'] + '/SenSig-val' + self.sig_ver
        dirs_test = dirs['gerdata'] + '/SenSig-test' + self.sig_ver

        dirs['sensig_train'] = [ 
            dirs_train + '/simulate', 
            dirs_train + '/BUTReverb', 
            dirs_train + '/ACE', 
            dirs_train + '/dEchorate',
            dirs_train + '/LOCATA'
            ]
        dirs['sensig_val'] = [ 
            dirs_val + '/simulate', 
            dirs_val + '/BUTReverb', 
            dirs_val + '/ACE', 
            dirs_val + '/dEchorate',
            dirs_val + '/LOCATA'
            ]
        dirs['sensig_test'] = [ 
            dirs_test + '/simulate',
            dirs_test + '/DCASE',           
            dirs_test + '/MIR',
            dirs_test + '/Mesh',
            dirs_test + '/BUTReverb', 
            dirs_test + '/ACE', 
            dirs_test + '/dEchorate',
            dirs_test + '/LOCATA',
            # dirs_test + '/LOCATA_task1_4s',
            ]
        
        dirs_rir = dirs['gerdata'] + '/RIR' + self.rir_ver
        dirs['rir'] = [
            dirs_rir + '/simulate',
            dirs_rir + '/BUTReverb',
            dirs_rir + '/BUTReverb_noise',
            dirs_rir + '/ACE',
            dirs_rir + '/ACE_noise',
            dirs_rir + '/dEchorate',
            dirs_rir + '/dEchorate_noise',
            dirs_rir + '/DCASE',
            dirs_rir + '/DCASE_noise',
            dirs_rir + '/Mesh',
            dirs_rir + '/MIR',
        ]

        return dirs

if __name__ == '__main__':

    dirs = opt().dir()
    print('code path:' + dirs['data']) 