import argparse
import time
import os
import numpy as np

class opt_pretrain():
    def __init__(self):
        time_stamp = time.time()
        local_time = time.localtime(time_stamp)
        self.time = time.strftime('%m%d%H%M', local_time)
        self.work_dir = r'~'
        self.work_dir = os.path.abspath(os.path.expanduser(self.work_dir))
        self.work_dir_local = os.path.abspath(os.path.expanduser(self.work_dir))
       
        self.sig_ver = ''
        self.rir_ver = ''
        self.noise_flag = ''
        self.time_ver = '0821'
        
        # Noise setting: w/ noise or w/o noise
        # self.wnoise = False
        self.wnoise = True
        if self.wnoise: 
            self.noise_flag = '-Noi15_30dB'
            snr_range = [15, 30]
            noise_type = ['diffuse_white'] 
        else:
            self.noise_flag = ''
            snr_range = [100, 100]
            noise_type = ['']
        self.noise_setting = {'noise_enable': self.wnoise, 'snr_range': snr_range, 'noise_type': noise_type,}
        
        # Array setting
        self.array_setting = {'nmic':  2}

        # Other acoustic setting
        self.acoustic_setting = {'speed': 343.0, 'fs': 16000}

        self.extra_info = '' # needs add '-' when used

        # Pretrain: simulated or real-world
        ##########################################
        # self.pretrain_sim = False
        self.pretrain_sim = True
        ##########################################

    def parse(self):
        """ Function: Define optional arguments
        """
        parser = argparse.ArgumentParser(description='Self-supervised learing for multi-channel audio processing')

        # for training and test stages
        parser.add_argument('--gpu-id', type=str, default='7', metavar='GPU', help='GPU ID (default: 7)')
        parser.add_argument('--workers', type=int, default=8, metavar='Worker', help='number of workers (default: 8)')
        parser.add_argument('--bs', type=int, nargs='+', default=[128, 128, 128], metavar='TrainValTestBatch', help='batch size for training, validation and test (default: 128, 128, 128)')
        parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training (default: False)')
        parser.add_argument('--use-amp', action='store_true', default=False, help='Use automatic mixed precision training (default: False)')
        parser.add_argument('--seed', type=int, default=1, metavar='Seed', help='random seed (default: 1)')
        
        parser.add_argument('--checkpoint-start', action='store_true', default=False, help='train model from saved checkpoints (default: False)')
        parser.add_argument('--time', type=str, default=self.time, metavar='Time', help='time flag')
        parser.add_argument('--work-dir', type=str, default=self.work_dir, metavar='WorkDir', help='work directory')

        parser.add_argument('--sources', type=int, nargs='+', default=[1], metavar='Sources', help='number of sources (default: 1)')
        parser.add_argument('--source-state', type=str, default='static', metavar='SourceState', help='state of sources (default: Static)') # ['static', 'mobile']
        parser.add_argument('--rir-gen', type=str, default='wo', metavar='RIRGen', help='mode of RIR generation (default: Without)') # ['wo', 'online', 'offline']

        parser.add_argument('--pretrain', action='store_true', default=False, help='change to pretrain stage (default: False)')
        parser.add_argument('--pretrain-frozen-encoder', action='store_true', default=False, help='change to pretrain stage (default: False)')
        parser.add_argument('--nepoch', type=int, default=30, metavar='Epoch', help='number of epochs to train (default: 30)')
        parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default:0.001)')
        
        parser.add_argument('--test', action='store_true', default=False, help='change to test stage of downstream tasks (default: False)')

        args = parser.parse_args()
        assert (args.pretrain + args.pretrain_frozen_encoder + args.test)==1, 'Pretraining stage (pretrain or test) is undefined'

        self.time = args.time
        self.work_dir = args.work_dir

        args.noise_setting = self.noise_setting
        args.array_setting = self.array_setting
        args.acoustic_setting = self.acoustic_setting

        self.rir_ver = ''
        self.sig_ver = self.rir_ver + self.noise_flag
        
        data = 'sim' if self.pretrain_sim else 'real'
        print('\ntime='+self.time, 'data version='+self.sig_ver, 'noise='+str(self.wnoise), 'data='+data)
        
        return args

    def dir(self):
        """ Function: Get directories of code, data and experimental results
        """ 
        work_dir = self.work_dir
        dirs = {}

        dirs['code'] = work_dir + '/Re-SSL/code'
        # dirs['data'] = work_dir + '/data'
        # dirs['gerdata'] = work_dir + '/Re-SSL/data'
        dirs['data'] = self.work_dir_local + '/data'
        dirs['gerdata'] = self.work_dir_local + '/Re-SSL/data'
        dirs['exp'] = work_dir + '/Re-SSL/exp'

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

        # RIR (+ noise) & mic signal
        dirs['simulate'] = dirs['gerdata'] + '/RIR-simulate' 
        dirs['DCASE'] = dirs['data'] + '/RIR/DCASE/TAU-SRIR_DB'
        dirs['MIR'] = dirs['data'] + '/RIR/MIRDB/Impulse_response_Acoustic_Lab_Bar-Ilan_University'
        dirs['Mesh'] = dirs['data'] + '/RIR/Mesh'
        dirs['BUTReverb'] = dirs['data'] + '/RIR/BUT_ReverbDB/RIRs'
        dirs['IRArni'] = dirs['data'] + '/RIR/IR_Arni/IR'
        dirs['dEchorate'] = dirs['data'] + '/RIR/dEchorate'
        dirs['ACE'] = dirs['data'] + '/RIR/ACE'
        dirs['LOCATA'] = dirs['data'] + '/SenSig/LOCATA'

        ## Generated mic signal
        dirs_pretrain = dirs['gerdata'] + '/SenSig-pretrain' + self.sig_ver
        dirs_preval = dirs['gerdata'] + '/SenSig-preval' + self.sig_ver
        dirs_pretest = dirs['gerdata'] + '/SenSig-test' + self.sig_ver
        dirs_pretest_ins = dirs['gerdata'] + '/SenSig-test-ins'
        if self.pretrain_sim:
            dirs['sensig_pretrain'] = [ dirs_pretrain + '/simulate' + self.time_ver ]
            dirs['sensig_pretrain_ratio'] = [1]
            dirs['sensig_preval'] = [ dirs_preval + '/simulate' + self.time_ver ] 
            dirs['sensig_pretest'] = [ dirs_pretest + '/simulate' + self.time_ver ]
            dirs['sensig_pretest_ins'] = [  dirs_pretest_ins + '/simulate0916-T200',
                                            dirs_pretest_ins + '/simulate0916-T500',
                                            dirs_pretest_ins + '/simulate0916-T1000',]
        else:
            dirs['sensig_pretrain'] = [      
                dirs_pretrain + '/DCASE',         
                dirs_pretrain.replace(self.noise_flag, '') + '/MIR',
                dirs_pretrain.replace(self.noise_flag, '')  + '/Mesh',
                dirs_pretrain + '/ACE',
                dirs_pretrain + '/dEchorate',
                dirs_pretrain + '/BUTReverb',
                dirs_pretrain.replace(self.noise_flag, '')  + '/LOCATA',
            ]
            dirs['sensig_pretrain_ratio'] = [1, 1, 1, 1, 1, 1, 1]
            # dirs['sensig_pretrain_ratio'] = [9, 3, 1, 7, 11, 9, 1]
            assert len(dirs['sensig_pretrain_ratio']) == len(dirs['sensig_pretrain']), 'dataset number and ratio unmatched~'
            dirs['sensig_preval'] = [    
                                dirs_preval + '/DCASE',           
                                dirs_preval.replace(self.noise_flag, '') + '/MIR',
                                dirs_preval.replace(self.noise_flag, '') + '/Mesh',
                                dirs_preval + '/ACE',
                                dirs_preval + '/dEchorate',
                                dirs_preval + '/BUTReverb',
                                dirs_preval.replace(self.noise_flag, '') + '/LOCATA',
                            ]
            dirs['sensig_pretest'] = [     
                                dirs_pretest + '/DCASE',           
                                dirs_pretest.replace(self.noise_flag, '') + '/MIR',
                                dirs_pretest.replace(self.noise_flag, '') + '/Mesh',
                                dirs_pretest + '/ACE',
                                dirs_pretest + '/dEchorate',
                                dirs_pretest + '/BUTReverb', 
                                dirs_pretest.replace(self.noise_flag, '') + '/LOCATA_task1_4s',
                            ]

        # Experimental data
        dirs['log_pretrain'] = dirs['exp'] + '/pretrain/' + self.time
        dirs['log_pretrain_frozen_encoder'] = dirs['exp'] + '/pretrain_frozen_encoder/' + self.time

        return dirs


class opt_downstream():
    def __init__(self):
        time_stamp = time.time()
        local_time = time.localtime(time_stamp)
        self.time = time.strftime('%m%d%H%M', local_time)
        self.work_dir = r'~'
        self.work_dir = os.path.abspath(os.path.expanduser(self.work_dir))
        self.work_dir_local = os.path.abspath(os.path.expanduser(self.work_dir))
       
        self.sig_ver = ''
        self.rir_ver = ''
        self.noise_flag = ''
        self.time_ver = '0821'
        
        # Noise setting: w/ noise or w/o noise
        self.wnoise = True
        if self.wnoise: 
            self.noise_flag = '-Noi15_30dB'
            snr_range = [15, 30]
            noise_type = ['diffuse_white'] 
        else:
            self.noise_flag = ''
            snr_range = [100, 100]
            noise_type = ['']
        self.noise_setting = {'noise_enable': self.wnoise, 'snr_range': snr_range, 'noise_type': noise_type,}
        
        # Array setting
        self.array_setting = {'nmic':  2}

        # Other acoustic setting
        self.acoustic_setting = {'speed': 343.0, 'fs': 16000}

        self.extra_info = '' 
        
        self.ds_token = ''
        self.ds_head = ''
        self.ds_embed = ''
        self.ds_nsimroom = 0

        # Downstream: simulated or real-world
        ##########################################
        downstream_sim = True
        ##########################################
        if downstream_sim:
            ds_task = ['TDOA', 'DRR', 'T60', 'C50', 'ABS']
            ds_data = 'sim'
            self.train_test_model = 'sim_' + self.time_ver 
            real_sim_ratio = [0, 1]
        else:
            ds_task = ['TDOA']
            # ds_task = ['DRR', 'T60', 'C50', 'ABS']
            if ('TDOA' in ds_task) & (len(ds_task)==1):
                ds_data = 'real_locata'
            else:
                ds_data = 'real_ace'
                # ds_data = 'real_dechorate'
                # ds_data = 'real_butreverb'
            #####################################
            # real_sim_ratio = [1, 0] # real
            real_sim_ratio = [1, 1] # real + sim
            # real_sim_ratio = [0, 1] # sim
            #####################################
            self.train_test_model = 'real_' + self.time_ver + '_train' + str(real_sim_ratio[0]) + 'real'+ str(real_sim_ratio[1]) + 'sim_valreal' 
            # self.train_test_model = 'real_' + self.time_ver + '_train0real1sim_valsim'  # only sim 

        self.ds_specifics = {
            'task': ds_task, 
            'data': ds_data,
            'real_sim_ratio': real_sim_ratio, }

    def parse(self):
        """ Function: Define optional arguments
        """
        parser = argparse.ArgumentParser(description='Self-supervised learing for multi-channel audio processing')

        # for training and test stages
        parser.add_argument('--gpu-id', type=str, default='7', metavar='GPU', help='GPU ID (default: 7)')
        parser.add_argument('--workers', type=int, default=8, metavar='Worker', help='number of workers (default: 8)')
        parser.add_argument('--bs', type=int, nargs='+', default=[128, 128, 128], metavar='TrainValTestBatch', help='batch size for training, validation and test (default: 128, 128, 128)')
        parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training (default: False)')
        parser.add_argument('--use-amp', action='store_true', default=False, help='Use automatic mixed precision training (default: False)')
        parser.add_argument('--seed', type=int, default=1, metavar='Seed', help='random seed (default: 1)')
        
        parser.add_argument('--checkpoint-start', action='store_true', default=False, help='train model from saved checkpoints (default: False)')
        parser.add_argument('--time', type=str, default=self.time, metavar='Time', help='time flag')
        parser.add_argument('--work-dir', type=str, default=self.work_dir, metavar='WorkDir', help='work directory')

        parser.add_argument('--sources', type=int, nargs='+', default=[1], metavar='Sources', help='number of sources (default: 1)')
        parser.add_argument('--source-state', type=str, default='static', metavar='SourceState', help='state of sources (default: Static)') # ['static', 'mobile']
        parser.add_argument('--rir-gen', type=str, default='wo', metavar='RIRGen', help='mode of RIR generation (default: Without)') # ['wo', 'online', 'offline']

        parser.add_argument('--ds-train', action='store_true', default=False, help='change to train stage of downstream tasks (default: False)')
        parser.add_argument('--ds-trainmode', type=str, default='finetune', metavar='DSTrainMode', help='how to train downstream models (default: finetune)') # ['scratchUP', 'scratchLOW', 'finetune', 'lineareval']
        parser.add_argument('--ds-task', type=str, nargs='+', default=self.ds_specifics['task'], metavar='DSTask', help='downstream task (default: TDOA, DRR, T60 estimation)')
        parser.add_argument('--ds-token', type=str, default='all', metavar='DSToken', help='downstream token (default: all)') # ['all', 'cls']
        parser.add_argument('--ds-head', type=str, default='mlp', metavar='DSHead', help='downstream head (default: mlp)') # ['mlp', 'crnn_in', 'crnn_med', 'crnn_out']
        parser.add_argument('--ds-embed', type=str, default='spat', metavar='DSEmbed', help='downstream embed (default: spat)') # ['spec_spat', 'spec', 'spat']
        parser.add_argument('--ds-nsimroom', type=int, default=0, metavar='DSSimRoom', help='number of simulated room used for downstream training (default: 0)') 

        parser.add_argument('--ds-test', action='store_true', default=False, help='change to test stage of downstream tasks (default: False)')

        args = parser.parse_args()
        assert (args.ds_train + args.ds_test)==1, 'Downstream stage (train or test) is not defined'
        assert args.ds_trainmode in ['sratchUP', 'scratchLOW', 'finetune', 'lineareval'], 'Downstream train mode in not defined'
        self.time = args.time
        self.work_dir = args.work_dir
        self.ds_token = args.ds_token
        self.ds_head = args.ds_head
        self.ds_embed = args.ds_embed
        self.ds_nsimroom = args.ds_nsimroom

        args.noise_setting = self.noise_setting
        args.array_setting = self.array_setting
        args.acoustic_setting = self.acoustic_setting
        args.ds_specifics = self.ds_specifics

        self.rir_ver = ''
        self.sig_ver = self.rir_ver + self.noise_flag
        data = self.ds_specifics['data']
        print('\ntime='+self.time, 'data version='+self.sig_ver, 'noise='+str(self.wnoise), 'data='+data, 'task='+str(args.ds_task), 'ds-embed='+self.ds_embed)

         # add argument
        if (args.ds_trainmode == 'scratchUP'):
            bs_set = [32] 
            nepoch = 200
            num = 5120*100
            ntrial = 1 
            args.ds_setting = {}
            args.ds_setting['TDOA'] = {'nepoch': nepoch, 'num': num, 'lr_set': [0.0001], 'bs_set': bs_set, 'ntrial': ntrial}
            args.ds_setting['DRR'] = {'nepoch': nepoch, 'num': num, 'lr_set': [0.001], 'bs_set': bs_set, 'ntrial': ntrial}
            args.ds_setting['C50'] = {'nepoch': nepoch, 'num': num, 'lr_set': [0.001], 'bs_set': bs_set, 'ntrial': ntrial}
            args.ds_setting['T60'] = {'nepoch': nepoch, 'num': num, 'lr_set': [0.001], 'bs_set': bs_set, 'ntrial': ntrial}
            args.ds_setting['ABS'] = {'nepoch': nepoch, 'num': num, 'lr_set': [0.001], 'bs_set': bs_set, 'ntrial': ntrial}
            args.ds_setting['SUR'] = {'nepoch': nepoch, 'num': num, 'lr_set': [0.001], 'bs_set': bs_set, 'ntrial': ntrial}
            args.ds_setting['VOL'] = {'nepoch': nepoch, 'num': num, 'lr_set': [0.001], 'bs_set': bs_set, 'ntrial': ntrial}
            args.ds_setting['SNR'] = {'nepoch': nepoch, 'num': num, 'lr_set': [0.0001, 0.001], 'bs_set': bs_set, 'ntrial': ntrial}
            args.ds_setting['TDOA-DRR-T60-C50-ABS'] = {'nepoch': nepoch, 'num': num, 'lr_set': [0.0001], 'bs_set': bs_set, 'ntrial': ntrial}
        else:
            ### use pretrained model for downstream tasks
            ## Simulate data
            if 'sim' in data:
                bs_set = [8] # simulated
                lr_set = [0.001, 0.0005, 0.0001, 0.00005] # simulated
                nepoch = 200
                num = args.ds_nsimroom * 100
                ntrial = np.maximum(1, round(32/(args.ds_nsimroom+10e-4)))
            else:
                ## Real-world data
                bs_set = [16] # real-world
                lr_set = [0.001, 0.0001] # real-world 
                nepoch = 200

                # set to infinite training epochs for plotting training curves
                # lr_set = [0.0001] # for TDOA estimation on real-world data 
                # nepoch = 60 # TDOA 
                # lr_set = [0.001,0.0001]
                # nepoch = 50 # T60
                # nepoch = 300 # DRR

            args.ds_setting = {}
            args.ds_setting['TDOA'] = {'nepoch': nepoch, 'num': num, 'lr_set': lr_set, 'bs_set': bs_set, 'ntrial': ntrial}
            args.ds_setting['DRR'] = {'nepoch': nepoch, 'num': num, 'lr_set': lr_set, 'bs_set': bs_set, 'ntrial': ntrial}
            args.ds_setting['C50'] = {'nepoch': nepoch, 'num': num, 'lr_set': lr_set, 'bs_set': bs_set, 'ntrial': ntrial}
            args.ds_setting['T60'] = {'nepoch': nepoch, 'num': num, 'lr_set': lr_set, 'bs_set': bs_set, 'ntrial': ntrial}
            args.ds_setting['ABS'] = {'nepoch': nepoch, 'num': num, 'lr_set': lr_set, 'bs_set': bs_set, 'ntrial': ntrial}
            args.ds_setting['SUR'] = {'nepoch': nepoch, 'num': num, 'lr_set': lr_set, 'bs_set': bs_set, 'ntrial': ntrial}
            args.ds_setting['VOL'] = {'nepoch': nepoch, 'num': num, 'lr_set': lr_set, 'bs_set': bs_set, 'ntrial': ntrial}
            args.ds_setting['SNR'] = {'nepoch': nepoch, 'num': num, 'lr_set': lr_set, 'bs_set': bs_set, 'ntrial': ntrial}
            args.ds_setting['DOA'] = {'nepoch': nepoch, 'num': num, 'lr_set': lr_set, 'bs_set': bs_set, 'ntrial': ntrial}
            if 'sim' in self.ds_specifics['data']:
                self.extra_info = 'R'+str(args.ds_nsimroom)

        return args

    def dir(self):
        """ Function: Get directories of code, data and experimental results
        """ 
        work_dir = self.work_dir
        dirs = {}

        dirs['code'] = work_dir + '/Re-SSL/code'
        # dirs['data'] = work_dir + '/data'
        # dirs['gerdata'] = work_dir + '/Re-SSL/data'
        dirs['data'] = self.work_dir_local + '/data'
        dirs['gerdata'] = self.work_dir_local + '/Re-SSL/data'
        dirs['exp'] = work_dir + '/Re-SSL/exp'

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

        # RIR (+ noise) & sensor signal
        dirs['simulate'] = dirs['gerdata'] + '/RIR-simulate' 
        dirs['DCASE'] = dirs['data'] + '/RIR/DCASE/TAU-SRIR_DB'
        dirs['MIR'] = dirs['data'] + '/RIR/MIRDB/Impulse_response_Acoustic_Lab_Bar-Ilan_University'
        dirs['Mesh'] = dirs['data'] + '/RIR/Mesh'
        dirs['BUTReverb'] = dirs['data'] + '/RIR/BUT_ReverbDB/RIRs'
        dirs['IRArni'] = dirs['data'] + '/RIR/IR_Arni/IR'
        dirs['dEchorate'] = dirs['data'] + '/RIR/dEchorate'
        dirs['ACE'] = dirs['data'] + '/RIR/ACE'
        dirs['LOCATA'] = dirs['data'] + '/SenSig/LOCATA'

        ## RIR / generated mic signal for downstream tasks
        dirs_train = dirs['gerdata'] + '/SenSig-train' + self.sig_ver
        dirs_val = dirs['gerdata'] + '/SenSig-val' + self.sig_ver
        dirs_test = dirs['gerdata'] + '/SenSig-test' + self.sig_ver
        dirs['sensig_train'] = []
        dirs['sensig_val'] = []
        dirs['sensig_test'] = []
        dirs['sensig_test_large'] = []

        if ('sim' in self.ds_specifics['data']):
            dirs_rir = dirs['gerdata'] + '/RIR' + self.rir_ver 
            dirs['rir'] = [
                dirs_rir + '/simulate' + self.time_ver,
            ]

            # ScrtchUP
            dirs_pretrain = dirs['gerdata'] + '/SenSig-pretrain' + self.sig_ver
            dirs_preval = dirs['gerdata'] + '/SenSig-preval' + self.sig_ver
            dirs_pretest = dirs['gerdata'] + '/SenSig-test' + self.sig_ver
            dirs['sensig_pretrain'] = [ dirs_pretrain + '/simulate' + self.time_ver ]
            dirs['sensig_preval'] = [ dirs_preval + '/simulate' + self.time_ver ] 
            dirs['sensig_pretest'] = [ dirs_pretest + '/simulate' + self.time_ver ]

            dirs['sensig_train'] = [dirs_train + '/simulate' + self.time_ver + 'R' + str(self.ds_nsimroom), ]
            dirs['sensig_val'] = [ dirs_val + '/simulate' + self.time_ver + 'R20', ]
            dirs['sensig_test'] = [ dirs_test + '/simulate' + self.time_ver +'R20', ]

        if ('real' in self.ds_specifics['data']):

            if ('locata' in self.ds_specifics['data']): 
                dirs['sensig_train'] += [ dirs_train.replace(self.noise_flag, '') + '/LOCATA',
                                          dirs_train + '/simulate' + self.time_ver + 'R1000', ] # add sim
                dirs['sensig_val'] += [ dirs_val.replace(self.noise_flag, '') + '/LOCATA', 
                                        dirs_val + '/simulate' + self.time_ver + 'R20', ] # add sim        
                dirs['sensig_test'] += [ dirs_test.replace(self.noise_flag, '') + '/LOCATA',
                                        dirs_test + '/simulate' + self.time_ver + 'R20', ]
                dirs['rir'] = ['']

            if 'ace' in self.ds_specifics['data']:
                dirs_rir = dirs['gerdata'] + '/RIR' + self.rir_ver 
                dirs['rir'] = [ dirs_rir + '/ACE',
                                dirs_rir + '/simulate' + self.time_ver, ] # add sim

        data_model_flag = self.train_test_model
 
        # Experimental data
        dirs['log_pretrain'] = dirs['exp'] + '/pretrain/' + self.time
        
        dirs['log_task'] = dirs['exp'] + '/' + 'TASK' + '/' + self.time

        dirs['log_task_scratchUP'] = dirs['log_task'] + '/scratchup-' + self.ds_token + '-' + self.ds_head + '-' + 'NUM' + '-' + 'LR-BAS-TRI' + '-' + self.ds_embed + '-' + data_model_flag + self.extra_info
        dirs['log_task_scratchLOW'] = dirs['log_task'] + '/scratchlow-' + self.ds_token + '-' + self.ds_head + '-' + 'NUM' + '-' + 'LR-BAS-TRI' + '-' + self.ds_embed + '-' + data_model_flag + self.extra_info

        dirs['log_task_finetune'] = dirs['log_task'] + '/finetune-' + self.ds_token + '-' + self.ds_head + '-' + 'NUM' + '-' + 'LR-BAS-TRI' + '-' + self.ds_embed + '-' + data_model_flag  + self.extra_info
        dirs['log_task_lineareval'] = dirs['log_task'] + '/lineareval-' + self.ds_token + '-' + self.ds_head + '-' + 'NUM' + '-' + 'LR-BAS-TRI' + '-' + self.ds_embed + '-' + data_model_flag + self.extra_info
        
        return dirs

if __name__ == '__main__':

    args = opt_pretrain().parse()
    dirs = opt_pretrain().dir()
    print('gpu-id: ' + str(args.gpu_id))
    print('code path:' + dirs['code']) 
