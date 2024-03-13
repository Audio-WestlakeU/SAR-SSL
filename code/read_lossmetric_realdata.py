""" 
    Read downstream results (MAEs of TDOA, DRR, T60, C50, SNR, ABS, SUR, VOL estimation) from log files for real-world data
"""

import os
import argparse
import numpy as np
import scipy.io
from opt import opt_pretrain
from read_lossmetric_simdata import read_result

parser = argparse.ArgumentParser(description='Reading downstream results')

opts = opt_pretrain()
dirs = opts.dir()

def get_specifed_result_realdata(task, ver_specify, ds_ver_name_pattern):

    # Get specified downstream results 
    ver_time_dir = dirs['exp'] + '/'+ task + '/'
    result = read_result(ver_specify, ver_time_dir)
    if 'loss' not in result['val'].keys():
        print('lack loss recording in validation stage!')
    
    atts_pattern = ds_ver_name_pattern.split('-')
    atts = ver_specify.split('-')
    time = atts[atts_pattern.index('TIME')]
    for i in range(len(atts_pattern)):
        if i < len(atts):
            globals()[atts_pattern[i]] = atts[i]
        else:
            globals()[atts_pattern[i]] = ''
    ver_name = ds_ver_name_pattern.replace('FT', FT).replace('TIME-', '').replace('TOKEN', TOKEN).replace('HEAD', HEAD).replace('NUM', NUM).replace('BAS', BAS).replace('LR', LR).replace('CV', CV).replace('EMBED', EMBED)

    best_epoch_idx = np.argmin(result['val']['metric'])
    test_loss = result['test']['loss'][best_epoch_idx]
    test_metric = result['test']['metric'][best_epoch_idx]
    best_epoch = best_epoch_idx+1
    print('downstream: ', time, task, ver_name, best_epoch, round(test_loss,2), round(test_metric,2))
    # scipy.io.savemat(ver_time_dir+time+'/test_result/loss_metric_' + ver_specify + '.mat', result)

    return result

def avg_loss_metric_over_cvs(result_list):

    epoch_idx = 0
    nepoch = 500
    nepoch_max = 0
    for cv_idx in range(ncv):
        nepoch_this_cv = len(result_list[cv_idx]['train']['loss'])
        nepoch = np.minimum(nepoch, nepoch_this_cv)
        nepoch_max = np.maximum(nepoch_max, nepoch_this_cv)

    # print(nepoch)
    train_loss = []
    train_metric = []
    test_loss = []
    test_metric = []
    for epoch_idx in range(nepoch):
        train_loss_cv = []
        train_metric_cv = []
        test_loss_cv = []
        test_metric_cv = []
        for cv_idx in range(ncv):
            train_loss_cv += [result_list[cv_idx]['train']['loss'][epoch_idx]]
            train_metric_cv += [result_list[cv_idx]['train']['metric'][epoch_idx]]
            test_loss_cv += [result_list[cv_idx]['test']['loss'][epoch_idx]]
            test_metric_cv += [result_list[cv_idx]['test']['metric'][epoch_idx]]
        train_loss += [np.mean(np.array(train_loss_cv))]
        train_metric += [np.mean(np.array(train_metric_cv))]
        test_loss += [np.mean(np.array(test_loss_cv))]
        test_metric += [np.mean(np.array(test_metric_cv))]

    for epoch_idx in range(nepoch, nepoch_max):
        train_loss_cv = []
        train_metric_cv = []
        test_loss_cv = []
        test_metric_cv = []
        for cv_idx in range(ncv):
            if epoch_idx>=len(result_list[cv_idx]['train']['loss']):
                train_loss_cv += [result_list[cv_idx]['train']['loss'][-1]]
                train_metric_cv += [result_list[cv_idx]['train']['metric'][-1]]
                test_loss_cv += [result_list[cv_idx]['test']['loss'][-1]]
                test_metric_cv += [result_list[cv_idx]['test']['metric'][-1]]
            else:
                train_loss_cv += [result_list[cv_idx]['train']['loss'][epoch_idx]]
                train_metric_cv += [result_list[cv_idx]['train']['metric'][epoch_idx]]
                test_loss_cv += [result_list[cv_idx]['test']['loss'][epoch_idx]]
                test_metric_cv += [result_list[cv_idx]['test']['metric'][epoch_idx]]
        train_loss += [np.mean(np.array(train_loss_cv))]
        train_metric += [np.mean(np.array(train_metric_cv))]
        test_loss += [np.mean(np.array(test_loss_cv))]
        test_metric += [np.mean(np.array(test_metric_cv))]

    return train_loss, train_metric, test_loss, test_metric

if __name__ == '__main__':

    ## Real
    # task = 'TDOA'
    # time = '10192210new'
    # real_sim_flag = 'train1real0sim_valreal'
    # finetune_datanum_flag = '80000'
    # scratch_datanum_flag = '80000'
    # finetune_lr_list = [0.0001]
    # scratch_lr_list = [0.0001]
    # ncv = 1

    # task = 'DRR'
    # time = '10192210new'
    # real_sim_flag = 'train1real0sim_valreal'
    # finetune_datanum_flag = '1600'
    # scratch_datanum_flag = '1600'
    # finetune_lr_list = [0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001]
    # scratch_lr_list = [0.001,0.0001,0.001,0.0001,0.001,0.001,0.001]
    # ncv = 7

    # task = 'T60'
    # time = '10192210new'
    # real_sim_flag = 'train1real0sim_valreal'
    # finetune_datanum_flag = '1600'
    # scratch_datanum_flag = '1600'
    # finetune_lr_list = [0.0001,0.0001,0.001,0.0001,0.001,0.0001,0.0001]
    # scratch_lr_list = [0.0001,0.001,0.0001,0.001,0.001,0.001,0.001]
    # ncv = 7

    # task = 'C50'
    # time = '10192210new'
    # real_sim_flag = 'train1real0sim_valreal'
    # finetune_datanum_flag = '1600'
    # scratch_datanum_flag = '1600'
    # finetune_lr_list = [0.0001,0.001,0.0001,0.0001,0.001,0.0001,0.0001]
    # scratch_lr_list = [0.0001,0.001,0.001,0.001,0.0001,0.0001,0.001]
    # ncv = 7

    # task = 'ABS'
    # time = '10192210new'
    # real_sim_flag = 'train1real0sim_valreal'
    # finetune_datanum_flag = '1600'
    # scratch_datanum_flag = '1600'
    # finetune_lr_list = [0.0001,0.001,0.001,0.0001,0.001,0.0001,0.0001]
    # scratch_lr_list = [0.0001,0.001,0.001,0.001,0.001,0.0001,0.0001]
    # ncv = 7


    # Sim
    # task = 'TDOA'
    # time = '10192210new'
    # real_sim_flag = 'train0real1sim_valreal'
    # finetune_datanum_flag = '80000'
    # scratch_datanum_flag = '80000'
    # finetune_lr_list = [0.0001]
    # scratch_lr_list = [0.0001]
    # ncv = 1

    # task = 'DRR'
    # time = '10192210new'
    # real_sim_flag = 'train0real1sim_valreal'
    # finetune_datanum_flag = '32000'
    # scratch_datanum_flag = '32000'
    # finetune_lr_list = [0.001,0.001,0.001,0.001,0.001,0.001,0.001]
    # scratch_lr_list = [0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001]
    # ncv = 7

    # task = 'T60'
    # time = '10192210new'
    # real_sim_flag = 'train0real1sim_valreal'
    # finetune_datanum_flag = '32000'
    # scratch_datanum_flag = '32000'
    # finetune_lr_list = [0.0001,0.0001,0.001,0.0001,0.001,0.0001,0.0001]
    # scratch_lr_list = [0.001,0.001,0.0001,0.0001,0.0001,0.0001,0.0001]
    # ncv = 7

    # Sim+Real
    # task = 'TDOA'
    # time = '10192210new'
    # real_sim_flag = 'train5real5sim_valreal'
    # finetune_datanum_flag = '80000'
    # scratch_datanum_flag = '80000'
    # finetune_lr_list = [0.0001]
    # scratch_lr_list = [0.0001]
    # ncv = 1

    # task = 'DRR'
    # time = '10192210new'
    # real_sim_flag = 'train5real5sim_valreal'
    # finetune_datanum_flag = '3200'
    # scratch_datanum_flag = '16000'
    # finetune_lr_list = [0.0001,0.001,0.001,0.0001,0.001,0.0001,0.0001]
    # scratch_lr_list = [0.001,0.001,0.001,0.001,0.001,0.001,0.001]
    # ncv = 7

    # task = 'T60'
    # time = '10192210new'
    # real_sim_flag = 'train5real5sim_valreal'
    # finetune_datanum_flag = '3200'
    # scratch_datanum_flag = '16000'
    # finetune_lr_list =  [0.001,0.0001,0.001,0.001,0.001,0.0001,0.001]
    # scratch_lr_list = [0.0001,0.001,0.0001,0.0001,0.001,0.001,0.001]
    # ncv = 7

    # task = 'C50'
    # time = '10192210new'
    # real_sim_flag = 'train5real5sim_valreal'
    # finetune_datanum_flag = '3200'
    # scratch_datanum_flag = '16000'
    # finetune_lr_list =  [0.0001,0.0001,0.0001,0.0001,0.001,0.001,0.0001]
    # scratch_lr_list = [0.0001,0.0001,0.0001,0.0001,0.0001,0.001,0.001]
    # ncv = 7

    # task = 'ABS'
    # time = '10192210new'
    # real_sim_flag = 'train5real5sim_valreal'
    # finetune_datanum_flag = '3200'
    # scratch_datanum_flag = '16000'
    # finetune_lr_list =  [0.0001,0.0001,0.0001,0.0001,0.0001,0.001,0.0001]
    # scratch_lr_list = [0.0001,0.0001,0.0001,0.0001,0.001,0.0001,0.001]
    # ncv = 7

    # task = 'DRR'
    # time = '10192210new'
    # real_sim_flag = 'train5real5sim_valreal'
    # finetune_datanum_flag = '16000'
    # scratch_datanum_flag = '16000'
    # finetune_lr_list = [0.0001,0.001,0.0001,0.0001,0.0001,0.0001,0.0001]
    # scratch_lr_list = [0.001,0.001,0.001,0.001,0.001,0.001,0.001]
    # ncv = 7

    # task = 'T60'
    # time = '10192210new'
    # real_sim_flag = 'train5real5sim_valreal'
    # finetune_datanum_flag = '16000'
    # scratch_datanum_flag = '16000'
    # finetune_lr_list = [0.0001,0.0001,0.001,0.0001,0.001,0.0001,0.0001]
    # scratch_lr_list = [0.0001,0.001,0.0001,0.0001,0.001,0.001,0.001]
    # ncv = 7

    # fix number of training iterations
    # Sim
    task = 'TDOA'
    time = '10192210inf'
    real_sim_flag = 'train0real1sim_valreal'
    finetune_datanum_flag = '80000'
    scratch_datanum_flag = '80000'
    finetune_lr_list = [0.0001]
    scratch_lr_list = [0.0001]
    ncv = 1

    task = 'T60'
    time = '10192210inf'
    real_sim_flag = 'train0real1sim_valreal'
    finetune_datanum_flag = '3200'
    scratch_datanum_flag = '3200'
    # finetune_lr_list = [0.001,0.001,0.001,0.001,0.001,0.001,0.001]
    # scratch_lr_list = [0.001,0.001,0.001,0.001,0.001,0.001,0.001]
    finetune_lr_list = [0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001]
    scratch_lr_list = [0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001]
    ncv = 7

    task = 'DRR'
    time = '10192210inf'
    real_sim_flag = 'train0real1sim_valreal'
    finetune_datanum_flag = '3200'
    scratch_datanum_flag = '3200'
    finetune_lr_list = [0.001,0.001,0.001,0.001,0.001,0.001,0.001]
    # scratch_lr_list = [0.001,0.001,0.001,0.001,0.001,0.001,0.001]
    # finetune_lr_list = [0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001]
    scratch_lr_list = [0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001]
    ncv = 7

    # Real+Sim
    # task = 'TDOA'
    # time = '10192210inf'
    # real_sim_flag = 'train5real5sim_valreal'
    # finetune_datanum_flag = '80000'
    # scratch_datanum_flag = '80000' 
    # finetune_lr_list = [0.0001]
    # scratch_lr_list = [0.0001]
    # ncv = 1

    # task = 'T60'
    # time = '10192210inf'
    # real_sim_flag = 'train5real5sim_valreal'
    # finetune_datanum_flag = '3200'
    # scratch_datanum_flag = '3200'
    # # finetune_lr_list = [0.001,0.001,0.001,0.001,0.001,0.001,0.001]
    # # scratch_lr_list = [0.001,0.001,0.001,0.001,0.001,0.001,0.001]
    # finetune_lr_list = [0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001]
    # scratch_lr_list = [0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001]
    # ncv = 7

    # task = 'DRR'
    # time = '10192210inf'
    # real_sim_flag = 'train5real5sim_valreal'
    # finetune_datanum_flag = '3200'
    # scratch_datanum_flag = '3200'
    # # finetune_lr_list = [0.001,0.001,0.001,0.001,0.001,0.001,0.001]
    # scratch_lr_list = [0.001,0.001,0.001,0.001,0.001,0.001,0.001]
    # finetune_lr_list = [0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001]
    # # scratch_lr_list = [0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001]
    # ncv = 7

    # Real
    # task = 'TDOA'
    # time = '10192210inf'
    # real_sim_flag = 'train1real0sim_valreal'
    # finetune_datanum_flag = '80000'
    # scratch_datanum_flag = '80000'
    # finetune_lr_list = [0.0001]
    # scratch_lr_list = [0.0001]
    # ncv = 1

    # task = 'T60'
    # time = '10192210inf'
    # real_sim_flag = 'train1real0sim_valreal'
    # finetune_datanum_flag = '3200'
    # scratch_datanum_flag = '3200'
    # # finetune_lr_list = [0.001,0.001,0.001,0.001,0.001,0.001,0.001]
    # # scratch_lr_list = [0.001,0.001,0.001,0.001,0.001,0.001,0.001]
    # finetune_lr_list = [0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001]
    # scratch_lr_list = [0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001]
    # ncv = 7

    # task = 'DRR'
    # time = '10192210inf'
    # real_sim_flag = 'train1real0sim_valreal'
    # finetune_datanum_flag = '3200'
    # scratch_datanum_flag = '3200'
    # # finetune_lr_list = [0.001,0.001,0.001,0.001,0.001,0.001,0.001]
    # scratch_lr_list = [0.001,0.001,0.001,0.001,0.001,0.001,0.001]
    # finetune_lr_list = [0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001]
    # # scratch_lr_list = [0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001]
    # ncv = 7
    
    ver_name_pattern_ori = 'TIME-FT-TOKEN-HEAD-NUM-LR-BAS-CV-EMBED' # according to the saved directory+file
    task_dir = dirs['exp'] + '/'+ task + '/'
    save_dir = task_dir+time+'/test_result/'
    exist_temp = os.path.exists(save_dir)
    if exist_temp==False:
        os.makedirs(save_dir)
        print('make dir: ' + save_dir)

    ft_list = []
    sc_list = []
    for cv_idx in range(ncv):
        ft_lr = str(finetune_lr_list[cv_idx])
        ft_result = get_specifed_result_realdata(task=task, ver_specify= time + '-finetune-all-mlp-' + finetune_datanum_flag + '-' + ft_lr + '-16-' + str(cv_idx) + '-spat-real_0821_' + real_sim_flag, ds_ver_name_pattern=ver_name_pattern_ori)
        ft_list += [ft_result]
        
        sc_lr = str(scratch_lr_list[cv_idx])
        sc_result = get_specifed_result_realdata(task=task, ver_specify= time + '-scratchlow-all-mlp-' + scratch_datanum_flag + '-' + sc_lr + '-16-' + str(cv_idx) + '-spat-real_0821_' + real_sim_flag, ds_ver_name_pattern=ver_name_pattern_ori)
        sc_list += [sc_result]
    
    train_loss, train_metric, test_loss, test_metric = avg_loss_metric_over_cvs(result_list=ft_list)
    scipy.io.savemat(save_dir + 'loss_metric_' + real_sim_flag + '_' + finetune_datanum_flag + '_finetune' + '.mat', {'train_loss': train_loss, 'train_metric':train_metric,'test_loss': test_loss, 'test_metric':test_metric})
    train_loss, train_metric, test_loss, test_metric = avg_loss_metric_over_cvs(result_list=sc_list)
    scipy.io.savemat(save_dir + 'loss_metric_' + real_sim_flag + '_' + scratch_datanum_flag + '_scratch' + '.mat', {'train_loss': train_loss, 'train_metric':train_metric,'test_loss': test_loss, 'test_metric':test_metric})