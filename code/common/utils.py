""" 
    Function:   Define some basic operations
"""

import os
import scipy.signal
import numpy as np
import torch
import random
import pickle
import soundfile 
import matplotlib.pyplot as plt
import copy
from torch.utils.flop_counter import FlopCounterMode
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA


def detect_infnan(data, mode='torch', info=''):
    """ Check whether there is inf/nan in the element of data or not
    """ 
    if mode == 'torch':
        inf_flag = torch.isinf(data)
        nan_flag = torch.isnan(data)
    elif mode == 'np':
        inf_flag = np.isinf(data)
        nan_flag = np.isnan(data)
    else:
        raise Exception('Detect infnan mode unrecognized')
    if (True in inf_flag):
        raise Exception('INF exists in data')
    if (True in nan_flag):
        print(info)
        raise Exception('NAN exists in data')


def set_seed(seed):
    """ Fix random seed
    """ 
    np.random.seed(seed) # python
    random.seed(seed)   # random
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed_all(seed) # multi-GPU
    torch.cuda.manual_seed(seed)    # current GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)


def get_nparams(model, param_key_list=[]):
    """ Get the number of parameters of specified key 
    """ 
    nparam_sum = 0.0
    nparam = {}
    for param_key_specified in param_key_list:
        nparam[param_key_specified] = 0
    for key, value in model.named_parameters():
        nparam_sum += value.numel()/1000000
        for param_key_specified in param_key_list:
            if param_key_specified in key:
                nparam[param_key_specified] += value.numel()/1000000

    return nparam, nparam_sum


def get_flops(model, input_shape):
    """ Get FLOPS (G/s)
    """ 
    model.cuda()
    x = torch.randn((input_shape)).cuda()
    with FlopCounterMode(model, display=False) as fcm:
        y = model(x)
        flops_forward_eval = fcm.get_total_flops()/4e9
    for k, v in fcm.get_flop_counts().items():
        ss = f"{k}: {{"
        for kk, vv in v.items():
            ss += f" {str(kk)}:{vv}"
        ss += " }"

    return flops_forward_eval, ss


def get_learning_rate(optimizer):
    """ Get learning rates from optimizer
    """ 
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def set_learning_rate(epoch, lr_init, step, gamma):
    """ Set exponentially-decay learning rate
    """ 
    lr = lr_init*pow(gamma, epoch/step)
    return lr


def create_learning_rate_schedule(total_steps, base, decay_type, warmup_steps, linear_end=1e-5):
    """ Creates learning rate schedule
        Currently only warmup + {linear,cosine} but will be a proper mini-language
        like preprocessing one in the future
    Args:   total_steps: The total number of steps to run
            base: The starting learning-rate (without warmup)
            decay_type: 'linear' or 'cosine'
            warmup_steps: how many steps to warm up for
            linear_end: Minimum learning rate
    Return: A function learning_rate(step): float
    Refs: https://github.com/Audio-WestlakeU/audiossl/blob/main/audiossl/modules/lr_scheduler.py
    """

    def step_fn(step):

        lr = base

        progress = (step - warmup_steps) / float(total_steps - warmup_steps)
        progress = np.clip(progress, 0.0, 1.0)
        if decay_type == 'linear':
            lr = linear_end + (lr - linear_end) * (1.0 - progress)
        elif decay_type == 'cosine':
            lr = lr * 0.5 * (1. + np.cos(np.pi * progress))
        else:
            raise ValueError(f'Unknown lr type {decay_type}')

        if warmup_steps:
            lr = lr * np.minimum(1., step / warmup_steps)

        return np.asarray(lr, dtype=np.float32)

    return step_fn

def forgetting_norm(input, num_frame_set=None):
    """
        Function: Using the mean value of the near frames to normalization
        Args:
            input: feature [B, C, F, T]
            num_frame_set: length of the training time frames, used for calculating smooth factor
        Returns:
            normed feature
        Ref: Online Monaural Speech Enhancement using Delayed Subband LSTM, INTERSPEECH, 2020
    """
    assert input.ndim == 4
    batch_size, num_channels, num_freqs, num_frames = input.size()
    input = input.reshape(batch_size, num_channels * num_freqs, num_frames)

    if num_frame_set == None:
        num_frame_set = copy.deepcopy(num_frames)

    mu = 0
    mu_list = []
    for frame_idx in range(num_frames):
        if num_frames<=num_frame_set:
            alpha = (frame_idx - 1) / (frame_idx + 1)
        else:
            alpha = (num_frame_set - 1) / (num_frame_set + 1)
        current_frame_mu = torch.mean(input[:, :, frame_idx], dim=1).reshape(batch_size, 1) # [B, 1]
        mu = alpha * mu + (1 - alpha) * current_frame_mu
        mu_list.append(mu)
    mu = torch.stack(mu_list, dim=-1) # [B, 1, T]
    output = mu.reshape(batch_size, 1, 1, num_frames)

    return output

def save_file(mic_signal, acoustic_scene, sig_path, acous_path):
    """ Save audio and annotation files
    """
    if (mic_signal is not None) & (sig_path is not None):
        soundfile.write(sig_path, mic_signal, acoustic_scene.fs)

    if (acoustic_scene is not None) & (acous_path is not None):
        file = open(acous_path,'wb')
        file.write(pickle.dumps(acoustic_scene.__dict__))
        file.close()


def load_file(acoustic_scene, sig_path, acous_path, sig_tar_fs=16000):
    """ Load audio and annotation files
        Note: Must import ArraySetup in main.py if arraysetup attribute is used
    """

    if sig_path is not None:
        mic_signal, fs = soundfile.read(sig_path)
        if (fs != sig_tar_fs):
            print('Loaded signals are resampled~')
            mic_signal = scipy.signal.resample_poly(mic_signal, sig_tar_fs, fs)

    if acous_path is not None:
        file = open(acous_path,'rb')
        dataPickle = file.read()
        file.close()
        acoustic_scene.__dict__ = pickle.loads(dataPickle)

    if (sig_path is not None) & (acous_path is not None):
        return mic_signal, acoustic_scene
    elif (sig_path is not None) & (acous_path is None):
        return mic_signal
    elif (sig_path is None) & (acous_path is not None):
        return acoustic_scene

    ## When reading mat file, the array_setup cannot present normally
    # data = scipy.io.loadmat(load_dir+'/'+name+'.mat')
	# mic_signals = data['mic_signals']
	# acoustic_scene0 =data['acoustic_scene'][0,0]
	# keys = acoustic_scene0.dtype.names
    # for idx in range(len(keys)):
	# 	key = keys[idx]
	# 	value = acoustic_scene0[key]
	# 	sh = value.shape
	# 	if len(sh)==2:
	# 		if (sh[0]==1) & (sh[1]==1):
	# 			value = value[0,0]
	# 		elif (sh[0]==1) & (sh[1]>1):
	# 			value = value[0,:]
	# 	print(key ,value)
	# 	acoustic_scene.__dict__[key] = value

def cross_validation_datadir(data_dir):
    """ Divid data into train, validation and test sets without overlap (perform cross validation), vaolidation and test set only with one room each
        Args:   data_dir
        Return: dirs -  list of cross validations, 
                        each element in list is a dataset dictionary, 
                        each dictionary value is a list of room dirs
    """
    room_names = os.listdir(data_dir)
    dirs = []
    for test_room_name in room_names:
        train_dir = []
        val_dir = []
        test_dir = []

        test_dir += [data_dir + '/' + test_room_name]

        trainval_room_names = copy.deepcopy(room_names)
        trainval_room_names.remove(test_room_name)
        val_room_name = random.sample(trainval_room_names, 1)[0]
        val_dir += [data_dir + '/' + val_room_name]

        train_room_names = copy.deepcopy(trainval_room_names)
        train_room_names.remove(val_room_name)
        for train_room_name in train_room_names:
            train_dir += [data_dir + '/' + train_room_name]

        dirs += [{'train': train_dir, 'val': val_dir, 'test': test_dir}]

    return dirs

def one_validation_datadir_simdata(data_dir, train_room_idx=[20,120], val_room_idx=[10,20], test_room_idx=[0,10]):
    """ Divide data into train, validation and test sets without overlap
        Args:   data_dir
        Return: dirs -  dataset dictionary, 
                        each element in list is a dataset dictionary, 
                        each dictionary value is a list of room dirs
    """
    room_names = os.listdir(data_dir)

    train_dir = []
    val_dir = []
    test_dir = []
    for room_name in room_names:
        if '.' not in room_name:
            room_idx = int(room_name.split('Room')[-1])
            if room_idx in range(train_room_idx[0], train_room_idx[1]):
                train_dir += [data_dir + '/' + room_name]

            if room_idx in range(val_room_idx[0], val_room_idx[1]):
                val_dir += [data_dir + '/' + room_name]

            if room_idx in range(test_room_idx[0], test_room_idx[1]):
                test_dir += [data_dir + '/' + room_name]
    dirs = {'train': train_dir, 'val': val_dir, 'test': test_dir}

    return dirs

def vis_TSNE(data, label):
    """ Visualize by TSNE
        Args:   data - (nins, dim)
                label - (nins, )
    """
    plt.switch_backend('agg')
    data_vis = TSNE(n_components=2, learning_rate=100).fit_transform(data)
    # data_vis = TSNE(n_components=2, learning_rate=100, init='pca', random_state=0).fit_transform(data)
    plt.figure(figsize=(4, 3.2))
    p = plt.scatter(data_vis[:, 0], data_vis[:, 1], c=label, s=15, marker='o', cmap='plasma') # 'rainbow' 'plasma'
    plt.colorbar(p)
    return plt, {'data':data_vis,'label':label}

def vis_time_fre_data(data, ins_idx, eps=10e-5):
    """ Visualize in time-frequency domain
        Args:   data - dict (nins, nf, nt, nmic) / (nins, nf, nt, nreim, nmic)
                ins_idx - the index of visualized instance
    """
    plt.switch_backend('agg')
    cmap = 'jet'
    nkey = 0
    keys = data.keys()
    for key in keys:
        nkey += 1
    idx = 0
    for key in keys:
        show_data = data[key][ins_idx, ...].cpu() # (nf, nt, nmic) / (nf, nt, nreim, nmic)
        shape = show_data.shape
        nf = shape[0]
        nt = shape[1]
        ncol = 4
        nreim = 2
        nch = show_data.shape[2]
        if len(shape)==3:
            for ch_idx in range(nch):
                plt.subplot(nkey, ncol, ncol*idx+nreim*ch_idx+1)
                plt.axis('on')
                plt.imshow(show_data[:,:,ch_idx], origin='lower', cmap='binary', interpolation='none', vmax=1, vmin=0, extent=(0, nt, 0, nf))
                plt.colorbar(shrink=1)
                plt.xlabel('Time frame')
                plt.ylabel('Frquency bin')
                
        elif len(shape)==4:
            show_real = show_data[:,:,0,:]
            show_imag = show_data[:,:,1,:]
            show_real_log = np.log(np.abs(show_real[:,:,ch_idx])+eps)
            show_imag_log = np.log(np.abs(show_imag[:,:,ch_idx])+eps)
            show_data_complex = show_real + 1j*show_imag
            show_mag = np.log(np.sqrt(show_real**2 + show_imag** 2)+eps)
            show_phase = np.angle(show_data_complex)
            for ch_idx in range(nch):
                plt.subplot(nkey, ncol, ncol*idx+nreim*ch_idx+1)
                plt.axis('on')
                plt.imshow(show_mag[:,:,ch_idx], origin='lower', cmap=cmap, vmax=5, vmin=-10, extent=(0, nt, 0, nf))
                plt.colorbar(shrink=1)
                plt.title('Magnitude')
                plt.xlabel('Time frame')
                plt.ylabel('Frquency bin')

                plt.subplot(nkey, ncol, ncol*idx+nreim*ch_idx+2)
                plt.axis('on')
                plt.imshow(show_phase[:,:,ch_idx], origin='lower', cmap=cmap, vmax=np.pi, vmin=-np.pi, extent=(0, nt, 0, nf))
                plt.colorbar(shrink=1)
                plt.title('Phase')
                plt.xlabel('Time frame')
                plt.ylabel('Frquency bin')

                # plt.subplot(nkey, ncol, ncol*idx+nreim*ch_idx+1)
                # plt.axis('on')
                # plt.imshow(show_real_log, origin='lower', cmap=cmap, vmax=5, vmin=-10, extent=(0, nt, 0, nf))
                # plt.colorbar(shrink=1)
                # plt.title('Complex number-real')
                # plt.xlabel('Time frame')
                # plt.ylabel('Frquency bin')

                # plt.subplot(nkey, ncol, ncol*idx+nreim*ch_idx+2)
                # plt.axis('on')
                # plt.imshow(show_imag_log, origin='lower', cmap=cmap, vmax=5, vmin=-10, extent=(0, nt, 0, nf))
                # plt.colorbar(shrink=1)
                # plt.title('Complex number-image')
                # plt.xlabel('Time frame')
                # plt.ylabel('Frquency bin')

        idx = idx+1
    return plt

# Check duration of code
def check_duration():
    from line_profiler import LineProfiler
    lp = LineProfiler()
    lp_wrap = lp(acoustic_scene.simulate)
    lp_wrap()
    lp.print_stats() 

if __name__ == "__main__":
    import torch
    import numpy as np
    import scipy.io

