"""
    Generate simulated room impulse responses and microphone signals for training and test 

    Examples: 
        python gen_simu.py --mode sig --stage pretrain --data_num 512000 --src_dir ../../../data/SrcSig/wsj0 --save_to ../../data/MicSig/simu --gpus [0,1]
        python gen_simu.py --mode sig --stage preval --data_num 4000 --src_dir ../../../data/SrcSig/wsj0 --save_to ../../data/MicSig/simu --gpus [0]
        python gen_simu.py --mode sig --stage pretest --data_num 4000 --src_dir ../../../data/SrcSig/wsj0 --save_to ../../data/MicSig/simu --gpus [0]
        python gen_simu.py --mode sig --stage pretest_ins_T1000 --data_num 10 --save_dp True --room_sz_range [[5,10],[3,6],[2.5,3]] --T60_range [1.0,1.0] --snr_range [20,20] --src_dir ../../../data/SrcSig/wsj0 --save_to ../../data/MicSig/simu --gpus [0]

        # python gen_simu.py --mode rir --stage train --data_num 1000 --save_to ../../data/RIR/simu --gpus [0,1]
        # python gen_simu.py --mode rir --stage val --data_num 20 --save_to ../../data/RIR/simu --gpus [0,1]
        # python gen_simu.py --mode rir --stage test --data_num 20 --save_to ../../data/RIR/simu --gpus [0,1]
"""
import os
cpu_num = 8
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)

import numpy as np
import tqdm
import inspect
import importlib
import multiprocessing as mp
from functools import partial
# import librosa # cause CPU overload, for data generation (scipy.signal.resample, librosa.resample) 
from jsonargparse import ArgumentParser
from typing import *
from pathlib import Path
from utils_simu_rir_sig import *
from utils_src import *
from utils_noise import * 
from utils_array import *

def GenerateRandomRIR(
    room_sz_range: Union[List[Tuple[float, float]], np.ndarray]=[(3,15), (3,10), (2.5,6)], 
    T60_range: Tuple[float,float]=(0.2, 1.3), 
    abs_weights_range: List[Tuple[float, float]]=[(0.5,1)]*6, 
    mic_array_cfg: Dict[str, Any]=mic_array_cfg_2ch, 
    array_pos_ratio_range: Union[List[Tuple[float, float]], np.ndarray]=[(0.2,0.8), (0.2,0.8), (0.1,0.5)], 
    num_source_range: Tuple[int,int]=(1,1), 
    source_state: str='static', 
    min_src_array_dist: float=0.3, 
    min_src_boundary_dist: float=0.3, 
    traj_pt_mode: str='time',
    fs: int=16000, 
    c: float=343.0, 
    ism_db: float=12, 
    T: float=4.112, 
    stage: str='train',
    data_num: int=1024,
    save_to: str='',
    gpus: List[int]=[0,0],
    use_gpu: bool=True,
    ): 

    if traj_pt_mode == 'time':
        if source_state =='static':
            nb_points = 1
        else:
            nb_points = int(T/0.1)
    else:
        nb_points = None
    if stage == 'pretrain':
        seed = 1
    elif stage == 'preval':
        seed = 2e6
    elif stage == 'pretest':
        seed = 3e6
    elif stage == 'train':
        seed = 4e6
    elif stage == 'val':
        seed = 5e6
    elif stage == 'test':
        seed = 6e6
    seed = int(seed)
    args = locals().copy()  # capture the parameters passed to this function or their edited values
    
    Path(save_to+'/'+stage).mkdir(parents=True, exist_ok=True)
    save_to_file = os.path.join(save_to, stage, f'all_info.npz')
    msg = None
    if os.path.exists(save_to_file):
        msg = input('all_info.npz already exists, sure to regenerate? (Enter (or y) for yes,  n for no)')
        if (msg == 'n'):
            info = dict(np.load(save_to, allow_pickle=True))
            sa_cfgs = info['cfgs']
            args = info['args'] 
            print('load rir cfgs from file ' + save_to_file)
            print('Args in npz: \n', args.item())
        
    if ~os.path.exists(save_to_file) | (msg == 'y') | (msg == ''):
        print('Args:')
        print(dict(args), '\n')

        # Generate random spatial acoustics  
        spatialacoustics = SpatialAcoustics()
        roomir = RoomImpulseResponse(fs=fs, c=c, ism_db=ism_db)
        sa_cfgs = []
        for idx in range(data_num):
            sa_cfg = spatialacoustics.generate_random_spatial_acoustics(
                room_sz_range=room_sz_range, 
                T60_range=T60_range, 
                abs_weights_range=abs_weights_range, 
                c=c,
                ism_db=ism_db,
                mic_array_cfg=mic_array_cfg,
                array_pos_ratio_range=array_pos_ratio_range,
                num_source_range=num_source_range, 
                source_state=source_state,
                min_src_array_dist=min_src_array_dist, 
                min_src_boundary_dist=min_src_boundary_dist, 
                traj_pt_mode=traj_pt_mode,
                nb_points=nb_points,
                room_cfg=None,
                seed=seed,
                idx=idx
            )
            sa_cfgs.append(sa_cfg)

        # Save to npz
        np.savez_compressed(save_to_file, 
                            args=args,
                            cfgs=sa_cfgs)

    # Define dataset
    mic_sig_or_rir = MicrophoneSignalOrRIR()

    # Generate room impulse responses  
    pbar = tqdm.tqdm(total=data_num)
    pbar.set_description('generating rirs')

    if use_gpu:
        def init_env_var(gpus: List[int]):
            i = queue.get()
            os.environ['CUDA_VISIBLE_DEVICES'] = str(i)
            import gpuRIR  # occupy this gpu
            importlib.reload(gpuRIR)  # reload gpuRIR to use another gpu

        queue = mp.Queue()
        for gid in gpus:
            queue.put(gid)

        p = mp.Pool(processes=len(gpus), initializer=init_env_var, initargs=(queue,))

        for _ in p.imap_unordered(
            partial(
                mic_sig_or_rir.generate_rir,
                sa_cfgs=sa_cfgs,
                fs=fs,
                c=c,
                roomir=roomir, 
                save_to=os.path.join(save_to, stage),
            ),
            range(data_num),
            chunksize=100,
        ):
            pbar.update()
        p.close()
        p.join()

    else:
        for idx in range(data_num):
            pbar.update()
            mic_sig_or_rir.generate_rir(
                idx=idx, 
                sa_cfgs=sa_cfgs, 
                fs=fs,
                c=c,
                roomir=roomir,  
                save_to=os.path.join(save_to, stage),
                )


def GenerateRandomMicSig(
    room_sz_range: Union[List[Tuple[float, float]], np.ndarray]=[(3,15), (3,10), (2.5,6)], 
    T60_range: Tuple[float,float]=(0.2, 1.3), 
    abs_weights_range: List[Tuple[float, float]]=[(0.5,1)]*6, 
    mic_array_cfg: Dict[str, Any]=mic_array_cfg_2ch, 
    array_pos_ratio_range: Union[List[Tuple[float, float]], np.ndarray]=[(0.2,0.8), (0.2,0.8), (0.1,0.5)], 
    num_source_range: Tuple[int,int]=(1,1), 
    source_state: str='static', 
    min_src_array_dist: float=0.3, 
    min_src_boundary_dist: float=0.3, 
    traj_pt_mode: str='time',
    snr_range: Tuple[float,float]=(15,30),
    noise_type: str='diffuse_white',
    fs: int=16000, 
    c: float=343.0, 
    ism_db: float=12, 
    T: float=4.112, 
    src_dir: str='', 
    noi_dir: str='',
    stage: str='pretrain',
    data_num: int=1,
    save_to: str='',
    save_dp: bool=False,
    gpu_conv: bool=False,
    gpus: List[int]=[0,0,1,1],
    use_gpu: bool=True,
    ): 
 
    if traj_pt_mode == 'time':
        if source_state =='static':
            nb_points = 1
        else:
            nb_points = int(T/0.1)
    else:
        nb_points = None
    assert (stage in ['pretrain', 'preval', 'pretest']) | ('pretest_ins' in stage), stage
    if stage == 'pretrain':
        seed = 1
    elif stage == 'preval':
        seed = 2e6
    elif stage == 'pretest':
        seed = 3e6
    elif 'pretest_ins' in stage:
        seed = 3e6
    # elif stage == 'train':
    #     seed = 4e6
    # elif stage == 'val':
    #     seed = 5e6
    # elif stage == 'test':
    #     seed = 6e6
    seed = int(seed)
    args = locals().copy()  # capture the parameters passed to this function or their edited values

    Path(save_to+'/'+stage).mkdir(parents=True, exist_ok=True)
    save_to_file = os.path.join(save_to, stage, f'all_info.npz')
    msg = None
    if os.path.exists(save_to_file):
        msg = input('all_info.npz already exists, sure to regenerate? (Enter (or y) for yes,  n for no)')
        if (msg == 'n'):
            info = dict(np.load(save_to, allow_pickle=True))
            sa_cfgs = info['cfgs']
            args = info['args'] 
            print('load rir cfgs from file ' + save_to_file)
            print('Args in npz: \n', args.item())
        
    if ~os.path.exists(save_to_file) | (msg == 'y') | (msg == ''):
        print('Args:')
        print(dict(args), '\n')

        # Generate random spatial acoustics 
        spatialacoustics = SpatialAcoustics()
        roomir = RoomImpulseResponse(fs=fs, c=c, ism_db=ism_db) 
        sa_cfgs = []
        for idx in range(data_num):
            sa_cfg = spatialacoustics.generate_random_spatial_acoustics(
                room_sz_range=room_sz_range, 
                T60_range=T60_range, 
                abs_weights_range=abs_weights_range, 
                c=c,
                ism_db=ism_db,
                mic_array_cfg=mic_array_cfg,
                array_pos_ratio_range=array_pos_ratio_range,
                num_source_range=num_source_range, 
                source_state=source_state,
                min_src_array_dist=min_src_array_dist, 
                min_src_boundary_dist=min_src_boundary_dist, 
                traj_pt_mode=traj_pt_mode,
                nb_points=nb_points,
                room_cfg=None,
                seed=seed,
                idx=idx
            )
            sa_cfgs.append(sa_cfg)

        # Save to npz
        np.savez_compressed(save_to_file, 
                            args=args, 
                            cfgs=sa_cfgs)

    # Define dataset
    mic_sig_or_rir = MicrophoneSignalOrRIR()
    if stage == 'pretrain':
        src_dir = src_dir + '/tr'
    elif stage == 'preval':
        src_dir = src_dir + '/dt'
    elif stage == 'pretest':
        src_dir = src_dir + '/et'
    elif 'pretest_ins' in stage:
        src_dir = src_dir + '/et'
    srcdataset = WSJ0Dataset(
        path = src_dir,
        T = T,
        fs = fs,
        num_source = max(num_source_range) 
    )
    noidataset = NoiseSignal(
        T = T,
        fs = fs,
        nmic = mic_array_cfg['mic_pos_relative'].shape[0],
        noise_type = noise_type,
        noise_path = noi_dir,
        c = c
    )

    # Generate microphone signals 
    pbar = tqdm.tqdm(total=data_num)
    pbar.set_description('generating rirs|microphone signals')

    if use_gpu:
        def init_env_var(gpus: List[int]):
            i = queue.get()
            os.environ['CUDA_VISIBLE_DEVICES'] = str(i)
            import gpuRIR  # occupy this gpu
            importlib.reload(gpuRIR)  # reload gpuRIR to use another gpu

        queue = mp.Queue()
        for gid in gpus:
            queue.put(gid)

        p = mp.Pool(processes=len(gpus), initializer=init_env_var, initargs=(queue,))

        for _ in p.imap_unordered(
            partial(
                mic_sig_or_rir.generate_microphone_signal,
                sa_cfgs=sa_cfgs,
                fs=fs,
                c=c,
                roomir=roomir, 
                srcdataset=srcdataset, 
                noidataset=noidataset,
                snr_range=snr_range,   
                save_to=os.path.join(save_to, stage),
                save_dp=save_dp,
                gpu_conv=gpu_conv,
                seed=seed,
            ),
            range(data_num),
            chunksize=100,
        ):
            pbar.update()
        p.close()
        p.join()

    else:
        for idx in range(data_num):
            pbar.update()
            mic_sig_or_rir.generate_microphone_signal(
                idx=idx, 
                sa_cfgs=sa_cfgs, 
                fs=fs,
                c=c,
                roomir=roomir, 
                srcdataset=srcdataset, 
                noidataset=noidataset,
                snr_range=snr_range,  
                save_to=os.path.join(save_to, stage),
                save_dp=save_dp,
                gpu_conv=gpu_conv,
                seed=seed,
                )


if __name__ == '__main__':
     
    parser = ArgumentParser(description='Generate microphone signals and RIRs')
    parser.add_function_arguments(GenerateRandomMicSig) 
    parser.add_argument('--mode', type=str, default='rir', metavar='Mode', help='Mode (default: rir)')
    args = parser.parse_args()

    # print(args)
    if args.mode == 'rir':
        # get paramters for function `GenerateRandomRIR`
        sign = inspect.signature(GenerateRandomRIR)
        args_for_generate_sig_cfg = dict()
        for param in sign.parameters.values():
            args_for_generate_sig_cfg[param.name] = getattr(args, param.name)

        # generate configuration & microphone signals
        GenerateRandomRIR(**args_for_generate_sig_cfg)

    elif args.mode =='sig':
        # get paramters for function `GenerateRandomMicSig`
        sign = inspect.signature(GenerateRandomMicSig)
        args_for_generate_sig_cfg = dict()
        for param in sign.parameters.values():
            args_for_generate_sig_cfg[param.name] = getattr(args, param.name)

        # generate configuration & microphone signals
        GenerateRandomMicSig(**args_for_generate_sig_cfg)


