"""
	Function:   Define some basic operations for generating multi-channel audio data 
"""

import numpy as np
import scipy.signal
import random
import soundfile
from scipy import stats
import matplotlib.pyplot as plt


def add_noise(mic_sig_won, noi_sig, snr, mic_sig_wonr=None, eps=1e-10):
    """ Add noise to clean microphone signals with a given signal-to-noise ratio (SNR)
        Args:       mic_sig_won   - clean microphone signals without any noise (nsample, nch)
                    noi_sig       - noise signals (nsample, nch)
                    snr           - specific SNR
                    mic_sig_won   - clean microphone signals without any noise and reverberation (nsample, nch)
        Returns:    mic_sig       - microphone signals with noise (nsample, nch)
    """ 
    nsample, _ = mic_sig_won.shape
    if mic_sig_wonr is None: # signal power includes reverberation
        av_pow = np.mean(np.sum(mic_sig_won**2, axis=0)/nsample, axis=0) 	# average mic power across all received signals
        av_pow_noise = np.mean(np.sum(noi_sig**2, axis=0)/nsample, axis=0)
        noise_wsnr = np.sqrt(av_pow / (10 ** (snr / 10)))/ (np.sqrt(av_pow_noise)+eps) * noi_sig
    else: # signal power do not include reverberation
        av_pow = np.mean(np.sum(mic_sig_wonr**2, axis=0)/nsample, axis=0) 	# average mic power across all received signals
        av_pow_noise = np.mean(np.sum(noi_sig**2, axis=0)/nsample, axis=0)
        noise_wsnr = np.sqrt(av_pow / (10 ** (snr / 10)))/ (np.sqrt(av_pow_noise)+eps) * noi_sig
    mic_sig = mic_sig_won + noise_wsnr
    return mic_sig


def sou_conv_rir(sou_sig, rir):
    """ Perform convolution between source signal and room impulse reponses (RIRs)
        Args:       sou_sig   - source signal (nsample, )
                    rir       - multi-channel RIR (nrirsample, nch)
        Returns:    mic_sig   - multi-channel microphone signals (nsample, nch)
    """ 
    nsample = sou_sig.shape[0]

    mic_sig_temp = scipy.signal.convolve(sou_sig[:, np.newaxis], rir, mode='full', method='fft')
    mic_sig = mic_sig_temp[0:nsample, :]

    return mic_sig


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
    st = random.randint(0, nsample - nsample_desired)
    ed = st + nsample_desired
    sig_pad_cut = sig[st:ed]

    return sig_pad_cut


def pad_cut_sig_samespk(utt_path_list, current_utt_idx, nsample_desired, fs_desired):
    """ Pad (by adding utterance of the same spearker) and cut signal to desired length
        Args:       utt_path_list             - 
                    current_utt_idx
                    nsample_desired - desired sample length
                    fs_desired
        Returns:    sig_pad_cut     - padded and cutted signal (nsample_desired,)
    """ 
    sig = np.array([])
    nsample = sig.shape[0]
    while nsample < nsample_desired:
        utterance, fs = soundfile.read(utt_path_list[current_utt_idx])
        if fs != fs_desired:
            utterance = scipy.signal.resample_poly(utterance, up=fs_desired, down=fs)
            raise Warning(f'Signal is downsampled from {fs} to {fs_desired}')
        sig = np.concatenate((sig, utterance), axis=0)
        nsample = sig.shape[0]
        current_utt_idx += 1
        if current_utt_idx >= len(utt_path_list): current_utt_idx=0
    st = random.randint(0, nsample - nsample_desired)
    ed = st + nsample_desired
    sig_pad_cut = sig[st:ed]

    return sig_pad_cut


def select_microphones(mic_poss, nmic_selected, mic_dist_range):
    """ Randomly select certain number of microphones within microphone distance range 
        Args:       mic_poss        - microphone postions (nmic, 3)
                    nmic_selected   - the number of selected microphones
                    mic_dist_range  - the range of microphone distance [lower, upper]
        Returns:    mic_idxes       - indexes of selected microphones
    """
    nmic = mic_poss.shape[0]
    dist = -1
    cnt = 0
    while( (dist < mic_dist_range[0]) | (dist > mic_dist_range[1]) ):
        mic_idxes = random.sample(range(0, nmic), nmic_selected)
        mic_pos = mic_poss[mic_idxes, :]
        dist = np.sqrt(np.sum((mic_pos[0, :]-mic_pos[1, :])**2))
        cnt += 1
        assert cnt<1000, 'the distance of selected microphones is not satisfied, microphone distance is ' + str(dist)

    return mic_idxes, mic_pos


def com_num_micpair(mic_pos, mic_dist_range):
    """ Compute the number of all possible microphone pairs
        Args: mic_pos - (nmic, 3)
        Return: num - the number of micophone pairs
    """
    num = 0
    nmic = mic_pos.shape[0]
    for mic_idx in range(nmic-1):
        for mic_idx2 in range(mic_idx+1, nmic):
            mic_dist = np.sqrt(np.sum((mic_pos[mic_idx, :] - mic_pos[mic_idx2, :])**2))
            if (mic_dist_range[0]<= mic_dist) & (mic_dist_range[1]>= mic_dist):
                print(mic_idx, mic_idx2, mic_dist)
                num += 1
    return num


def micpair_dist_in_range(mic_pos, mic_dist_range):
    """ Check whether the distance between microhphone pairs is the predefined range 
        Args:   mic_pos - (2, 3)
                mic_dist_range - the range of microhphone distance
        Return: True or False
    """
    dist = np.sqrt(np.sum((mic_pos[0, :]-mic_pos[1, :])**2))
    return (dist >= mic_dist_range[0]) & (dist <= mic_dist_range[1])


# def stft_ham(insig, winsize=256, fftsize=512, hopsize=128):
#     nb_dim = len(np.shape(insig))
#     lSig = int(np.shape(insig)[0])
#     nCHin = int(np.shape(insig)[1]) if nb_dim > 1 else 1
#     x = np.arange(0,winsize)
#     nBins = int(fftsize/2 + 1)
#     nWindows = int(np.ceil(lSig/(2.*hopsize)))
#     nFrames = int(2*nWindows+1)
    
#     winvec = np.zeros((len(x),nCHin))
#     for i in range(nCHin):
#         winvec[:,i] = np.sin(x*(np.pi/winsize))**2
    
#     frontpad = winsize-hopsize
#     backpad = nFrames*hopsize-lSig

#     if nb_dim > 1:
#         insig_pad = np.pad(insig,((frontpad,backpad),(0,0)),'constant')
#         spectrum = np.zeros((nBins, nFrames, nCHin),dtype='complex')
#     else:
#         insig_pad = np.pad(insig,((frontpad,backpad)),'constant')
#         spectrum = np.zeros((nBins, nFrames),dtype='complex')

#     idx=0
#     nf=0
#     if nb_dim > 1:
#         while nf <= nFrames-1:
#             insig_win = np.multiply(winvec, insig_pad[idx+np.arange(0,winsize),:])
#             inspec = scipy.fft.fft(insig_win,n=fftsize,norm='backward',axis=0)
#             #inspec = scipy.fft.fft(insig_win,n=fftsize,axis=0)
#             inspec=inspec[:nBins,:]
#             spectrum[:,nf,:] = inspec
#             idx += hopsize
#             nf += 1
#     else:
#         while nf <= nFrames-1:
#             insig_win = np.multiply(winvec[:,0], insig_pad[idx+np.arange(0,winsize)])
#             inspec = scipy.fft.fft(insig_win,n=fftsize,norm='backward',axis=0)
#             #inspec = scipy.fft.fft(insig_win,n=fftsize,axis=0)
#             inspec=inspec[:nBins]
#             spectrum[:,nf] = inspec
#             idx += hopsize
#             nf += 1
    
#     return spectrum

# def ctf_ltv_direct(sig, irs, ir_times, fs, win_size):
#     """ Args:
#             sig: 
#             irs: (lrir, nch, nrir)/(lrir, nrir)
#         Refs: https://github.com/danielkrause/DCASE2022-data-generator/blob/main/utils.py
#     """
#     convsig = []
#     win_size = int(win_size)
#     hop_size = int(win_size / 2)
#     fft_size = win_size*2
#     nBins = int(fft_size/2)+1
    
#     # IRs
#     ir_shape = np.shape(irs)
#     sig_shape = np.shape(sig)
    
#     lIr = ir_shape[0]

#     if len(ir_shape) == 2:
#         nIrs = ir_shape[1]
#         nCHir = 1
#     elif len(ir_shape) == 3: 
#         nIrs = ir_shape[2]
#         nCHir = ir_shape[1]
    
#     if nIrs != len(ir_times):
#         return ValueError('Bad ir times')
    
#     # number of STFT frames for the IRs (half-window hopsize)
    
#     nIrWindows = int(np.ceil(lIr/win_size))
#     nIrFrames = 2*nIrWindows+1
#     # number of STFT frames for the signal (half-window hopsize)
#     lSig = sig_shape[0]
#     nSigWindows = np.ceil(lSig/win_size)
#     nSigFrames = 2*nSigWindows+1
    
#     # quantize the timestamps of each IR to multiples of STFT frames (hopsizes)
#     tStamps = np.round((ir_times*fs+hop_size)/hop_size)
    
#     # create the two linear interpolator tracks, for the pairs of IRs between timestamps
#     nIntFrames = int(tStamps[-1])
#     Gint = np.zeros((nIntFrames, nIrs))
#     for ni in range(nIrs-1):
#         tpts = np.arange(tStamps[ni],tStamps[ni+1]+1,dtype=int)-1
#         ntpts = len(tpts)
#         ntpts_ratio = np.arange(0,ntpts)/(ntpts-1)
#         Gint[tpts,ni] = 1-ntpts_ratio
#         Gint[tpts,ni+1] = ntpts_ratio
    
#     # compute spectra of irs
#     if nCHir == 1:
#         irspec = np.zeros((nBins, nIrFrames, nIrs),dtype=complex)
#     else:
#         temp_spec = stft_ham(irs[:, :, 0], winsize=win_size, fftsize=2*win_size,hopsize=win_size//2)
#         irspec = np.zeros((nBins, np.shape(temp_spec)[1], nCHir, nIrs),dtype=complex)
    
#     for ni in range(nIrs):
#         if nCHir == 1:
#             irspec[:, :, ni] = stft_ham(irs[:, ni], winsize=win_size, fftsize=2*win_size,hopsize=win_size//2)
#         else:
#             spec = stft_ham(irs[:, :, ni], winsize=win_size, fftsize=2*win_size,hopsize=win_size//2)
#             irspec[:, :, :, ni] = spec # np.transpose(spec, (0, 2, 1))
    
#     #compute input signal spectra
#     sigspec = stft_ham(sig, winsize=win_size,fftsize=2*win_size,hopsize=win_size//2)
#     #initialize interpolated time-variant ctf
#     Gbuf = np.zeros((nIrFrames, nIrs))
#     if nCHir == 1:
#         ctf_ltv = np.zeros((nBins, nIrFrames),dtype=complex)
#     else:
#         ctf_ltv = np.zeros((nBins,nIrFrames,nCHir),dtype=complex)
    
#     S = np.zeros((nBins, nIrFrames),dtype=complex)
    
#     # processing loop
#     idx = 0
#     nf = 0
#     inspec_pad = sigspec
#     nFrames = int(np.min([np.shape(inspec_pad)[1], nIntFrames]))
    
#     convsig = np.zeros((win_size//2 + nFrames*win_size//2 + fft_size-win_size, nCHir))
    
#     while nf <= nFrames-1:
#         #compute interpolated ctf
#         Gbuf[1:, :] = Gbuf[:-1, :]
#         Gbuf[0, :] = Gint[nf, :]
#         if nCHir == 1:
#             for nif in range(nIrFrames):
#                 ctf_ltv[:, nif] = np.matmul(irspec[:,nif,:], Gbuf[nif,:].astype(complex))
#         else:
#             for nch in range(nCHir):
#                 for nif in range(nIrFrames):
#                     ctf_ltv[:,nif,nch] = np.matmul(irspec[:,nif,nch,:],Gbuf[nif,:].astype(complex))
#         inspec_nf = inspec_pad[:, nf]
#         S[:,1:nIrFrames] = S[:, :nIrFrames-1]
#         S[:, 0] = inspec_nf
        
#         repS = np.tile(np.expand_dims(S,axis=2), [1, 1, nCHir])
#         convspec_nf = np.squeeze(np.sum(repS * ctf_ltv,axis=1))
#         first_dim = np.shape(convspec_nf)[0]
#         convspec_nf = np.vstack((convspec_nf, np.conj(convspec_nf[np.arange(first_dim-1, 1, -1)-1,:])))
#         convsig_nf = np.real(scipy.fft.ifft(convspec_nf, fft_size, norm='forward', axis=0)) ## get rid of the imaginary numerical error remain
#         # convsig_nf = np.real(scipy.fft.ifft(convspec_nf, fft_size, axis=0))
#         #overlap-add synthesis
#         convsig[idx+np.arange(0,fft_size),:] += convsig_nf
#         #advance sample pointer
#         idx += hop_size
#         nf += 1
    
#     convsig = convsig[(win_size):(nFrames*win_size)//2,:]

#     return convsig

## Caculate T60

def rt60_with_sabine(room_sizes, absorption):
    x, y, z = room_sizes
    volumes = x*y*z

    surf_east = y*z
    surf_west = y*z
    surf_north = x*z
    surf_south = x*z
    surf_ceiling = x*y
    surf_floor = x*y

    abs_south = absorption['south']
    abs_east = absorption['east']
    abs_west = absorption['west']
    abs_north = absorption['north']
    abs_south = absorption['south']
    abs_ceiling = absorption['ceiling']
    abs_floor = absorption['floor']

    equivalent_absorption_surface = (surf_east * abs_east) \
        + (surf_west * abs_west) \
        + (surf_north * abs_north) \
        + (surf_south * abs_south) \
        + (surf_ceiling * abs_ceiling) \
        + (surf_floor * abs_floor)

    rt60 = 0.161*volumes/equivalent_absorption_surface

    return rt60


def envelope(x):
    return np.abs(scipy.signal.hilbert(x))


def cal_edc(RIR, eps=1e-10):
    # Schroeder Integration method
    max_idx = np.argmax(RIR)
    EDC = 10.0 * np.log10(np.cumsum(RIR[::-1]**2)[::-1]/(np.sum(RIR[max_idx:]**2)+eps)+eps)

    return EDC


def cal_rt60(EDC, fs, edc_st_list=list(range(-5,-20,-2)), edc_duration_list=list(range(-10,-30,-2)), vis=False, eps=1e-10):
    """ add extra [edc_st, edc_ed] pairs
    """
    
    def find_nearest_value(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    t60_list = []
    r_list = []
    x_list = []
    y_list = []
    for edc_st0 in edc_st_list:
        for edc_duration in edc_duration_list: 
            edc_st = find_nearest_value(EDC, edc_st0)
            edc_st = np.where(EDC == edc_st)[0][0]

            edc_ed = find_nearest_value(EDC, edc_st0+edc_duration)
            edc_ed = np.where(EDC == edc_ed)[0][0]
            
            # Perform linear regression
            if abs(edc_st-edc_ed)>1:
                times = np.arange(len(EDC))/fs
                x = times[edc_st:edc_ed]
                y = EDC[edc_st:edc_ed]
 
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                # assert (slope != np.inf) & (slope != np.nan), 'inf/nan exists~'
                t60_list += [-60/(slope+eps)]
                r_list += [r_value]
                x_list += [x]
                y_list += [y]
            else:
                t60_list += [np.nan]
                r_list += [0]
                x_list += [np.nan]
                y_list += [np.nan]

    # Compute the final value
    idx = np.argmax(abs(np.array(r_list)))
    r = r_list[idx]
    t60 = t60_list[idx]
    x = x_list[idx]
    y = y_list[idx]

    if vis:
        plt.switch_backend('agg')
        plt_all = plt.scatter(times, EDC, label='all', c='lightgray', marker='.', linewidth=1, zorder=10)
        plt_used = plt.scatter(x, y, label='used', c='whitesmoke', marker='.', linewidth=1, zorder=10)
        plt.legend(handles = [plt_all, plt_used])
        plt.xlabel('Time sample')
        plt.ylabel('Value')
        plt.savefig('edc_curve')
        # plt.show()

    return t60, r


def rt60_from_rirs(h, Fs, vis=False):
    """ 
        Refs: https://github.com/Chutlhu/dEchorate
    """
    edc = cal_edc(h)
    rt60, r = cal_rt60(edc, Fs, vis=vis)
    
    return rt60, r


def dpRIR_from_RIR(rir, dp_time, fs):
    """ 
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


def acoustic_power(s):
    """ Acoustic power of after removing the silences
	"""
    w = 512  # Window size for silent detection
    o = 256  # Window step for silent detection

    # Window the input signal
    s = np.ascontiguousarray(s)
    sh = (s.size - w + 1, w)
    st = s.strides * 2
    S = np.lib.stride_tricks.as_strided(s, strides=st, shape=sh)[0::o]

    window_power = np.mean(S ** 2, axis=-1)
    th = 0.01 * window_power.max()  # Threshold for silent detection
    return np.mean(window_power[np.nonzero(window_power > th)])


def cart2sph(cart):
    """ cart [x,y,z] → sph [azi,ele,r]
	"""
    xy2 = cart[:,0]**2 + cart[:,1]**2
    sph = np.zeros_like(cart)
    sph[:,0] = np.arctan2(cart[:,1], cart[:,0])
    sph[:,1] = np.arctan2(np.sqrt(xy2), cart[:,2]) # Elevation angle defined from Z-axis down
    sph[:,2] = np.sqrt(xy2 + cart[:,2]**2)

    return sph


def sph2cart(sph):
    """ sph [azi,ele,r] → cart [x,y,z]
	"""
    if sph.shape[-1] == 2: sph = np.concatenate((sph, np.ones_like(sph[..., 0]).unsqueeze(-1)), dim=-1)
    x = sph[..., 2] * np.sin(sph[..., 1]) * np.cos(sph[..., 0])
    y = sph[..., 2] * np.sin(sph[..., 1]) * np.sin(sph[..., 0])
    z = sph[..., 2] * np.cos(sph[..., 1])

    return np.stack((x, y, z)).transpose(1, 0)


if __name__ == "__main__":
    source_signal = np.random.rand(16000,)
    noise_signal = np.random.rand(16000,4)
    snr = 10
    rir = np.random.rand(10,4)
    sensor_signal_won = sou_conv_rir(source_signal, rir)
    sensor_signal = add_noise(sensor_signal_won, noise_signal, snr)
    print(source_signal.shape, rir.shape, sensor_signal.shape)

