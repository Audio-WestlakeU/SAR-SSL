import os
import random
import numpy as np
import scipy
import scipy.io
import scipy.signal
import soundfile
import webrtcvad
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
    st = np.random.randint(0, nsample - nsample_desired)
    ed = st + nsample_desired
    sig_pad_cut = sig[st:ed]

    return sig_pad_cut

class WSJ0Dataset(Dataset):
    """ WSJ0Dataset (after 20240403)
        train: /tr 81h (both speaker independent and dependent)
        val: /dt 5h
        test: /et 5h
        spk/wav
    """
    def __init__(self, path, T, fs, num_source=1, size=None):

        self.corpus, self.paths = explore_corpus(path, 'wav')
        self.spkWAVs = []
        self.spkIDs = []
        for spks in list(self.corpus.values()):
            self.spkWAVs += list(spks.values())
            self.spkIDs += list(spks.keys())

        # self.paths.sort()
        # random.shuffle(self.paths)
        self.fs = fs
        self.T = T
        self.sum_source = num_source
        self.sz = len(self.spkIDs) if size is None else size 

    def __len__(self):
        return self.sz

    def __getitem__(self, idx):
        if idx < 0: idx = len(self.spkIDs) + idx
        elif idx >= len(self.spkIDs): idx = idx % len(self.spkIDs)

        # random speaker IDs
        spkID_list = [self.spkIDs[idx]]
        idx_list = [idx]
        while(len(set(spkID_list))<self.sum_source):
            idx_othersources = np.random.randint(0, len(self.spkIDs))
            spkID_list += [self.spkIDs[idx_othersources]]
            idx_list += [idx_othersources]

        # read speech signals
        s_shape_desired = int(self.T * self.fs)
        s_sources = []
        for source_idx in range(self.sum_source):
            spkID = spkID_list[source_idx]
            spkWAVs = self.spkWAVs[idx_list[source_idx]]
            utt_paths = list(spkWAVs.values())
            # Get a random speech utterance from specific speaker
            utt_idx = np.random.randint(0, len(utt_paths))
            s, fs = soundfile.read(self.paths[utt_idx], dtype='float32')
            if fs != self.fs:
                s = scipy.signal.resample_poly(s, up=self.fs, down=fs)
                raise Warning('WSJ0 is downsampled to requrired frequency~')
            s = pad_cut_sig_samespk(utt_paths, utt_idx, s_shape_desired, self.fs) # pad by the same spk
            s -= s.mean()

            s_sources += [s]
        s_sources = np.array(s_sources).transpose(1,0)

        return s_sources #, np.ones_like(s_sources)
    

class LibriSpeechDataset(Dataset):
    """ LibriSpeechDataset (about 1000h)
        https://www.openslr.org/12
        spk/chapter/spk-chapter-utterance.flac
    """

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
        self.corpus, _ = explore_corpus(path, 'flac')
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
        spkID_list = []

        for source_idx in range(self.num_source):
            if source_idx==0:
                chapter = self.chapterList[idx]
                utts = list(chapter.keys())
                spkID = utts[0].split('-')[0]
                spkID_list += [spkID]
            else:
                while(len(set(spkID_list))<=source_idx):
                    idx_othersources = np.random.randint(0, len(self.chapterList))
                    chapter = self.chapterList[idx_othersources]
                    utts = list(chapter.keys())
                    spkID = utts[0].split('-')[0]
                    spkID_list += [spkID]

            utt_paths = list(chapter.values())
            s_shape_desired = int(self.T * self.fs)
            s_clean = np.zeros((s_shape_desired, 1)) # random initialization
            while np.sum(s_clean) == 0: # avoid full-zero s_clean
                # Get a random speech segment from the selected chapter
                utt_idx = np.random.randint(0, len(chapter))
                s = pad_cut_sig_samespk(utt_paths, utt_idx, s_shape_desired, self.fs) # pad by the same spk & chapter
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
    