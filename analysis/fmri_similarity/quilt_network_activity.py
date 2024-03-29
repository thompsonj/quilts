"""Saves quilted versions of network activations to match the speech stimuli.

Goal: save one activation quilt per speaker which corresponds to the speech
quilts presented to participants during the quilts fmri experiment

Ingredients:
- .npy of network activations to utterances that were used in the quilts, one
sample every two frames of the audio features, so every 20 ms (ish)
- .npy of segmentation info for each utterance that was used for quilting.
These came from the Nuance seg files they provided to me at the beginning of
the project. Contains 'phn_start' and 'phn_end' to define segments in
milliseconds in chronological order. Segments labelled as silence were not used
in the quilting process so the first segment always begins after 0ms and
- .wav original sound files from Nuance for each utterance
- .mat containing 'final_seg_order' : a list of indices into a list of segments

Steps:
- Turn ms segment start/end times into indices into the network activations.
  - load activations
  - load segmentations
  - load audio to know total length of utterance
  - divide start/end by total length to convert to relative to total length
  (0 is start, 1 is end)
  - multiply by the number of feature frames for that utterance, assuming that
  feature frames are evenly distributed throughout the total audio file
  - round to nearest integer to get indices into the network activations
  - test: verify that the acoustic features correspond to approximately the
  length of the audio and that the selected audio for quilting is slightly less
  than the length of the audio (because segments labelled silence are not
  included in the quilts)
- Add the to-be-quilted segments of network activations in a dict with one key
per layer of the network
- Load the seg order file which provides indices into the concatenated set of
segments from all speakers assembled in the previous step. These indices allow
us to reorder the segments according to the order which was used when
generating the speech quilts.
- The speech quilts were originally 60 seconds long and then trimmed out 59
seconds after they'd been constructed. So, pickle only the first 59 seconds of
the quilted activations.
- This process produces one 59 second quilt for each speaker. To call from
command line:
quilt_network_activity.py [input language ('enu', 'deu', or 'nld')] [model name
(e.g. enu-enu-freeze/enu-enu-freeze)]
"""
# env: python2.7 due to old files
import sys
import os
import time
import pickle
import numpy as np
from scipy.io import loadmat
import pandas

QUILT_LEN = 59000.0  # each speech quilt is 59 seconds
FRAME_RATE = 20.0  # 20 milliseconds
N_FRAMES_PER_QUILT = int(QUILT_LEN/FRAME_RATE)
LAYERS = ['cnn1', 'cnn2', 'cnn3', 'cnn4', 'cnn5', 'cnn6', 'cnn7', 'cnn8',
          'cnn9', 'fc1', 'fc2']
LAYERS_INPUT = ['input'] + LAYERS
STIM_DIR = '/data0/Dropbox/quilts/fmri_exp/generate_stimuli/'
ACT_DIR = '/data0/Dropbox/quilts/network_activations'
INPUT_DIR = '/data0/Dropbox/quilts/models_and_input'
HOME = '/data0/Dropbox/quilts/analysis/fmri_similarity'
OUT_DIR = HOME + '/quilted_activities'


class Quilter():
    
    
    def __init__(self, input_lang, model):
        """Initialize instance attributes.

        Parameters
        ----------
        input_lang : str
            Input speech language identifier. 'enu', 'deu', or 'nld'
        model : str
            Network identifier corresponding to two nested directories. 
            e.g. 'enu-enu-freeze/enu-enu-freeze'

        Returns
        -------
        None

        """
        self.in_lang = input_lang
        self.model = model
        audio_len_file = HOME + '/{}_audio_lengths.pkl'
        with open(audio_len_file.format(input_lang), 'rb') as alf:
            self.len_audio = pickle.load(alf)
        self.out_dir = '{}/{}/{}'.format(OUT_DIR, input_lang, model)


    def quilt_all_speakers(self):
        """Save all activation quilts for all speakers for this lang and model.

        Returns
        -------
        None

        """
        fname = '{}/{}/selected_speakers.txt'.format(STIM_DIR, self.in_lang)
        with open(fname, 'r') as fil:
            speakers = [name.split('.')[0] for name in fil.read().splitlines()]
        
        n_spk = len(speakers)
        for spkr_no, spkr in enumerate(speakers):
            # if spkr == 'nrc_ajm':
                # continue
            print('Processing speaker:{} {}/{}'.format(spkr, spkr_no+1, n_spk))
            tic = time.time()
            self.quilt_network_activations(speaker=spkr)
            toc = time.time()
            print("Quilting took {} seconds".format(toc - tic))


    def quilt_network_activations(self, speaker='inh_100cor_1'):
        """Save all quilted activations for a given speaker, model and language.

        Segment network activations based on phonetic boundaries. For each
        layer in the network, reorder segments based on stored segment order
        used when generating the experimental stimuli. Save one pickle file
        containing the network activations at each layer corresponding to the
        response to each of the 180 stimulus speech quilts.

        Parameters
        ----------
        speaker : str
            Speaker identifier used to load the input features and segment
            order.

        Returns
        -------
        None
        """
        # Don't redo if the quilt is alredy made
        out_file = '{}/{}_quilted.pkl'.format(self.out_dir, speaker)
        # if os.path.isfile(out_file):
        #     return
        all_act = self.segment_activations(speaker)
        all_act['input'] = self.segment_input(speaker)

        # Load the order of segments used when generating stimuli quilts
        mat_file = '{}/{}/quilts/order/{}_60s_order.mat'.format(STIM_DIR, self.in_lang, speaker)
        mat = loadmat(mat_file)
        seg_order = mat['final_seg_order'] - 1  # to correct for MATLAB 1-based indexing

        # Concatenate segments according to the quilt order
        quilted_acts = {}
        for layer in LAYERS_INPUT:
            # Initialize with first segment
            quilted_acts[layer] = all_act[layer][seg_order[0][0]]
            for i in seg_order[0][1:]:
                quilted_acts[layer] = np.append(quilted_acts[layer], all_act[layer][i], axis=0)
        for layer in LAYERS_INPUT:
            quilted_acts[layer] = quilted_acts[layer][:N_FRAMES_PER_QUILT, :]
            minn = np.min(quilted_acts[layer])
            if minn < 0:
                print('min(quilted_acts[{}]): {}'.format(layer, minn))
        # Save the quilted network activations
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)
        out_file = '{}/{}_quilted.pkl'.format(self.out_dir, speaker)
        with open(out_file, 'wb') as file_:
            pickle.dump(quilted_acts, file_, pickle.HIGHEST_PROTOCOL)


    def add_input_to_quilted_acts(self, speaker):
        """Auxillary method to add the input features to the saved activations.

        (just because I didn't think to include them initially)

        Parameters
        ----------
        speaker : str
            Speaker identifier used to load the input features and segment
            order.

        Returns
        -------
        None

        """
        # Load input segments
        all_input = self.segment_input(speaker)

        # Load the order of segments used when generating stimuli quilts
        mat_file = '{}/{}/quilts/order/{}_60s_order.mat'.format(STIM_DIR, self.in_lang, speaker)
        mat = loadmat(mat_file)
        seg_order = mat['final_seg_order'] - 1  # to correct for MATLAB 1-based indexing

        # Load previously saved quilted activations
        in_file = '{}/{}_quilted.pkl'.format(self.out_dir, speaker)
        with open(in_file, 'rb') as file_:
            quilted_acts = pickle.load(file_)

        quilted_acts['input'] = all_input[seg_order[0][0]]
        for i in seg_order[0][1:]:
            quilted_acts['input'] = np.append(quilted_acts['input'], all_input[i], axis=0)

        with open(in_file, 'wb') as file_:
            pickle.dump(quilted_acts, file_, pickle.HIGHEST_PROTOCOL)


    def segment_input(self, speaker):
        """Segment the input features according to phonetic boundaries.

        Parameters
        ----------
        speaker : str
            Speaker identifier to use when loading the input features.

        Returns
        -------
        list
            Segments of input features.

        """
        all_input = []
        with open('{}/{}/proposed_utts/{}.txt'.format(STIM_DIR, self.in_lang, speaker), 'r') as fil:
            utterance_nos = fil.read().splitlines()
        # Divide the activations into the segments that were used in the speech quilts
        for utt_no in utterance_nos:
            if speaker == 'nrc_ajm' and utt_no == '38':
                # An short utterance containing only silence for which the input
                # was not recorded needs to be added manually
                n_segments_38 = 4
                # segs = [np.empty((1, all_input[0].shape[1])) for i in range(n_segments_38)]
                segs = [np.zeros((1, all_input[0].shape[1])) for i in range(n_segments_38)]
                all_input = all_input + segs
            else:
                fname = '{0}/{1}/inputs/{2}/{2}_{3}.npy'
                input_file = fname.format(INPUT_DIR, self.in_lang, speaker, utt_no)
                input_feats = np.load(input_file).squeeze()[::2, :]

                n_frames = input_feats.shape[0]
                # Load and concatenate segmentation info (in ms)
                seg = np.load('{0}/{1}/seg/{2}/{2}_{3}.npy'.format(STIM_DIR, self.in_lang,
                                                                   speaker, utt_no),
                              encoding='latin1', allow_pickle=True)
                length_ms = self.len_audio[speaker][utt_no]
                frame_times = np.arange(0, n_frames*FRAME_RATE, FRAME_RATE)

                # Convert ms boundaries to indices into activations
                phn_start = pandas.to_timedelta(seg[()]['phn_start'], unit='ms')
                phn_end = pandas.to_timedelta(seg[()]['phn_end'], unit='ms')

                index = pandas.to_timedelta(frame_times, unit='ms')
                input_df = pandas.DataFrame(input_feats, index=index)
                segs = [input_df[strt:stp].to_numpy() for strt, stp in zip(phn_start, phn_end)]
                all_input = all_input + segs
        print(len(all_input))
        return all_input


    def segment_activations(self, speaker):
        """Segment network activations according to saved boundaries.

        Parameters
        ----------
        speaker : str
            Speaker identifier to use when loading the input features.

        Returns
        -------
        dict of list
            One list of activation segments for each layer.

        """
        all_act = {layer:[] for layer in LAYERS}
        total_frames = 0
        total_segs = 0
        with open('{}/{}/proposed_utts/{}.txt'.format(STIM_DIR, self.in_lang, speaker), 'r') as fil:
            utterance_nos = fil.read().splitlines()
        # Divide the activations into the segments that were used in the speech quilts
        for utt_no in utterance_nos:
            if speaker == 'nrc_ajm' and utt_no == '38':
                n_segments_38 = 4
                for layer in LAYERS:
                    segs = [np.zeros((1, all_act[layer][0].shape[1])) for i in range(n_segments_38)]
                    all_act[layer] = all_act[layer] + segs
                total_segs += n_segments_38
            else:
                # print('utterance number:', utt_no)
                # Load and concatenate network activations
                act = np.load('{0}/{1}/{2}/{3}/{3}_{4}.npz'.format(ACT_DIR, self.in_lang,
                                                                   self.model, speaker,
                                                                   utt_no))
                n_frames = act['cnn1'].shape[0]
                total_frames = total_frames + n_frames
                # Load and concatenate segmentation info (in ms)
                seg = np.load('{0}/{1}/seg/{2}/{2}_{3}.npy'.format(STIM_DIR, self.in_lang,
                                                                   speaker, utt_no),
                              encoding='latin1', allow_pickle=True)
                length_ms = self.len_audio[speaker][utt_no]
                frame_times = np.arange(0, n_frames*FRAME_RATE, FRAME_RATE)

                # Convert ms boundaries to indices into activations
                phn_start = pandas.to_timedelta(seg[()]['phn_start'], unit='ms')
                total_segs = total_segs + len(phn_start)
                phn_end = pandas.to_timedelta(seg[()]['phn_end'], unit='ms')
                for layer in LAYERS:
                    index = pandas.to_timedelta(frame_times, unit='ms')
                    act_df = pandas.DataFrame(act[layer], index=index)
                    segs = [act_df[strt:stp].to_numpy() for strt, stp in zip(phn_start, phn_end)]
                    # print('segs[0].shape: {}'.format(segs[0].shape))
                    all_act[layer] = all_act[layer] + segs

        assert_str = 'len(all_act[cnn1]):{}, total_segs:{}'.format(len(all_act['cnn1']), total_segs)
        assert len(all_act['cnn1']) == total_segs, assert_str
        return all_act


def main():
    """Process input arguments and call quilter."""
    args = sys.argv
    if len(args) < 3:
        print("Usage: quilt_network_activity.py input_language model_name")
        # e.g. python quilt_network_activity.py enu enu-enu-freeze/enu-enu-freeze
    else:
        input_lang = args[1]
        model = args[2]
        quilter = Quilter(input_lang, model)
        quilter.quilt_all_speakers()

if __name__ == '__main__':
    main()
