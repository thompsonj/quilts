"""Saves quilted versions of network activations to match the speech quilt stimuli."""
# env: activ? python2.7 due to old files
import sys
import os
import time
import pickle
import numpy as np
from scipy.io import loadmat

QUILT_LEN = 59000  # each speech quilt is 59 seconds
LAYERS = ['cnn1', 'cnn2', 'cnn3', 'cnn4', 'cnn5', 'cnn6', 'cnn7', 'cnn8', 'cnn9', 'fc1', 'fc2']
STIM_DIR = '/data0/Dropbox/quilts/fmri_exp/generate_stimuli/'
ACT_DIR = '/data0/Dropbox/quilts/network_activations'
OUT_DIR = '/data0/Dropbox/quilts/analysis/fmri_similarity/quilted_activities'
# ACT_DIR = {'enu': '/media/jessica/My Book/activations/enu/enu-enu-freeze/enu-enu-freeze',
#             'deu': '/data0/Dropbox/quilts/network_activations/deu/enu-enu-freeze/enu-enu-freeze',
#             'nld': '/media/jessica/My Book/activations/nld/enu-enu-freeze/enu-enu-freeze'}

class Quilter():

    def __init__(self, input_lang, model):
        self.input_lang = input_lang
        self.model = model
        audio_len_file = '/data0/Dropbox/quilts/analysis/fmri_similarity/{}_audio_lengths.pkl'.format(input_lang)
        with open(audio_len_file, 'rb') as af:
            self.len_audio = pickle.load(af)
        self.out_dir = '{}/{}/{}'.format(OUT_DIR, input_lang, model)


    def quilt_all_speakers(self):
        """Given a lanaguage and model, save all activation quilts for all speakers."""
        with open('{}/{}/selected_speakers.txt'.format(STIM_DIR, self.input_lang), 'r') as fil:
            speakers = [name.split('.')[0] for name in fil.read().splitlines()]

        for n_spkr, spkr in enumerate(speakers):
            print('Processing speaker: {} {}/{}     \r\r'.format(spkr, n_spkr+1, len(speakers)))
            tic = time.time()
            self.quilt_network_activations(speaker=spkr)
            toc = time.time()
            print("Quilting took {} seconds".format(toc - tic))


    def quilt_network_activations(self, speaker='inh_100cor_1'):
        """Save the 1s quilted network activations for a given speaker, model and language."""
        all_act, time_per_frame = self.segment_activations(speaker)

        # Load the order of segments used when generating stimuli quilts
        mat_file = '{}/{}/quilts/order/{}_60s_order.mat'.format(STIM_DIR, self.input_lang, speaker)
        mat = loadmat(mat_file)
        seg_order = mat['final_seg_order'] - 1  # to correct for MATLAAB 1-based indexing

        # Concatenate segments according to the quilt order
        quilted_acts = {}
        for layer in LAYERS:
            # Initialize with first segment
            quilted_acts[layer] = all_act[layer][0]
            for i in seg_order[0][1:]:
                quilted_acts[layer] = np.append(quilted_acts[layer], all_act[layer][i], axis=0)
        for layer in LAYERS:
            quilted_acts[layer] = quilted_acts[layer][:int(round(QUILT_LEN/np.mean(time_per_frame))), :]
        # Save the quilted network activations
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)
        out_file = '{}/{}_quilted.pkl'.format(self.out_dir, speaker)
        with open(out_file, 'wb') as f:
            pickle.dump(quilted_acts, f, pickle.HIGHEST_PROTOCOL)


    def segment_activations(self, speaker):
        """Split network activations into segments according to saved boundaries."""
        all_act = {layer:[] for layer in LAYERS}
        total_frames = 0
        time_per_frame = []
        total_segments = 0
        with open('{}/{}/proposed_utts/{}.txt'.format(STIM_DIR, self.input_lang, speaker), 'r') as fil:
            utterance_nos = fil.read().splitlines()
        # Divide the activations into the segments that were used in the speech quilts
        for utt_no in utterance_nos:
            # print('utterance number:', utt_no)
            # Load and concatenate network activations
            act = np.load('{0}/{1}/{2}/{3}/{3}_{4}.npz'.format(ACT_DIR, self.input_lang,
                                                               self.model, speaker,
                                                               utt_no))
            n_frames = act['cnn1'].shape[0]
            total_frames = total_frames + n_frames
            # Load and concatenate segmentation info (in ms)
            seg = np.load('{0}/{1}/seg/{2}/{2}_{3}.npy'.format(STIM_DIR, self.input_lang,
                                                               speaker, utt_no),
                          encoding='latin1', allow_pickle=True)
            length_ms = self.len_audio[speaker][utt_no]
            time_per_frame.append((length_ms)/n_frames)

            # Convert ms boundaries to indices into activations
            phn_start = np.round((seg[()]['phn_start']/length_ms)*n_frames).astype(int)
            total_segments = total_segments + len(phn_start)
            phn_end = np.round(seg[()]['phn_end']/length_ms*n_frames).astype(int)
            for layer in LAYERS:
                segs = [act[layer][start:stop, :] for start, stop in zip(phn_start, phn_end)]
                all_act[layer] = all_act[layer] + segs
        assert len(all_act['cnn1']) == total_segments
        return all_act, time_per_frame


def main():
    """Process input arguments and call quilter."""
    args = sys.argv
    if len(args) < 3:
        print("Usage: quilt_network_activity.py input_language model_name")
        # e.g. quilt_network_activity.py enu enu-enu-freeze/enu-enu-freeze
    else:
        input_lang = args[1]
        model = args[2]
        quilter = Quilter(input_lang, model)
        quilter.quilt_all_speakers()

if __name__ == '__main__':
    main()