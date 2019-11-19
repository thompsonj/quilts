"""This module provides the SimilarityMatrix class for using CKA to compare
activations from biological and artificial neural networks."""
import os
import sys
import time
import pickle
import pandas
import multiprocessing as mp
import numpy as np
from nistats.hemodynamic_models import glover_hrf
import matplotlib.pyplot as plt
# from scipy.stats import norm
# import nibabel as nib
import cka  # Simon Kornblith's colab 
#https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb

#  Paths
DATA_DIR = '/data1/quilts/speechQuiltsfMRIdata'
HOME = '/data0/Dropbox/quilts/analysis/fmri_similarity'
QUILT_DIR = HOME + '/quilted_activities'
RESULTS_DIR = HOME + '/similarity_matrices'
ROI_FILE = DATA_DIR + '/derivatives/nilearn/sub-{0}/ses-{1}/sub-{0}_ses-{1}_task-QuiltLanguage_run-{2}_desc-preproc_bold_{3}.npy'
EVENTS_FILE = '{0}/sub-{1}/ses-{2}/func/sub-{1}_ses-{2}_task-QuiltLanguage_run-{3}_events.tsv'
# Constants
TR = 1.7
FRAME_RATE = 0.020
LAYERS = ['cnn1', 'cnn2', 'cnn3', 'cnn4', 'cnn5', 'cnn6', 'cnn7', 'cnn8', 'cnn9', 'fc1', 'fc2']
DIMS = [1024, 256, 256, 128, 128, 128, 64, 64, 64, 600, 190]
DIMS = {LAYERS[i]:DIMS[i] for i in range(len(LAYERS))}
# TEST_LAYERS = ['cnn8', 'cnn9']
# LAYERS=TEST_LAYERS
ROIS = ['LIC', 'RIC', 'LMGN', 'RMGN']
SLICE_TIME_REF = 0
N_EXAMPLES = 609300


class SimilarityMatrix:
    '''
    The SimilarityMatrix object calculates a CKA similarity matrix for a particular subject and model.
    '''

    def __init__(self, subject, model, rois, overwrite):
        self.model = model
        self.subject = subject
        self.rois = rois
        self.n_rois = len(rois)
        self.n_layers = len(LAYERS)
        self.res_dir = '{}/sub-{}/{}'.format(RESULTS_DIR, self.subject, self.model.split('/')[0])
        self.file = '{}/sub-{}_model-{}_similarity_matrix.npy'.format(self.res_dir,
                                                                      self.subject,
                                                                      self.model.replace('/', '_'))
        if os.path.isfile(self.file) and not overwrite:
            self.sim_mat = np.load(self.file)
        else:
            self.sim_mat = np.zeros((self.n_rois, self.n_layers))


    def calculate_similarity_matrix(self):
        pool = mp.Pool(mp.cpu_count())
        rl_pairs = [(i*(self.n_layers) + j, roi, layer) for i, roi in enumerate(self.rois) for j, layer in enumerate(LAYERS)]
        res_objs = [pool.apply_async(self.run_cka, args=roi_layer) for roi_layer in rl_pairs]
        # res_objs is a list of pool.ApplyResult objects
        results = [r.get() for r in res_objs]
        pool.close()
        print(results)
        results.sort(key=lambda x: x[0])
        results_final = np.array([r for i, r in results])
        self.sim_mat = np.reshape(results_final, (self.n_rois, self.n_layers))

        # Save similarity matrix
        self.save_sim_mat()
        self.plot_sim_mat()


    def save_sim_mat(self):
        # Create the results directory if it does not exist
        if not os.path.isdir(self.res_dir):
            os.makedirs(self.res_dir)
        np.save(self.file, self.sim_mat)


    def plot_sim_mat(self):
        # Create the results directory if it does not exist
        if not os.path.isdir(self.res_dir):
            os.makedirs(self.res_dir)
        fig = plt.figure()
        im = plt.imshow(self.sim_mat.T)
        plt.ylabel('Network Layers')
        plt.yticks(np.arange(len(LAYERS)), LAYERS)
        plt.xlabel('Auditory Regions of Interest')
        plt.xticks(np.arange(self.n_rois), self.rois)
        fig.colorbar(im)
        save_file = '{}/sub-{}_model-{}_similarity_matrix.png'.format(self.res_dir,
                                                                      self.subject,
                                                                      self.model.replace('/', '_'))
        plt.savefig(save_file, bbox_inches='tight', dpi=300)


    def run_cka(self, mp_i, roi, layer):
        # Initialize matrices to avoid growing them in a loop
        act_allruns = np.zeros((N_EXAMPLES, DIMS[layer]))
        roi_size = self.get_roi_size(roi)
        fmri_allruns = np.zeros((N_EXAMPLES, roi_size))
        # Fill matrices with concatenated activities from all 20 runs
        start = 0
        for ses in [1, 2]:
            for run in range(1, 11):
                tic = time.time()
                fmri_1run, act_1run = self.get_X_and_Y_matrices(ses, run, roi, layer)
                toc = time.time()
                print('Preparing matrices took {} seconds.'.format(toc - tic))
                len_this_run = fmri_1run.shape[0]
                end = start + len_this_run
                fmri_allruns[start:end, :] = fmri_1run
                act_allruns[start:end, :] = act_1run
                start += len_this_run
                # if ses==1 and run==1:
                #     fmri_allruns, act_allruns = self.get_X_and_Y_matrices(ses, run, roi, layer)
                # else:
                #     fmri_1run, act_1run = self.get_X_and_Y_matrices(ses, run, roi, layer)
                #     fmri_allruns = np.append(fmri_allruns, fmri_1run, axis=0)
                #     act_allruns = np.append(act_allruns, act_1run, axis=0)
        # print('act_allruns.shape: {}'.format(act_allruns.shape))
        # print('fmri_allruns.shape: {}'.format(fmri_allruns.shape))
        # Calculate debiased linear Centered Kernel Alignment
        cka_from_features = cka.feature_space_linear_cka(act_allruns, fmri_allruns, debiased=True)
        return (mp_i, cka_from_features)


    def get_roi_size(self, roi):
        fmri = np.load(ROI_FILE.format(self.subject, 1, 1, roi))
        return fmri.shape[1]

    def get_X_and_Y_matrices(self, ses, run, roi, layer):
        # Load fmri from one ROI and one run
        fmri = np.load(ROI_FILE.format(self.subject, ses, run, roi))
        # bold = nib.load(bold_file.format(DATA_DIR, sub, ses, run))
        # Resample to 20ms sample rate
        n_scan = fmri.shape[0]
        start_time = SLICE_TIME_REF * TR
        end_time = (n_scan - 1 + SLICE_TIME_REF) * TR
        frame_times = np.linspace(start_time, end_time, n_scan)
        len(frame_times)
        fmri_df = pandas.DataFrame(fmri, index=pandas.to_timedelta(frame_times, unit='s'))
        fmri_df_20 = fmri_df.resample('20ms').pad()
        len(fmri_df_20 )
        n_rows = len(fmri_df_20)
        # Load events file
        events = pandas.read_csv(EVENTS_FILE.format(DATA_DIR, self.subject, ses, run), sep='\t')
        events['onset'] = pandas.to_timedelta(events['onset'], unit='s')
        events['duration'] = pandas.to_timedelta(events['duration'], unit='s')
        # Get the event info for all runs (what stimuli were played when)
        langs = [x.split('/')[1] for x in events['stim_file']]
        spkrs = [x.split('/')[-1].split('_60s')[0] for x in events['stim_file']]

        # Load quilted activations
        n_stim = len(spkrs)
        assert n_stim == 9
        quilted_acts = {'enu':{}, 'deu':{}, 'nld':{}}
        for stim in range(n_stim):
            quilt_file = '{}/{}/{}/{}_quilted.pkl'.format(QUILT_DIR, langs[stim],
                                                          self.model, spkrs[stim])
            with open(quilt_file, 'rb') as qfile:
                quilted_acts[langs[stim]][spkrs[stim]] = pickle.load(qfile)

        # Assemble quilted activations in dataframe of same length as fmri
        # layer = 'cnn9'
        dim = quilted_acts[langs[0]][spkrs[0]][layer].shape[1]
        activities = pandas.DataFrame(np.zeros((n_rows, dim)), index=fmri_df_20.index)
        for stim in range(n_stim):
            onset = events['onset'][stim]
            len_acts = quilted_acts[langs[stim]][spkrs[stim]][layer].shape[0]
            offset = onset + pandas.to_timedelta(.020*len_acts, unit='s')
            len1 = activities[onset:offset].shape[0]
            len2 = quilted_acts[langs[stim]][spkrs[stim]][layer].shape[0]
            while len1 > len2:
                offset -= pandas.to_timedelta(0.005, unit='s')
                len1 = activities[onset:offset].shape[0]
            # print('len1: {}, len2: {}'.format(len1, len2))
            activities[onset:offset] = quilted_acts[langs[stim]][spkrs[stim]][layer]
            len(activities)
            # Apply HRF to activations
            hrf = glover_hrf(FRAME_RATE, oversampling=1, time_length=32., onset=0.)
            # plt.plot(hrf)
            # plt.plot(activities[0])
            activities_hrf = activities.apply(np.convolve, args=(hrf,), axis=0)
            len(activities_hrf)
            X = fmri_df_20.to_numpy()
            Y = activities_hrf.to_numpy()
            Y = Y[:X.shape[0], :]
            # X.shape
            # Y.shape
        # plt.plot(activities_hrf[0])
        return X, Y

def main():
    """Process input arguments and call quilter."""
    args = sys.argv
    if len(args) < 3:
        print("Usage: cka_similarity.py subject model_name \n e.g. ")
        # e.g. cka_similarity.py 4 enu-enu-freeze/enu-enu-freeze
    else:
        # rois = ['LIC', 'RIC', 'LMGN', 'RMGN']
        rois = ROIS
        subject = args[1]
        model = args[2]
        sim_mat = SimilarityMatrix(subject, model, rois, True)
        tic = time.time()
        sim_mat.calculate_similarity_matrix()
        toc = time.time()
        print("Calculating sim mat took {} minutes".format((toc - tic)/60))

if __name__ == '__main__':
    main()

# sim_mat = SimilarityMatrix('4', 'enu-enu-freeze/enu-enu-freeze', ['RIC'], False)
# sim_mat.sim_mat
# sim_mat.plot_sim_mat()
