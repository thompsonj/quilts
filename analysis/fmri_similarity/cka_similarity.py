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
LAYER_SIZES = {LAYERS[i]:DIMS[i] for i in range(len(LAYERS))}
N_LAYERS = len(LAYERS)
# TEST_LAYERS = ['cnn8', 'cnn9']
# LAYERS=TEST_LAYERS
ROI_ORDER = ['LCN', 'RCN', 'LSOC', 'RSOC', 'LIC', 'RIC', 'LMGN', 'RMGN', 'HG', 'PP', 'PT', 'STGa', 'STGp']
SLICE_TIME_REF = 0
N_EXAMPLES = 609300


class SimilarityMatrix:
    '''
    The SimilarityMatrix object calculates a CKA similarity matrix for a particular subject and model.
    '''

    def __init__(self, subject, model, rois):
        """Loads the saved SimilarityMatrix object if it exists and overwrite is not True. 
        Otherwise initializes a set of variables associated with the preparation of a single similarity matrix.
        """

        self.model = model
        self.subject = subject
        self.res_dir = '{}/sub-{}/{}'.format(RESULTS_DIR, self.subject, self.model.split('/')[0])
        self.file = '{}/sub-{}_model-{}_similarity_matrix'.format(self.res_dir,
                                                                      self.subject,
                                                                      self.model.replace('/', '_'))


        self.rois = rois
        self.new_rois = rois
        self.roi_sizes = {roi:self.get_roi_size(roi) for roi in self.rois} 
        self.n_rois = len(rois)
        self.layer_sizes = LAYER_SIZES
        self.similarities = {roi:np.ones((N_LAYERS))*-1 for roi in self.rois}
        # similarities = {roi:np.ones((len(LAYERS)))*-1 for roi in ['RIC']}
        self.sim_mat = np.ones((N_LAYERS, self.n_rois))*-1


    def calculate_similarity_matrix(self):
        pool = mp.Pool(mp.cpu_count())
        # rl_pairs = [(i*(N_LAYERS) + j, roi, layer) for i, roi in enumerate(self.rois) for j, layer in enumerate(LAYERS)]
        rl_pairs = [(i,  j, roi, layer) for i, roi in enumerate(self.new_rois) for j, layer in enumerate(LAYERS)]
        print(rl_pairs)
        res_objs = [pool.apply_async(self.run_cka, args=roi_layer) for roi_layer in rl_pairs]
        # res_objs is a list of pool.ApplyResult objects
        results = [r.get() for r in res_objs]
        for res in results:
            _, j, roi, layer, sim = res
            self.similarities[roi][j] = sim
        pool.close()
        print(results)
        # results.sort(key=lambda x: x[0])
        # results_final = np.array([r for i, r in results])
        # self.similarities = np.reshape(results_final, (self.n_rois, N_LAYERS))

        # Save similarity matrix
        self.save_sim_mat()
        self.plot_sim_mat()


    def save_sim_mat(self):
        # Create the results directory if it does not exist
        if not os.path.isdir(self.res_dir):
            os.makedirs(self.res_dir)
        col = 0
        for roi in ROI_ORDER:
            if roi in self.similarities:
                self.sim_mat[:, col] = self.similarities[roi]
                col += 1
                
        np.save(self.file, self.sim_mat)


    def plot_sim_mat(self):
        # Create the results directory if it does not exist
        if not os.path.isdir(self.res_dir):
            os.makedirs(self.res_dir)
        fig = plt.figure()
        im = plt.imshow(self.sim_mat, origin='lower')
        plt.ylabel('Network Layers')
        plt.yticks(np.arange(len(LAYERS)), LAYERS)
        plt.xlabel('Auditory Regions of Interest')
        plt.xticks(np.arange(self.n_rois), self.rois)
        fig.colorbar(im)
        save_file = '{}/sub-{}_model-{}_similarity_matrix.png'.format(self.res_dir,
                                                                      self.subject,
                                                                      self.model.replace('/', '_'))
        plt.savefig(save_file, bbox_inches='tight', dpi=300)


    def run_cka(self, i, j, roi, layer):
        # Prepare input matrices, one for network activations, one for fMRI
        act_allruns, fmri_allruns = self.get_matrices_for_cka(roi, layer)

        # Remove any dead units, if present
        n_units = act_allruns.shape[1]
        zero = [not bool(np.count_nonzero(act_allruns[:, i])) for i in range(n_units)]
        n_dead = np.sum(zero)
        print('Removing {} dead units from layer {}.'.format(n_dead, layer))
        self.layer_sizes[layer] = n_units - n_dead
        cols_to_delete = np.where(zero)
        act_allruns_exfoliated = np.delete(act_allruns, cols_to_delete, 1)
        
        # Calculate debiased linear Centered Kernel Alignment
        cka_from_features = cka.feature_space_linear_cka(act_allruns_exfoliated,
                                                         fmri_allruns, debiased=True)
        return (i, j, roi, layer, cka_from_features)

    def get_matrices_for_cka(self, roi, layer):
        if self.model == 'random':
            # print('self.model is "random"')
            act_allruns = np.random.rand(N_EXAMPLES, self.layer_sizes[layer])
            fmri_allruns = np.random.rand(N_EXAMPLES, self.roi_sizes[roi])
        else:
            # Initialize matrices to avoid growing them in a loop
            act_allruns = np.zeros((N_EXAMPLES, self.layer_sizes[layer]))
            roi_size = self.roi_sizes[roi]
            fmri_allruns = np.zeros((N_EXAMPLES, roi_size))
            # Fill matrices with concatenated activities from all 20 runs
            start = 0
            for ses in [1, 2]:
                for run in range(1, 11):
                    tic = time.time()
                    fmri_1run, act_1run = self.get_matrices_per_run(ses, run, roi, layer)
                    toc = time.time()
                    # print('Preparing matrices took {} seconds.'.format(toc - tic))
                    len_this_run = fmri_1run.shape[0]
                    end = start + len_this_run
                    fmri_allruns[start:end, :] = fmri_1run
                    act_allruns[start:end, :] = act_1run
                    start += len_this_run
                    # if ses==1 and run==1:
                    #     fmri_allruns, act_allruns = self.get_matrices_per_run(ses, run, roi, layer)
                    # else:
                    #     fmri_1run, act_1run = self.get_matrices_per_run(ses, run, roi, layer)
                    #     fmri_allruns = np.append(fmri_allruns, fmri_1run, axis=0)
                    #     act_allruns = np.append(act_allruns, act_1run, axis=0)
        # print('act_allruns.shape: {}'.format(act_allruns.shape))
        # print('fmri_allruns.shape: {}'.format(fmri_allruns.shape))
        return act_allruns, fmri_allruns
        
    def get_roi_size(self, roi):
        fmri = np.load(ROI_FILE.format(self.subject, 1, 1, roi))
        size = fmri.shape[1]
        return size
    
    
    def update_rois(self, rois):
        
        self.new_rois = []
        for roi in rois:
            print('is {} in {}?'.format(roi, self.rois))
            if roi not in self.rois:
                print('{} is NOT in {}?'.format(roi, self.rois))
                self.rois = np.append(self.rois, roi)
                self.roi_sizes[roi] = self.get_roi_size(roi)
                self.new_rois = np.append(self.new_rois, roi)
                self.similarities[roi] = np.ones((N_LAYERS))
                self.n_rois += 1
        self.sim_mat = np.ones((N_LAYERS, self.n_rois))*-1
        print('self.new_rois: {}'.format(self.new_rois))
        

    def get_matrices_per_run(self, ses, run, roi, layer):
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
        subject = args[1]
        model = args[2]
        rois = args[3].strip('[]').replace(" ", "").split(',')
        print('rois in main: {}'.format(rois))
        if len(args) > 4:
            if args[4] == 'True':
                overwrite = True
            else:
                overwrite = False
        else:
            overwrite = False
        sim_mat = SimilarityMatrix(subject, model, rois)
        
        # Load previously saved SimilarityMatrix if it exists
        infile = sim_mat.file + '.pickle'
        if os.path.isfile(infile) and not overwrite:
            print("Loading SimilarityMatrix instance from {}".format(infile))
            with open(infile, 'rb') as pkl:
                sim_mat = pickle.load(pkl)
            sim_mat.update_rois(rois)
                
                
        
        # Calculate similarity matrix: run cka for all roi-layer pairs
        tic = time.time()
        sim_mat.calculate_similarity_matrix()
        toc = time.time()
        print("Calculating sim mat took {} minutes".format((toc - tic)/60))
        
        # Save this SimilarityMatrix instance
        outfile = sim_mat.file + '.pickle'
        if not os.path.isdir(sim_mat.res_dir):
            os.makedirs(sim_mat.res_dir)
        print("Pickling SimilarityMatrix object to {}".format(outfile))
        with open(outfile, 'wb') as pkl:
            pickle.dump(sim_mat, pkl)

if __name__ == '__main__':
    main()
# 
# sim_mat = SimilarityMatrix('4', 'deu-deu-freeze/deu-deu-freeze', ['RIC'])
# sim_mat = SimilarityMatrix('4', 'random', ['RIC'])
# sim_mat.calculate_similarity_matrix()
# sim_mat = SimilarityMatrix('4', 'random', ['RIC'])
# # Load previously saved SimilarityMatrix if it exists
# infile = sim_mat.file + '.pickle'
# if os.path.isfile(infile) and not overwrite:
#     print("Loading SimilarityMatrix instance from {}".format(infile))
#     with open(infile, 'rb') as pkl:
#         sim_mat = pickle.load(infile)
#     sim_mat.update_rois(rois)
# 
# X, Y = sim_mat.get_matrices_for_cka('RIC', 'cnn2')
# 
# outfile = sim_mat.file + '.pickle'
# 
# n_units = X.shape[1]
# zero = [not bool(np.count_nonzero(X[:, i])) for i in range(n_units)]
# n_dead = np.sum(zero)
# print('Removing {} dead units.'.format(n_dead))
# cols_to_del = np.where(zero)
# X_exfoliated = np.delete(X, cols_to_del, 1)
# X_exfoliated.shape
# 
# 
# 
# sim_mat.sim_mat
# X, Y = sim_mat.get_matrices_per_run(1, 1, 'RIC', 'cnn2')
# plt.hist(Y.flatten())
# Y.shape
# nonzero = [True if np.count_nonzero(Y[:,i]) else False for i in range(Y.shape[1])]
# np.count_nonzero(Y[:,9])
# Y[:,9]
# # sim_mat.plot_sim_mat()
