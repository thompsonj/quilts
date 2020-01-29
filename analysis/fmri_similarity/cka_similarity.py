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
from scipy.stats import describe
# from scipy.stats import norm
# import nibabel as nib
import cka  # Simon Kornblith's colab 
#https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb

#  Paths
DATA_DIR = '/data1/quilts/speechQuiltsfMRIdata'
HOME = '/data0/Dropbox/quilts/analysis/fmri_similarity'
QUILT_DIR = HOME + '/quilted_activities'
RESULTS_DIR = HOME + '/similarity_matrices'
# ROI_FILE = DATA_DIR + '/derivatives/nilearn/sub-{0}/ses-{1}/sub-{0}_ses-{1}_task-QuiltLanguage_run-{2}_desc-preproc_bold_{3}.npy'
ROI_FILE = DATA_DIR + '/derivatives/nilearn/sub-{0}/ses-{1}/sub-{0}_ses-{1}_task-QuiltLanguage_run-{2}_mask-{3}_desc-preproc_bold.npy'
EVENTS_FILE = '{0}/sub-{1}/ses-{2}/func/sub-{1}_ses-{2}_task-QuiltLanguage_run-{3}_events.tsv'
# Constants
TR = 1.7
FRAME_RATE = 0.020
N_FRAMES_PER_QUILT = 2950
LAYERS = ['input', 'cnn1', 'cnn2', 'cnn3', 'cnn4', 'cnn5', 'cnn6', 'cnn7', 'cnn8', 'cnn9', 'fc1', 'fc2']
DIMS = [45, 1024, 256, 256, 128, 128, 128, 64, 64, 64, 600, 190]
LAYER_SIZES = {LAYERS[i]:DIMS[i] for i in range(len(LAYERS))}
N_LAYERS = len(LAYERS)
# TEST_LAYERS = ['cnn8', 'cnn9']
# LAYERS=TEST_LAYERS
ROI_ORDER = ['LCN', 'RCN', 'LSOC', 'RSOC', 'LIC', 'RIC', 'LMGN', 'RMGN', 'HG', 'PP', 'PT', 'STGa', 'STGp']
MERGED_ROI_ORDER = ['CN', 'SOC', 'IC', 'MGN', 'HG', 'PP', 'PT', 'STGa', 'STGp']
SLICE_TIME_REF = 0
# N_EXAMPLES = 609300
N_EXAMPLES = 610980


class SimilarityMatrix:
    '''
    The SimilarityMatrix object calculates a CKA similarity matrix for a particular subject and model.
    '''

    def __init__(self, subject, model, rois, shuffle):
        """Loads the saved SimilarityMatrix object if it exists and overwrite is not True. 
        Otherwise initializes a set of variables associated with the preparation of a single similarity matrix.
        """

        self.model = model
        self.subject = subject
        self.res_dir = '{}/sub-{}/{}'.format(RESULTS_DIR, self.subject, self.model.split('/')[0])
        # Create the results directory if it does not exist
        if not os.path.isdir(self.res_dir):
            os.makedirs(self.res_dir)
        self.file = '{}/sub-{}_model-{}_similarity_matrix_610980'.format(self.res_dir,
                                                                      self.subject,
                                                                      self.model.replace('/', '_'))
        self.shuffle = False
        self.shuffle_run = False
        if shuffle == 'shuffle':
            self.file = self.file + '_shuffled'
            self.shuffle = True
        if shuffle == 'shuffle_run':
            self.file = self.file + '_shuffle_run'
            self.shuffle_run = True


        self.rois = rois
        self.new_rois = rois
        self.merged_rois = [roi for roi in rois if roi in MERGED_ROI_ORDER]
        self.roi_sizes = {roi:self.get_roi_size(roi) for roi in self.rois} 
        self.n_rois = len(rois)
        self.layer_sizes = LAYER_SIZES
        self.similarities = {roi:np.ones((N_LAYERS))*-1 for roi in self.rois}
        # similarities = {roi:np.ones((len(LAYERS)))*-1 for roi in ['RIC']}
        self.sim_mat = np.ones((N_LAYERS, len(self.merged_rois)))*-1
        self.max_similarity = None
        self.min_similarity = None
        self.network_activities = {layer:np.zeros((N_EXAMPLES, self.layer_sizes[layer])) for layer in LAYERS}
        self.fmri_activities = {roi:np.zeros((N_EXAMPLES, self.roi_sizes[roi])) for roi in self.new_rois}
        
        self.fmri_stats = {}
        self.net_stats = {}
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle baz
        del state["network_activities"]
        del state["fmri_activities"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Add baz back since it doesn't exist in the pickle
        self.network_activities = {layer:np.zeros((N_EXAMPLES, self.layer_sizes[layer])) for layer in LAYERS}
        self.fmri_activities = {roi:np.zeros((N_EXAMPLES, self.roi_sizes[roi])) for roi in self.new_rois}


    def calculate_similarity_matrix(self):

        # Load all data
        self.load_data()
        self.center_data()
        # print(self.network_activities)
        # print(self.fmri_activities)
        # if len(self.new_rois) > 1 or self.new_rois[0] != 'STGp':
            # Don't bother running the test if STGp is the only new roi
            # self.test_data()

        for roi in self.new_rois:
            for j, layer in enumerate(LAYERS):
                # Check for NaNs
                if np.isnan(self.network_activities[layer]).any():
                    print('Layer {} contains NaNs.'.format(layer))
                if np.isnan(self.fmri_activities[roi]).any():
                    print('ROI {} contains NaNs.'.format(roi))
                # Call CKA
                sim = cka.feature_space_linear_cka(self.network_activities[layer],
                                                   self.fmri_activities[roi], debiased=True)
                print(sim)
                # Store similarity value and update min and max
                self.similarities[roi][j] = sim
                if self.max_similarity is None or sim > self.max_similarity:
                    self.max_similarity = sim
                if self.min_similarity is None or sim < self.min_similarity:
                    self.min_similarity = sim
            # To free memory, get rid of the fmri data once you're done with it
            self.fmri_activities[roi] = []


        # rl_pairs = [(i*(N_LAYERS) + j, roi, layer) for i, roi in enumerate(self.rois) for j, layer in enumerate(LAYERS)]
        # rl_pairs = [(i, j, roi, layer) for i, roi in enumerate(self.new_rois) for j, layer in enumerate(LAYERS)]
        # print(rl_pairs)


        # pool = mp.Pool(mp.cpu_count())
        # res_objs = [pool.apply_async(self.run_cka, args=(i, j, roi, layer)) for (i, j, roi, layer) in rl_pairs]
        # # res_objs is a list of pool.ApplyResult objects
        # results = [r.get() for r in res_objs]
        # for _, j, roi, layer, sim in results:
        #     self.similarities[roi][j] = sim
        #     if self.max_similarity is None or sim > self.max_similarity:
        #         self.max_similarity = sim
        #     if self.min_similarity is None or sim < self.min_similarity:
        #         self.min_similarity = sim
        # pool.close()
        # print(results)

        # results.sort(key=lambda x: x[0])
        # results_final = np.array([r for i, r in results])
        # self.similarities = np.reshape(results_final, (self.n_rois, N_LAYERS))

        # Save similarity matrix
        self.save_sim_mat()
        self.plot_sim_mat()
        
    def test_data(self):
        n_bins = 50
        if not self.net_stats:
            fig, axs = plt.subplots(len(LAYERS), 6, figsize=(9, 14), facecolor='w', edgecolor='k')
            plt.tick_params(axis='both', which='minor', labelsize=8)
            # for ax in axs.ravel():
            #     ax.tick_params(axis='both', which='major', labelsize=10)
            
            # CNN
            for i, layer in enumerate(LAYERS):
                # Check for NaNs and Infs
                n_nans = np.isnan(self.network_activities[layer]).sum()
                n_infs = np.isinf(self.network_activities[layer]).sum()
                if n_nans or n_infs:
                    print('Layer {} contains {} NaNs and {} Infs.'.format(layer, n_nans, n_infs))
                    np.nan_to_num(self.network_activities[layer], copy=False)

                # if not layer in self.net_stats:
                print('Calculating descriptive stats...')
                nobs, (mn, mx), meen, var, skw, kur = describe(self.network_activities[layer])
                self.net_stats[layer] = {'nobs':nobs, 'min':mn, 'max':mx,
                                         'mean':meen, 'var':var, 'skew':skw, 'kurt':kur}
                # else:
                #     nobs = self.net_stats[layer]['nobs']
                #     mn = self.net_stats[layer]['min']
                #     mx = self.net_stats[layer]['max']
                #     meen = self.net_stats[layer]['mean']
                #     var = self.net_stats[layer]['var']
                #     skw = self.net_stats[layer]['skew']
                    # kur = self.net_stats[layer]['kurt']
                axs[i, 0].hist(mn, bins=n_bins)
                axs[i, 0].set_ylabel(layer)
                axs[i, 0].set_yticks([])
                axs[i, 1].hist(mx, bins=n_bins)
                axs[i, 1].set_yticks([])
                axs[i, 2].hist(meen, bins=n_bins)
                axs[i, 2].set_yticks([])
                # axs[i, 3].hist(var, bins=n_bins)
                axs[i, 3].set_yticks([])
                axs[i, 4].hist(skw, bins=n_bins)
                axs[i, 4].set_yticks([])
                axs[i, 5].hist(kur, bins=n_bins)
                axs[i, 5].set_yticks([])
                if i == 0:
                    axs[i, 0].set_title('Min')
                    axs[i, 1].set_title('Max')
                    axs[i, 2].set_title('Mean')
                    axs[i, 3].set_title('Variance')
                    axs[i, 4].set_title('Skew')
                    axs[i, 5].set_title('Kurtosis')
                print('Layer {0}: nobs={1}. min={2:.3f}. max={3:.3f}, mean={4:.3f}, var={5:.5f}, skw={6:.3f}, kur={7:.3f}'.format(layer, nobs, min(mn), max(mx), meen.mean(), var.mean(), skw.mean(), kur.mean()))
                print('mean.shape: {}'.format(meen.shape))
            plt.subplots_adjust(hspace=.75)
            plt.savefig(self.res_dir + '/network_summary.png', dpi=300)
            plt.close()

        # fMRI
        # fig2, axs2 = plt.subplots(len(self.rois), 6, figsize=(9, 14), facecolor='w', edgecolor='k')
        # # for ax in axs2.ravel():
        # #     ax.tick_params(axis='both', which='major', labelsize=10)
        # plt.tick_params(axis='both', which='minor', labelsize=8)
        # for j, roi in enumerate(self.rois):
        #     if roi == 'STGp':
        #         break
        #     if not roi in self.fmri_stats:
        #         nobs, (mn, mx), meen, var, skw, kur = describe(self.fmri_activities[roi])
        #         self.fmri_stats[roi] = {'nobs':nobs, 'min':mn, 'max':mx,
        #                                 'mean':meen, 'var':var, 'skew':skw, 'kurt':kur}
        #     else:
        #         nobs = self.fmri_stats[roi]['nobs']
        #         mn = self.fmri_stats[roi]['min']
        #         mx = self.fmri_stats[roi]['max']
        #         meen = self.fmri_stats[roi]['mean']
        #         var = self.fmri_stats[roi]['var']
        #         skw = self.fmri_stats[roi]['skew']
        #         kur = self.fmri_stats[roi]['kurt']
        #     axs2[j, 0].hist(mn, bins=n_bins)
        #     axs2[j, 0].set_ylabel(roi)
        #     axs2[j, 0].set_yticks([])
        #     axs2[j, 1].hist(mx, bins=n_bins)
        #     axs2[j, 1].set_yticks([])
        #     axs2[j, 2].hist(meen, bins=n_bins)
        #     axs2[j, 2].set_yticks([])
        #     axs2[j, 3].hist(var, bins=n_bins)
        #     axs2[j, 3].set_yticks([])
        #     axs2[j, 4].hist(skw, bins=n_bins)
        #     axs2[j, 4].set_yticks([])
        #     axs2[j, 5].hist(kur, bins=n_bins)
        #     axs2[j, 5].set_yticks([])
        #     if j == 0:
        #         axs2[j, 0].set_title('Min')
        #         axs2[j, 1].set_title('Max')
        #         axs2[j, 2].set_title('Mean')
        #         axs2[j, 3].set_title('Variance')
        #         axs2[j, 4].set_title('Skew')
        #         axs2[j, 5].set_title('Kurtosis')
        # 
        #     print('ROI {0}: nobs={1}. min={2:.3f}. max={3:.3f}, mean={4:.3f}, var={5:.5f}, skw={6:.3f}, kur={7:.3f}'.format(roi, nobs, min(mn), max(mx), meen.mean(), var.mean(), skw.mean(), kur.mean()))
        #     print('mean.shape: {}'.format(meen.shape))
        # plt.savefig(self.res_dir + '/fmri_summary.png', dpi=300)

    def run_cka(self, i, j, roi, layer):
        # Calculate debiased linear Centered Kernel Alignment
        
        cka_from_features = cka.feature_space_linear_cka(self.network_activities[layer],
                                                         self.fmri_activities[roi], debiased=True)
        return (i, j, roi, layer, cka_from_features)

    def load_data(self):
        tic = time.time()
        indexes = {1:{}, 2:{}}
        # FMRI
        print('Loading fMRI...')
        for roi in self.new_rois:
            if self.model == 'random':
                # print('self.model is "random"')
                self.fmri_activities[roi] = np.random.rand(N_EXAMPLES, self.roi_sizes[roi])
            else:
                # Fill matrices with concatenated activities from all 20 runs
                start = 0
                for ses in [1, 2]:
                    for run in range(1, 11):
                        fmri_1run, indexes[ses][run] = self.get_fmri_matrix_per_run(ses, run, roi)
                        # print('Preparing matrices took {} seconds.'.format(toc - tic))
                        len_this_run = fmri_1run.shape[0]
                        end = start + len_this_run
                        self.fmri_activities[roi][start:end, :] = fmri_1run
                        start += len_this_run


        # CNN
        print('Loading CNN activations...')
        # for layer in LAYERS:
            # print("Layer: {}".format(layer))
        if self.model == 'random':
            # print('self.model is "random"')
            self.network_activities = {layer:np.random.rand(N_EXAMPLES, self.layer_sizes[layer]) for layer in LAYERS}
        else:

            act_file = '{}/design_matrices/{}/{}_convolved.npy'
            if self.shuffle_run:
                act_file = '{}/design_matrices/{}/{}_convolved_shuffle_run.npy'
            if os.path.isfile(act_file.format(QUILT_DIR, self.model, 'input')):  # Load matrix if it exists
                print('Loading previously saved convolved activations...')
                for layer in LAYERS:
                    self.network_activities[layer] = np.load(act_file.format(QUILT_DIR, self.model, layer))
                    # Shuffle frames if random permutation baseline model
                    if self.shuffle:
                        print('Shuffling...')
                        idx = np.arange(self.network_activities[layer].shape[0])
                        np.random.shuffle(idx)
                        self.network_activities[layer] = self.network_activities[layer][idx, :]
            else:  # Otherwise prepare activations matrixs
                # Fill matrices with concatenated activities from all 20 runs
                start = 0
                for ses in [1, 2]:
                    print('Session: {}'.format(ses))
                    for run in range(1, 11):
                        print('Run: {}'.format(run))
                        act_1run = self.get_cnn_matrix_per_run(ses, run, indexes[ses][run])
                        # print('Preparing matrices took {} seconds.'.format(toc - tic))
                        len_this_run = act_1run['input'].shape[0]
                        end = start + len_this_run
                        for layer in LAYERS:
                            self.network_activities[layer][start:end, :] = act_1run[layer]
                        start += len_this_run
                for layer in LAYERS:
                    # Remove any dead units, if present
                    n_units = self.network_activities[layer].shape[1]
                    zero = [not bool(np.count_nonzero(self.network_activities[layer][:, i])) for i in range(n_units)]
                    n_dead = np.sum(zero)
                    print('Removing {} dead units from layer {}.'.format(n_dead, layer))
                    self.layer_sizes[layer] = n_units - n_dead
                    cols_to_delete = np.where(zero)
                    self.network_activities[layer] = np.delete(self.network_activities[layer], cols_to_delete, 1)
                    # Shuffle frames if random permutation baseline model
                    if self.shuffle:
                        print('Shuffling...')
                        idx = np.arange(self.network_activities[layer].shape[0])
                        np.random.shuffle(idx)
                        self.network_activities[layer] = self.network_activities[layer][idx, :]
                    # Save activations matrix
                    outdir = '{}/design_matrices/{}'.format(QUILT_DIR, self.model)
                    if not os.path.isdir(outdir):
                        os.makedirs(outdir)
                    if self.shuffle:
                        outfile = act_file.format(QUILT_DIR, self.model, layer) + '_shuffle'
                    else:
                        outfile = act_file.format(QUILT_DIR, self.model, layer)
                    print("Saving {}".format(outfile))
                    np.save(outfile, self.network_activities[layer])
        toc = time.time()
        print('Took {} minutes to load all data'.format((toc - tic)/60))
    
    def center_data(self):
        for layer in LAYERS:
            self.network_activities[layer] = self.network_activities[layer] - self.network_activities[layer].mean(axis=0, keepdims=True)
        for roi in self.new_rois:
            self.fmri_activities[roi] = self.fmri_activities[roi] - self.fmri_activities[roi].mean(axis=0, keepdims=True)
    
    def get_fmri_matrix_per_run(self, ses, run, roi):
        # Load fmri from one ROI and one run
        fmri_file = ROI_FILE.format(self.subject, ses, run, roi)
        print('Loading from {}...'.format(fmri_file))
        fmri = np.load(fmri_file)
        # bold = nib.load(bold_file.format(DATA_DIR, sub, ses, run))
        # Resample to 20ms sample rate
        n_scan = fmri.shape[0]
        start_time = SLICE_TIME_REF * TR
        # end_time = (n_scan - 1 + SLICE_TIME_REF) * TR
        end_time = ((n_scan + SLICE_TIME_REF) * TR) - 0.02
        frame_times = np.linspace(start_time, end_time, n_scan)
        # len(frame_times)
        fmri_df = pandas.DataFrame(fmri, index=pandas.to_timedelta(frame_times, unit='s'))
        fmri_df_20 = fmri_df.resample('20ms').pad()
        # len(fmri_df_20 )
        X = fmri_df_20.to_numpy()
        return X, fmri_df_20.index
        
    def get_cnn_matrix_per_run(self, ses, run, index):
        matrices = {}
        n_rows = len(index)
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
        quilts_this_run = {}
        for stim in range(n_stim):
            if self.model == 'random_features':
                quilted_acts[langs[stim]][spkrs[stim]] = {layer:np.random.rand(N_FRAMES_PER_QUILT, self.layer_sizes[layer]) for layer in LAYERS}
            else:
                quilt_file = '{}/{}/{}/{}_quilted.pkl'.format(QUILT_DIR, langs[stim],
                                                              self.model, spkrs[stim])
                with open(quilt_file, 'rb') as qfile:
                    quilt = pickle.load(qfile)
                # for layer in LAYERS:
                #     quilt[layer] = quilt[layer] - quilt[layer].min(axis=1, keepdims=True)
                #     quilt[layer] = quilt[layer] / quilt[layer].max(axis=1, keepdims=True)
                quilted_acts[langs[stim]][spkrs[stim]] = quilt
                for layer in LAYERS:
                    if stim == 0:
                        quilts_this_run[layer] = quilt[layer]
                    else:
                        # print(quilts_this_run[layer].shape)
                        # print(quilt[layer].shape)
                        quilts_this_run[layer] = np.concatenate((quilts_this_run[layer], quilt[layer]), axis=0)

        # Assemble quilted activations in dataframe of same length as fmri
        # layer = 'cnn9'
        for layer in LAYERS:
            if self.shuffle_run:
                print('Shuffling run {} layer {}.'.format(run, layer))
                idx = np.arange(quilts_this_run[layer].shape[0])
                np.random.shuffle(idx)
                quilts_this_run[layer] = quilts_this_run[layer][idx, :]
                start = 0
                for stim in range(n_stim):
                    shape0 = quilted_acts[langs[stim]][spkrs[stim]][layer].shape[0]
                    end = start + shape0
                    quilted_acts[langs[stim]][spkrs[stim]][layer] = quilts_this_run[layer][start:end, :]
                    start += shape0
            dim = quilted_acts[langs[0]][spkrs[0]][layer].shape[1]
            activities = pandas.DataFrame(np.zeros((n_rows, dim)), index=index)
            for stim in range(n_stim):
                minn = np.min(quilted_acts[langs[stim]][spkrs[stim]][layer])
                maxx = np.max(quilted_acts[langs[stim]][spkrs[stim]][layer])
                if minn != 0.0:
                    print('quilt {} {} {} min: {}'.format(langs[stim], spkrs[stim], layer, minn))
                print('quilt {} {} {} max: {}'.format(langs[stim], spkrs[stim], layer, maxx))
                onset = events['onset'][stim]
                len_acts = quilted_acts[langs[stim]][spkrs[stim]][layer].shape[0]
                offset = onset + pandas.to_timedelta(.020*len_acts, unit='s')
                len1 = activities[onset:offset].shape[0]
                len2 = quilted_acts[langs[stim]][spkrs[stim]][layer].shape[0]
                # if len1 != N_FRAMES_PER_QUILT or len2 != N_FRAMES_PER_QUILT:
                    # print('onset:offset yields {} rows while quilted activities have {} for speaker {}'.format(len1, len2, spkrs[stim]))
                while len1 > len2:
                    offset -= pandas.to_timedelta(0.005, unit='s')
                    len1 = activities[onset:offset].shape[0]
                    # print('new len1: {}'.format(len1))
                while len1 < len2:
                    offset += pandas.to_timedelta(0.005, unit='s')
                    len1 = activities[onset:offset].shape[0]
                    # print('new len1: {}'.format(len1))
                # print('len1: {}, len2: {}'.format(len1, len2))

                activities[onset:offset] = quilted_acts[langs[stim]][spkrs[stim]][layer]
                len(activities)
            # Apply HRF to activations
            hrf = glover_hrf(FRAME_RATE, oversampling=1, time_length=32., onset=0.)
            # plt.plot(hrf)
            # plt.plot(activities[0])
            activities_hrf = activities.apply(np.convolve, args=(hrf,), axis=0)
            Y = activities_hrf.to_numpy()
            print('Y convolved min layer-{} ses-{} run-{} min: {}'.format(layer, ses, run, np.min(Y)))
            matrices[layer] = Y[:n_rows, :]

        return matrices
        
    
    def save_sim_mat(self):
        
        col = 0
        for roi in MERGED_ROI_ORDER:
            if roi in self.similarities:
                self.sim_mat[:, col] = self.similarities[roi]
                col += 1
                
        np.save(self.file, self.sim_mat)


    def plot_sim_mat(self):
        """Call this function after save_sim_mat has been called."""
        # Create the results directory if it does not exist
        if not os.path.isdir(self.res_dir):
            os.makedirs(self.res_dir)
        fig = plt.figure()
        im = plt.imshow(self.sim_mat, origin='lower')
        plt.ylabel('Network Layers')
        plt.yticks(np.arange(len(LAYERS)), LAYERS)
        plt.xlabel('Auditory Regions of Interest')
        plt.xticks(np.arange(self.n_rois), self.rois, rotation='vertical')
        plt.title('{} sub-{}'.format(self.model.split('/')[0], self.subject))
        fig.colorbar(im)
        if self.shuffle:
            save_file = '{}/sub-{}_model-{}_similarity_matrix_shuffle.png'.format(self.res_dir,
                                                                          self.subject,
                                                                          self.model.replace('/', '_'))
        elif self.shuffle_run:
            save_file = '{}/sub-{}_model-{}_similarity_matrix_shuffle_run.png'.format(self.res_dir,
                                                                          self.subject,
                                                                          self.model.replace('/', '_'))
        else:
            save_file = '{}/sub-{}_model-{}_similarity_matrix.png'.format(self.res_dir,
                                                                      self.subject,
                                                                      self.model.replace('/', '_'))
        plt.savefig(save_file, bbox_inches='tight', dpi=300)




    # def get_matrices_for_cka(self, roi, layer):
    #     if self.model == 'random':
    #         # print('self.model is "random"')
    #         act_allruns = np.random.rand(N_EXAMPLES, self.layer_sizes[layer])
    #         fmri_allruns = np.random.rand(N_EXAMPLES, self.roi_sizes[roi])
    #     else:
    #         # Initialize matrices to avoid growing them in a loop
    #         act_allruns = np.zeros((N_EXAMPLES, self.layer_sizes[layer]))
    #         roi_size = self.roi_sizes[roi]
    #         fmri_allruns = np.zeros((N_EXAMPLES, roi_size))
    #         # Fill matrices with concatenated activities from all 20 runs
    #         start = 0
    #         for ses in [1, 2]:
    #             for run in range(1, 11):
    #                 tic = time.time()
    #                 fmri_1run, act_1run = self.get_matrices_per_run(ses, run, roi, layer)
    #                 toc = time.time()
    #                 # print('Preparing matrices took {} seconds.'.format(toc - tic))
    #                 len_this_run = fmri_1run.shape[0]
    #                 end = start + len_this_run
    #                 fmri_allruns[start:end, :] = fmri_1run
    #                 act_allruns[start:end, :] = act_1run
    #                 start += len_this_run
    #                 # if ses==1 and run==1:
    #                 #     fmri_allruns, act_allruns = self.get_matrices_per_run(ses, run, roi, layer)
    #                 # else:
    #                 #     fmri_1run, act_1run = self.get_matrices_per_run(ses, run, roi, layer)
    #                 #     fmri_allruns = np.append(fmri_allruns, fmri_1run, axis=0)
    #                 #     act_allruns = np.append(act_allruns, act_1run, axis=0)
    #     # print('act_allruns.shape: {}'.format(act_allruns.shape))
    #     # print('fmri_allruns.shape: {}'.format(fmri_allruns.shape))
    #     return act_allruns, fmri_allruns
        
    def get_roi_size(self, roi):
        fmri = np.load(ROI_FILE.format(self.subject, 1, 1, roi))
        size = fmri.shape[1]
        return size


    def update_rois(self, rois):
        """Identifies any new ROIs for which similarities haven't yet been calculated.
        
        This function is called after loading a previously pickled SimilarityMatrix.
        After calling this method, a call to calculate_similarity_matrix will
        calculate CKA similarity for any newly added ROIs.
        """
        self.new_rois = []
        for roi in rois:
            # print('is {} in {}?'.format(roi, self.rois))
            if roi not in self.rois:
                # print('{} is NOT in {}?'.format(roi, self.rois))
                self.rois.append(roi)
                self.roi_sizes[roi] = self.get_roi_size(roi)
                self.new_rois.append(roi)
                self.similarities[roi] = np.ones((N_LAYERS))
                if roi in MERGED_ROI_ORDER:
                    self.merged_rois.append(roi)
                self.fmri_activities[roi] = np.zeros((N_EXAMPLES, self.roi_sizes[roi]))
                self.n_rois += 1
                
        self.sim_mat = np.ones((N_LAYERS, len(self.merged_rois)))*-1
        print('New ROIs to run CKA on: {}'.format(self.new_rois))
        

    def get_matrices_per_run(self, ses, run, roi, layer):
        # Load fmri from one ROI and one run
        fmri = np.load(ROI_FILE.format(self.subject, ses, run, roi))
        # bold = nib.load(bold_file.format(DATA_DIR, sub, ses, run))
        # Resample to 20ms sample rate
        n_scan = fmri.shape[0]
        start_time = SLICE_TIME_REF * TR
        # end_time = (n_scan - 1 + SLICE_TIME_REF) * TR
        end_time = ((n_scan + SLICE_TIME_REF) * TR) - .02 # subtract one frame rather than one TR: 
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
        # e.g. python cka_similarity.py 4 enu-enu-freeze/enu-enu-freeze [CN,SOC,IC,MGN,HG,PP,PT,STGa]
    else:
        # rois = ['LIC', 'RIC', 'LMGN', 'RMGN']
        subject = args[1]
        model = args[2]
        rois = args[3].strip('[]').replace(" ", "").split(',')
        print('rois in main: {}'.format(rois))
        overwrite = False
        shuffle = None
        if len(args) > 4:
            if 'overwrite' in args:
                overwrite = True
            if 'shuffle' in args:
                print('Shuffle')
                shuffle = 'shuffle'
            if 'shuffle_run' in args:
                print('Shuffle_run')
                shuffle = 'shuffle_run'
            if args[4] == 'True':
                overwrite = True

        sim_mat = SimilarityMatrix(subject, model, rois, shuffle)
        
        # Load previously saved SimilarityMatrix if it exists
        infile = sim_mat.file + '.pickle'
        if os.path.isfile(infile) and not overwrite:
            print("Loading SimilarityMatrix instance from {}".format(infile))
            with open(infile, 'rb') as pkl:
                sim_mat = pickle.load(pkl)
            sim_mat.update_rois(rois)
            print('Number of new ROIS: {}'.format(len(sim_mat.new_rois)))
            if len(sim_mat.new_rois) == 0:
                sim_mat.save_sim_mat()
                sim_mat.plot_sim_mat()
                return

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
