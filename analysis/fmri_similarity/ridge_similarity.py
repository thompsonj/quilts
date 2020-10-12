"""This module provides the SimilarityMatrix class for using CKA to compare
activations from biological and artificial neural networks."""
import os
import sys
import time
import pickle
import pandas
from argparse import ArgumentParser
import numpy as np
from nistats.hemodynamic_models import glover_hrf
import matplotlib.pyplot as plt
from scipy.stats import describe
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

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
SC_ROI_FILE = DATA_DIR + '/derivatives/nilearn/sub-{0}/ses-{1}/sub-{0}_ses-{1}_task-QuiltLanguage_run-{2}_mask-{3}_desc-preproc_bold.npy'
ROI_FILE = DATA_DIR + '/derivatives/nilearn/sub-{0}/ses-{1}/sub-{0}_ses-{1}_task-QuiltLanguage_run-{2}_mask-{3}+active_desc-preproc_bold.npy'
EVENTS_FILE = '{0}/sub-{1}/ses-{2}/func/sub-{1}_ses-{2}_task-QuiltLanguage_run-{3}_events.tsv'
# Constants
TR = 1.7
QUILT_LEN = 59000.0  # each speech quilt is 59 seconds
FRAME_RATE = 0.020  # in seconds, as needed by glover_hrf
N_FRAMES_PER_QUILT = 2950
LAYERS = ['input', 'cnn1', 'cnn2', 'cnn3', 'cnn4', 'cnn5', 'cnn6', 'cnn7',
          'cnn8', 'cnn9', 'fc1', 'fc2']
DIMS = [45, 1024, 256, 256, 128, 128, 128, 64, 64, 64, 600, 190]
LAYER_SIZES = {LAYERS[i]: DIMS[i] for i in range(len(LAYERS))}
N_LAYERS = len(LAYERS)
RUNS = np.arange(1, 11)
# TEST_LAYERS = ['cnn8', 'cnn9']
# LAYERS=TEST_LAYERS
ROI_ORDER = ['LCN', 'RCN', 'LSOC', 'RSOC', 'LIC', 'RIC', 'LMGN', 'RMGN', 'HG',
             'PP', 'PT', 'STGa', 'STGp']
MERGED_ROI_ORDER = ['CN', 'SOC', 'IC', 'MGN', 'HG', 'PP', 'PT', 'STGa', 'STGp']
SLICE_TIME_REF = 0
N_EXAMPLES = 515940
BLOCK_LEN = 8599


class SimilarityMatrix:
    '''
    Calculates a CKA similarity matrix for a particular subject and model.
    '''

    def __init__(self, subject, model, rois, shuffle=None, specifier=None):
        """Initialize SimilarityMatrix object.

        Parameters
        ----------
        subject : str
            Subject identifier. A numeral 1--6
        model : str
            Network model identifier e.g. 'enu-enu-freeze/enu-enu-freeze' or 
            'enu-deu-freeze/enu-deu-freeze'.
        rois : list of str
            One or more ROI identifiers 
            e.g. ['CN', 'SOC', 'IC', 'MGN', 'HG', 'PP', 'PT', 'STGa']
        shuffle : str (Default None)
            Which type of random permutation to perform. 'shuffle' to permute
            all time points. 'shuffle_run' to permute only within a run.
        specifier : str (Default None)
            Additional specifier string to distinguish different analysis
            variations.

        Returns
        -------
        None

        """

        self.model = model
        self.subject = subject
        self.specifier = specifier
        self.metric = 'ridge'
        self.res_dir = '{}/sub-{}/{}'.format(RESULTS_DIR, self.subject, self.model.split('/')[0])
        # Create the results directory if it does not exist
        if not os.path.isdir(self.res_dir):
            os.makedirs(self.res_dir)
        self.file = '{}/sub-{}_model-{}_{}_similarity_matrix_{}'.format(self.res_dir,
                                                                     self.subject,
                                                                     self.model.replace('/', '_'),
                                                                     self.metric,
                                                                     self.specifier)
        self.shuffle = False
        self.shuffle_run = False
        if shuffle == 'shuffle':
            self.file = self.file + '_shuffled'
            self.shuffle = True
        elif shuffle == 'shuffle_run':
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
        # self.net_activities = {layer:np.zeros((N_EXAMPLES, self.layer_sizes[layer])) for layer in LAYERS}
        # self.fmri_activities = {roi:np.zeros((N_EXAMPLES, self.roi_sizes[roi])) for roi in self.new_rois}
        self.net_activities = {layer:{} for layer in LAYERS}
        self.fmri_activities = {roi:{} for roi in self.new_rois}
        self.fmri_stats = {}
        self.net_stats = {}
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle baz
        del state["net_activities"]
        del state["fmri_activities"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Add baz back since it doesn't exist in the pickle
        # self.net_activities = {layer:np.zeros((N_EXAMPLES, self.layer_sizes[layer])) for layer in LAYERS}
        # self.fmri_activities = {roi:np.zeros((N_EXAMPLES, self.roi_sizes[roi])) for roi in self.new_rois}
        self.net_activities = {layer:{} for layer in LAYERS}
        self.fmri_activities = {roi:{} for roi in self.new_rois}

    def calculate_similarity_matrix(self):
        """Fit sklearn Ridge Regression for all new roi-layer pairs.
        
        Loads data, centers data, checks for NaNs, calculates leave out run out
        cross validated Ridge regression R-squared, saves and
        plots similarity matrices of max R-squared per roi.

        Returns
        -------
        None

        """
        # Load all data
        self.load_data()

        # print(self.net_activities)
        # print(self.fmri_activities)
        # if len(self.new_rois) > 1 or self.new_rois[0] != 'STGp':
            # Don't bother running the test if STGp is the only new roi
            # self.test_data()
        reg = Ridge(fit_intercept = False)
        mse = {lay: {roi: np.zeros((20,)) for roi in self.new_rois} for lay in LAYERS}
        r2 = {lay: {roi: np.zeros((20, self.roi_sizes[roi])) for roi in self.new_rois} for lay in LAYERS}
        for roi in self.new_rois:
            print(f'ROI: {roi}')
            for j, layer in enumerate(LAYERS):
                print(f'--Layer: {layer}')
                nvoxel = self.roi_sizes[roi]
                cv_res = np.empty((nvoxel, 20))
                cv_folds = np.arange(1, 21)
                for test_run in cv_folds:
                    test_fmri = self.fmri_activities[roi][test_run]
                    test_cnn = self.net_activities[layer][test_run]
                    train_runs = cv_folds[:]
                    train_runs = np.delete(train_runs, test_run-1)
                    # print(f'Train runs: {train_runs}')
                    train_fmri = self.fmri_activities[roi][train_runs[0]]
                    train_cnn = self.net_activities[layer][train_runs[0]]
                    for run in train_runs[1:]:
                        train_fmri = np.concatenate((train_fmri, self.fmri_activities[roi][run]), axis=0)
                        # import pdb; pdb.set_trace()
                        train_cnn = np.concatenate((train_cnn, self.net_activities[layer][run]), axis=0)
                    # Standardize data based on training data only
                    fmri_mean = train_fmri.mean(axis=0, keepdims=True)
                    net_mean = train_cnn.mean(axis=0, keepdims=True)
                    fmri_var = train_fmri.var(axis=0, keepdims=True)
                    net_var = train_cnn.var(axis=0, keepdims=True)
                    train_fmri = (train_fmri - fmri_mean)/fmri_var
                    test_fmri = (test_fmri - fmri_mean)/fmri_var
                    train_cnn = (train_cnn - net_mean)/net_var
                    test_cnn = (test_cnn - net_mean)/net_var
                    
                
                # Check for NaNs
                # if np.isnan(self.net_activities[layer]).any():
                #     print('Layer {} contains NaNs.'.format(layer))
                # if np.isnan(self.fmri_activities[roi]).any():
                #     print('ROI {} contains NaNs.'.format(roi))
                    reg.fit(train_cnn, train_fmri)
                    pred_fmri = reg.predict(test_cnn)
                    mse[layer][roi][test_run - 1] = mean_squared_error(test_fmri, pred_fmri)
                    r2[layer][roi][test_run - 1, :] = r2_score(test_fmri, pred_fmri, multioutput='raw_values')
                #     for voxel in np.arange(nvoxel):
                #         reg.fit(train_cnn, train_fmri[:, voxel])
                #         # R2[layer][roi][voxel, test_run] 
                #         cv_res[voxel, test_run - 1] = reg.score(test_cnn, test_fmri[:, voxel])
                # # Average over cv_folds
                # R2[layer][roi] = cv_res.mean(axis=1)
                # import pdb; pdb.set_trace()
                sim = r2[layer][roi].mean(axis=0).max()
                print(sim)
                # Store similarity value and update min and max
                self.similarities[roi][j] = sim
                if self.max_similarity is None or sim > self.max_similarity:
                    self.max_similarity = sim
                if self.min_similarity is None or sim < self.min_similarity:
                    self.min_similarity = sim
            # Save all R-squares for all voxels
            rois = '-'.join(self.new_rois)
            with open(f'{self.file}_allR2s_{rois}.pickle', 'wb') as ffile:
                pickle.dump(r2, ffile)
            with open(f'{self.file}_allMSE_{rois}.pickle', 'wb') as ffile:
                pickle.dump(mse, ffile)
            # To free memory, get rid of the fmri data once you're done with it
            self.fmri_activities[roi] = []

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
                n_nans = np.isnan(self.net_activities[layer]).sum()
                n_infs = np.isinf(self.net_activities[layer]).sum()
                if n_nans or n_infs:
                    print('Layer {} contains {} NaNs and {} Infs.'.format(layer, n_nans, n_infs))
                    np.nan_to_num(self.net_activities[layer], copy=False)

                # if not layer in self.net_stats:
                print('Calculating descriptive stats...')
                nobs, (mn, mx), meen, var, skw, kur = describe(self.net_activities[layer])
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


    def run_cka(self, i, j, roi, layer):
        # Calculate debiased linear Centered Kernel Alignment
        
        cka_from_features = cka.feature_space_linear_cka(self.net_activities[layer],
                                                         self.fmri_activities[roi], debiased=True)
        return (i, j, roi, layer, cka_from_features)

    def load_data(self):
        """Prepare the DNN and BOLD activiation matrices on which to run CKA.
        
        Inserts the to-be-analyzed data from both domains into pre-initialized
        instance attributes self.fmri_actvities and self.net_activities.
        
        A number of baseline models are also implemented. The random model 
        replaces the to-be-analyzed data with uniform random data. The 
        'shuffle' condition loads the true data but randomly permutes 
        observations. 'shuffle_run' permutes only within runs.

        Returns
        -------
        None

        """
        tic = time.time()
        indexes = {1:{}, 2:{}}
        # FMRI
        print('Loading fMRI...')
        for roi in self.new_rois:
            if self.model == 'random':
                print('not implemented')
                # self.fmri_activities[roi] = np.random.rand(N_EXAMPLES, self.roi_sizes[roi])
            else:
                # Fill matrices with concatenated activities from all 20 runs
                # start = 0
                run20 = 1
                for ses in [1, 2]:
                    for run in RUNS:
                        # Load data and put in DataFrame
                        fmri_df_20 = self.load_fmri(ses, run, roi)
                        indexes[ses][run] = fmri_df_20.index
                        # Prepare events DataFrame containing onsets/offsets
                        events = self.get_events(ses, run, fmri_df_20.index)
                        # Get the numpy matrix of to-be-analyzed activity
                        fmri_1run = self.get_fmri_matrix(fmri_df_20, events)
                        
                        # fmri_1run, indexes[ses][run] = self.get_fmri_matrix_per_run(ses, run, roi)
                        # print('Preparing matrices took {} seconds.'.format(toc - tic))
                        # len_this_run = fmri_1run.shape[0]
                        # end = start + len_this_run
                        self.fmri_activities[roi][run20] = fmri_1run
                        run20 +=1 
                        # start += len_this_run


        # CNN
        print('Loading CNN activations...')
        # for layer in LAYERS:
            # print("Layer: {}".format(layer))
        if self.model == 'random':
            print('not implemented')
            # self.net_activities = {layer:np.random.rand(N_EXAMPLES, self.layer_sizes[layer]) for layer in LAYERS}
        else:
            act_file = '{}/design_matrices/{}/sub-{}/{}_convolved'
            if self.specifier:
                act_file = act_file + '_' + self.specifier
            if self.shuffle_run:
                act_file = act_file + '_shuffle_run'
            act_file = act_file + '.npy'
            # Load matrix if it exists
            if os.path.isfile(act_file.format(QUILT_DIR, self.model,
                                              self.subject, 'cnn1')):
                print('Loading previously saved convolved activations...')
                for layer in LAYERS:
                    file_ = act_file.format(QUILT_DIR, self.model,
                                            self.subject, layer)
                    # import pdb; pdb.set_trace()
                    self.net_activities[layer] = np.load(file_, allow_pickle=True).item()
                    # Shuffle frames if random permutation baseline model
                    if self.shuffle:
                        print('not implemented')
                        # print('Shuffling...')
                        # idx = np.arange(self.net_activities[layer].shape[0])
                        # np.random.shuffle(idx)
                        # self.net_activities[layer] = self.net_activities[layer][idx, :]
            else:  # Otherwise prepare activations matrixs
                # Fill matrices with concatenated activities from all 20 runs
                # start = 0
                run20 = 1
                for ses in [1, 2]:
                    print('Session: {}'.format(ses))
                    for run in RUNS:
                        print('Run: {}'.format(run))
                        events = self.get_events(ses, run, indexes[ses][run])
                        act_1run = self.get_cnn_matrix(ses, run,
                                                       indexes[ses][run],
                                                       events)
                        # print('Preparing matrices took {} seconds.'.format(toc - tic))
                        # len_this_run = act_1run['input'].shape[0]
                        # end = start + len_this_run
                        for layer in LAYERS:
                            self.net_activities[layer][run20] = act_1run[layer]
                        run20 += 1
                        # start += len_this_run
                for layer in LAYERS:
                    # Remove any dead units, if present
                    n_units = self.net_activities[layer][1].shape[1]
                    zero = [not bool(np.count_nonzero(self.net_activities[layer][1][:, i])) for i in range(n_units)]
                    n_dead = np.sum(zero)
                    print('Removing {} dead units from layer {}.'.format(n_dead, layer))
                    self.layer_sizes[layer] = n_units - n_dead
                    cols_to_delete = np.where(zero)
                    for run in range(1, 21):
                        self.net_activities[layer][run] = np.delete(self.net_activities[layer][run], cols_to_delete, 1)
                    # Shuffle frames if random permutation baseline model
                    if self.shuffle:
                        print('not implemented')
                        # print('Shuffling...')
                        # idx = np.arange(self.net_activities[layer].shape[0])
                        # np.random.shuffle(idx)
                        # self.net_activities[layer] = self.net_activities[layer][idx, :]
                    # Save activations matrix
                    outdir = '{}/design_matrices/{}/sub-{}'.format(QUILT_DIR, self.model, self.subject)
                    if not os.path.isdir(outdir):
                        os.makedirs(outdir)
                    outfile = act_file.format(QUILT_DIR, self.model, self.subject, layer)
                    if self.shuffle: # probably move this up to the other act_file handling
                        outfile = outfile + '_shuffle'
                    print("Saving {}".format(outfile))
                    np.save(outfile, self.net_activities[layer])
        toc = time.time()
        print('Took {} minutes to load all data'.format((toc - tic)/60))
    
    # def center_data(self):
    #     for layer in LAYERS:
    #         self.net_activities[layer] = self.net_activities[layer] - self.net_activities[layer].mean(axis=0, keepdims=True)
    #     for roi in self.new_rois:
    #         self.fmri_activities[roi] = self.fmri_activities[roi] - self.fmri_activities[roi].mean(axis=0, keepdims=True)
    def get_events(self, ses, run, index):
        """Load the events file and calculate stimulus offsets.
        
        Since we are using time indexing with less than infinite precision,
        offsets need to be set appropriately so that the activations for each
        quilt correspond to the same number of rows.

        Parameters
        ----------
        ses : int
            Session identifer (1 or 2).
        run : int
            Run identifier (1--10).
        index : pandas TimeDelta index
            Index from the DataFrame contaning the BOLD activity.

        Returns
        -------
        None

        """
        events = pandas.read_csv(EVENTS_FILE.format(DATA_DIR, self.subject,
                                 ses, run), sep='\t')
        events['onset'] = pandas.to_timedelta(events['onset'], unit='s')
        events['duration'] = pandas.to_timedelta(events['duration'], unit='s')
        events['offset'] = events['onset'] + events['duration']
        n_stim = 9
        mock_df = pandas.DataFrame(index=index)
        for stim in range(n_stim):
            onset = events['onset'][stim]
            offset = onset + pandas.to_timedelta(.020*N_FRAMES_PER_QUILT,
                                                 unit='s')
            len1 = mock_df[onset:offset].shape[0]
            # if len1 != N_FRAMES_PER_QUILT or len_acts != N_FRAMES_PER_QUILT:
                # print('onset:offset yields {} rows while quilted activities have {} for speaker {}'.format(len1, len_acts, spkrs[stim]))
            while len1 > N_FRAMES_PER_QUILT:
                offset -= pandas.to_timedelta(0.005, unit='s')
                len1 = mock_df[onset:offset].shape[0]
            while len1 < N_FRAMES_PER_QUILT:
                offset += pandas.to_timedelta(0.005, unit='s')
                len1 = mock_df[onset:offset].shape[0]
            events['offset'][stim] = offset
        return events

    def load_fmri(self, ses, run, roi):
        """Load and resample the BOLD activity of a single run.

        Parameters
        ----------
        ses : int
            Session identifer (1 or 2).
        run : int
            Run identifier (1--10).
        roi : str
            ROI identifier. One of 'CN', 'SOC', 'MGN', 'IC', 'HG', 'PP', 'PT',
            'STGa', 'STGp'

        Returns
        -------
        pandas DataFrame
            BOLD activity resampled to one frame every 20 ms.

        """
        # Load fmri from one ROI and one run
        if roi in ['CN', 'SOC', 'IC', 'MGN']:
            fmri_file = SC_ROI_FILE.format(self.subject, ses, run, roi)
        else:
            fmri_file = ROI_FILE.format(self.subject, ses, run, roi)
        fmri = np.load(fmri_file)
        # Resample to 20ms sample rate
        n_scan = fmri.shape[0]
        start_time = SLICE_TIME_REF * TR
        end_time = ((n_scan + SLICE_TIME_REF) * TR) - 0.02
        frame_times = np.linspace(start_time, end_time, n_scan)
        fmri_df = pandas.DataFrame(fmri, index=pandas.to_timedelta(frame_times,
                                                                   unit='s'))
        fmri_df_20 = fmri_df.resample('20ms').pad()
        return fmri_df_20

    def get_fmri_matrix(self, fmri_df_20, events):
        """Extract stimulation blocks from fMRI.

        Parameters
        ----------
        fmri_df_20 : pandas DataFrame
             contains fMRI activity for a single run and ROI
        events : pandas DataFrame
            contains stimuli timing and duration info

        Returns
        -------
        numpy array
            to-be-analyzed fMRI activity. Rows are timepoints. Columns are 
            voxels. 
        """
        # Cut out just the time points corresponding to auditory stimulation
        block0 = fmri_df_20[events['onset'][0] + pandas.to_timedelta(6, unit='s'):events['offset'][2]].to_numpy()
        block1 = fmri_df_20[events['onset'][3] + pandas.to_timedelta(6, unit='s'):events['offset'][5]].to_numpy()
        block2 = fmri_df_20[events['onset'][6] + pandas.to_timedelta(6, unit='s'):events['offset'][8]].to_numpy()
        X = np.concatenate([block0[:BLOCK_LEN, :], block1[:BLOCK_LEN, :], block2[:BLOCK_LEN, :]], axis=0)
        return X
        
    def get_cnn_matrix(self, ses, run, index, events):
        """Load quilted activations and arrange in DataFrame matched to fMRI.

        Parameters
        ----------
        ses : int
            MRI experimental session. either 1 or 2
        run : int
            Experimental run. 1-10
        index : pandas.TimeDelta
            The pandas index from the corresponding DataFrame containing the
            BOLD activity

        Returns
        -------
        dict
            One matrix for each layer of the network containing the layer
            activations for a single run

        """
        # print('Start of get_cnn_matrix')
        matrices = {}
        n_rows = len(index)

        # Get the event info for all runs (what stimuli were played when)
        langs = [x.split('/')[1] for x in events['stim_file']]
        spkrs = [x.split('/')[-1].split('_60s')[0] for x in events['stim_file']]

        # Load quilted activations
        n_stim = len(spkrs)
        assert n_stim == 9
        quilted_acts = {'enu': {}, 'deu': {}, 'nld': {}}
        quilts_this_run = {}
        for stim in range(n_stim):
            if self.model == 'random_features':
                quilted_acts[langs[stim]][spkrs[stim]] = {layer:np.random.rand(N_FRAMES_PER_QUILT, self.layer_sizes[layer]) for layer in LAYERS}
            else:
                quilt_file = '{}/{}/{}/{}_quilted.pkl'.format(QUILT_DIR, langs[stim],
                                                              self.model, spkrs[stim])
                print(f'Opening {quilt_file}')
                with open(quilt_file, 'rb') as qfile:
                    quilt = pickle.load(qfile)
                quilted_acts[langs[stim]][spkrs[stim]] = quilt
                for layer in LAYERS:
                    if stim == 0:
                        quilts_this_run[layer] = quilt[layer]
                    else:
                        quilts_this_run[layer] = np.concatenate((quilts_this_run[layer], quilt[layer]), axis=0)

        # Assemble quilted activations in dataframe of same length as fmri
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
            # loop over the 9 stimuli quilts in the run
            for stim in range(n_stim):
                minn = np.min(quilted_acts[langs[stim]][spkrs[stim]][layer])
                maxx = np.max(quilted_acts[langs[stim]][spkrs[stim]][layer])
                onset = events['onset'][stim]
                offset = events['offset'][stim]
                activities[onset:offset] = quilted_acts[langs[stim]][spkrs[stim]][layer]
                len(activities)

            # Apply HRF to activations
            # print('Applying HRF')
            hrf = glover_hrf(FRAME_RATE, oversampling=1, time_length=32.0, onset=0.0)
            # print('After applying HRF')
            # plt.plot(hrf)
            # plt.plot(activities[0])
            activities_hrf = activities.apply(np.convolve, args=(hrf,), axis=0)
            # Convert to timedelta again
            nrows = activities_hrf.shape[0]
            # import pdb; pdb.set_trace()
            # fr_ms = FRAME_RATE*1000
            time = np.arange(0, nrows+1*FRAME_RATE, FRAME_RATE)
            time = time[:nrows]  # to make sure we always have the right length
            assert len(time) == len(activities_hrf.index)
            activities_hrf.index = pandas.to_timedelta(time, unit='s')

            # Cut out just the timepoints corresponding to auditory stimulation
            block0 = activities_hrf[events['onset'][0] + pandas.to_timedelta(6, unit='s'):events['offset'][2]].to_numpy()
            block1 = activities_hrf[events['onset'][3] + pandas.to_timedelta(6, unit='s'):events['offset'][5]].to_numpy()
            block2 = activities_hrf[events['onset'][6] + pandas.to_timedelta(6, unit='s'):events['offset'][8]].to_numpy()
            Y = np.concatenate([block0[:BLOCK_LEN, :], block1[:BLOCK_LEN, :], block2[:BLOCK_LEN, :]], axis=0)

            # Y = activities_hrf.to_numpy()
            # print('Y convolved min layer-{} ses-{} run-{} min: {}'.format(layer, ses, run, np.min(Y)))

            # matrices[layer] = Y[:n_rows, :]
            matrices[layer] = Y

        return matrices

    def save_sim_mat(self):
        """Save the calculated CKA similarity matrix as a numpy array.

        The columns of the similarity matrix are reordered so that every saved
        matrix has the ROIs in the same order, even if they were originally
        calculated in a different order. 

        Returns
        -------
        None

        """
        col = 0
        for roi in MERGED_ROI_ORDER:
            if roi in self.similarities:
                self.sim_mat[:, col] = self.similarities[roi]
                col += 1
        np.save(self.file, self.sim_mat)


    def plot_sim_mat(self):
        """Plot CKA similarity matrix after all values have been calculated.

        This method is called after same_sim_mat has been called and stores the
        resulting figure in the same results directory as the pickled
        SimilarityMatrix.

        Returns
        -------
        None

        """
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
        if self.specifier:
            spec = '_' + self.specifier
        else:
            spec = ''
        if self.shuffle:
            name = '{}/sub-{}_model-{}_{}_similarity_matrix{}_shuffle.png'
            save_file = name.format(self.res_dir, self.subject,
                                    self.model.replace('/', '_'), self.metric, spec)
        elif self.shuffle_run:
            name = '{}/sub-{}_model-{}_{}_similarity_matrix{}_shuffle_run.png'
            save_file = name.format(self.res_dir, self.subject,
                                    self.model.replace('/', '_'), self.metric, spec)
        else:
            name = '{}/sub-{}_model-{}_{}_similarity_matrix{}.png'
            save_file = name.format(self.res_dir, self.subject,
                                    self.model.replace('/', '_'), self.metric, spec)
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
    #             for run in RUNS:
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
        """Returns the number of voxels in the given ROI.
        
        Loads the first run of the first session to access the size of the
        stored numpy array.

        Parameters
        ----------
        roi : str
            ROI identifier (e.g. 'CN', 'HG' pr 'PP')

        Returns
        -------
        int
            Number of voxels in the extracted ROI

        """
        # Subcortical ROIs
        if roi in ['CN', 'SOC', 'IC', 'MGN']:
            fmri = np.load(SC_ROI_FILE.format(self.subject, 1, 1, roi))
        # Cortical ROIs
        else:
            fmri = np.load(ROI_FILE.format(self.subject, 1, 1, roi))
        size = fmri.shape[1]
        return size

    def update_rois(self, rois):
        """Identify ROIs for which similarities haven't yet been calculated.
        
        This function is called after loading a previously pickled
        SimilarityMatrix. After calling this method, a call to
        calculate_similarity_matrix will calculate CKA similarity for any newly
        added ROIs, skipping ROIs for which CKA similarity has already been
        stored.

        Parameters
        ----------
        rois : list of str
            ROI identifiers (e.g. CN, SOC, HG, etc.) to be included in this
            SimilarityMatrix. 

        Returns
        -------
        None

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
                self.fmri_activities[roi] = dict()
                # self.fmri_activities[roi] = np.zeros((N_EXAMPLES, self.roi_sizes[roi]))
                self.n_rois += 1
                
        self.sim_mat = np.ones((N_LAYERS, len(self.merged_rois)))*-1
        print('New ROIs to run CKA on: {}'.format(self.new_rois))

    # def get_matrices_per_run(self, ses, run, roi, layer):
    #     # Load fmri from one ROI and one run
    #     fmri = np.load(ROI_FILE.format(self.subject, ses, run, roi))
    #     # bold = nib.load(bold_file.format(DATA_DIR, sub, ses, run))
    #     # Resample to 20ms sample rate
    #     n_scan = fmri.shape[0]
    #     start_time = SLICE_TIME_REF * TR
    #     # end_time = (n_scan - 1 + SLICE_TIME_REF) * TR
    #     end_time = ((n_scan + SLICE_TIME_REF) * TR) - .02 # subtract one frame rather than one TR: 
    #     frame_times = np.linspace(start_time, end_time, n_scan)
    #     len(frame_times)
    #     fmri_df = pandas.DataFrame(fmri, index=pandas.to_timedelta(frame_times, unit='s'))
    #     fmri_df_20 = fmri_df.resample('20ms').pad()
    #     len(fmri_df_20 )
    #     n_rows = len(fmri_df_20)
    #     # Load events file
    #     events = pandas.read_csv(EVENTS_FILE.format(DATA_DIR, self.subject, ses, run), sep='\t')
    #     events['onset'] = pandas.to_timedelta(events['onset'], unit='s')
    #     events['duration'] = pandas.to_timedelta(events['duration'], unit='s')
    #     # Get the event info for all runs (what stimuli were played when)
    #     langs = [x.split('/')[1] for x in events['stim_file']]
    #     spkrs = [x.split('/')[-1].split('_60s')[0] for x in events['stim_file']]
    # 
    #     # Load quilted activations
    #     n_stim = len(spkrs)
    #     assert n_stim == 9
    #     quilted_acts = {'enu':{}, 'deu':{}, 'nld':{}}
    #     for stim in range(n_stim):
    #         quilt_file = '{}/{}/{}/{}_quilted.pkl'.format(QUILT_DIR, langs[stim],
    #                                                       self.model, spkrs[stim])
    #         with open(quilt_file, 'rb') as qfile:
    #             quilted_acts[langs[stim]][spkrs[stim]] = pickle.load(qfile)
    # 
    #     # Assemble quilted activations in dataframe of same length as fmri
    #     # layer = 'cnn9'
    #     dim = quilted_acts[langs[0]][spkrs[0]][layer].shape[1]
    #     activities = pandas.DataFrame(np.zeros((n_rows, dim)), index=fmri_df_20.index)
    #     for stim in range(n_stim):
    #         onset = events['onset'][stim]
    #         len_acts = quilted_acts[langs[stim]][spkrs[stim]][layer].shape[0]
    #         offset = onset + pandas.to_timedelta(.020*len_acts, unit='s')
    #         len1 = activities[onset:offset].shape[0]
    #         len2 = quilted_acts[langs[stim]][spkrs[stim]][layer].shape[0]
    #         while len1 > len2:
    #             offset -= pandas.to_timedelta(0.005, unit='s')
    #             len1 = activities[onset:offset].shape[0]
    #         # print('len1: {}, len2: {}'.format(len1, len2))
    #         activities[onset:offset] = quilted_acts[langs[stim]][spkrs[stim]][layer]
    #         len(activities)
    #         # Apply HRF to activations
    #         hrf = glover_hrf(FRAME_RATE, oversampling=1, time_length=32., onset=0.)
    #         # plt.plot(hrf)
    #         # plt.plot(activities[0])
    #         activities_hrf = activities.apply(np.convolve, args=(hrf,), axis=0)
    #         len(activities_hrf)
    #         X = fmri_df_20.to_numpy()
    #         Y = activities_hrf.to_numpy()
    #         Y = Y[:X.shape[0], :]
    #         # X.shape
    #         # Y.shape
    #     # plt.plot(activities_hrf[0])
    #     return X, Y

def main():
    """Process input arguments and call quilter."""
    # Set input arguments
    parser = ArgumentParser(description='Calculate CKA similiarity matrix for one convnet and one human subject.')
    # Positional arguments
    parser.add_argument("subject", help="<Required> Subject identifier. A numeral 1--6")
    parser.add_argument("model", help="<Required> Network model identifier e.g. enu-enu-freeze/enu-enu-freeze or enu-deu-freeze/enu-deu-freeze")
    parser.add_argument("rois", nargs='+', help="<Required> One or more ROI identifiers e.g. CN SOC IC MGN HG PP PT STGa")
    # Optional arguments
    parser.add_argument("-o", "--overwrite",
                        help='Whether to overwrite previously saved SimilarityMatrix, if it exists.',
                        dest="overwrite", action='store_true')
    parser.add_argument("-s", "--shuffle", type=str,
                        help="Which type of random permutation to perform. 'shuffle' to permute all time points. 'shuffle_run' to permute only within a run.", 
                        dest="shuffle",
                        default=None)
    parser.add_argument("-p", "--specifier", type=str,
                        help="Additional specifier string to distinguish different analysis variations.",
                        dest="specifier",
                        default=None)
    parser.add_argument("-u", "--subject2", type=int,
                        help="subject with with to compare.", dest="sub2",
                        default=None)
    args = parser.parse_args()

    sim_mat = SimilarityMatrix(args.subject, args.model, args.rois,
                               args.shuffle, args.specifier)
        
    # Load previously saved SimilarityMatrix if it exists
    infile = sim_mat.file + '.pickle'
    if os.path.isfile(infile) and not args.overwrite:
        print("Loading SimilarityMatrix instance from {}".format(infile))
        with open(infile, 'rb') as pkl:
            sim_mat = pickle.load(pkl)
        sim_mat.update_rois(args.rois)
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
