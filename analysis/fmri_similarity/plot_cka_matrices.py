import pickle
import os
import numpy as np
from matplotlib import pyplot as plt
# from cka_similarity import SimilarityMatrix

MERGED_ROI_ORDER = ['CN', 'SOC', 'IC', 'MGN', 'HG', 'PP', 'PT', 'STGa', 'STGp']
# MERGED_ROI_ORDER = ['CN', 'SOC', 'IC', 'MGN', 'HG', 'PP', 'PT', 'STGa']
LAYERS = ['input', 'cnn1', 'cnn2', 'cnn3', 'cnn4', 'cnn5', 'cnn6', 'cnn7', 'cnn8', 'cnn9', 'fc1', 'fc2']
home = os.path.expanduser('~/Dropbox/quilts/analysis/fmri_similarity/similarity_matrices/')
subject_lang = {1:'Dutch', 2:'German', 3:'German', 4:'English', 5:'German', 6:'Dutch'}
minval = {1:-2, 2:-2, 3:-2, 4:-2, 5:-2, 6:-2}
maxval = {1:2, 2:2, 3:2, 4:2, 5:2, 6:2}
# subject_lang = {2:'German', 3:'German', 4:'English', 6:'Dutch'}
# minval = {2:-2, 3:-2, 4:-2, 6:-2}
# maxval = {2:2, 3:2, 4:2, 6:2}
# Load all daya
# Find min and max values over all subjects
simmats = {sub:{} for sub in subject_lang}
simmats_sh = {sub:{} for sub in subject_lang}
simmats_sh_run = {sub:{} for sub in subject_lang}
untrained = {}
rand_acts = {}
full_rand = {}
in_specifier = 'active'
out_specifier = 'active_range_per_sub'
# in_specifier = '610980'
# out_specifier = '610980_active_range_per_sub'

def update_range(min_, max_):
    if min_ < minval[sub]:
        minval[sub] = min_
    if max_ > maxval[sub]:
        maxval[sub] = max_

for sub in subject_lang:
# base1_file = 'fmri_similarity/similarity_matrices/sub-4/untrained/sub-4_model-untrained_untrained_similarity_matrix.pickle'
# with open(base1_file, 'rb') as bsfl:
#     base1 = pickle.load(bsfl)
# enu_file = 'similarity_matrices/sub-4/enu-enu-freeze/sub-4_model-enu-enu-freeze_enu-enu-freeze_similarity_matrix.pickle'
# deu_file = 'similarity_matrices/sub-4/deu-deu-freeze/sub-4_model-deu-deu-freeze_deu-deu-freeze_similarity_matrix.pickle'
# nld_file = 'similarity_matrices/sub-4/nld-nld-freeze/sub-4_model-nld-nld-freeze_nld-nld-freeze_similarity_matrix.pickle'
    untrained_file = home + 'sub-{0}/untrained/sub-{0}_model-untrained_untrained_similarity_matrix_{1}.npy'.format(sub, in_specifier)
    # base1_file = 'analysis/fmri_similarity/similarity_matrices/sub-4/untrained/sub-4_model-untrained_untrained_similarity_matrix.npy'
    untrained[sub] = np.load(untrained_file)
    if minval[sub] == -2:
        minval[sub] = untrained[sub].min()
    if maxval[sub] == 2:
        maxval[sub] = untrained[sub].max()

    # rand_acts_file = home + 'sub-{0}/random_features/sub-{0}_model-random_features_similarity_matrix_{1}.npy'.format(sub, in_specifier)
    # rand_acts[sub] = np.load(rand_acts_file)
    # if rand_acts[sub].min() < minval[sub]:
    #     minval[sub] = rand_acts[sub].min()
    # if rand_acts[sub].max() > maxval[sub]:
    #     maxval[sub] = rand_acts[sub].max()

    # full_rand_file = home + 'sub-{0}/random_features/sub-{0}_model-random_similarity_matrix_610980.npy'.format(sub)
    # full_rand[sub] = np.load(full_rand_file)
    # if full_rand[sub].min() < minval[sub]:
    #     minval[sub] = full_rand[sub].min()
    # if full_rand[sub].max() > maxval[sub]:
    #     maxval[sub] = full_rand[sub].max()

    for lang in ['enu', 'deu', 'nld']:
        file = home + 'sub-{0}/{1}-{1}-freeze/sub-{0}_model-{1}-{1}-freeze_{1}-{1}-freeze_similarity_matrix_{2}.npy'.format(sub, lang, in_specifier)
        mat = np.load(file)
        simmats[sub][lang] = mat
        if mat.min() < minval[sub]:
            minval[sub] = mat.min()
        if mat.max() > maxval[sub]:
            maxval[sub] = mat.max()
        # file_sh = home + 'sub-{0}/{1}-{1}-freeze/sub-{0}_model-{1}-{1}-freeze_{1}-{1}-freeze_similarity_matrix_{2}_shuffled.npy'.format(sub, lang, in_specifier)
        # mat_sh = np.load(file_sh)
        # simmats_sh[sub][lang] = mat_sh
        # if mat_sh.min() < minval[sub]:
        #     minval[sub] = mat_sh.min()
        # if mat_sh.max() > maxval[sub]:
        #     maxval[sub] = mat_sh.max()
        file_sh_run = home + 'sub-{0}/{1}-{1}-freeze/sub-{0}_model-{1}-{1}-freeze_{1}-{1}-freeze_similarity_matrix_{2}_shuffle_run.npy'.format(sub, lang, in_specifier)
        mat_sh_run = np.load(file_sh_run)
        simmats_sh_run[sub][lang] = mat_sh_run
        if mat_sh_run.min() < minval[sub]:
            minval[sub] = mat_sh_run.min()
        if mat_sh_run.max() > maxval[sub]:
            maxval[sub] = mat_sh_run.max()

# for sub in subject_lang:
for sub in [5]:
    # base1_file = home + 'sub-{0}/untrained/sub-{0}_model-untrained_untrained_similarity_matrix_610980.npy'.format(sub)
    # base = np.load(base1_file)
    # base2_file = home + 'sub-{0}/random_features/sub-{0}_model-random_features_similarity_matrix_610980.npy'.format(sub)
    # base2 = np.load(base2_file)
    # simmats = {}
    # simmats_sh = {}
    # for lang in ['enu', 'deu', 'nld']:
    #     file = home + 'sub-{0}/{1}-{1}-freeze/sub-{0}_model-{1}-{1}-freeze_{1}-{1}-freeze_similarity_matrix_610980.npy'.format(sub, lang)
    #     mat = np.load(file)
    #     simmats[lang] = mat
    #     file_sh = home + 'sub-{0}/{1}-{1}-freeze/sub-{0}_model-{1}-{1}-freeze_{1}-{1}-freeze_similarity_matrix_610980_shuffled.npy'.format(sub, lang)
    #     mat_sh = np.load(file_sh)
    #     simmats_sh[lang] = mat_sh

    fig, axs = plt.subplots(2, 4, figsize=(15, 11), facecolor='w', edgecolor='k')
    # True Models

    im = axs[0, 0].imshow(simmats[sub]['enu'], origin='lower', cmap='magma', vmin=minval[sub], vmax=maxval[sub])
    axs[0, 0].set_xticks(range(len(MERGED_ROI_ORDER)))
    axs[0, 0].set_xticklabels(MERGED_ROI_ORDER, rotation='vertical')
    axs[0, 0].set_yticks(range(len(LAYERS)))
    axs[0, 0].set_yticklabels(LAYERS)
    axs[0, 0].set_title('English Freeze')
    axs[0, 0].set_xlabel('Auditory ROIs')
    axs[0, 0].set_ylabel('Network Layers')
    im = axs[0, 1].imshow(simmats[sub]['deu'], origin='lower', cmap='magma', vmin=minval[sub], vmax=maxval[sub])
    axs[0, 1].set_xticks(range(len(MERGED_ROI_ORDER)))
    axs[0, 1].set_yticks([])
    axs[0, 1].set_xticklabels(MERGED_ROI_ORDER, rotation='vertical')
    axs[0, 1].set_title('German Freeze')
    axs[0, 1].set_xlabel('Auditory ROIs')
    im = axs[0, 2].imshow(simmats[sub]['nld'], origin='lower', cmap='magma', vmin=minval[sub], vmax=maxval[sub])
    axs[0, 2].set_xticks(range(len(MERGED_ROI_ORDER)))
    axs[0, 2].set_yticks([])
    axs[0, 2].set_xticklabels(MERGED_ROI_ORDER, rotation='vertical')
    axs[0, 2].set_title('Dutch Freeze')
    axs[0, 2].set_xlabel('Auditory ROIs')
    axs[0, 3].axis('off')
    # axs[0, 4].axis('off')
    # Model specific design preserving baselines
    im = axs[1, 0].imshow(simmats_sh_run[sub]['enu'], origin='lower', cmap='magma', vmin=minval[sub], vmax=maxval[sub])
    axs[1, 0].set_xticks(range(len(MERGED_ROI_ORDER)))
    axs[1, 0].set_xticklabels(MERGED_ROI_ORDER, rotation='vertical')
    axs[1, 0].set_yticks(range(len(LAYERS)))
    axs[1, 0].set_yticklabels(LAYERS)
    axs[1, 0].set_title('English Freeze \n Shuffled Within Run')
    axs[1, 0].set_xlabel('Auditory ROIs')
    axs[1, 0].set_ylabel('Network Layers')
    im = axs[1, 1].imshow(simmats_sh_run[sub]['deu'], origin='lower', cmap='magma', vmin=minval[sub], vmax=maxval[sub])
    axs[1, 1].set_xticks(range(len(MERGED_ROI_ORDER)))
    axs[1, 1].set_yticks([])
    axs[1, 1].set_xticklabels(MERGED_ROI_ORDER, rotation='vertical')
    axs[1, 1].set_title('German Freeze \n Shuffled Within Run')
    axs[1, 1].set_xlabel('Auditory ROIs')
    im = axs[1, 2].imshow(simmats_sh_run[sub]['nld'], origin='lower', cmap='magma', vmin=minval[sub], vmax=maxval[sub])
    axs[1, 2].set_xticks(range(len(MERGED_ROI_ORDER)))
    axs[1, 2].set_yticks([])
    axs[1, 2].set_xticklabels(MERGED_ROI_ORDER, rotation='vertical')
    axs[1, 2].set_title('Dutch Freeze \n Shuffled Within Run')
    axs[1, 2].set_xlabel('Auditory ROIs')
    # Model specific non design preserving baselines
    # im = axs[2, 0].imshow(simmats_sh[sub]['enu'], origin='lower', cmap='magma', vmin=minval[sub], vmax=maxval[sub])
    # axs[2, 0].set_xticks(range(len(MERGED_ROI_ORDER)))
    # axs[2, 0].set_xticklabels(MERGED_ROI_ORDER, rotation='vertical')
    # axs[2, 0].set_yticks(range(len(LAYERS)))
    # axs[2, 0].set_yticklabels(LAYERS)
    # axs[2, 0].set_title('English Freeze \n Shuffled')
    # axs[2, 0].set_xlabel('Auditory ROIs')
    # axs[2, 0].set_ylabel('Network Layers')
    # im = axs[2, 1].imshow(simmats_sh[sub]['deu'], origin='lower', cmap='magma', vmin=minval[sub], vmax=maxval[sub])
    # axs[2, 1].set_xticks(range(len(MERGED_ROI_ORDER)))
    # axs[2, 1].set_yticks([])
    # axs[2, 1].set_xticklabels(MERGED_ROI_ORDER, rotation='vertical')
    # axs[2, 1].set_title('German Freeze \n Shuffled')
    # axs[2, 1].set_xlabel('Auditory ROIs')
    # im = axs[2, 2].imshow(simmats_sh[sub]['nld'], origin='lower', cmap='magma', vmin=minval[sub], vmax=maxval[sub])
    # axs[2, 2].set_xticks(range(len(MERGED_ROI_ORDER)))
    # axs[2, 2].set_yticks([])
    # axs[2, 2].set_xticklabels(MERGED_ROI_ORDER, rotation='vertical')
    # axs[2, 2].set_title('Dutch Freeze \n Shuffled')
    # axs[2, 2].set_xlabel('Auditory ROIs')
    # Null models that aren't model specific
    # im = axs[1, 3].imshow(full_rand[sub], origin='lower', cmap='magma', vmin=minval[sub], vmax=maxval[sub])
    # axs[1, 3].set_xticks(range(len(MERGED_ROI_ORDER)))
    # axs[1, 3].set_xticklabels(MERGED_ROI_ORDER, rotation='vertical')
    # axs[1, 3].set_yticks([])
    # axs[1, 3].set_title('100% Uniform Random')
    # axs[1, 3].set_xlabel('Auditory ROIs')
    # axs[2, 3].axis('off')
    # axs[2, 4].axis('off')
    # im = axs[1, 3].imshow(rand_acts[sub], origin='lower', cmap='magma', vmin=minval[sub], vmax=maxval[sub])
    # axs[1, 3].set_xticks(range(len(MERGED_ROI_ORDER)))
    # axs[1, 3].set_xticklabels(MERGED_ROI_ORDER, rotation='vertical')
    # axs[1, 3].set_yticks([])
    # axs[1, 3].set_title('Uniform Random \n Activations')
    # axs[1, 3].set_xlabel('Auditory ROIs')
    im = axs[1, 3].imshow(untrained[sub], origin='lower', cmap='magma', vmin=minval[sub], vmax=maxval[sub])
    axs[1, 3].set_xticks(range(len(MERGED_ROI_ORDER)))
    axs[1, 3].set_xticklabels(MERGED_ROI_ORDER, rotation='vertical')
    axs[1, 3].set_yticks([])
    axs[1, 3].set_yticklabels(LAYERS)
    axs[1, 3].set_title('Untrained Network')
    axs[1, 3].set_xlabel('Auditory ROIs')



    # cbar = fig.colorbar(im_c, ax=axs.ravel().tolist(), shrink=0.99, pad=0.01)
    cbar = fig.colorbar(im, ax=axs[0, 3])
    # plt.suptitle('Subject {} ({} speaker) active voxels only (p < 0.05 uncorrected)'.format(sub, subject_lang[sub]), size=16)
    plt.suptitle('Subject {} ({} speaker) all voxels'.format(sub, subject_lang[sub]), size=16)
    fig_file_name = home + '../figures/sub-{}_mono_freeze_models_{}'.format(sub, out_specifier)
    plt.savefig(fig_file_name, dpi=300)
