""" """
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
from scipy.misc import imresize

# LAYERS = ['cnn1', 'cnn2', 'cnn3', 'cnn4', 'cnn5', 'cnn6', 'cnn7', 'cnn8',
          # 'cnn9', 'fc1', 'fc2']
LAYERS = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8',
          'c9', 'fc1', 'fc2']
NLAYERS = 11

def plot_all_random_sim_mats():
    nmats = 10

    # Load similarity matrices and accuracies
    all_rv_mats = np.zeros((nmats, NLAYERS, NLAYERS))
    accuracy = np.zeros((nmats,))
    for i in range(nmats):
        mat = 'RV2_results_enu-base_vs_enu-rand_{}_lang-enu.npy'.format(i)
        acc = '../enu/models/exp5_random_feats/rand_netpars7_{}/accuracy.npy'.format(i)
        try:
            accuracy[i] = np.load(acc)[-1, 2]
            all_rv_mats[i, :NLAYERS, :NLAYERS] = np.load(mat)[:NLAYERS, :NLAYERS]
        except:
            print('Skipping matrix {}'.format(i))

    min_rv = all_rv_mats.min()
    max_rv = all_rv_mats.max()

    fig, axs = plt.subplots(2, 5, figsize=(14.7, 6.3), facecolor='w', edgecolor='k')
    for a, ax in enumerate(axs.ravel()):
        im = ax.imshow(all_rv_mats[a, :, :], origin='lower', cmap='magma',
                      vmin=min_rv, vmax=max_rv)
        ax.set_xticks(range(NLAYERS))
        ax.set_xticklabels(LAYERS, rotation='vertical')
        ax.set_yticks(range(NLAYERS))
        ax.set_title('Accuracy: {:.2f}%'.format(accuracy[a]*100))
        if a==0 or a==5:
            ax.set_yticklabels(LAYERS)
            ax.set_ylabel('Standard Net', fontsize=14)
        else:
            ax.set_yticks([])
        # ax.set_xlabel('Only layers > {} trained'.format(LAYERS[a]))
        ax.set_xlabel('Random Net {}'.format(a+1), fontsize=14)
        plt.subplots_adjust(wspace=.05, hspace=.35)
    # fig.subplots_adjust(left=0.1, right=0.9,
    #                     wspace=0.02, hspace=0.02)
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.99, pad=0.01)
    # fig.suptitle('Similarity of Activation Vectors (modified RV coefficient)', x=0.44, y=.9, horizontalalignment='center', fontsize=14)

    plt.savefig('figures/all_mats_{}_2rows_relabeled'.format(nmats), bbox_inches='tight', dpi=300)


def plot_input():
    fig, axs = plt.subplots(2, 1, figsize=(15, 10), facecolor='w', edgecolor='k')
    # for utt in np.range(778,861):
    dpath = '../deu/inputs/accont1/accont1_778.npy'
    dfeat = np.load(dpath).squeeze()
    epath = '../enu/inputs/wsj_01b/wsj_01b_76.npy'
    efeat = np.load(epath).squeeze()
    vmin = min(min(efeat.flatten()), min(dfeat.flatten()))
    vmax = max(max(efeat.flatten()), max(dfeat.flatten()))
    axs[0].imshow(np.squeeze(dfeat).T, vmin=vmin, vmax=vmax)
    im = axs[1].imshow(np.squeeze(efeat).T, vmin=vmin , vmax=vmax)
    plt.imshow(np.squeeze(efeat).T)
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.58, pad=0.01)

def plot_enu_sim_mats():
    eb_eb_path = 'RV2_results_enu-base_vs_enu-base_lang-enu.npy'
    eb_nb_path = 'RV2_results_enu-base_vs_nld-base_lang-enu.npy'
    eb_eef_path = 'RV2_results_enu-base_vs_enu-enu-freeze_lang-enu.npy'
    eb_nef_path = 'RV2_results_enu-base_vs_nld-enu-freeze_lang-enu.npy'
    nb_nef_path = 'RV2_results_nld-base_vs_nld-enu-freeze_lang-enu.npy'
    un_un_path = 'RV2_results_untrained_vs_untrained_lang-enu.npy'
    eb_un_path = 'RV2_results_enu-base_vs_untrained_lang-enu.npy'
    eef_un_path = 'RV2_results_enu-enu-freeze_vs_untrained_lang-enu.npy'
    nef_un_path = 'RV2_results_enu-enu-freeze_vs_untrained_lang-enu.npy'
    eb_eb = np.load(eb_eb_path)
    eb_nb = np.load(eb_nb_path)
    eb_eef = np.load(eb_eef_path)
    eb_nef = np.load(eb_nef_path)
    nb_nef = np.load(nb_nef_path)
    un_un = np.load(un_un_path)
    eb_un = np.load(eb_un_path)
    eef_un = np.load(eef_un_path)
    nef_un = np.load(nef_un_path)

    # vmax = np.max([np.max(eb_eb), np.max(eb_ef), np.max(un_un), np.max(eb_un), np.max(ef_un)])
    # vmin = np.min([np.min(eb_eb), np.min(eb_ef), np.min(un_un), np.min(eb_un), np.min(ef_un)])
    vmax = np.max([np.max(eb_nb), np.max(eb_nef), np.max(eb_nef), np.max(un_un), np.max(eb_un), np.max(nef_un)])
    vmin = np.min([np.min(eb_nb), np.min(eb_nef), np.max(eb_nef), np.min(un_un), np.min(eb_un), np.min(nef_un)])
    fs=16
    fs_tks = 15
    fig, axs = plt.subplots(2, 3, figsize=(9, 7), facecolor='w', edgecolor='k')
    fig.subplots_adjust(left=0, right=1,
                        wspace=0.02, hspace=.4)
    kwargs = {'origin':'lower', 'cmap':'magma', 'vmin':vmin, 'vmax':vmax}
    axs[0, 0].imshow(eb_nb, **kwargs)
    axs[0, 0].set_ylabel('English Std (47.6%)', fontsize=fs)
    axs[0, 0].set_xlabel('Dutch Std', fontsize=fs)
    axs[0, 0].set_xticks(range(NLAYERS))
    axs[0, 0].set_xticklabels(LAYERS, rotation='vertical', fontsize=fs_tks)
    axs[0, 0].set_yticks(range(NLAYERS))
    axs[0, 0].set_yticklabels(LAYERS, fontsize=fs_tks)

    im = axs[0, 1].imshow(eb_nef, **kwargs)
    axs[0, 1].set_ylabel('English Std (47.6%)', fontsize=fs)
    axs[0, 1].set_xlabel('Dut->Eng Frz (52.6%)', fontsize=fs)
    axs[0, 1].set_xticks(range(NLAYERS))
    axs[0, 1].set_xticklabels(LAYERS, rotation='vertical', fontsize=fs_tks)
    axs[0, 1].set_yticks([])
    # axs[0, 1].set_yticks(range(NLAYERS))
    # axs[0, 1].set_yticklabels(LAYERS)

    im = axs[0, 2].imshow(nb_nef, **kwargs)
    axs[0, 2].set_ylabel('Dutch Std', fontsize=fs)
    axs[0, 2].set_xlabel('Dut->Eng Frz (52.6%)', fontsize=fs)
    axs[0, 2].set_xticks(range(NLAYERS))
    axs[0, 2].set_xticklabels(LAYERS, rotation='vertical', fontsize=fs_tks)
    axs[0, 2].set_yticks([])
    # axs[0, 2].set_yticks(range(NLAYERS))
    # axs[0, 2].set_yticklabels(LAYERS)

    # axs[0, 2].axis('off')
    # cbar = fig.colorbar(im, cax=axs[0,2], ax = axs.ravel().tolist(),use_gridspec=True, aspect=20)

    axs[1, 0].imshow(un_un, **kwargs)
    axs[1, 0].set_ylabel('Untrained', fontsize=fs)
    axs[1, 0].set_xlabel('Untrained', fontsize=fs)
    axs[1, 0].set_xticks(range(NLAYERS))
    axs[1, 0].set_xticklabels(LAYERS, rotation='vertical', fontsize=fs_tks)
    axs[1, 0].set_yticks(range(NLAYERS))
    axs[1, 0].set_yticklabels(LAYERS, fontsize=fs_tks)

    axs[1, 1].imshow(eb_un, **kwargs)
    axs[1, 1].set_ylabel('English Std (47.6%)', fontsize=fs)
    axs[1, 1].set_xlabel('Untrained', fontsize=fs)
    axs[1, 1].set_xticks(range(NLAYERS))
    axs[1, 1].set_xticklabels(LAYERS, rotation='vertical', fontsize=fs_tks)
    axs[1, 1].set_yticks([])
    # axs[1, 1].set_yticks(range(NLAYERS))
    # axs[1, 1].set_yticklabels(LAYERS)

    axs[1, 2].imshow(nef_un, **kwargs)
    axs[1, 2].set_ylabel('Dut->Eng Frz (52.6%)', fontsize=fs)
    axs[1, 2].set_xlabel('Untrained', fontsize=fs)
    axs[1, 2].set_xticks(range(NLAYERS))
    axs[1, 2].set_xticklabels(LAYERS, rotation='vertical', fontsize=fs_tks)
    axs[1, 2].set_yticks([])

    plt.subplots_adjust(wspace=.19, hspace=.37)
    cbar = fig.colorbar(im, ax = axs.ravel().tolist(), pad=.02)
    # axs[1, 2].set_yticks(range(NLAYERS))
    # axs[1, 2].set_yticklabels(LAYERS)
    # plt.tight_layout()
    plt.savefig('figures/Dutch_to_English_Freeze_and_Untrained_bigfont', bbox_inches='tight', dpi=300)


def plot_hypothesis():
    kwargs = {'origin':'lower', 'cmap':'magma'}
    eb_eb_path = 'RV2_results_enu-base_vs_enu-base_lang-enu.npy'
    eb_eb = np.load(eb_eb_path)
    selfsim = eb_eb
    fs_title = 16
    fs_label = 15
    fs_tk = 14
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), facecolor='w', edgecolor='k')
    # selfsim = np.array([
    #    [1., 0.8, 0.6, 0.4, 0.2, 0., 0., 0., 0., 0., 0.],
    #    [0.8, 1., 0.8, 0.6, 0.4, 0.2, 0., 0., 0., 0., 0.],
    #    [0.6, 0.8, 1., 0.8, 0.6, 0.4, 0.2, 0., 0., 0., 0.],
    #    [0.4, 0.6, 0.8, 1., 0.8, 0.6, 0.4, 0.2, 0., 0., 0.],
    #    [0.2, 0.4, 0.6, 0.8, 1., 0.8, 0.6, 0.4, 0.2, 0., 0.],
    #    [0., 0.2, 0.4, 0.6, 0.8, 1., 0.8, 0.6, 0.4, 0.2, 0.],
    #    [0., 0., 0.2, 0.4, 0.6, 0.8, 1., 0.8, 0.6, 0.4, 0.2],
    #    [0., 0., 0., 0.2, 0.4, 0.6, 0.8, 1., 0.8, 0.6, 0.4],
    #    [0., 0., 0., 0., 0.2, 0.4, 0.6, 0.8, 1., 0.8, 0.6],
    #    [0., 0., 0., 0., 0., 0.2, 0.4, 0.6, 0.8, 1., 0.8],
    #    [0., 0., 0., 0., 0., 0., 0.2, 0.4, 0.6, 0.8, 1.]])

    im = axs[0].imshow(selfsim, **kwargs)
    axs[0].set_xticks(range(NLAYERS))
    axs[0].set_xticklabels(LAYERS, rotation='vertical', fontsize=fs_tk)
    axs[0].set_yticks(range(NLAYERS))
    axs[0].set_yticklabels(LAYERS, fontsize=fs_tk)
    axs[0].set_title('True Self Similarity', fontsize=fs_title)
    axs[0].set_ylabel('English Std (all layers trained)', fontsize=fs_label)
    axs[0].set_xlabel('English Std (all layers trained)', fontsize=fs_label)
    cbar = fig.colorbar(im, ax = axs.ravel().tolist(),use_gridspec=True, shrink=0.8, pad=0.01)
    hyp = np.array([
       [0., 0., 0., 0., 1., 0.8, 0.6, 0.4, 0.2, 0., 0.],
       [0., 0., 0., 0., 0.8, 1., 0.8, 0.6, 0.4, 0.2, 0.],
       [0., 0., 0., 0., 0.8, 1., 0.8, 0.6, 0.4, 0.2, 0.],
       [0., 0., 0., 0., 0.6, 0.8, 1., 0.8, 0.6, 0.4, 0.2],
       [0., 0., 0., 0., 0.6, 0.8, 1., 0.8, 0.6, 0.4, 0.2],
       [0., 0., 0., 0., 0.4, 0.6, 0.8, 1., 0.8, 0.6, 0.4],
       [0., 0., 0., 0., 0.2, 0.4, 0.6, 0.8, 1., 0.8, 0.6],
       [0., 0., 0., 0., 0.2, 0.4, 0.6, 0.8, 1., 0.8, 0.6],
       [0., 0., 0., 0., 0., 0.2, 0.4, 0.6, 0.8, 1., 0.8],
       [0., 0., 0., 0., 0., 0.2, 0.4, 0.6, 0.8, 1., 0.8],
       [0., 0., 0., 0., 0., 0., 0.2, 0.4, 0.6, 0.8, 1.]])

    axs[1].imshow(hyp, **kwargs)
    axs[1].set_title('Hypothesized \n Random Similarity', fontsize=fs_title)
    axs[1].set_xticks(range(NLAYERS))
    axs[1].set_xticklabels(LAYERS, rotation='vertical', fontsize=fs_tk)
    axs[1].set_yticks(range(NLAYERS))
    axs[1].set_yticklabels([])
    axs[1].set_ylabel('Standard Net (all layers trained)', fontsize=fs_label)
    axs[1].set_xlabel('Random Net 4 \n (only layers above cnn4 trained)', fontsize=fs_label)
    plt.savefig('figures/hypothesis', bbox_inches='tight', dpi=300)
