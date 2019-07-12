"""Calculate RV similarity between two networks."""
import copy
import os
import multiprocessing as mp
import time
import datetime
from operator import isub
from argparse import ArgumentParser
# from nbmultitask import ThreadWithLogAndControls
import numpy as np
from matplotlib import pyplot as plt
# %matplotlib inline
from callrv2 import rv2
start = time.time()

# Set input arguments
parser = ArgumentParser(description='Calculate RV similiarity matrix for two networks.')
parser.add_argument("-p", "--n_processes", type=int,
                    help='Number of processes to run in parallel', dest="nprocs",
                    default=2)
parser.add_argument("-n1", "--net1_name", type=str,
                    help="Name of model 1", dest="net1_name",
                    default=None)
parser.add_argument("-n2", "--net2_name", type=str,
                    help="Name of model 2", dest="net2_name",
                    default=None)
parser.add_argument("-m1", "--net1_path", type=str,
                    help="Path to model 1", dest="net1_path",
                    default=None)
parser.add_argument("-m2", "--net2_path", type=str,
                    help="Path to model 2", dest="net2_path",
                    default=None)
parser.add_argument("-d", "--featDir", type=str,
                    help="Working directory where input features are found",
                    dest="feat_dir", default=None)
parser.add_argument("-s", "--skip", type=int,
                    help="Load activations to every skpth frame",
                    dest="skp", default=20)
parser.add_argument("-l", "--lang", type=str,
                    help="Language of speech passed when saving activations",
                    dest="lang", default=None)
args = parser.parse_args()
feat_dir = args.feat_dir
putts_dir = feat_dir + '/proposed_utts/'
nprocs = args.nprocs
pool = mp.Pool(nprocs)
skp = args.skp
# leave out logits layer
layers = ['cnn1', 'cnn2', 'cnn3', 'cnn4', 'cnn5', 'cnn6', 'cnn7', 'cnn8',
          'cnn9', 'fc1', 'fc2']#, 'fc3']
nlayers = len(layers)
frms_dict = {1:459329, 5:92392, 10:46508, 20:23582}
frms = frms_dict[skp]
n_outputs = [1024, 256, 256, 128, 128, 128, 64, 64, 64, 600, 190, 9000]
get_ndims = iter(n_outputs)

# Initialize dictionaries to store activation vectors
net1 = {}
net2 = {}
for layer in layers:
    net1[layer] = np.zeros((frms, next(get_ndims)), dtype='float32')
net2 = copy.deepcopy(net1)

# Load activation vectors
total_nframes = 0
with open(feat_dir + '/selected_speakers.txt', 'r') as f:
    spkrs = [name.split('.')[0] for name in f.read().splitlines()]
    # spkr = spkrs[0]
for s, spkr in enumerate(spkrs):
    print('Processing speaker: {} {}/{}\r'.format(spkr, s+1, len(spkrs)))
    with open(putts_dir + spkr + '.txt', 'r') as f:
        utt_ind = f.read().splitlines()
    for utt_no in utt_ind:
        indir_net1 = '%s/%s/' % (args.net1_path, spkr)
        indir_net2 = '%s/%s/' % (args.net2_path, spkr)
        inname = '%s_%s.npz' % (spkr, utt_no)
        # inname = '%s_%s_noavg.npz' % (spkr, utt_no)
        model_net1 = np.load(indir_net1 + inname)
        if indir_net1 == indir_net2:
            model_net2 = model_net1
        else:
            model_net2 = np.load(indir_net2 + inname)
        nframes = model_net1['cnn1'][::skp, :].shape[0]
        for layer in layers:
            net1[layer][total_nframes:total_nframes+nframes, :] = model_net1[layer][::skp, :]
            net2[layer][total_nframes:total_nframes+nframes, :] = model_net2[layer][::skp, :]
        total_nframes += nframes
    print('Total num frames: ', total_nframes)
    # break
print('Total num frames: ', total_nframes)

# Center activation vectors at zero
print('Centering...')
for layer in layers:
    print('-',layer)
     # in-place subtraction
    net1[layer] = isub(net1[layer], np.mean(net1[layer], axis=0, keepdims=True))
    net2[layer] = isub(net2[layer], np.mean(net2[layer], axis=0, keepdims=True))

llayers = layers.copy()
# llayers[-1] = 'logits'
rv_results = np.zeros((len(layers), len(layers)))

# Calculate RV correlation coefficient for all layer pairs
# Store and plot resulting similarity matrix
for i, l1 in enumerate(layers):
    print('Calculating RV similarity for layer ', l1)
    rv = np.zeros((nlayers,))

    res = [pool.apply_async(rv2, args=(net1[l1], net2[l2], l2)) for l2 in layers]
    rv_results[i, :] = [p.get() for p in res]
    print(rv_results[i, :])
    res_save_name = 'RV2_results_{}_vs_{}_lang-{}'.format(args.net1_name, args.net2_name, args.lang)
    np.save(res_save_name, rv_results)

    print('Plotting...')
    plt.figure()
    plt.imshow(rv_results, origin='lower', cmap='magma') # cmap = 'magma'
    plt.xticks(range(nlayers), llayers, rotation='vertical')
    plt.yticks(range(nlayers), llayers)
    plt.xlabel(args.net2_name)
    plt.ylabel(args.net1_name)
    plt.title('Activations to {} Speech'.format(args.lang))
    plt.colorbar()
    if not os.path.isdir('figures'):
        os.makedirs('figures')
    fig_name = 'figures/RV2_{}_vs_{}_lang-{}.png'.format(args.net1_name, args.net2_name, args.lang)
    plt.savefig(fig_name, bbox_inches='tight')

    now = time.time()
    print('Elapsed time: {} minutes'.format((now - start)/60))

now = time.time()
print('Finished {} vs {} in {} hours'.format(args.net1_name, args.net2_name, (now - start)/3600))
print('current time: {}'.format(datetime.datetime.now()))
