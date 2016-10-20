"""Set randomized stimuli order for one subject in deep quilts fMRI exp.

Each subject will be scanned twice.
Each session contains 10 runs.
Each run contains 3 blocks, one for each language: deu, enu, nld

1. Set order of languages within each run, covering each permutation at least
three times.
2. Set gender order. The gender of the speaker will alternate every quilt. So
each subject can be a male order or female order speaker, meaning their first
run begins with a male or female speaker. Each run is then a male order or
female order run. The first subject will have female order, the second subject
male order, and so on.

Once 1 and 2 are determined, we can create the stimuli order from selected_spkr
text files.

3. Set placement of task question presentation. Here we have 6 types of runs.
    0: no resp
    1: 1 resp at the end after the last block
    2: 1 resp after either the 1st or 2nd block. We differentiate between 1 and
       2 beacuse 1 can be a few seconds shorter than 2.
    3: 2 resps where one happens after the last block
    4: 2 resps, one after each of the first two blocks
    5: 3 resps, one after each block
For the sake of time, we want the runs with fewer resps and the runs with resps
after the last block to occur more frequently.

The output is:
* A list of stimuli file and path names for each run.
* A file containing 3 binary digits for each run indicating when task question
should be presented.
"""
import itertools
import sys
import os
from os.path import join
import numpy as np
import numpy.matlib as matlib
from scipy import io

n_runs = 20
n_blocks = 3
n_quilts = 3
home = '/Users/jthompson/Dropbox/quilts/'
stimdir = home + 'stimuli/'


def save_order(sub_no):
    """Save information needed to run fmri experiment for one subject.

    Saves one npy file per run per subject in fmri_exp/[subject]

    Numerical indicators of language identity always follow alphabetical order
    of the English version of the languages: Dutch (0), English (1), German (2)
    """
    expdir = join(home, 'code', 'fmri_exp', 's' + str(sub_no))
    if not os.path.isdir(expdir):
        os.mkdir(expdir)
    langs = np.array(['nld', 'enu', 'deu'])
    allstims = {'nld': {'female': get_spkrs('nld', 'female'),
                        'male': get_spkrs('nld', 'male')},
                'enu': {'female': get_spkrs('enu', 'female'),
                        'male': get_spkrs('enu', 'male')},
                'deu': {'female': get_spkrs('deu', 'female'),
                        'male': get_spkrs('deu', 'male')}}

    lang_order = set_lang_order()
    gender = set_gender(sub_no)
    all_runs_resp, all_runs_resp_vol = set_response_type()
    for run in range(n_runs):
        order = np.array([])
        for block in range(n_blocks):
            lang = langs[lang_order[run, block]]
            for quilt in range(n_quilts):
                g = gender[run, block, quilt]
                stims = allstims[lang][g]
                idxtoadd = np.random.choice(len(stims))
                stimtoadd = stims[idxtoadd]
                order = np.append(order, stimtoadd)
                np.delete(allstims[lang][g], idxtoadd)
        to_save = {'stimuli': order, 'sub': sub_no, 'run': run+1,
                   'resp': all_runs_resp[run, :],
                   'vols': all_runs_resp_vol[run, 0],
                   'lang': lang_order[run, :]}
        fname = join(expdir, 's' + str(sub_no) + '_run' + str(run+1) + 'order')
        np.save(fname, to_save)
        io.savemat(fname, to_save)
    print all_runs_resp_vol
    return to_save


def get_spkrs(lang, gen):
    """."""
    spkfile = os.path.join(stimdir, lang, 'selected_' + gen + '_spkrs.txt')
    spkrs = list(np.loadtxt(spkfile, dtype='|S25'))
    filtdir = join('stimuli', lang, gen)
    spkrs = [join(filtdir, i[:-4] + '_60s_filtered.wav') for i in spkrs]
    return spkrs


def set_lang_order():
    """Set order of language blocks within each run for a single subject.

    Since there are 3 languages, there are 6 possible orderings of language
    blocks. Since we have 20 runs, each of those 6 ordering should be used at
    least 3 times. The remaining 2 orders are selected randomly. Then all
    orders are shuffled to set the final order for each run.

    This doesn't take into acount the two sessions.
    For the sake of easy permutations, maybe it makes sense to just select the
    lang order randomly independently in each run?
    """
    perm3 = np.array(list(itertools.permutations([0, 1, 2])))  # 6 perms of 3
    allperms = np.concatenate((perm3, perm3), 0)  # 12 perms
    allperms = np.concatenate((allperms, perm3), 0)  # 18 perms
    # Add 2 more random permutations to get to 20 runs
    for i in range(n_runs - allperms.shape[0]):
        # Select lang order of remaining runs randomly
        newperm = np.matrix(np.random.permutation(3))
        allperms = np.concatenate((allperms, newperm), 0)  # 20 perms
    # Randomize the permutations across runs
    np.random.shuffle(allperms)
    return allperms


def set_gender(sub_no):
    """Set the gender of the first quilt of each run."""
    if np.mod(sub_no, 2):
        gender = matlib.repmat(['male', 'female'], 1, (n_runs * n_blocks *
                                                       n_quilts)/2)
    else:
        gender = matlib.repmat(['female', 'male'], 1, (n_runs * n_blocks *
                                                       n_quilts)/2)
    gender = np.reshape(gender, (n_runs, n_blocks, n_quilts))
    return gender


def set_response_type(nresp0=3, nresp1end=2, nresp1=2, nresp2end=1,
                      nresp2=1, nresp3=1):
    """Set which block will be followed by a task response period.

    nresp0=3
    nresp1end=2
    nresp1=2
    nresp2end=1
    nresp2=1
    nresp3=1

    nresp0 = 3 implies 3 runs with no response per session

    save txt file or mat file per run
    """
    assert nresp0 + nresp1end + nresp1 + nresp2end + nresp2 + nresp3 == n_runs/2

    resp0 = matlib.repmat([0, 0, 0], nresp0, 1)
    resp0vol = matlib.repmat(353, nresp0, 1)
    resp1end = matlib.repmat([0, 0, 1], nresp1end, 1)
    resp1endvol = matlib.repmat(355, nresp1end, 1)
    resp1_1 = matlib.repmat([1, 0, 0], np.floor(nresp1/2), 1)
    resp1_1vol = matlib.repmat(361, np.floor(nresp1/2), 1)
    resp1_2 = matlib.repmat([0, 1, 0], np.ceil(nresp1/2), 1)
    resp1_2vol = matlib.repmat(361, np.ceil(nresp1/2), 1)
    resp2end_1 = matlib.repmat([0, 1, 1], np.floor(nresp2end/2), 1)
    resp2end_1vol = matlib.repmat(363, np.floor(nresp2end/2), 1)
    resp2end_2 = matlib.repmat([1, 0, 1], np.ceil(nresp2end), 1)
    resp2end_2vol = matlib.repmat(363, np.ceil(nresp2end), 1)
    resp2 = matlib.repmat([1, 1, 0], nresp2, 1)
    resp2vol = matlib.repmat(369, nresp2, 1)
    resp3 = matlib.repmat([1, 1, 1], nresp3, 1)
    resp3vol = matlib.repmat(371, nresp3, 1)

    run_resps = np.concatenate([resp0, resp1end, resp1_1, resp1_2, resp2,
                                resp2end_1, resp2end_2, resp3], 0)
    run_resps_vol = np.concatenate([resp0vol, resp1endvol, resp1_1vol,
                                    resp1_2vol, resp2vol,
                                    resp2end_1vol, resp2end_2vol, resp3vol], 0)

    def get_random_order():
        consec_noresp = True
        while consec_noresp:
            idx = np.arange(run_resps.shape[0])
            np.random.shuffle(idx)
            rand_run_resps = run_resps[idx].copy()
            rand_run_resps_vol = run_resps_vol[idx].copy()
            # Get number of consecutive instances of the same run type
            grouped = np.array([[x[0], sum(1 for i in y)] for x, y in
                                itertools.groupby(rand_run_resps_vol)])
            noresps = grouped[grouped[:, 0] == 353, 1]
            # Re-randomize if there are two no-response runs back to back
            consec_noresp = sum(noresps > 1)
        return rand_run_resps, rand_run_resps_vol

    run_resps_s1, run_resps_vol_s1 = get_random_order()
    run_resps_s2, run_resps_vol_s2 = get_random_order()

    allruns_resps = np.concatenate([run_resps_s1, run_resps_s2], 0)
    allruns_resps_vol = np.concatenate([run_resps_vol_s1, run_resps_vol_s2], 0)
    return allruns_resps, allruns_resps_vol

if __name__ == "__main__":
    sub_no = sys.argv[1]
    save_order(sub_no)
