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
    0: no task
    1: 1 task at the end after the last block
    2: 1 task after either the 1st or 2nd block. We differentiate between 1 and
       2 beacuse 1 can be a few seconds shorter than 2.
    3: 2 tasks where one happens after the last block
    4: 2 tasks, one after each of the first two blocks
    5: 3 tasks, one after each block
For the sake of time, we want the runs with fewer tasks and the runs with tasks
after the last block to occur more frequently.

The output is:
* A list of stimuli file and path names for each run.
* A file containing 3 binary digits for each run indicating when task question
should be presented.
"""
import numpy as np
import itertools

n_runs = 20
langs = np.array(['deu', 'enu', 'nld'])


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

# def set_lang(sub_no):
#
#     return
