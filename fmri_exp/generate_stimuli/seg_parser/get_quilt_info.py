"""."""
import os.path
import numpy as np
import dill
import cPickle as pickle
import parse_seg as ps
from glob import glob

def save_all_utts(segdir, skpdir):
    # this could be a lot faster if we operated on unique names instead of parsing the same seg file over and over
    segs = glob(segdir + '*.seg')
    for segfile in segs:
        seg = ps.parse_seg(segfile)
        print seg.name
        base_segdir = os.path.join(segdir, seg.name)
        seg.save_all_utts_as_mat(segdir, skpdir)
    print 'done'

def save_longest_utts(segdir, skpdir, savedir):
    """."""
    name, utt_idx, utt_len = ps.get_lengths(segdir, skpdir)
    print '\nSaving the longest unskipped utterances adding up to 1hr'
    total_time = np.cumsum(utt_len)  # in secs, starting from longest
    idx = np.where(total_time < 3600)
    idx = idx[0][-1]+1  # funky shit
    if idx < 30:
        idx = 30
    print 'total length of segs to save: ', total_time[idx]

    for i, name in enumerate(name[:idx]):
        segfile = segdir + name + '.seg'
        seg = ps.parse_seg(segfile)
        base_segdir = os.path.join(segdir, seg.name)
        seg.utterances[utt_idx[i]].save_as_mat(base_segdir, utt_idx[i])

    np.save(savedir+'name_1hr.npy', name[:idx])
    unique = np.unique(name[:idx])
    np.savetxt(savedir+'unique_name_1hr.txt', unique, fmt='%s')
    np.save(savedir+'utt_idx_1hr.npy', utt_idx[:idx])
    np.save(savedir+'utt_len_1hr.npy', utt_len[:idx])

def get_length_per_spkr(segdir, skpdir, savedir):
    name, utt_idx, utt_len = ps.get_lengths(segdir, skpdir)
    unique_name = np.unique(name)
    lengths = np.zeros(len(unique_name))
    for i in range(len(unique_name)):
        idx = np.where(name == unique_name[i])
        lengths[i] = np.sum(utt_len[idx])

    return unique_name, lengths


def select_audio_to_quilt(langname, langdir, min_phn, savedir, length_to_quilt = 120, n_spkr = 120):
    """Save audio clips to be used for quilting.

    Loop over utterances sorted in ascending length (because we want to make
    quilts out of long clips). If that utterance is the desired length for
    quilting, save it as is. Otherwise, try to concatenate adjacent utterances
    until the total length reaches desired length. Concatenate audio and save
    concatenated version with a record of which utterances were concatenated.
    Continue until you have saved desired length clips for desired number of
    unique speakers.


    """
    # length_to_quilt = 120  # Want to pass the quilting algorithm 2m of audio
    # n_spkr = 120  # maybe increase to 120 just so that you can exclude some by listening after
    # name, utt_idx, utt_len = ps.get_lengths(segdir, skpdir)
    langfile = langdir + langname + '_min' + str(min_phn) + '.pkl'
    f = open(langfile, 'rb')
    print 'loading ', langfile
    # lang = dill.load(f)
    lang = pickle.load(f)
    f.close()
    name = lang.spkrs
    utt_len = lang.utt_len
    utt_idx = lang.utt_idx
    sortidx = np.argsort(utt_len)[::-1]  # descending order, take long segments first
    name = name[sortidx]
    utt_len = utt_len[sortidx]
    utt_idx = utt_idx[sortidx]
    spkr_list = []
    # length = 0
    print 'searching for utterances.'
    for i in range(len(utt_len)):
        if len(spkr_list) == n_spkr:
            break
        spkr_idx = np.where(name == name[i])[0]  # index of all utterances from this speaker
        all_len_spkr = utt_len[spkr_idx]
        # print 'try utt: ', i
        if name[i] in spkr_list:
            # if this speaker already has a quilt, move to the next utterance
            print name[i], ' in spkr_list'
            continue  # unique speakers only
        elif 'wsj_' not in name[i]:  # only look at nrc_ speech
            # print name[i]
            continue
        # elif utt_len[i] >= length_to_quilt:
        elif utt_len[i] >=  length_to_quilt*1000:
            # save audio to savedir
            print 'save utt: ', utt_idx[i]
            print 'length: ', utt_len[i]
            spkr_list.append(name[i])
            np.savetxt(savedir + name[i] + '.txt', np.matrix(utt_idx[i]), fmt='%i')
            continue
        elif np.sum(all_len_spkr) < length_to_quilt*1000:
            # If this speaker does not have long enough utterances, move on
            print 'not long enough: ', np.sum(all_len_spkr), '<', length_to_quilt
            continue
        else:
            spkr_utt_idx = utt_idx[spkr_idx]  # indexes utterances for this speaker
            utts_to_cat = utt_idx[i]
            print 'begin while'
            # utt_idx refers to the index within the utterances of that speaker
            quilt_len = utt_len[spkr_idx[spkr_utt_idx==utt_idx[i]][0]]
            while quilt_len < length_to_quilt*1000:
                # if this is not the last utterance of this speaker
                if np.max(utts_to_cat) != np.max(spkr_utt_idx):
                    new_utt = np.min(spkr_utt_idx[spkr_utt_idx > np.max(utts_to_cat)])
                    utts_to_cat = np.append(utts_to_cat, new_utt)
                    quilt_len = quilt_len + utt_len[spkr_idx[spkr_utt_idx==new_utt][0]]
                # if this is not the first utterance of this speaker
                elif np.min(utts_to_cat) != np.min(spkr_utt_idx):
                    new_utt = np.max(spkr_utt_idx[spkr_utt_idx < np.min(utts_to_cat)])
                    utts_to_cat = np.append(new_utt, utts_to_cat)
                    quilt_len = quilt_len + utt_len[spkr_idx[spkr_utt_idx==new_utt][0]]

                else:
                    # error: speaker had more than enough but we didn't find it
                    print 'error: speaker had enough audio but concat failed.'
                    print np.sum(all_len_spkr)
                    print quilt_len
                    print np.max(utts_to_cat)+1
                    print np.min(utts_to_cat)-1
                    print spkr_utt_idx
                    break
            assert quilt_len >= length_to_quilt*1000
            # save audio to savedir
            print 'save utts: ', utts_to_cat
            print 'quilt_len: ', quilt_len
            spkr_list.append(name[i])
            np.savetxt(savedir + name[i] + '.txt', utts_to_cat, fmt='%i')
    np.savetxt(langdir + 'selected_spkrs.txt', spkr_list, fmt='%s')
