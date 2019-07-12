#!/usr/bin/python
"""Parse Nuance format segmentation file.

methods:
parse_seg - read contents of seg file into a Segmentation object
parse_seg_list - parse each seg file in seg list
get_total_length - get total length of audio for a set of segfiles

"""
import sys
import os.path
import glob
import cPickle as pickle
import numpy as np
import segmentation
import parse_skp as pskp


def parse_seg(segfile):
    """Get segmentation info from a segfile and fill Segmentation object.

    Arguments:
    segfile - path to a Nuance segmentation file to be parsed

    Returns:
    seg - Segmentation object filled with seg info
    """
    # This base name will be used to match segmentations to wav files
    base = os.path.splitext(os.path.basename(segfile))[0]
    seg = segmentation.Segmentation(base)
    try:
        f = open(segfile, 'r')
        txt = f.readlines()
        f.close()
    except IOError as e:
        print "I/O error({0}): {1}".format(e.errno, e.strerror)
    table = False
    # Parse seg file line by line in a single pass
    for line in txt:
        e = line.split()
        if len(e) > 0:
            if e[0] == 'frame_rate':
                seg.set_frame_rate(float(e[2]))  # e.g. frame_rate = 100
            # adding an utterance updates the pointer to the current utterance
            elif e[0] == 'utterance_number':
                seg.add_utterance(e[2])  # e.g. utterance_number = 0
            elif e[0] == 'prompt':
                prompt = line.split('"')[1]
                seg.curr_utt.set_prompt(prompt)
            elif e[0] == 'transcription':
                transcript = line.split('"')[1]
                seg.curr_utt.set_transcript(transcript)
            elif e[0] == 'number_of_phonemes':
                seg.curr_utt.set_nphone(e[2])
            elif e[0] == 'number_of_pels':
                seg.curr_utt.set_npel(e[2])
            elif e[0] == 'score':
                seg.curr_utt.set_score(e[2])
            elif e[0] == 'TABLE':
                table = True
            elif e[0] == 'ENDTABLE':
                table = False
            elif table and e[0] != ';':
                curr_word = seg.curr_utt.curr_word
                # If this line is the start of a new word, create new Word
                # Initialization of a new Word, automatically creates a new
                # Phone and a new Pel
                if curr_word is None or curr_word.no != int(e[0]):
                    seg.curr_utt.add_word(e[0], e[1], e[2], e[3], e[4], e[5])
                # if this line is the start of a new phoneme, create new Phone
                # Initialization of a new Phone automatically creates a new Pel
                elif curr_word.curr_phn.no != int(e[1]):
                    curr_word.add_phone(e[1], e[2], e[3], e[4], e[5])
                # Each line of the table represents a new Pel
                else:
                    curr_word.curr_phn.add_pel(e[3], e[4], e[5])
    return seg


def parse_seg_list(seglist):
    """Return a list of Segmentation objects for each seg in seglist.

    This will take up a lot of memory if seglist contains many segs.

    Arguments:
    seglist - a directory containing seg files or a text file containing the
        path to one seg file per line

    Returns:
    segs - list of Segmentation objects
    """
    # Get list of seg files to parse
    if os.path.isdir(seglist):
        files = glob.glob(os.path.join(seglist, '*.seg'))
    else:
        try:
            f = open(seglist, 'r')
            files = f.readlines()
            f.close()
        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
    # Parse each segfile
    # segs = []
    for segfile in files:
        print segfile
        seg = parse_seg(segfile)
        
        fname = seglist + seg.name + '.pkl'
        f = open(fname, 'wb')
        pickle.dump(seg, f)
        f.close()
        # segs.append(seg)
    print 'done'

def get_lengths(seglist, skpdir):
    """Get list of all utterances sorted by their length in ascending order.

    Arguments:
    seglist - a directory containing seg files or a text file containing the
        path to one seg file per line

    Returns:
    name - name of segmentation for each utterance, sorted according to
        utterance length
    utt_idx_srt - index of utterances within their segmentation, sorted
        according to utterance length
    utt_len_srt - list of sorted lengths (in seconds) for all utterances
    """
    print '\nSorting lengths of all utterances'
    # Get list of seg files to parse
    files = get_seg_file_list(seglist)
    name = []
    utt_idx = []
    utt_len = []
    for segfile in files:
        seg = parse_seg(segfile)
        skpfile = skpdir + seg.name + '.skp'
        skp = pskp.parse_skp(skpfile)
        for utt in seg.utterances:
            if utt.number in skp.skipped:
                continue
            else:
                name.append(seg.name)
                utt_idx.append(i)
                utt_len.append(utt.get_length_frame()/seg.frame_rate)
    # Sort name, utt_index, and utt_length based to utt_length
    name = np.array(name)
    utt_idx = np.array(utt_idx)
    utt_len = np.array(utt_len)
    # Get index to sort all three arrays based on length
    sort_idx = np.argsort(utt_len)
    utt_len_srt = utt_len[sort_idx]
    utt_idx_srt = utt_idx[sort_idx]
    name_srt = name[sort_idx]
    # Reverse order to make descending order
    name_srt = name_srt[::-1]
    utt_idx_srt = utt_idx_srt[::-1]
    utt_len_srt = utt_len_srt[::-1]
    return name_srt, utt_idx_srt, utt_len_srt


def save_longest_utts(segdir, skpdir, savedir):
    """."""
    name_srt, utt_idx_srt, utt_len_srt = get_lengths(segdir, skpdir)
    print '\nSaving the longest unskipped utterances adding up to 1hr'
    total_time = np.cumsum(utt_len_srt)  # in secs, starting from longest
    idx = np.where(total_time < 3600)
    idx = idx[0][-1]+1  # funky shit
    if idx < 30:
        idx = 30
    print 'total length of segs to save: ', total_time[idx]

    for i, name in enumerate(name_srt[:idx]):
        segfile = segdir + name + '.seg'
        seg = parse_seg(segfile)
        base_segdir = os.path.join(segdir, seg.name)
        seg.utterances[utt_idx_srt[i]].save_as_mat(base_segdir, utt_idx_srt[i])

    np.save(savedir+'name_srt_1hr.npy', name_srt[:idx])
    unique = np.unique(name_srt[:idx])
    np.savetxt(savedir+'unique_name_srt_1hr.txt', unique, fmt='%s')
    np.save(savedir+'utt_idx_srt_1hr.npy', utt_idx_srt[:idx])
    np.save(savedir+'utt_len_srt_1hr.npy', utt_len_srt[:idx])


def get_total_length(seglist):
    """Get total length of audio (seconds) for segmentations in seglist.

    Arguments:
    seglist - a directory containing seg files or a text file containing the
        path to one seg file per line

    Returns:
    length - the total length in seconds of the audio corresponding to the
        segmentations in seglist
    """
    # Get list of seg files to parse
    files = get_seg_file_list(seglist)

    # Parse each segfile
    length = 0
    for segfile in files:
        print segfile
        seg = parse_seg(segfile)
        length = length + seg.get_length_s()
    print length
    return length


def write_skp_file(segdir, skpsegdir, skpdir):
    """Compare transcriptions of two sets to seg files to write skp files.

    Assumes that skp files exist but are empty (contain only the number of
    utterances).

    Arguments:
    segdir - list of segfiles or directory of seg files
    skpseg - directory of segfiles with skipped utterances
    skpdir - directory where to append to skpfiles
    """
    segfiles = get_seg_file_list(segdir)

    for segfile in segfiles:
        print segfile
        seg = parse_seg(segfile)
        skpsegfile = skpsegdir + seg.name + '.seg'
        print skpsegfile
        skpseg = parse_seg(skpsegfile)
        skp_trans_list = skpseg.get_transcription_list()
        try:
            f = open(skpdir + seg.name + '.skp', 'a')
        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)

        skp_utts = []
        for i, utt in enumerate(seg.utterances):
            if utt.transcript not in skp_trans_list:
                skp_utts.append(str(i) + '\n')
        print skp_utts
        print 'number of skipped utterances: ', len(skp_utts)
        f.writelines(skp_utts)
        f.close()


def get_seg_file_list(seglist):
    """Get list of seg files.

    Arguments:
    seglist - a directory containing seg files or a text file containing the
        path to one seg file per line

    Returns:
    files - a list of seg file names
    """
    # if os.path.isdir(seglist):
    files = glob.glob(seglist + '*.seg')
    # else:
    #     try:
    #         f = open(seglist, 'r')
    #         files = f.readlines()
    #         f.close()
    #     except IOError as e:
    #         print "I/O error({0}): {1}".format(e.errno, e.strerror)
    return files

if __name__ == "__main__":
    """First argument assumed to be the fullpath to the seg file to parse."""
    segfile = sys.argv[1]
    print segfile
    parse_seg(segfile)
