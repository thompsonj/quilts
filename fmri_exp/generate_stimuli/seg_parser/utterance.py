#!/usr/bin/python
"""Stores information for a single utterance."""
import os
from scipy import io
import numpy as np
import word


class Utterance:
    """Container object for an utterance."""

    def __init__(self, name, frame_rate, number):
        """Initialize Utterance object."""
        self.name = name
        self.frame_rate = float(frame_rate)
        self.number = int(number)
        self.prompt = ''
        self.transcript = ''
        self.nphone = 0
        self.npel = 0
        self.score = 0
        self.words = []
        self.curr_word = None

    def set_number(self, number):
        self.number = number

    def set_prompt(self, prompt):
        self.prompt = prompt

    def set_transcript(self, transcript):
        self.transcript = transcript

    def set_nphone(self, prompt):
        self.prompt = prompt

    def set_npel(self, npel):
        self.npel = npel

    def set_score(self, score):
        self.score = score

    def add_word(self, word_no, phn_no, phn, pel, start, stop):
        new_word = word.Word(word_no, phn_no, phn, pel, start, stop)
        self.words.append(new_word)
        self.curr_word = new_word
    #
    # def get_length_frame(self):
    #     """Return length of audio (frames) referenced by this Utterance."""
    #     if len(self.words) > 0:
    #         # length = self.words[-1].end - self.words[0].start # WRONG
    #     else:
    #         length = 0
    #     return length

    def get_all_phn_bounds_ms(self):
        """Get phoneme star and end points in ms (excluding silence)."""
        phn_start = [wd.get_phn_start_ms(self.frame_rate) for wd in self.words if wd.phones[0].phn != 'sil']  # exclude silence from segmentations
        phn_start_flat = [item for sublist in phn_start for item in sublist]
        phn_end = [wd.get_phn_end_ms(self.frame_rate) for wd in self.words if wd.phones[0].phn != 'sil']
        phn_end_flat = [item for sublist in phn_end for item in sublist]
        return np.array(phn_start_flat), np.array(phn_end_flat)

    def get_phn_lengths(self):
        phn_start, phn_end = self.get_all_phn_bounds_ms()
        lengths = phn_end - phn_start
        return lengths

    def get_length_ms(self):
        phn_start, phn_end = self.get_all_phn_bounds_ms()
        length = np.sum(phn_end - phn_start)
        return length

    def get_phns(self):
        phns = [wd.get_phns() for wd in self.words if wd.phones[0].phn != 'sil']
        phns_flat = [item for sublist in phns for item in sublist]
        return phns_flat

    def save_as_mat(self, base_seg_dir):
        """Save utterance contents as matlab .mat file."""
        if not os.path.isdir(base_seg_dir):
            os.mkdir(base_seg_dir)
        phn_start, phn_end = self.get_all_phn_bounds_ms()

        leng = phn_end - phn_start
        # Don't save segments less than 20ms,
        # Future: add short segments to neighboring segments
        # short_segments = np.where(leng < 20)[0]
        # if len(short_segments) > 0:
            # print len(short_segments), 'short segments encountered in utt .', self.number
            # print '...', short_segments
            # np.delete(phn_start, short_segments)
            # np.delete(phn_end, short_segments)
        phns = self.get_phns()
        to_save = {'phn_start': phn_start, 'phn_end': phn_end, 'phns': phns}
        fname = os.path.join(base_seg_dir, self.name + "_" + str(self.number))
        io.savemat(fname, to_save)
        np.save(fname, to_save)
        return to_save
