#!/usr/bin/python
"""Container object for a language."""
import os
import numpy as np
import parse_skp as ps


class Language:
    """Container object for language.
    """

    def __init__(self, name, min_phn_length):
        """Initialise with name of language.

        Input arguments:
        name:   'enu', 'deu', or 'nld'
        """
        self.lang = name
        self.min_phn_len = min_phn_length
        self.spkrs = np.array([])
        self.utt_idx = np.array([])
        self.utt_len = np.array([])
        self.phn_count = dict()
        self.phn_len = dict()
        self.phn_len_count = dict()
        self.n_utts = 0

    def add_utt_info(self, seg, skpdir):
        """Add information from Segmentation object seg."""
        skp = ps.parse_skp(skpdir + seg.name + '.skp')
        for utt in seg.utterances:
            if utt.number not in skp.skipped:
                self.n_utts += 1
                self.spkrs = np.append(self.spkrs, seg.name)
                self.utt_idx = np.append(self.utt_idx, utt.number) # does utt.number really match up with the suffixes?
                self.utt_len = np.append(self.utt_len, utt.get_length_ms())
                lengths = utt.get_phn_lengths()
                for i, phn in enumerate(utt.get_phns()):
                    if lengths[i] >= self.min_phn_len:
                        if lengths[i] in self.phn_len_count:
                            if phn in self.phn_len_count[lengths[i]]:
                                self.phn_len_count[lengths[i]][phn] += 1
                            else:
                                self.phn_len_count[lengths[i]][phn] = 1
                        else:
                            self.phn_len_count[lengths[i]] = dict()
                            self.phn_len_count[lengths[i]][phn] = 1
                        if phn in self.phn_count:
                            self.phn_count[phn] += 1
                        else:
                            self.phn_count[phn] = 1
                        if phn in self.phn_len:
                            self.phn_len[phn] += lengths[i]
                        else:
                            self.phn_len[phn] = lengths[i]

    def get_len_per_phn(self):
        mean_phn_len = dict()
        assert len(self.phn_count) == len(self.phn_len)
        for phn in self.phn_count:
            mean_phn_len[phn] = self.phn_len[phn]/self.phn_count[phn]
        return mean_phn_len
