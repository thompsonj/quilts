#!/usr/bin/python
"""Container object for a segmentation."""
import utterance
import os
import parse_skp as ps


class Segmentation:
    """Container object for segmentation.
    """

    def __init__(self, name):
        """Initialise with basename of seg file.

        Input arguments:
        name:   Basename of the segmentation, corresponds to the nwv with the
                same basename
        """
        self.name = name
        self.frame_rate = None
        self.utterances = []
        self.curr_utt = None

    def set_name(self, name):
        """Update the base name."""
        self.name = name

    def set_frame_rate(self, fr):
        """Update the frame rate.
        Input arguments:
        fr:     Frame rate in number of frames per seconds
        """
        self.frame_rate = float(fr)

    def add_utterance(self, utt_no):
        new_utt = utterance.Utterance(self.name, self.frame_rate, utt_no)
        self.utterances.append(new_utt)
        self.curr_utt = new_utt

    def get_length_s(self):
        """Return length of audio (seconds) referenced in this Segmentation."""
        length = 0
        for utt in self.utterances:
            length = length + utt.get_length_frame()
        length_s = length/self.frame_rate
        return length_s

    def save_all_utts_as_mat(self, segdir, skpdir):
        """Save matlab .mat files for each utterance."""
        # Create a directory <basename> in segdir if it does not already exist
        base_seg_dir = os.path.join(segdir, self.name)
        skpfile = skpdir + self.name + '.skp'
        skp = ps.parse_skp(skpfile)
        if not os.path.isdir(base_seg_dir):
            os.mkdir(base_seg_dir)
        for utt in self.utterances:
            if utt.number not in skp.skipped:
                utt.save_as_mat(base_seg_dir)

    # def save_utterances(self, segdir, utt_idx):
    #     """"""
    #     # Create a directory <basename> in segdir if it does not already exist
    #     base_seg_dir = os.path.join(segdir, self.name)
    #     if not os.path.isdir(base_seg_dir):
    #         os.mkdir(base_seg_dir)
    #     for i, utt in enumerate(self.utterances[utt_idx]):
    #         utt.save_as_mat(base_seg_dir, utt_idx[i])

    def get_transcription_list(self):
        """Return list of all transcriptions in this segmentation."""
        trans_list = []
        for utt in self.utterances:
            trans_list.append(utt.transcript)
        return trans_list
