#!/usr/bin/python
"""Container object for a phoneme."""
import phone as ph


class Word:
    """Container object for a Word."""

    def __init__(self, word_no, phn, phn_no, pel, start, end):
        """ """
        self.no = int(word_no)
        self.start = int(start)
        self.end = int(end)
        self.phones = [ph.Phone(phn, phn_no, pel, start, end)]
        self.curr_phn = self.phones[0]  # consider renaming

    def add_phone(self, phn, phn_no, pel, start, end):
        """ """
        new_phn = ph.Phone(phn, phn_no, pel, start, end)
        self.phones.append(new_phn)
        self.curr_phn = new_phn
        self.end = int(end)

    def get_phn_start_frame(self):
        phone_start_frame = [phn.get_start_frame for phn in self.phones]
        return phone_start_frame

    def get_phn_start_ms(self, frame_rate):
        phone_start_ms = [phn.get_start_ms(frame_rate) for phn in self.phones]
        return phone_start_ms

    def get_phn_end_frame(self):
        phone_end_frame = [phn.get_end_frame for phn in self.phones]
        return phone_end_frame

    def get_phn_end_ms(self, frame_rate):
        phone_end_ms = [phn.get_end_ms(frame_rate) for phn in self.phones]
        return phone_end_ms

    def get_phns(self):
        phns = [phn.phn for phn in self.phones]
        return phns
