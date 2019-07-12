#!/usr/bin/python
"""Container object for a phoneme."""
import pel as p


class Phone:
    """Container object for a phoneme."""

    def __init__(self, phn_no, phn, pel, start, end):
        """ """
        self.no = int(phn_no)
        self.phn = phn
        self.start = int(start)
        self.end = int(end)
        self.pels = [p.Pel(pel, start, end)]

    def add_pel(self,  pel, start, end):
        """ """
        new_pel = p.Pel(pel, start, end)
        self.pels.append(new_pel)
        self.end = int(end)

    def get_start_frame(self):
        return self.start

    def get_start_ms(self, frame_rate):
        frame_rate_ms = frame_rate/1000.0
        return self.start / frame_rate_ms

    def get_end_frame(self):
        return self.end

    def get_end_ms(self, frame_rate):
        frame_rate_ms = frame_rate/1000.0
        return (self.end+1) / frame_rate_ms # +1 because frame has length

    def get_length_ms(self, frame_rate):
        length = self.get_end_ms(frame_rate) - self.get_start_ms(frame_rate)
        return length
