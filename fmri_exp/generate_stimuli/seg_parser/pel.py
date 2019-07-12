#!/usr/bin/python
"""Container object for a pel."""


class Pel:
    """Container object for a pel."""

    def __init__(self, pel, start, end):
        self.pel = int(pel)
        self.start = int(start)
        self.end = int(end)
