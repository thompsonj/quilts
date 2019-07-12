#!/usr/bin/python
"""Container object for a skip file."""


class Skip:
    """Container object for skip file."""

    def __init__(self, name,  n_utterances, skipped):
        """Initialise with basename of skp file.

        Arguments:
        name - Basename of the skp file, corresponds to the nwv with the
            same basename
        n_utterances - number of utterances
        skipped - list of skipped utterances, refers to order in nwv file
        """
        self.name = name
        self.n_utterances = n_utterances
        self.skipped = skipped
