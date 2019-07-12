#!/usr/bin/python
"""Parse Nuance format skip file.

methods:

"""
import sys
import os
import skip


def parse_skp(skpfile):
    """Parse Nuance format skip file."""
    # This base name will be used to match skp, seg, wav files
    base = os.path.splitext(os.path.basename(skpfile))[0]
    print base
    try:
        f = open(skpfile, 'r')
        txt = f.readlines()
    except IOError as e:
        print "I/O error({0}): {1}".format(e.errno, e.strerror)
    lines = [i[:-1] for i in txt]
    n_utterances = lines[0]
    skipped = [int(i) for i in lines[1:]]
    skp = skip.Skip(base, n_utterances, skipped)
    return skp

if __name__ == "__main__":
    """First argument assumed to be the fullpath to the skp file to parse."""
    skpfile = sys.argv[1]
    print skpfile
    parse_skp(skpfile)
