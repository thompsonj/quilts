from glob import glob
import cPickle as pickle
import dill
import os
import numpy as np
import language


def parse_lang(segdir, skpdir, lang_name, min_seg, langdir):
    lang = language.Language(lang_name, min_seg)
    pickles = glob(os.path.join(segdir, '*.pkl'))
    for p in pickles:
        # if lang.n_utts >= 8826 and len(np.unique(lang.spkrs))>=120:
            # as many utts for nld,
            # 120 spkrs so hopefully we find male and female
            # Don't save all because laptop could not handle
            # break
        f = open(p, 'rb')
        seg = pickle.load(f)
        f.close()
        lang.add_utt_info(seg, skpdir)
    savename = os.path.join(langdir, lang_name + '_min' + str(min_seg) + '.pkl')
    s = open(savename, 'wb')
    print 'saving ', savename
    # dill.dump(lang, s)
    pickle.dump(lang, s) # this should work now because I found I was passing a method instead of calling it with ()
    s.close()
    return lang
