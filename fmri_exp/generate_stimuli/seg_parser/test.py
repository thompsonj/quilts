import parse_seg as ps
import get_quilt_info as gqi
import parse_lang as pl
from glob import glob
import os
import cPickle as pickle
import dill
# segdir = '/Users/jthompson/data/enu/inh/seg/'
# skpdir = '/Users/jthompson/data/enu/inh/skp/'
# lang_name = 'nld'
# lang_name = 'deu'
lang_name = 'enu'
langdir = '/Users/jthompson/data/' + lang_name +'/'
segdir = langdir + 'seg/'
skpdir = langdir + '/skp/'
# savedir = '/Users/jthompson/data/' + lang + '/'
savedir = '/Users/jthompson/Dropbox/quilts/stimuli/' + lang_name + '/proposed_utts/'

# skpsegdir = '/Users/jthompson/data/enu/inh/seg_filtered/'
#
# skpfile = skpdir + 'hub4_0000_f1.skp'
#
# s = ps.parse_seg('/Users/jthompson/data/enu/seg/hub4_0000_f1.seg')
#
# phn_start_flat, phn_end_flat = s.utterances[0].get_all_phn_bounds_ms()
#
# s.save_all_utts_as_mat('/Users/jthompson/data/enu/inh/seg/')
#
# ps.write_skp_file(segdir, skpsegdir, skpdir)
#
# name_srt, utt_idx_srt, utt_len_srt = ps.get_lengths(segdir, skpdir)


name, utt_idx, utt_len = ps.get_lengths(segdir, skpdir)


ps.save_longest_utts(segdir, skpdir, savedir)

segs = glob(segdir + '*.seg')
for seg in segs:
    (path, f) = os.path.split(seg)
    os.rename(seg, path + '/' + f[4:])

skps = glob(skpdir + '*.skp')
for skp in skps:
    (path, f) = os.path.split(skp)
    os.rename(skp, path + '/' + f[4:])

# Save mat files for all utterances
gqi.save_all_utts(segdir, skpdir)

# Pickle all segmentation objects
ps.parse_seg_list(segdir)


# parse language
min_seg=20
lang20 = pl.parse_lang(segdir, skpdir, lang_name, min_seg, langdir)
mean_phn_len = lang.get_len_per_phn()

pickle.dump(mean_phn_len, (open(langdir + 'mean_phn_len.pkl', 'wb')))
lang.phn_len_count[10]
pickle.dump(lang.phn_len_count, (open(langdir + 'phn_len_count.pkl', 'wb')))

gqi.select_audio_to_quilt(lang_name, langdir, min_seg, savedir)
