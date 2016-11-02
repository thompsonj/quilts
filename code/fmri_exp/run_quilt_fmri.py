#!/usr/bin/env python
"""Executes one full run of the phone-quilts fMRI experiment.

Stimuli for this experiment consist of 180 1-minute speech quilts made from
German, Dutch, and English speech.

The experiment uses a continuous design with the following
configuration (TR = 1.7s, no silent gap)

A different random order of stimulus presentation is used for each subject.

"""

from os import path, mkdir
import time
import sys
import numpy as np
from psychopy import core, visual, sound, gui, event, logging, prefs
from psychopy.tools.filetools import fromFile, toFile
from psychopy.hardware.emulator import launchScan


__author__ = "Jessica Thompson"


def run():
    """Execute one experimental run of the experiment."""
    settings = initialize_run()
    stimuli = load_stimuli(settings)
    present_stimuli(stimuli, settings)


def initialize_run():
    """Initalize settings and log file for this run of the experiment.

    Returns
    -------
    settings : dict
        Contains various experimental settings such as MR imaging parameters

        subject : Subject code use for loading/saving data, e.g. 's1'
        run : integer from 1-20
        debug : If true, don't display in full-screen mode
        home_dir : Parent directory, assumed to contain a directory 'fmri_exp'
                   containing one directory per subject
        TR : Time between acquisitions
        volumes : Number of whole-brain 3D volumes to collect this run
        sync : Character to use as the sync timing event; assumed to come at
               start of a volume
        resp : binary array indicating which blocks should be followed by a
               response probe
        skip : Number of volumes lacking a sync pulse at start of scan (for T1
               stabilization)
        scan_sound : In test mode only, play a tone as a reminder of scanner
                     noise
    """
    logging.console.setLevel(logging.DEBUG)
    if prefs.general['audioLib'][0] == 'pyo':
        # if pyo is the first lib in the list of preferred libs then we could
        # use small buffer
        # pygame sound is very bad with a small buffer though
        sound.init(16384, buffer=128)
    print 'Using %s(with %s) for sounds' % (sound.audioLib, sound.audioDriver)

    # settings for launchScan:
    try:
        settings = fromFile('settings.pickle')
    except:
        settings = {
            'subject': 's0',  # Subject code use for loading/saving data
            'run': 1,  # int from 1-20
            'debug': True,  # If true, print extra info
            'home_dir': '..',  # Parent directory, assumed to contain a
            # directory 'fmri_exp' containing one directory
            # per subject
            'TR': 1.7,  # Time between acquisitions
            'volumes': 371,  # Number of whole-brain 3D volumes / frames
            # this will be updated when known
            'sync': '5',  # Character to use as the sync timing event;
            # assumed to come at start of a volume
            'resp': ['0', '0', '0'],  # Blocks after which response is made
            'skip': 0,  # Number of volumes lacking a sync pulse at
            # start of scan (for T1 stabilization)
            'sound': True  # In test mode only, play a tone as a
            # reminder of scanner noise
        }
    # First, confirm subject number and run number
    subandrun = {'sub': settings['subject'], 'run':  settings['run']}
    info_dlg = gui.DlgFromDict(subandrun)
    sub = subandrun['sub']
    run = subandrun['run']
    # Load order info from file
    run_info = np.load(path.join(sub, sub + '_run' + str(run) + 'order.npy')).item()
    settings['subject'] = sub
    settings['run'] = run
    settings['volumes'] = int(run_info['vols'])
    settings['resp'] = run_info['resp']
    # Confirm all settings
    info_dlg = gui.DlgFromDict(settings, title='settings',
                               order=['subject', 'run', 'volumes', 'debug'])
    # Save settings for next run
    if info_dlg.OK:
        next_settings = settings.copy()
        if settings['run'] == 20:
            next_settings['run'] = 1  # Reset run when experiment is over
        else:
            next_settings['run'] += 1  # Increment for the next run
        toFile('settings.pickle', next_settings)
    else:
        core.quit()
    sub = settings['subject']
    run = settings['run']
    # Load order info again incase sub/run was altered in previous dialog box
    run_info = np.load(path.join(sub, sub + '_run' + str(run) +
                                 'order.npy')).item()
    settings['stimuli'] = run_info['stimuli']
    settings['lang'] = run_info['lang']
    if not path.isdir(settings['home_dir']):
        mkdir(settings['home_dir'])

    # Create dated log file
    date_str = time.strftime("%b_%d_%H%M", time.localtime())
    logfname = path.join('logs',
                         "%s_run%s_log_%s.log" %
                         (settings['subject'], settings['run'], date_str))
    log = logging.LogFile(logfname, level=logging.INFO, filemode='a')

    return settings


def load_stimuli(settings):
    """Load wav files of stimuli for a given run and subject.

    All stimuli are loaded at the beginning of the run to avoid long
    computations during the presentation of the stimuli.

    All stimuli are assumed to be located in the directory 'stimuli' in
    the parent directory.

    Assumes that the following files are in a directory named [subject]:
        s[subject_no]_run[run]order.npy

    Parameters
    ----------
    settings : dict
        Contains various experimental settings such as MR imaging parameters

    Returns
    -------
    stimuli :  ndarray of psychopy.sound.SoundPyo objects
        Sounds to be presented in this run in the order in which they

    """
    sub = settings['subject']
    run = settings['run']
    run_info = np.load(path.join(sub, sub + '_run' + str(run) +
                       'order.npy')).item()
    fnames = run_info['stimuli']
    stimuli = np.array(
        [sound.SoundPyo(f) for f in fnames])

    return stimuli


def present_stimuli(stimuli, settings):
    """Presents stimuli in the specified order timed by scanner triggers.

    Uses psychopy's launchScan module to add a test/scan switch on the main
    window. If in test mode, launchScan immitates the scanner by sending
    triggers (key press '5') and playing scanner noise when the scanner would
    be acquiring a volume.

    Saves the following log files to the subject's folder in a directory called
    'data' in the parent directory.
        [subject]_run[run]_key_presses_[date].txt
        [subject]_run[run]_resp_trials_[date].txt
        [subject]_run[run]_stimuli_timing_[date].txt
        [subject]_run[run]_stimuli_onsets_[date].npy
        [subject]_run[run]_volumes_onsets_[date].npy

    Saves the following date stamped copies of the order and jitter used for
    this run in the subject directory in the current directory:
        [subject]_run[run]_order_[date].npy
        [subject]_run[run]_jitter_[date].npy

    Parameters
    ----------
    stimuli :  ndarray of psychopy.sound.SoundPyo objects
        Sounds to be presented in this run in the order in which they were
        listed in either SimpleWavFileNames.txt or MixWavFileNames.txt
    order : ndarray
        Specifies order of presentation as indices into the stimuli array
    jitter : ndarray
        Inter-stimuli-intervals in multiples of TRs (2, 3 or 4)
    settings : dict
        Contains various experimental settings such as MR imaging parameters

    """
    if settings['debug']:
        win = visual.Window(fullscr=False)
    else:
        win = visual.Window(allowGUI=False, monitor='testMonitor',
                            fullscr=True)
    fixation = visual.ShapeStim(win,
                                vertices=((0, -0.055), (0, 0.055), (0, 0),
                                          (-0.045, 0), (0.045, 0)),
                                lineWidth=2,
                                closeShape=False,
                                lineColor='black')
    global_clock = core.Clock()

    # summary of run timing, for each key press:
    output = u'vol    onset key\n'
    for i in range(-1 * settings['skip'], 0):
        output += u'%d prescan skip (no sync)\n' % i
    volume_onsets = np.zeros(settings['volumes'])

    # Stimulus log
    stim_log = u'onset  stimulus_idx\n'
    stim_onsets = np.zeros(len(stimuli))

    # Response log
    n_resp = sum(settings['resp'])
    correct = 0
    resp_log = u'resp_onset block correct/incorrect \n'

    # Key press log
    key_code = settings['sync']
#    resp_text = visual.TextStim(win, height=.1, pos=(0, 0), color=win.rgb + 0.5)
    resp_text = visual.TextStim(win, height=.15, pos=(0, 0), color='black', bold=True)
    resp_text.setText(u"Which language did the last block sound like?  Press: \n 1 (Dutch) \n 2 (English) \n 3 (German)")

    output += u"  0    0.000 %s  [Start of scanning run, vol 0]\n" % key_code

    # Best if your script timing works without this, but this might be useful
    # sometimes
    infer_missed_sync = False

    # How long to allow before treating a "slow" sync as missed. Any slippage
    # is almost certainly due to timing issues with your script or PC, and not
    # MR scanner
    max_slippage = .5

    # variables
    n_tr_seen = 0
    block = 0
    base1tr = 5
    stimtr = 35
    waithrftr = 6
    responsetr = 2
    resttr = 11
    base2tr = 11
    # Dict to maintain state of the program
    states = {'sync_now': False,
              'rest': False,
              'response': False,
              'playblock': False,
              'baseline_start': True,
              'baseline_end': False,
              'listen_for_resp': False,
              'responded': False}
    # wait 5 TRs after the last stimulus
    duration = settings['volumes'] * settings['TR']
    # launch: operator selects Scan or Test (emulate); see API documentation
    vol = launchScan(win, settings, globalClock=global_clock)
    fixation.draw()
    win.flip()

    def parse_keys(states, output, resp_log, n_tr_seen, vol,
                   volume_onsets):
        core.wait(.0001, hogCPUperiod=0.00001)
        now = global_clock.getTime()
        all_keys = event.getKeys()
        # Log all key preses
        for key in all_keys:
            if key != settings['sync']:
                output += u"%3d  %7.3f %s\n" % (vol - 1, now,
                                                unicode(key))
        # Let the user close the program
        if 'escape' in all_keys:
            output += u'user cancel, '
            sys.exit()
        # Detect sync or infer it should have happened:
        if settings['sync'] in all_keys:
            states['sync_now'] = key_code
            onset = now
        if infer_missed_sync:
            expected_onset = vol * settings['TR']
            now = global_clock.getTime()
            if now > expected_onset + max_slippage:
                states['sync_now'] = u'(inferred onset)'
                onset = expected_onset
        # Save lang info so that you can mark responses as hit or not
        # Detect button press from subject, record performance on resp trials
        # if settings['resp'] in all_keys:
        #     output += (u"%3d  %7.3f %s\n" % (vol - 1, global_clock.getTime(),
        #                                      unicode(settings['resp'])))
        #     if resp:
        #         resp_log += u' hit\n'
        #     if resp and (global_clock.getTime() - resp_onset) > 3:
        #         resp_log += u' miss\n'
        #         resp = False
        if states['sync_now']:
            output += u"%3d  %7.3f %s\n" % (vol, onset, states['sync_now'])
            volume_onsets[vol - 1] = onset  # are we sure about -1 here?
            # Count tiggers to know when to present stimuli
            n_tr_seen += 1
            vol += 1
            print 'vol:', vol
            states['sync_now'] = False
        if states['listen_for_resp']:
            true_lang = unicode(settings['lang'][0, block-1]+1)
            if true_lang in all_keys:
                print 'correct response'
                resp_log += u'%7.3f %1f correct\n' % (now, block-1)
                output += (u"%3d  %7.3f correct identification of %s \n" %
                           (vol - 1, now, true_lang))
                np.add(correct, 1)
                states['responded'] = True
            else:
                for r in ['1', '2', '3']:
                    if r in all_keys:
                        print 'incorrect response', r
                        print 'true lang: ', true_lang
                        resp_log += u'%7.3f %1f incorrect\n' % (now, block-1)
                        output += (u"%3d  %7.3f incorrect:") % (vol - 1, now)
                        output += (u" Pressed %s instead of %s \n" % (r,
                                    true_lang))
                        states['responded'] = True

        return states, output, resp_log, n_tr_seen, vol, volume_onsets

    print 'duration: ', duration
    print 'vol %f of %f' % (vol, settings['volumes'])
    while vol < settings['volumes']:
        if global_clock.getTime() > duration:
            print 'overdue'
            break
        states, output, resp_log, n_tr_seen, vol, volume_onsets = parse_keys(states, output, resp_log, n_tr_seen, vol, volume_onsets)
        if states['baseline_start']:
            while n_tr_seen < base1tr:
                states, output, resp_log, n_tr_seen, vol, volume_onsets = parse_keys(states, output, resp_log, n_tr_seen, vol, volume_onsets)
            n_tr_seen = 0
            states['baseline_start'] = False
            states['playblock'] = True
        if states['playblock']:
            print 'playblock'
            for stimi in range(block*3, (block*3)+3):
                print settings['stimuli'][stimi]
                stim = stimuli[stimi]
                now = global_clock.getTime()
                stim.play()
                # Log onset time of stimulus presentation
                stim_log += u"%7.3f  %s\n" % (now, settings['stimuli'][stimi])
                print u"%7.3f  %s\n" % (now, settings['stimuli'][stimi])
                stim_onsets[stimi] = now
                while n_tr_seen < stimtr:
                    # Wait longer while stimuli are playing to let CPU catch up
                    core.wait(.01, hogCPUperiod=0.001)
                    states, output, resp_log, n_tr_seen, vol, volume_onsets = parse_keys(states, output, resp_log, n_tr_seen, vol, volume_onsets)
                n_tr_seen = 0
            states['playblock'] = False
            if settings['resp'][block]:
                print settings['resp'][block]
                states['response'] = True
            elif block == 2 and not settings['resp'][block]:
                states['baseline_end'] = True
            else:
                states['rest'] = True
            block += 1
        if states['response']:
            print 'response'
            # Wait for HRF
            while n_tr_seen < waithrftr:
                states, output, resp_log, n_tr_seen, vol, volume_onsets = parse_keys(states, output, resp_log, n_tr_seen, vol, volume_onsets)
            states['listen_for_resp'] = True
            # Present task question
            n_tr_seen = 0
            resp_text.draw()
            win.flip()
            while n_tr_seen < responsetr:
                states, output, resp_log, n_tr_seen, vol, volume_onsets = parse_keys(states, output, resp_log, n_tr_seen, vol, volume_onsets)
            n_tr_seen = 0
            fixation.draw()
            win.flip()
            states['response'] = False
            if block >= 3:  # block is incremented at the end of play block so after last block will be 3
                states['baseline_end'] = True
            else:
                states['rest'] = True
        if states['rest']:
            print 'rest'
            while n_tr_seen < resttr:
                if global_clock.getTime() > duration:
                    print 'overdue'
                    break
                states, output, resp_log, n_tr_seen, vol, volume_onsets = parse_keys(states, output, resp_log, n_tr_seen, vol, volume_onsets)
            n_tr_seen = 0
            states['rest'] = False
            states['playblock'] = True
            # If no response was made, note as missed
            if states['listen_for_resp'] and not states['responded']:
                true_lang = unicode(settings['lang'][0, block-1]+1)
                print 'no response'
                print 'true lang: ', true_lang
                resp_log += u'%7.3f %1f no response\n' % (now, block-1)
                output += (u"%3d  %7.3f no response:") % (vol - 1, now)
                output += (u" Should have pressed %s \n" % (true_lang))

            states['responded'] = False
            states['listen_for_resp'] = False
        if states['baseline_end']:
            print 'begin baseline_end'
#            core.wait(settings['TR']*base2tr, hogCPUperiod=0.2)
            print 'vol %f of %f' % (vol, settings['volumes'])
            # if a response was asked on the last block, don't need to wait as long
            if block == 3 and settings['resp'][2]:
                base2tr = 5
            while n_tr_seen < base2tr:
                states, output, resp_log, n_tr_seen, vol, volume_onsets = parse_keys(states, output, resp_log, n_tr_seen, vol, volume_onsets)
                # In the event that we are still waiting for trs that are not comming, get out
                if global_clock.getTime() > duration:
                    break
            n_tr_seen = 0
            states['baseline_end'] = False
            # If no response was made, note as missed
            if states['listen_for_resp'] and not states['responded']:
                true_lang = unicode(settings['lang'][0, block-1]+1)
                print 'no response'
                print 'true lang: ', true_lang
                resp_log += u'%7.3f %1f no response\n' % (now, block-1)
                output += (u"%3d  %7.3f no response:") % (vol - 1, now)
                output += (u" Should have pressed %s \n" % (true_lang))
            print 'vol %f of %f' % (vol, settings['volumes'])
            print 'end baseline_end'
            break

#            if settings['debug']:
#                counter.setText(u"%d volumes\n%.3f seconds" % (vol, onset))
#                counter.draw()
#                win.flip()

    # Save copy order and jitter used in this run
    # date_str = time.strftime("%b_%d_%H%M", time.localtime())
    # np.save(path.join(settings['subject'],
    #                   "%s_run%s" %
    #                   (settings['subject'], settings['run'], date_str)))
    print 'out of while loop'
    output += (u"End of scan (vol 0..%d = %d of %s).\n" %
               (vol - 1, vol, settings['volumes']))
    total_dur = global_clock.getTime()
    date_str = time.strftime("%b_%d_%H%M", time.localtime())
    output += u"Total duration = %7.3f sec (%7.3f min)" % (total_dur,
                                                           total_dur / 60)
    print "Total duration = %7.3f sec (%7.3f min)" % (total_dur, total_dur / 60)

    # Save data and log files
    # datapath = path.join(settings['home_dir'], 'data', settings['subject'])
    datapath = path.join(settings['subject'], 'run_data')
    if not path.isdir(datapath):
        mkdir(datapath)

    # Write log of key presses/volumes
    fname = path.join(datapath,
                      "%s_run%s_all_keys_%s" %
                      (settings['subject'], settings['run'], date_str))

    data_file = open(fname + '.txt', 'w')
    data_file.write(output)
    data_file.close()

    # Wite log of resp trials
    fname = path.join(datapath,
                      "%s_run%s_resp_blocks_%s" %
                      (settings['subject'], settings['run'], date_str))

    data_file = open(fname + '.txt', 'w')
    data_file.write(resp_log)
    data_file.close()

    # Write log of stimulus presentations
    fname = path.join(datapath,
                      "%s_run%s_stimuli_timing_%s" %
                      (settings['subject'], settings['run'], date_str))

    data_file = open(fname + '.txt', 'w')
    data_file.write(stim_log)
    data_file.close()

    # Save numpy arrays of onset timing for easy analysis
    fname = path.join(datapath,
                      "%s_run%s_stimuli_onsets_%s" %
                      (settings['subject'], settings['run'], date_str))
    np.save(fname, stim_onsets)

    # data4prts = np.concatenate(
    #     (np.matrix(np.floor(stim_onsets / settings['TR'])),
    #      np.matrix(order))).T
    # savedict = {'stimuliinfo': data4prts}
    # fname = path.join(datapath,
    #                   "%s_run%02d_data4prts_%s" %
    #                   (settings['subject'], int(settings['run']), date_str))
    # sio.savemat(fname, savedict)

    fname = path.join(datapath,
                      "%s_run%s_volume_onsets_%s" %
                      (settings['subject'], settings['run'], date_str))
    np.save(fname, volume_onsets)

    # Save settings for this run
    fname = path.join(datapath,
                      "%s_run%s_settings_%s" %
                      (settings['subject'], settings['run'], date_str))
    np.save(fname, settings)
    core.quit()


def main():
    run()
    print('Done')


if __name__ == '__main__':
    main()
