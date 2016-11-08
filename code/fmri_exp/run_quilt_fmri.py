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
    run_info = np.load(path.join(sub, sub + '_run' + str(run) +
                       'order.npy')).item()
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
    the present working directory.

    Assumes that order info is contained in numpy files in one directory per
    subject, for example:
        s1/s1_run1_order.npy
        s1/s1_run2_order.npy
        ...
        s1/s1_run20_order.npy

    Parameters
    ----------
    settings : dict
        Contains various experimental settings such as MR imaging parameters

    Returns
    -------
    stimuli :  ndarray of psychopy.sound.SoundPyo objects
        Sounds to be presented in this run in the order in which they

    """
    fnames = settings['stimuli']
    stimuli = np.array(
        [sound.SoundPyo(f) for f in fnames])

    return stimuli


def present_stimuli(stimuli, settings):
    """Present stimuli in the specified order timed by scanner triggers.

    Uses psychopy's launchScan module which adds a test/scan switch on the
    first window. If in test mode, launchScan immitates the scanner by sending
    triggers (key press '5') and playing scanner noise when the scanner would
    be acquiring a volume (if settings['sound'] == True).

    The inner function parse_keys() is called regularly to listen for key
    presses and update the dictionary of state variables accordingly. The rest
    of the function uses various state variables to determine what actions to
    take, e.g. play a sound, present a response probe, rest.

    Saves the following files in a directory called 'run_data' in the subject
    directory.
        [subject]_run[run]_all_keys_[date].txt: All key presses recorded
        [subject]_run[run]_resp_blocks_[date].txt: Task performance
        [subject]_run[run]_stimuli_timing_[date].txt: Stim name and onset time
        [subject]_run[run]_settings_[date].npy: settings dict used this run
        [subject]_run[run]_stimuli_onsets_[date].npy: Stim name and onset time
        [subject]_run[run]_vol_onsets_[date].npy: Onset time of all triggers

    An automated log file is also saved in logs/

    Parameters
    ----------
    stimuli :  ndarray of psychopy.sound.SoundPyo objects
        Sounds to be presented in this run in the order in which they are to be
        presented. As determined by order.py
    settings : dict
        Contains various experimental settings such as MR imaging parameters

    TODO:
    - present stimuli is too complex
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
    out = u'vol    onset key\n'
    for i in range(-1 * settings['skip'], 0):
        out += u'%d prescan skip (no sync)\n' % i
    vol_onsets = np.zeros(settings['volumes'])

    # Stimulus log
    stim_log = u'onset  stimulus_idx\n'
    stim_onsets = np.zeros(len(stimuli))

    # Response log
    n_resp = sum(settings['resp'])
    cor = 0
    resp_log = u'resp_onset block correct/incorrect \n'

    # Key press log
    key_code = settings['sync']
    resp_text = visual.TextStim(win, height=.15, pos=(0, 0), color='black',
                                bold=True)
    question = u"Which language did the last block sound like?  "
    options = u"Press: \n 1 (Dutch) \n 2 (English) \n 3 (German)"
    resp_text.setText(question + options)

    out += u"  0    0.000 %s  [Start of scanning run, vol 0]\n" % key_code

    # Best if your script timing works without this, but this might be useful
    # sometimes
    infer_missed_sync = False

    # How long to allow before treating a "slow" sync as missed. Any slippage
    # is almost certainly due to timing issues with your script or PC, and not
    # MR scanner
    max_slippage = .5

    # Timing variables
    n_tr = 0
    block = 0
    base1tr = 5
    stimtr = 35
    waithrftr = 6
    responsetr = 2
    resttr = 11
    base2tr = 11
    stim = None
    # Dict to maintain state of the program
    state = {'sync_now': False,
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

    def parse_keys():
        core.wait(.0001, hogCPUperiod=0.00001)
        all_keys = event.getKeys()
        new_ntr = n_tr
        new_vol = vol
        newcorrect = cor
        newout = out
        newrl = resp_log
        now = global_clock.getTime()
        # Log all key preses
        for key in all_keys:
            if key != settings['sync']:
                newout = newout + u"%3d  %7.3f %s\n" % (vol - 1, now, str(key))
        # Let the user close the program
        if 'escape' in all_keys:
            newout = newout + u'user cancel, '
            if stim:
                stim.stop()
            core.quit()
            sys.exit()
        # Detect sync or infer it should have happened:
        if settings['sync'] in all_keys:
            state['sync_now'] = key_code
            onset = now
        if infer_missed_sync:
            expected_onset = vol * settings['TR']
            now = global_clock.getTime()
            if now > expected_onset + max_slippage:
                state['sync_now'] = u'(inferred onset)'
                onset = expected_onset
        if state['sync_now']:
            newout = newout + u"%3d  %7.3f %s\n" % (vol, onset,
                                                    state['sync_now'])
            vol_onsets[vol - 1] = onset  # are we sure about -1 here?
            # Count tiggers to know when to present stimuli
            new_ntr = n_tr + 1
            new_vol = vol + 1
            print 'vol:', vol
            state['sync_now'] = False
        # Detect button press from subject, record performance on probes
        if state['listen_for_resp']:
            true_lang = unicode(settings['lang'][0, block-1]+1)
            if true_lang in all_keys:
                print 'correct response'
                newrl = resp_log + u'%7.3f %1.0f correct\n' % (now, block-1)
                newout = newout + (u"%3d  %7.3f correct ID of %s \n" %
                                   (vol - 1, now, true_lang))
                newcorrect = newcorrect + 1
                state['responded'] = True
            else:
                for r in ['1', '2', '3']:
                    if r in all_keys:
                        print 'incorrect response', r
                        print 'true lang: ', true_lang
                        newrl = newrl + u'%7.3f %1.0f incorrect\n' % (now,
                                                                      block-1)
                        newout = newout + (u"%3d %7.3f incorrect:") % (vol-1,
                                                                       now)
                        newout = newout + (u" Pressed %s instead of %s \n" %
                                           (r, true_lang))
                        state['responded'] = True

        return state, newout, newrl, vol_onsets, new_ntr, new_vol, newcorrect

    print 'duration: ', duration
    print 'vol %f of %f' % (vol, settings['volumes'])
    while vol < settings['volumes']:
        if global_clock.getTime() > duration:
            print 'overdue'
            break
        state, out, resp_log, vol_onsets, n_tr, vol, cor = parse_keys()
        if state['baseline_start']:  # Wait a few volumes before first stim
            while n_tr < base1tr:
                state, out, resp_log, vol_onsets, n_tr, vol, cor = parse_keys()
            n_tr = 0
            state['baseline_start'] = False
            state['playblock'] = True
        if state['playblock']:  # Play stimulus block
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
                while n_tr < stimtr:
                    # Wait longer while stimuli are playing to let CPU catch up
                    core.wait(.01, hogCPUperiod=0.001)
                    state, out, resp_log, vol_onsets, n_tr, vol, cor = \
                        parse_keys()
                n_tr = 0
            # Update state variables
            state['playblock'] = False
            if settings['resp'][block]:
                print settings['resp'][block]
                state['response'] = True
            elif block == 2 and not settings['resp'][block]:
                state['baseline_end'] = True
            else:
                state['rest'] = True
            block += 1
        if state['response']:  # Ask subject to respond to probe question
            print 'response'
            # Wait for HRF
            while n_tr < waithrftr:  # Wait a few volumes before probe
                state, out, resp_log, vol_onsets, n_tr, vol, cor = parse_keys()
            state['listen_for_resp'] = True
            # Display task question
            n_tr = 0
            resp_text.draw()
            win.flip()
            while n_tr < responsetr:
                state, out, resp_log, vol_onsets, n_tr, vol, cor = parse_keys()
            n_tr = 0
            fixation.draw()
            win.flip()
            # Update state variables
            state['response'] = False
            # block will equal 3 after the last block of stims have played
            if block >= 3:
                state['baseline_end'] = True
            else:
                state['rest'] = True
        if state['rest']:  # Rest (do nothing) between blocks
            print 'rest'
            while n_tr < resttr:
                if global_clock.getTime() > duration:
                    print 'overdue'
                    break
                state, out, resp_log, vol_onsets, n_tr, vol, cor = parse_keys()
            n_tr = 0
            state['rest'] = False
            state['playblock'] = True
            # If no response was made, note as missed
            if state['listen_for_resp'] and not state['responded']:
                true_lang = unicode(settings['lang'][0, block-1]+1)
                print 'no response'
                print 'true lang: ', true_lang
                resp_log += u'%7.3f %1.0f no response\n' % (now, block-1)
                out += (u"%3d  %7.3f no response:") % (vol - 1, now)
                out += (u" Should have pressed %s \n" % (true_lang))

            state['responded'] = False
            state['listen_for_resp'] = False
        if state['baseline_end']:  # Wait a few volumes at the end of the run
            print 'begin baseline_end'
#            core.wait(settings['TR']*base2tr, hogCPUperiod=0.2)
            print 'vol %f of %f' % (vol, settings['volumes'])
            # if a response was asked on the last block, don't wait as long
            if block == 3 and settings['resp'][2]:
                base2tr = 5
            while n_tr < base2tr:
                state, out, resp_log, vol_onsets, n_tr, vol, cor = parse_keys()
                # In the event that we are still waiting for trs that are not
                # coming, get out of loop
                if global_clock.getTime() > duration:
                    break
            n_tr = 0
            state['baseline_end'] = False
            # If no response was made, note as missed
            if state['listen_for_resp'] and not state['responded']:
                true_lang = unicode(settings['lang'][0, block-1]+1)
                print 'no response'
                print 'true lang: ', true_lang
                resp_log += u'%7.3f %1.0f no response\n' % (now, block-1)
                out += (u"%3d  %7.3f no response:") % (vol - 1, now)
                out += (u" Should have pressed %s \n" % (true_lang))
            print 'vol %f of %f' % (vol, settings['volumes'])
            print 'end baseline_end'
            break

#            if settings['debug']:
#                counter.setText(u"%d volumes\n%.3f seconds" % (vol, onset))
#                counter.draw()
#                win.flip()

    out += (u"End of scan (vol 0..%d = %d of %s).\n" %
            (vol - 1, vol, settings['volumes']))
    total_dur = global_clock.getTime()
    date_str = time.strftime("%b_%d_%H%M", time.localtime())
    out += u"Total duration = %7.3f sec (%7.3f min)" % (total_dur,
                                                        total_dur / 60)
    print "Total duration = %7.3f s (%7.3f min)" % (total_dur, total_dur / 60)

    # Save data and log files
    datapath = path.join(settings['subject'], 'run_data')
    if not path.isdir(datapath):
        mkdir(datapath)

    # Write log of key presses/volumes
    fname = path.join(datapath,
                      "%s_run%s_all_keys_%s" %
                      (settings['subject'], settings['run'], date_str))

    data_file = open(fname + '.txt', 'w')
    data_file.write(out)
    data_file.close()

    # Write log of resp trials
    resp_log += u'%1.0f out of %1.0f correct\n' % (cor, n_resp)
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
                      "%s_run%s_vol_onsets_%s" %
                      (settings['subject'], settings['run'], date_str))
    np.save(fname, vol_onsets)

    # Save settings for this run
    fname = path.join(datapath,
                      "%s_run%s_settings_%s" %
                      (settings['subject'], settings['run'], date_str))
    np.save(fname, settings)
    core.quit()


if __name__ == '__main__':
    run()
    print('Done')
