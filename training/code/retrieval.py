
from __future__ import division
from __future__ import print_function

import pylink
import os
import platform
import random
import time
import sys
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy
from psychopy import visual, core, event, monitors, gui, data
from datetime import date
from PIL import Image
from string import ascii_letters, digits
import os.path as op
import numpy as np
import pandas as pd
from psychopy.tools.filetools import fromFile, toFile
import pickle
from psychopy.hardware.emulator import launchScan
from math import fabs, hypot
from psychopy.tools.monitorunittools import posToPix, pix2deg
from datetime import datetime
import imageio

trainParams = {
    'subject': 0,
    'run': 0,
    'npos': 16,
    'ntrials': 5,
    'itis': [250, 500],
    'ecc': 7,
    'cuedur': 0.5,
    'delaydur': 1,
    'saccadedur': 1,
    'feedbackdur': 2,
    'time': datetime.now().strftime("%m-%d-%y_%H-%M"),
    'saccadeinput': 'Mouse',
    'debug': True,
    
    'distance': 83.5,
    'width': 64.35,
    'resolution': [1920, 1080],
    'fullscreen': False,
    'useretina': False,
    }

def generate_positions(params):
    subject = params['subject']
    npos = params['npos']
    ecc = params['ecc']
    
    pa_fname = '../positions/sub-%03d_%02dpos.tsv' % (subject, npos)
    
    if not op.exists(pa_fname):
        print("Generating new set of positions...")
        xstart = np.arange(0, 2*np.pi, 2*np.pi / npos)
        xstop = xstart + 2*np.pi/npos
        angles = np.random.uniform(xstart, xstop)
        positions = ecc * np.array([np.sin(angles), np.cos(angles)]).T
        print(positions.shape)
        
        cues = os.listdir('../../cues')
        cues = random.sample(cues, npos)
        
        PA_tsv = pd.DataFrame(data = {
                                      'pos_x': positions[:, 0], 
                                      'pos_y': positions[:, 1], 
                                      'radians': angles, 
                                      'degrees': np.degrees(angles),
                                      'cues': cues
                                      })
        PA_tsv.to_csv(pa_fname, sep = '\t', mode='w', header=True)
        
    else:
        print("Found previously generated set of positions, loading...")
        PA_tsv = pd.read_csv(pa_fname, sep = '\t')
        angles = list(PA_tsv.radians)
        positions = np.zeros([npos, 2])
        positions[:, 0] = list(PA_tsv.pos_x)
        positions[:, 1] = list(PA_tsv.pos_y)
        cues = list(PA_tsv.cues)
    
    return positions, angles, cues
    

def generate_trials(trainParams, positions, angles, cues, win):
    subject = trainParams['subject']
    run = trainParams['run']
    npos = trainParams['npos']
    itis = trainParams['itis']
    cuedur = trainParams['cuedur']
    delaydur = trainParams['delaydur']
    saccadedur = trainParams['saccadedur']
    feedbackdur = trainParams['feedbackdur']
    time = trainParams['time'],
    ntrials = trainParams['ntrials']
    
    # Set filenames and paths to be used
    tsv_filename_trial = '/sub-wlsubj%03d_run-%02d_retrieval_%s_trialdesign.tsv' % (subject, run, time[0])
    subj_dir = op.join('../design/', "sub-%03d" % subject)
    
    if not op.exists(subj_dir):
        os.makedirs(subj_dir)
    
    trialnums = list(range(npos)) * ntrials
    random.shuffle(trialnums)
    itis = itis * int(npos / len(itis)) * ntrials
    random.shuffle(itis)
    
    # Load cue images
    cues_dict = {}
    for cue_name in cues:
        cue_path = op.join('../../cues', cue_name)
        cues_dict[cue_name] = visual.ImageStim(win, cue_path, size = 2, units = 'deg')

    
    # Initialize dictionaries
    trialParams = []
    trial_design = pd.DataFrame(columns =         
        ['run', 'trialNum', 'ITIDur', 'gratingPosX', 'gratingPosY', 'gratingOri', 'gratingAng', 'cue'])

    # Loop through each trial and generate trial-specific parameters
    for i, trial in enumerate(trialnums):
        pos = positions[trial]
        ori = np.degrees(angles[trial]) + 90
        cue_name = cues[trial]
        cue = cues_dict[cue_name]
        
        if ori >= 360: ori -= 360
        
        # Save generated trial parameters
        trialParams.append({
            'trialNum': str(i),
            'cueDur': cuedur,
            'delayDur': delaydur,
            'saccadeDur': saccadedur,
            'feedbackDur': feedbackdur,
            'ITIDur': itis[trial],
            'gratingPos': pos,
            'gratingOri': ori,
            'gratingAng': np.degrees(angles[trial]),
            'cue': cue,
            'debug': trainParams['debug']
        })
        
        # Append design info to design DataFrames
        trial_design = trial_design.append({
            'run': run,
            'trialNum': str(i),
            'ITIDur': itis[trial],
            'gratingPosX': pos[0],
            'gratingPosY': pos[1],
            'gratingOri': ori,
            'gratingAng': np.degrees(angles[trial]),
            'cue': cue_name
            }, ignore_index = True)
            
    trial_design.to_csv(subj_dir + tsv_filename_trial, sep = '\t', mode='w', header=True)
    
    return trialParams


def create_stimuli(params, win, positions, angles, factor):
    # Create stimuli
    grating = visual.GratingStim(
        win, sf=2, size=3, mask='gauss', maskParams = {'sd': 5},
        pos=[-4,0], ori=0, units = 'deg')
    fixation = visual.TextStim(win, text='+', height=2 * factor, 
                               color=(-1, -1, -1))
    guide = visual.Circle(win, radius = params['ecc'], lineWidth = 0.5, lineColor = 'white', 
                          units = 'deg')
    correct = visual.Circle(win, radius = 0.3, lineWidth = 0, lineColor = 'white', 
                            units = 'deg', fillColor = 'red')
    sacc = visual.Circle(win, radius = 0.3, lineWidth = 0, lineColor = 'white', 
                            units = 'deg', fillColor = 'white')
    
    debug = []
    for i,p in enumerate(positions):
        label = "%d" % i
        d_stim = visual.TextStim(win, text = label, height = 0.01, 
                                 color = (-1, -1, -1), pos = p, units = 'deg')
        debug.append(d_stim)
    
    return grating, fixation, guide, correct, sacc, debug
    
    
def connect_eyelink(params=trainParams):
    edf_fname = "Is%02d_r%02d" % (params['subject'], params['run'])
    results_folder = '../results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    session_identifier = "sub-wlsubj%03d" % params['subject']

    session_folder = os.path.join(results_folder, session_identifier)
    if not os.path.exists(session_folder):
        os.makedirs(session_folder)

    if params['saccadeinput'] == 'Mouse':
        el_tracker = pylink.EyeLink(None)
    elif params['saccadeinput'] == 'EyeLink':
        try:
            el_tracker = pylink.EyeLink("100.1.1.1")
        except RuntimeError as error:
            print('ERROR:', error)
            core.quit()
            sys.exit()
    
    return el_tracker, session_folder


def create_EDF(el_tracker, params=trainParams):
    edf_fname = "Is%02d_r%02d" % (params['subject'], params['run'])
    edf_file = edf_fname + ".EDF"
    try:
        el_tracker.openDataFile(edf_file)
    except RuntimeError as err:
        print('ERROR:', err)
        # close the link if we have one open
        if el_tracker.isConnected():
            el_tracker.close()
        core.quit()
        sys.exit()
    
    preamble_text = 'RECORDED BY %s' % op.basename(__file__)
    el_tracker.sendCommand("add_file_preamble_text '%s'" % preamble_text)
    
    return edf_file


def configure_eyelink(el_tracker, params=trainParams):
    el_tracker.setOfflineMode()
    eyelink_ver = 0  # set version to 0, in case running in Dummy mode
    
    if params['saccadeinput'] == 'EyeLink':
        vstr = el_tracker.getTrackerVersionString()
        eyelink_ver = int(vstr.split()[-1].split('.')[0])
        print('Running experiment on %s, version %d' % (vstr, eyelink_ver))

    file_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'
    link_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT'

    if eyelink_ver > 3:
        file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,GAZERES,BUTTON,STATUS,INPUT'
        link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT'
    else:
        file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,GAZERES,BUTTON,STATUS,INPUT'
        link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT'
    el_tracker.sendCommand("file_event_filter = %s" % file_event_flags)
    el_tracker.sendCommand("file_sample_data = %s" % file_sample_flags)
    el_tracker.sendCommand("link_event_filter = %s" % link_event_flags)
    el_tracker.sendCommand("link_sample_data = %s" % link_sample_flags)

    el_tracker.sendCommand("calibration_type = HV9")
    el_tracker.sendCommand("button_function 5 'accept_target_fixation'")

    el_tracker.sendCommand('calibration_area_proportion 0.40 0.80')
    el_tracker.sendCommand('validation_area_proportion 0.40 0.80')
    
    # Register Eye uses
    eye_used = el_tracker.eyeAvailable()
    print(eye_used)
    if eye_used == 1:
        el_tracker.sendMessage("EYE_USED 1 RIGHT")
    elif eye_used == 0 or eye_used == 2:
        el_tracker.sendMessage("EYE_USED 0 LEFT")
        eye_used = 0
    else:
        print("Error in getting the eye information!")
        return pylink.TRIAL_ERROR
    params['eye_used'] = eye_used
    
    return el_tracker


def setup_graphics(el_tracker, params=trainParams):
#    mon = monitors.Monitor('myMonitor', distance = expParams['Screen Distance'], width = expParams['Screen Width'])
#    win = visual.Window(expParams['Screen Resolution'],
#                        fullscr=False,
#                        monitor=mon,
#                        allowGUI = True,
#                        units='deg')
    mon = monitors.Monitor('testMonitor', distance = params['distance'], width = params['width'])
    win = visual.Window(
        params['resolution'], allowGUI=True, monitor=mon, units='deg',
        fullscr = params['fullscreen'])

    scn_width, scn_height = win.size

    if 'Darwin' in platform.system():
        if params['useretina']:
            scn_width = int(scn_width/2.0)
            scn_height = int(scn_height/2.0)

    el_coords = "screen_pixel_coords = 0 0 %d %d" % (scn_width - 1, scn_height - 1)
    el_tracker.sendCommand(el_coords)

    dv_coords = "DISPLAY_COORDS  0 0 %d %d" % (scn_width - 1, scn_height - 1)
    el_tracker.sendMessage(dv_coords)

    genv = EyeLinkCoreGraphicsPsychoPy(el_tracker, win)
    print(genv) 

    foreground_color = (-1, -1, -1)
    background_color = win.color
    genv.setCalibrationColors(foreground_color, background_color)

    genv.setTargetType('picture')
    genv.setPictureTarget(os.path.join('images', 'fixTarget.bmp'))

    genv.setCalibrationSounds('', '', '')

    if params['useretina']:
        genv.fixMacRetinaDisplay()

    pylink.openGraphicsEx(genv)
    
    return mon, win, genv


def clear_screen(win, genv):
    """ clear up the PsychoPy window""" 

    win.fillColor = genv.getBackgroundColor()
    win.flip()


def show_msg(win, text, genv, wait_for_keypress=True):
    """ Show task instructions on screen""" 
    scn_width, scn_height = win.size
    msg = visual.TextStim(win, text,
                          color=genv.getForegroundColor(),
                          wrapWidth=scn_width/2)
    clear_screen(win, genv)
    msg.draw()
    win.flip()

    # wait indefinitely, terminates upon any key press
    if wait_for_keypress:
        event.waitKeys()
        clear_screen(win, genv)

def write_trialdata(trialParams, session_folder, params):
    format_str = "sub-wlsubj%03d_run-%02d_retrieval_%s_trialdata.tsv"
    filename =  format_str % (params['subject'], params['run'], params['time'])
    filepath = op.join(session_folder, filename)
    
    df = pd.DataFrame.from_dict(trialParams)
    df.to_csv(filepath, sep = '\t')


def terminate_task(win, session_folder, edf_file, genv, trialParams, params):
    """ Terminate the task gracefully and retrieve the EDF data file

    file_to_retrieve: The EDF on the Host that we would like to download
    win: the current window used by the experimental script
    """

    el_tracker = pylink.getEYELINK()

    if el_tracker.isConnected():
        # Terminate the current trial first if the task terminated prematurely
        error = el_tracker.isRecording()
        if error == pylink.TRIAL_OK:
            abort_trial(win, genv)

        # Put tracker in Offline mode
        el_tracker.setOfflineMode()

        # Clear the Host PC screen and wait for 500 ms
        el_tracker.sendCommand('clear_screen 0')
        pylink.msecDelay(500)

        # Close the edf data file on the Host
        el_tracker.closeDataFile()

        # Show a file transfer message on the screen
        msg = 'EDF data is transferring from EyeLink Host PC...'
        show_msg(win, msg, genv, wait_for_keypress=False)

        # Download the EDF data file from the Host PC to a local data folder
        # parameters: source_file_on_the_host, destination_file_on_local_drive
        local_edf = os.path.join(session_folder, edf_file)
        try:
            el_tracker.receiveDataFile(edf_file, local_edf)
        except RuntimeError as error:
            print('ERROR:', error)

        # Close the link to the tracker.
        el_tracker.close()
    
    # Write to tsv
    write_trialdata(trialParams, session_folder, params)
    
    # close the PsychoPy window
    win.close()

    # quit PsychoPy
    core.quit()
    sys.exit()


def abort_trial(win, genv):
    """Ends recording """

    el_tracker = pylink.getEYELINK()

    # Stop recording
    if el_tracker.isRecording():
        # add 100 ms to catch final trial events
        pylink.pumpDelay(100)
        el_tracker.stopRecording()

    # clear the screen
    clear_screen(win, genv)
    # Send a message to clear the Data Viewer screen
    bgcolor_RGB = (116, 116, 116)
    el_tracker.sendMessage('!V CLEAR %d %d %d' % bgcolor_RGB)

    # send a message to mark trial end
    el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_ERROR)

    return pylink.TRIAL_ERROR

# WILL HAVE TO TWEAK THIS
def disp_coords(win, mon, sac_end_pos, grating, retina):
    scn_width, scn_height = win.size
    print(win.size)
    stim_pix = posToPix(grating)
    if retina:
        scn_width = int(scn_width/2.0)
        scn_height = int(scn_height/2.0)
    sac_pix = [sac_end_pos[0] - scn_width/2, sac_end_pos[1] - scn_height/2]
    sac_deg = pix2deg(np.asarray(sac_pix), mon)
    
    print("------\nStim Pix:", stim_pix, "Sac Pix:", sac_pix, "\n", "Stim deg:", grating.pos, 
        "Sac Deg:", sac_deg, "\n------")


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def iti(fixation, guide, win, scan_clock, parameters, el_tracker, session_folder, 
        edf_file, genv, debug, params, trialParams):
    iti_dur = parameters['ITIDur']
    fixation.mask = 'cross'
    fixation.color = 'black'
    fixation.size = 0.5

    fixation.draw()
    guide.draw()
    if parameters['debug']: 
        for d in debug: 
            d.draw()
    win.flip()
    
    iti_onsetTime = scan_clock.getTime()
    iti_dur /= 1000
    while scan_clock.getTime() - iti_onsetTime <= iti_dur:
        # check for keyboard events
        for keycode, modifier in event.getKeys(modifiers=True):
            # Terminate the task if Ctrl-c
            if keycode == 'c' and (modifier['ctrl'] is True):
                el_tracker.sendMessage('terminated_by_user')
                terminate_task(win, session_folder, edf_file, genv, trialParams, params)
    
    ready = False
    while not ready:
        # check for keyboard events
        for keycode, modifier in event.getKeys(modifiers=True):
            # Terminate the task if Ctrl-c
            if keycode == 'c' and (modifier['ctrl'] is True):
                el_tracker.sendMessage('terminated_by_user')
                terminate_task(win, session_folder, edf_file, genv, trialParams, params)
            
            elif keycode == 'space':
                ready = True


def stimulus(fixation, guide, win, scan_clock, parameters, grating, 
             session_folder, edf_file, genv, debug, params, trialParams):
    cue_dur = parameters['cueDur']
    delay_dur = parameters['delayDur']

    grating.pos = parameters['gratingPos']
    grating.ori = parameters['gratingOri']
    
    parameters['cue'].draw()
    guide.draw()
    if parameters['debug']: 
        for d in debug: 
            d.draw()
    win.flip()
    stim_onsetTime = scan_clock.getTime()
    while scan_clock.getTime() - stim_onsetTime <= cue_dur:
        guide.draw()
        parameters['cue'].draw()
        if parameters['debug']: 
            for d in debug: 
                d.draw()
        win.flip()
        
        for keycode, modifier in event.getKeys(modifiers=True):
            # Terminate the task if Ctrl-c
            if keycode == 'c' and (modifier['ctrl'] is True):
                el_tracker.sendMessage('terminated_by_user')
                terminate_task(win, session_folder, edf_file, genv, trialParams, params)
    
    stim_onsetTime = scan_clock.getTime()
    while scan_clock.getTime() - stim_onsetTime <= delay_dur:
        guide.draw()
        fixation.draw()
        if parameters['debug']: 
            for d in debug: 
                d.draw()
        win.flip()
        
        for keycode, modifier in event.getKeys(modifiers=True):
            # Terminate the task if Ctrl-c
            if keycode == 'c' and (modifier['ctrl'] is True):
                el_tracker.sendMessage('terminated_by_user')
                terminate_task(win, session_folder, edf_file, genv, trialParams, params)


def saccade(fixation, guide, win, scan_clock, parameters, el_tracker, genv, 
            session_folder, edf_file, params, mon, debug, trialParams):
    eye_used = params['eye_used']
    sacc_dur = parameters["saccadeDur"]
    fixation.color = 'green'
    fixation.draw()
    guide.draw()
    if parameters['debug']: 
        for d in debug: 
            d.draw()
    win.flip()
    
    got_sac = False
    sac_start_time = -1
    SRT = -1  # initialize a variable to store saccadic reaction time (SRT)
    land_err = -1  # landing error of the saccade
    acc = 0  # hit the correct region or not
    sac_end_pos = np.asarray(parameters['gratingPos'] + np.random.randn(2))

    event.clearEvents()  # clear all cached events if there are any
    sacc_onsetTime = scan_clock.getTime()
    while scan_clock.getTime() - sacc_onsetTime <= sacc_dur:
        # abort the current trial if the tracker is no longer recording
        error = el_tracker.isRecording()
        if error is not pylink.TRIAL_OK:
            el_tracker.sendMessage('tracker_disconnected')
            abort_trial(win, genv)
            return error

        # check for keyboard events
        for keycode, modifier in event.getKeys(modifiers=True):
            # Terminate the task if Ctrl-c
            if keycode == 'c' and (modifier['ctrl'] is True):
                el_tracker.sendMessage('terminated_by_user')
                terminate_task(win, session_folder, edf_file, genv, trialParams, params)
                return pylink.ABORT_EXPT

        # grab the events in the buffer, for more details,
        # see the example script "link_event.py"
        eye_ev = el_tracker.getNextData()
        if (eye_ev is not None) and (eye_ev == pylink.ENDSACC):
            eye_dat = el_tracker.getFloatData()
            if eye_dat.getEye() == eye_used:
                sac_amp = eye_dat.getAmplitude()  # amplitude
                sac_start_time = eye_dat.getStartTime()  # onset time
                sac_end_time = eye_dat.getEndTime()  # offset time
                sac_start_pos = eye_dat.getStartGaze()  # start position
                sac_end_pos = eye_dat.getEndGaze()  # end position

                # a saccade was initiated
                if sac_start_time <= sacc_onsetTime:
                    sac_start_time = -1
                    pass  # ignore saccades occurred before target onset
                elif hypot(sac_amp[0], sac_amp[1]) > 1.5:
                    # log a message to mark the time at which a saccadic
                    # response occurred; note that, here we are detecting a
                    # saccade END event; the saccade actually occurred some
                    # msecs ago. The following message has an additional
                    # time offset, so Data Viewer knows when exactly the
                    # "saccade_resp" event actually happened
                    offset = int(el_tracker.trackerTime()-sac_start_time)
                    sac_response_msg = '{} saccade1_resp'.format(offset)
                    el_tracker.sendMessage(sac_response_msg)
                    SRT = sac_start_time - sacc_onsetTime
                    disp_coords(win, mon, sac_end_pos, grating, expParams['use_retina'])

    # send a 'TRIAL_RESULT' message to mark the end of trial, see Data
    # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
    el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_OK)
    return SRT, sac_end_pos


def feedback(fixation, guide, correct, win, scan_clock, parameters, el_tracker, 
             genv, session_folder, edf_file, params, mon, accuracy, sac_end_pos,
             sacc, debug, trialParams):
    eye_used = params['eye_used']
    feed_dur = parameters["feedbackDur"]
    sacc.pos = sac_end_pos
    correct.pos = parameters["gratingPos"]
    
    if accuracy:
        correct.color = 'green'
    else:
        correct.color = 'red'
    
    fixation.color = 'green'
    fixation.draw()
    guide.draw()
    sacc.draw()
    correct.draw()
    if parameters['debug']: 
        for d in debug: 
            d.draw()
    win.flip()
    
    got_sac = False
    sac_start_time = -1
    SRT = -1  # initialize a variable to store saccadic reaction time (SRT)
    land_err = -1  # landing error of the saccade
    acc = 0  # hit the correct region or not

    event.clearEvents()  # clear all cached events if there are any
    sacc_onsetTime = scan_clock.getTime()
    while scan_clock.getTime() - sacc_onsetTime <= feed_dur:
        error = el_tracker.isRecording()
        if error is not pylink.TRIAL_OK:
            el_tracker.sendMessage('tracker_disconnected')
            abort_trial(win, genv)
            return error
        for keycode, modifier in event.getKeys(modifiers=True):
            if keycode == 'c' and (modifier['ctrl'] is True):
                el_tracker.sendMessage('terminated_by_user')
                terminate_task(win, session_folder, edf_file, genv, trialParams, params)
                return pylink.ABORT_EXPT

        eye_ev = el_tracker.getNextData()
        if (eye_ev is not None) and (eye_ev == pylink.ENDSACC):
            eye_dat = el_tracker.getFloatData()
            if eye_dat.getEye() == eye_used:
                sac_amp = eye_dat.getAmplitude()  # amplitude
                sac_start_time = eye_dat.getStartTime()  # onset time
                sac_end_time = eye_dat.getEndTime()  # offset time
                sac_start_pos = eye_dat.getStartGaze()  # start position
                sac_end_pos = eye_dat.getEndGaze()  # end position

                if sac_start_time <= sacc_onsetTime:
                    sac_start_time = -1
                    pass  # ignore saccades occurred before target onset
                elif hypot(sac_amp[0], sac_amp[1]) > 1.5:
                    offset = int(el_tracker.trackerTime()-sac_start_time)
                    sac_response_msg = '{} saccade2_resp'.format(offset)
                    el_tracker.sendMessage(sac_response_msg)
                    SRT = sac_start_time - sacc_onsetTime
                    disp_coords(win, mon, sac_end_pos, grating, expParams['use_retina'])

    el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_OK)
    return SRT, sac_end_pos


def check_saccade(sac_end_pos, target_ang, target_ecc, angles, ecc_boundary):
    sac_ecc, sac_ang = cart2pol(sac_end_pos[1], sac_end_pos[0])
    sac_ang = np.degrees(sac_ang)
    if sac_ang < 0: sac_ang += 360
    
    if np.abs(sac_ecc - target_ecc) <= ecc_boundary:
        closest_ang_idx = np.abs(sac_ang - angles).argmin()
        tar_ang_idx = np.where(angles == target_ang)[0][0]
        
        if closest_ang_idx == tar_ang_idx:
            accuracy = 1
        else:
            accuracy = 0
    else:
        accuracy = 0
    
    return accuracy
    

def run_trial(trialParams, trial_index, grating, fixation, guide, win, scan_clock,
              session_folder, edf_file, genv, params, mon, correct, angles, sacc, 
              debug):
    parameters = trialParams[trial_index]
    print("TRIAL START")
    # get a reference to the currently active EyeLink connection
    el_tracker = pylink.getEYELINK()

    # send a "TRIALID" message to mark the start of a trial
    el_tracker.sendMessage('TRIALID %d' % trial_index)

    # For illustration purpose,
    # send interest area messages to record in the EDF data file
    # here we draw a rectangular IA, for illustration purposes
    # format: !V IAREA RECTANGLE <id> <left> <top> <right> <bottom> [label]
    # for all supported interest area commands, see the Data Viewer Manual,
    # "Protocol for EyeLink Data to Viewer Integration"
    scn_width, scn_height = win.size
    left = int(scn_width/2.0) - 50
    top = int(scn_height/2.0) - 50
    right = int(scn_width/2.0) + 50
    bottom = int(scn_height/2.0) + 50
    ia_pars = (1, left, top, right, bottom, 'screen_center')
    el_tracker.sendMessage('!V IAREA RECTANGLE %d %d %d %d %d %s' % ia_pars)
    
    # ITI
    print("ITI")
    iti(fixation, guide, win, scan_clock, parameters, el_tracker, 
        session_folder, edf_file, genv, debug, params, trialParams)
    
    # Stimulus
    print("STIMULUS")
    stimulus(fixation, guide, win, scan_clock, parameters, grating, 
             session_folder, edf_file, genv, debug, params, trialParams)

    # Saccade Response
    SRT1, sac_end_pos1 = saccade(fixation, guide, win, scan_clock, parameters, 
                               el_tracker, genv, session_folder, edf_file, params,
                               mon, debug, trialParams)

    accuracy = check_saccade(sac_end_pos1, parameters['gratingAng'], 
                             params['ecc'], np.degrees(angles), 2)

    # Feedback + Correction
    SRT2, sac_end_pos2 = feedback(fixation, guide, correct, win, scan_clock, 
                                parameters, el_tracker, genv, session_folder, 
                                edf_file, params, mon, accuracy, sac_end_pos1,
                                sacc, debug, trialParams)
    
    # Trial end
    el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_OK)
    
    return SRT1, sac_end_pos1, accuracy, SRT2, sac_end_pos2


def run(params=trainParams):
    subject = params['subject']
    run = params['run']
    npos = params['npos']
    itis = params['itis']
    ecc = params['ecc']
    
    # Setup EyeLink & Window
    el_tracker, session_folder = connect_eyelink(params)
    edf_file = create_EDF(el_tracker, params)
    el_tracker = configure_eyelink(el_tracker, params)
    mon, win, genv = setup_graphics(el_tracker, params)
    
    # Calibrate EyeLink
    if params['saccadeinput'] == 'EyeLink':
        task_msg = 'Press ENTER to calibrate tracker'
        #show_msg(win, task_msg, genv)
        
        try:
            el_tracker.doTrackerSetup()
        except RuntimeError as err:
            print('ERROR:', err)
            el_tracker.exitCalibration()
    # Generate trials
    positions, angles, cues = generate_positions(params)
    trialParams = generate_trials(params, positions, angles, cues, win)
    
    if params['saccadeinput'] == 'Mouse': 
        factor = 20 
    else: 
        factor = 1
    
    # Create stimuli
    grating, fixation, guide, correct, sacc, debug = create_stimuli(
        params, win, positions, angles, factor)
    
    # Display instructions
    instructions = "Press SPACEBAR to move on."
    msg = visual.TextStim(win, pos=[0, 5 * factor], text=instructions)

    msg.draw()
    fixation.draw()
    guide.draw()
    win.flip()

    # Wait for a response
    event.waitKeys(keyList=['space', 0])
    
    print('STARTING')
    # Start timing
    scan_clock = core.Clock()
    globalClock = core.Clock()
    
    el_tracker.setOfflineMode()

    # Start recording, at the beginning of a new run
    # arguments: sample_to_file, events_to_file, sample_over_link,
    # event_over_link (1-yes, 0-no)
    try:
        el_tracker.startRecording(1, 1, 1, 1)
    except RuntimeError as error:
        print("ERROR:", error)
        terminate_task(win, session_folder, edf_file, genv, trialParams, params)

    # Allocate some time for the tracker to cache some samples
    pylink.pumpDelay(100)
    
    print('REGISTERING EYES')
    # Register Eye uses
    eye_used = el_tracker.eyeAvailable()
    print(eye_used)
    if eye_used == 1:
        el_tracker.sendMessage("EYE_USED 1 RIGHT")
    elif eye_used == 0 or eye_used == 2:
        el_tracker.sendMessage("EYE_USED 0 LEFT")
        eye_used = 0
    else:
        print("Error in getting the eye information!")
        return pylink.TRIAL_ERROR
    
    
    # reset the global clock to compare stimulus timing
    # to time 0 to make sure each trial is 6-sec long
    # this is known as "non-slip timing"
    scan_clock.reset()
    print('TRIAL LOOP')
    # Trial loop
    for trial in range(params['ntrials'] * params['npos']):
        srt, spos, accuracy, frt, fpos = run_trial(
            trialParams, trial, grating, fixation, guide, win, scan_clock, 
            session_folder, edf_file, genv, params, mon, correct, angles, sacc, 
            debug)
            
        trialParams[trial]['srt'] = srt
        trialParams[trial]['sac_x'] = spos[0]
        trialParams[trial]['sac_y'] = spos[1]
        trialParams[trial]['acc'] = accuracy
        trialParams[trial]['frt'] = frt
        trialParams[trial]['feed_x'] = fpos[0]
        trialParams[trial]['feed_y'] = fpos[1]
        
    terminate_task(win, session_folder, edf_file, genv, trialParams, trialParams, params)
    print("DONE")

if __name__ == '__main__':
    run()
    
