
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
    'trialdur': 0.5,
    'time': datetime.now().strftime("%m-%d-%y_%H-%M"),
    'saccadeinput': 'Mouse',
    
    'monitor': 'drytortugas',
    
    'distance': 61,
    'width': 60,
    'resolution': [2560, 1440],
    'fullscreen': False,
    'useretina': True,
    }

def generate_positions(params):
    subject = params['subject']
    npos = params['npos']
    ecc = params['ecc']
    
    pa_fname = '../positions/sub-%03d_%02dpos.tsv' % (subject, npos)
    
    if not op.exists(pa_fname):
        print("Generating new set of positions...")
        
        task = ['perception', 'ltm', 'wm']
        positions = []
        angles = []
        tasks = []
        for t in task:
            xstart = np.arange(0, 2*np.pi, 2*np.pi / npos)
            xstop = xstart + 2*np.pi/npos
            A = np.random.uniform(xstart, xstop)
            P = ecc * np.array([np.sin(A), np.cos(A)]).T
            T = [t] * npos
            positions.append(P)
            angles.append(A)
            tasks.append(T)
            
        angles = np.concatenate(angles)
        positions = np.concatenate(positions)
        tasks = np.concatenate(tasks)

        cues = os.listdir('../../cues')
        cues = random.sample(cues, npos*3)
        
        PA_tsv = pd.DataFrame(data = {
                                      'task': tasks,
                                      'pos_x': positions[:, 0], 
                                      'pos_y': positions[:, 1], 
                                      'radians': angles, 
                                      'degrees': np.degrees(angles),
                                      'cues': cues,
                                      'cond': np.arange(npos*3)
                                      })
        PA_tsv.to_csv(pa_fname, sep = '\t', mode='w', header=True)
        

    print("Loading stimuli...")
    PA_tsv = pd.read_csv(pa_fname, sep = '\t')
    PA_tsv = PA_tsv[PA_tsv.task == 'ltm']
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
    trialdur = trainParams['trialdur']
    time = trainParams['time'],
    ntrials = trainParams['ntrials']
    conds = np.arange(npos, npos*2)
    
    # Set filenames and paths to be used
    tsv_filename_trial = '/sub-wlsubj%03d_run-%02d_study_%s_trialdesign.tsv' % (subject, run, time[0])
    subj_dir = op.join('../design/', "sub-%03d" % subject)
    
    if not op.exists(subj_dir):
        os.makedirs(subj_dir)
    
    trialnums = list(range(npos)) * ntrials
    random.shuffle(trialnums)
    itis = itis * int(npos / len(itis)) * ntrials
    random.shuffle(itis)
    
    # Load cue images
    cues_dict = {}
    for i, cue_name in enumerate(cues):
        msg = 'Loading stimuli... \n(%d/%d) %d %%' % (i+1, npos, (i+1) / npos * 100)
        msg = visual.TextStim(win, pos=[0, 0], text=msg, units = 'deg')
        msg.draw()
        win.flip()
        cue_path = op.join('../../cues', cue_name)
        cues_dict[cue_name] = visual.ImageStim(win, cue_path, size = 1, units = 'deg')

    
    # Initialize dictionaries
    trialParams = {}
    trial_design = pd.DataFrame(columns =         
        ['run', 'trialNum', 'ITIDur', 'gratingPosX', 'gratingPosY', 'gratingOri', 'gratingAng', 'cue', 'cond'])

    # Loop through each trial and generate trial-specific parameters
    for i, trial in enumerate(trialnums):
        pos = positions[trial]
        ori = np.degrees(angles[trial]) + 90
        cue_name = cues[trial]
        cue = cues_dict[cue_name]
        cond = conds[trial]
        
        if ori >= 360: ori -= 360
        
        # Save generated trial parameters
        trialParams[str(i)] = {
            'trialNum': str(i),
            'trialDuration': trialdur,
            'ITIDur': itis[trial],
            'gratingPos': pos,
            'gratingOri': ori,
            'gratingAng': np.degrees(angles[trial]),
            'cue': cue
        }
        
        # Append design info to design DataFrames
        trial_design = trial_design.append({
            'run': run,
            'trialNum': str(i),
            'ITIDur': itis[trial],
            'gratingPosX': pos[0],
            'gratingPosY': pos[1],
            'gratingOri': ori,
            'gratingAng': np.degrees(angles[trial]),
            'cue': cue_name,
            'cond': cond
            }, ignore_index = True)
            
    trial_design.to_csv(subj_dir + tsv_filename_trial, sep = '\t', mode='w', header=True)
    
    return trialParams
    
    
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
    
    return el_tracker


def setup_graphics(el_tracker, params=trainParams):
#    mon = monitors.Monitor('myMonitor', distance = expParams['Screen Distance'], width = expParams['Screen Width'])
#    win = visual.Window(expParams['Screen Resolution'],
#                        fullscr=False,
#                        monitor=mon,
#                        allowGUI = True,
#                        units='deg')
    mon = monitors.Monitor(params['monitor'], distance = params['distance'], width = params['width'])
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


def terminate_task(win, session_folder, edf_file, genv):
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


def iti(fixation, guide, win, scan_clock, iti_dur, el_tracker, session_folder, 
        edf_file, genv):
    fixation.mask = 'cross'
    fixation.color = 'black'
    fixation.size = 0.5

    fixation.draw()
    guide.draw()
    win.flip()
    
    iti_onsetTime = scan_clock.getTime()
    iti_dur /= 1000
    while scan_clock.getTime() - iti_onsetTime <= iti_dur:
        # check for keyboard events
        for keycode, modifier in event.getKeys(modifiers=True):
            # Terminate the task if Ctrl-c
            if keycode == 'c' and (modifier['ctrl'] is True):
                el_tracker.sendMessage('terminated_by_user')
                terminate_task(win, session_folder, edf_file, genv)
    
    ready = False
    while not ready:
        # check for keyboard events
        for keycode, modifier in event.getKeys(modifiers=True):
            # Terminate the task if Ctrl-c
            if keycode == 'c' and (modifier['ctrl'] is True):
                el_tracker.sendMessage('terminated_by_user')
                terminate_task(win, session_folder, edf_file, genv)
            
            elif keycode == 'space':
                ready = True


def stimulus(fixation, guide, win, scan_clock, parameters, grating, 
             session_folder, edf_file, genv):
    trial_dur = parameters['trialDuration']

    grating.pos = parameters['gratingPos']
    grating.ori = parameters['gratingOri']
    
    parameters['cue'].draw()
    guide.draw()
    win.flip()
    stim_onsetTime = scan_clock.getTime()
    while scan_clock.getTime() - stim_onsetTime <= trial_dur:
        grating.setPhase(0.05, '+')
        
        guide.draw()
        grating.draw()
        parameters['cue'].draw()
        win.flip()
        
        for keycode, modifier in event.getKeys(modifiers=True):
            # Terminate the task if Ctrl-c
            if keycode == 'c' and (modifier['ctrl'] is True):
                el_tracker.sendMessage('terminated_by_user')
                terminate_task(win, session_folder, edf_file, genv)


def run_trial(trialParams, trial_index, grating, fixation, guide, win, scan_clock,
              session_folder, edf_file, genv):
    parameters = trialParams[str(trial_index)]
    print("TRIAL START")
    # get a reference to the currently active EyeLink connection
    el_tracker = pylink.getEYELINK()

    # send a "TRIALID" message to mark the start of a trial
    el_tracker.sendMessage('TRIALID %d' % trial_index)
    
    # ITI
    print("ITI")
    iti(fixation, guide, win, scan_clock, parameters["ITIDur"], el_tracker, 
        session_folder, edf_file, genv)
    
    # Stimulus
    print("STIMULUS")
    stimulus(fixation, guide, win, scan_clock, parameters, grating, 
             session_folder, edf_file, genv)
    
    # Trial end
    el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_OK)


def run(params=trainParams):
    subject = params['subject']
    run = params['run']
    npos = params['npos']
    itis = params['itis']
    ecc = params['ecc']
    trial_dur = params['trialdur']
    
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
        
    # Create stimuli
    if params['saccadeinput'] == 'Mouse': 
        factor = 20 
    else: 
        factor = 1
    grating = visual.GratingStim(
        win, sf=2, size=3, mask='gauss', maskParams = {'sd': 5},
        pos=[-4,0], ori=0, units = 'deg')
    fixation = visual.TextStim(win, text='+', height=0.5, 
                               color=(-1, -1, -1), units = 'deg')
    guide = visual.Circle(win, radius = ecc, lineWidth = 1, lineColor = 'white', 
                          units = 'deg', fillColor = None, interpolate = True)
                          
    
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
        terminate_task(win, session_folder, edf_file, genv)

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
        run_trial(trialParams, trial, grating, fixation, guide, win, 
                  scan_clock, session_folder, edf_file, genv)
        
    terminate_task(win, session_folder, edf_file, genv)
    print("DONE")

if __name__ == '__main__':
    run()
    
