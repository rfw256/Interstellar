
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

trainParams = {
    'subject': 0,
    'run': 0,
    'npos': 16,
    'itis': [250, 500],
    'ecc': 7,
    'trialdur': 1
    }

def generate_positions(subject, nPos, ecc):
    xstart = np.arange(0, 2*np.pi, 2*np.pi / nPos)
    xstop = xstart + 2*np.pi/nPos
    angles = np.random.uniform(xstart, xstop)
    positions = ecc * np.array([np.sin(angles), np.cos(angles)]).T 
    print(angles.shape, positions.shape)
    
    PA_tsv = pd.DataFrame(data = {'pos_x': positions[:, 0], 'pos_y': positions[:, 1], 'radians': angles, 'degrees': np.degrees(angles)})
    
    PA_tsv.to_csv('sub-%03d_positions%02d.tsv' % (subject, nPos), sep = '\t', mode='w', header=True)
    
    return positions, angles
    

def generate_trials(subject, run, nPos, itis, positions, angles, trial_dur):
    # Set filenames and paths to be used
    tsv_filename_trial = '/sub-%03d_run-%02d_trialdesign.tsv' % (subject, run)
    subj_dir = op.join('../design/', "sub-%03d" % subject)
    
    trialnums = list(range(nPos))
    random.shuffle(trialnums)
    itis = itis * int(nPos / len(itis))
    random.shuffle(itis)
    
    # Initialize dictionaries
    trialParams = {}
    trial_design = pd.DataFrame(columns =         
        ['run', 'trialNum', 'ITIDur', 'gratingPosX', 'gratingPosY', 'gratingOri', 'gratingAng'])

    # Loop through each trial and generate trial-specific parameters
    for i, trial in enumerate(trialnums):
        pos = positions[trial]
        ori = np.degrees(angles[trial]) + 90
        
        if ori >= 360: ori -= 360

        # Save generated trial parameters
        trialParams[str(i)] = {
            'trialNum': str(i),
            'trialDuration': trial_dur,
            'ITIDur': itis[trial],
            'gratingPos': pos,
            'gratingOri': ori,
            'gratingAng': np.degrees(angles[(positions == pos)[:, 0]])[0]
        }
        
        # Append design info to design DataFrames
        trial_design = trial_design.append({
            'run': run,
            'trialNum': str(i),
            'ITIDur': itis[trial],
            'gratingPosX': pos[0],
            'gratingPosY': pos[1],
            'gratingOri': ori,
            'gratingAng': np.degrees(angles[(positions == pos)[:, 0]])[0]
            }, ignore_index = True)
            
    #trial_design.to_csv(subj_dir + tsv_filename_trial, sep = '\t', mode='w', header=True)
    
    return trialParams
    

def run(params=trainParams):
    subject = params['subject']
    run = params['run']
    nPos = params['npos']
    itis = params['itis']
    ecc = params['ecc']
    trial_dur = params['trialdur']
    
    positions, angles, = generate_positions(subject, nPos, ecc)
    trialParams = generate_trials(subject, run, nPos, itis, positions, angles, trial_dur)
    
    print("DONE")
    
run()
    
