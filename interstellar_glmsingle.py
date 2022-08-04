import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import glob
import nibabel as nib
import warnings
from tqdm import tqdm
import time
from pprint import pprint
from scipy.stats import sem
warnings.filterwarnings('ignore')
import pickle

import glmsingle
from glmsingle.glmsingle import GLM_single

'''
PARAMETERS
'''
wlsubj = 139
n_conds = 48
stimdur = 11.5 

n_TR = [280, 314]
sacc_dur = [1, 1.5]
tr = 1

design_dir = '~/mnt/winawer/Projects/Interstellar/task/Interstellar/main/design/'
fmriprep_dir = '~/mnt/winawer/Projects/Interstellar/derivatives/fmriprep'
output_dir = '~/mnt/winawer/Projects/Interstellar/analyses/GLMsingle'
session = 'nyu3t01'

design_filenames = [
    'sub-wlsubj139_run-01_perception_07-16-22_14-25_trialdesign.tsv',
    'sub-wlsubj139_run-02_ltm_07-16-22_14-33_trialdesign.tsv',
    'sub-wlsubj139_run-03_wm_07-16-22_14-39_trialdesign.tsv',
    'sub-wlsubj139_run-04_perception_07-16-22_14-45_trialdesign.tsv',
    'sub-wlsubj139_run-05_ltm_07-16-22_14-50_trialdesign.tsv',
    'sub-wlsubj139_run-06_wm_07-16-22_14-56_trialdesign.tsv'
]

'''
HELPER FUNCTIONS
'''
def designMatrix(design_filename, n_TR, trial_dur, initial_delay = 10, plot = False, save = True):
    d = pd.read_csv(design_filename, sep = '\t')
    conds = np.unique(d.gratingAng)
    conds.sort()

    n_pos = len(conds)
    n_conds = n_pos * 2

    D = np.zeros([n_TR, n_conds])
    run_time = initial_delay

    for i, trial in d.iterrows():
        ang = trial.gratingAng
        iti = trial.ITIDur

        row = int(run_time)
        if trial.saccadeType == 'Saccade':
            col = np.where(ang == conds)[0][0]
        else: 
            col = np.where(ang == conds)[0][0] + n_pos

        run_time += trial_dur + iti

        D[row, col] = 1

    if plot:
        plt.imshow(D,aspect='auto',interpolation='none')
        plt.gcf().set_size_inches(10, 10)
        
    if save:
        save_path = '../designs/%s/' % design_filename.split('/')[-2]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        filename = design_filename.split('/')[-1].split('designTrial.tsv')[0] + 'design.npy'
        np.save(os.path.join(save_path, filename), D)
        
    return D


def generateDesignMatrix(design_filepath, n_TR, n_conds, sacc_dur, exclude_trial = []):
    pre_scan_delay = 10
    cue = 0.5
    delay = 11.5
    trial_dur = cue+delay+sacc_dur

    D = np.zeros([n_TR, n_conds])

    design = pd.read_csv(d, sep = '\t')
    run_time = pre_scan_delay
    conds = design.cond.values

    for j, trial in design.iterrows():
        if j not in exclude_trial:
            row = int(run_time)
            col = trial.cond

            run_time += trial_dur + int(trial.ITIDur/1000)

            D[row, col] = 1
            
    return D, conds


def get_funcfilenames(sub_dir, wlsubj, session, save = True):
    datafiles = glob.glob(os.path.join(sub_dir, "sub-wlsubj%03d/ses-%s/func/sub-wlsubj%03d*mgz" % (wlsubj, session, wlsubj)))
    datafiles.sort()
    
    return datafiles
    

def load_hemidata(filenames, wlsubj, save = True):
    data = []
    for i, filename in enumerate(filenames):
        print('Nib. Loading:\n\t', filename)
        scan = nib.load(filename)
        print(filename, '\n\t', scan.shape)
        
        if i % 2:
            data[int(i/2)] = np.concatenate([data[int(i/2)], scan.get_fdata()])
            
            if save:
                subj_id = "sub-wlsubj%03d" % wlsubj
                prefix = filename.split('/')[-1].split('space')[0]
                
                save_path = '../voxels/%s/' % subj_id
                
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                save_name = prefix + 'hemi-both.npy'
                np.save(os.path.join(save_path, save_name), data[int(i/2)])
                print('Saved to: %s' % save_name)

        else:
            data.append(scan.get_fdata())
    
    return data


'''
MAIN CODE
'''
if __name__ == "__main__":
    # Get design files and generate design matrix
    design_dir = os.path.expanduser(design_dir)
    design_filepaths = [os.path.join(design_dir, "sub-wlsubj%03d" % wlsubj, filename) for filename in design_filenames]
    design_filepaths.sort()
    design = []
    conds = []

    for i, d in enumerate(design_filepaths):
        if i < 6: 
            D, c = generateDesignMatrix(d, n_TR[0], n_conds, sacc_dur[0], exclude_trial = [15])
        else:
            D, c = generateDesignMatrix(d, n_TR[1], n_conds, sacc_dur[1])
            
        design.append(D)
        conds.append(c)
            
    # Get functional filenames and load data from both hemispheres
    fmriprep_dir = os.path.expanduser(fmriprep_dir)
    funcs_filepaths = get_funcfilenames(fmriprep_dir, wlsubj, session)
    
    data = load_hemidata(funcs_filepaths, wlsubj, save = True)
    
    # print some relevant metadata
    nblocks = np.sum(np.concatenate(design))
    
    print(f'Data has {len(data)} runs\n')
    print(f'There are {nblocks} total blocks in runs 1-%d\n' % len(data))
    print(f'Shape of data from each run is: {data[0].shape}\n')
    print(f'XYZ dimensionality is: {data[0].shape[:3]} (one slice only)\n')
    print(f'N = {data[0].shape[3]} TRs per run\n')
    print(f'Numeric precision of data is: {type(data[0][0,0,0,0])}\n')

    # create a directory for saving GLMsingle outputs
    outputdir_glmsingle = os.path.expanduser(os.path.join(output_dir, 'sub-wlsubj%03d' % wlsubj))
    opt = dict()

    # set important fields for completeness 
    opt['wantlibrary'] = 1
    opt['wantglmdenoise'] = 1
    opt['wantfracridge'] = 1

    # we will keep the relevant outputs in memory
    # and also save them to the disk
    opt['wantfileoutputs'] = [1,1,1,1]
    opt['wantmemoryoutputs'] = [1,1,1,1]

    # Create a GLM_single object
    glmsingle_obj = GLM_single(opt)

    # visualize all the hyperparameters
    pprint(glmsingle_obj.params)
    
    # if these outputs don't already exist, we will perform the time-consuming call to GLMsingle;
    # otherwise, we will just load from disk.
    start_time = time.time()

    if not os.path.exists(outputdir_glmsingle):
        print("Creating output directory located at:", outputdir_glmsingle)
        os.makedirs(outputdir_glmsingle, 0o666)

        print(f'running GLMsingle...')

        # run GLMsingle
        results_glmsingle = glmsingle_obj.fit(
           design,
           data,
           stimdur,
           tr,
           outputdir=outputdir_glmsingle)

    else:
        print(f'loading existing GLMsingle outputs from directory:\n\t{outputdir_glmsingle}')

        # load existing file outputs if they exist
        results_glmsingle = dict()
        results_glmsingle['typea'] = np.load(
            os.path.join(outputdir_glmsingle,'TYPEA_ONOFF.npy'),allow_pickle=True).item()
        results_glmsingle['typeb'] = np.load(
            os.path.join(outputdir_glmsingle,'TYPEB_FITHRF.npy'),allow_pickle=True).item()
        results_glmsingle['typec'] = np.load(
            os.path.join(outputdir_glmsingle,'TYPEC_FITHRF_GLMDENOISE.npy'),allow_pickle=True).item()
        results_glmsingle['typed'] = np.load(
            os.path.join(outputdir_glmsingle,'TYPED_FITHRF_GLMDENOISE_RR.npy'),allow_pickle=True).item()

    elapsed_time = time.time() - start_time

        # Since GLM Single outputs are nVoxels x trial, with trials in chronological order, 
    # We save conditions in trial order for later analyses
    conds = np.asarray(conds).flatten()
    conds_df = pd.DataFrame({'conds': conds})
    conds_df.to_csv(os.path.join(os.path.expanduser(outputdir_glmsingle), 'sub-wlsubj%03d_conds.tsv' % wlsubj), sep = '\t')
    
    
    print(
        '\telapsed time: ',
        f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
    )
