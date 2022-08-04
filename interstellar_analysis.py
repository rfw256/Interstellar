import itertools
import os.path as op
import glob
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem
import scipy.signal as sp
from scipy.interpolate import interp2d
from scipy.interpolate import griddata
import os
import pickle
import warnings
import seaborn as sns
from scipy.special import iv
from scipy.optimize import curve_fit
warnings.filterwarnings('ignore')

'''
PARAMETERS
'''
wlsubjects = [139]
glm_dir = '~/mnt/winawer/Projects/Interstellar/analyses/GLMsingle'
prf_dir = '~/mnt/winawer/Projects/Interstellar/analyses/prf_vista'
output_dir = '~/mnt/winawer/Projects/Interstellar/analyses/voxels'
pos_dir = '~/mnt/winawer/Projects/Interstellar/task/Interstellar/training/positions/'


ret_params = ['x', 'y', 'eccen', 'angle', 'sigma', 'vexpl']
models = ['perception', 'ltm', 'wm']
space = 'surf'
load_hemis = ['lh', 'rh']
hemis = ['lh', 'rh', 'b']
labels = ['V1']

stim_ecc = 7

'''
HELPER FUNCTIONS
'''
def load_retinotopy(subj_prf_directory, load_hemis, ret_params):
    ret = {p:{'lh':[], 'rh':[]} for p in ret_params}

    for hemi, param in itertools.product(load_hemis, ret_params):
        rfile = glob.glob(op.join(subj_prf_directory, '%s*%s.mgz') % (hemi, param))
        print(rfile)
        r = nib.load(rfile[0]).get_fdata().squeeze()
        ret[param][hemi].append(r)
    
    for param in ret_params:
        ret[param]['b'] = [np.concatenate(b) for b in zip(ret[param]['lh'], ret[param]['rh'])]
        
    return ret


def load_glmsingle(filename, params = ''):
    g = np.load(filename, allow_pickle = True).item()
    
    if params == 'all':
        gparams = dict()
        for p, values in g.items():
            gparams[p] = g[p]
            
    elif params:
        gparams = dict()
        for p in params:
            gparams[p] = g[p]          
    else:
        gparams = {
            'beta': g['betasmd'],
            'R2': g['R2'],
            'se': g['glmbadness'],
            'rse': g['rrbadness']
        }

    return gparams


def load_conds(pos_dir, wlsubj):
    pos_filepath = os.path.join(pos_dir, "sub-wlsubj%03d_16pos.tsv" % wlsubj)
    conds = pd.read_csv(pos_filepath, sep = "\t", index_col = 0)
    
    return conds



def fix_deg(x):
    x = x - np.floor(x / 360 + 0.5) * 360
    
    return x


def voxels_df(ret, betas, n_vox, rois = [], hemi = 'b', stim_ecc = 7, sd_ecc = True, surf_labels = [], save = ''):
    # Pull retinotopy data
    voxels = {
        'x': ret['x'][hemi][0][:n_vox],
        'y': ret['y'][hemi][0][:n_vox],
        'eccen': ret['eccen'][hemi][0][:n_vox],
        'angle': ret['angle'][hemi][0][:n_vox],
        'sigma': ret['sigma'][hemi][0][:n_vox],
        'vexpl': ret['vexpl'][hemi][0][:n_vox],
        # 'roi': ret['ROIs_V1-3'][hemi][0][:n_vox]
    }

    voxels = pd.DataFrame(voxels)
    voxels = pd.DataFrame(betas).join(voxels)
    
    voxels['angle_adj'] = np.arctan2(voxels.y, voxels.x)
    
    
    # Filter by ROIs and restrict eccentricity
    if surf_labels:
        voxels['surf_label'] = ''

        for lab in surf_labels:
            for h in ['lh', 'rh']:
                label_path = "~/mnt/winawer/Projects/Retinotopy_NYU_3T/derivatives/freesurfer/sub-wlsubj%03d/label/%s.%s_exvivo.label" % (wlsubj, h, lab)
                label_path = os.path.expanduser(label_path)

                label_indices = nib.freesurfer.io.read_label(label_path)
                voxels['surf_label'].iloc[label_indices] = lab
        
        voxels = voxels[voxels.surf_label.isin(surf_labels)]

    if rois:
        voxels = voxels[voxels.roi.isin(rois)]
        
    if sd_ecc:
        voxels = voxels[np.abs(stim_ecc - voxels.eccen) <= voxels.sigma]
        
    if save:
        voxels.to_csv(save, sep = '\t')
    return voxels


def voxelsMeanCond(voxels, conditions):
    conds = np.unique(conditions)
    conds.sort()
    
    for i, c in enumerate(conds):
        indices = np.where(conditions == c)[0]
        v = voxels.iloc[:, indices]
        voxels_mean = v.mean(axis = 1)
        voxels['cond%02d' % i] = voxels_mean
        
    return voxels


def voxelsAngDist(wlsubj, voxels, angles, output_dir, save = True):
    for i, angle in enumerate(angles):
        # angle = -angle + 90
        angular_dist = np.degrees(voxels.angle) - angle
        angular_dist = angular_dist.apply(fix_deg)
        
        voxels['dist_cond%02d' % i] = angular_dist
        
    if save:
        filename = os.path.join(
            output_dir, 'sub-wlsubj%03d' % wlsubj, 'sub-wlsubj%03d_voxels.npy' % wlsubj)
        print(filename)
        np.save(filename, voxels, allow_pickle=True)
        
    return voxels



if __name__ == "__main__":
    glm_dir = os.path.expanduser(glm_dir)
    prf_dir = os.path.expanduser(prf_dir)
    output_dir = os.path.expanduser(output_dir)
    pos_dir = os.path.expanduser(pos_dir)

    for wlsubj in wlsubjects:
        ret = load_retinotopy(
            os.path.join(prf_dir, "sub-wlsubj%03d" % wlsubj), load_hemis, ret_params)
        gparams = load_glmsingle(
            os.path.join(glm_dir, "sub-wlsubj%03d" % wlsubj, "TYPED_FITHRF_GLMDENOISE_RR.npy"))
        conds_by_trials = pd.read_csv(
            os.path.join(glm_dir, "sub-wlsubj%03d" % wlsubj, "sub-wlsubj%03d_conds.tsv" % wlsubj), sep = '\t').conds
        conditions = load_conds(pos_dir, wlsubj)
        
        betas = gparams['beta'] 
        betas = betas.squeeze()
        
        n_vox = betas.shape[0]
        print(n_vox)
        angles = conditions.degrees.values
        
        voxels = voxels_df(ret, betas, n_vox, stim_ecc = stim_ecc, surf_labels=labels, save = os.path.join(
            output_dir, 'sub-wlsubj%03d' % wlsubj, 'sub-wlsubj%03d_pre-voxels_subconds.tsv' %  wlsubj))
        print(voxels.shape)
        voxels = voxelsMeanCond(voxels, conds_by_trials)
  
        voxels = voxelsAngDist(wlsubj, voxels, angles, output_dir)
    
        filename = os.path.join(
            output_dir, 'sub-wlsubj%03d' % wlsubj, 'sub-wlsubj%03d_voxels_subconds.tsv' %  wlsubj)
        voxels.to_csv(filename, sep = '\t')
        
    
    
    