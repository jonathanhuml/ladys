import torch
import numpy as np
import pandas as pd
import h5py
import os
from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors, make_eval_target_tensors, save_to_h5
from nlb_tools.evaluation import evaluate
import scipy.signal.windows as signal
import matplotlib.pyplot as plt

if __name__ == "__main__":
# ---- Default params ---- #
    default_dict = { # [latent_dim, alpha1, alpha2]
        'mc_maze': [52, 0.01, 0.0],
        'mc_rtt': [36, 0.0, 0.0],
        'area2_bump': [22, 0.0001, 0.0],
        'dmfc_rsg': [32, 0.0001, 0.0],
        'mc_maze_large': [44, 0.01, 0.0],
        'mc_maze_medium': [28, 0.0, 0.0],
        'mc_maze_small': [18, 0.01, 0.0],
    }

    # ---- Run Params ---- #
    dataset_name = "dmfc_rsg" # one of {'area2_bump', 'dmfc_rsg', 'mc_maze', 'mc_rtt', 
                                # 'mc_maze_large', 'mc_maze_medium', 'mc_maze_small'}
    bin_size_ms = 5
    # replace defaults with other values if desired
    latent_dim = default_dict[dataset_name][0]
    alpha1 = default_dict[dataset_name][1]
    alpha2 = default_dict[dataset_name][2]
    phase = 'val' # one of {'test', 'val'}

    # ---- Data locations ---- #
    datapath_dict = {
        'mc_maze': '~/data/000128/sub-Jenkins/',
        'mc_rtt': '~/data/000129/sub-Indy/',
        'area2_bump': '~/data/000127/sub-Han/',
        'dmfc_rsg': '~/data/000130/sub-Haydn/',
        'mc_maze_large': '~/data/000138/sub-Jenkins/',
        'mc_maze_medium': '~/data/000139/sub-Jenkins/',
        'mc_maze_small': '~/data/000140/sub-Jenkins/',
    }
    prefix_dict = {
        'mc_maze': '*full',
        'mc_maze_large': '*large',
        'mc_maze_medium': '*medium',
        'mc_maze_small': '*small',
    }
    # datapath = datapath_dict[dataset_name]
    # prefix = prefix_dict.get(dataset_name, '')


    cwd = os.getcwd()
    datapath = os.environ["HOME"] + '/000130/sub-Haydn/'
    # prefix = f'*ses-large'
    prefix = ''
    # savepath = f'{dataset_name}{"" if bin_size_ms == 5 else f"_{bin_size_ms}"}_smoothing_output_{phase}.h5'
    # ---- Load data ---- #
    dataset = NWBDataset(datapath, skip_fields=['hand_pos', 'cursor_pos', 'eye_pos', 'muscle_vel', 'muscle_len', 'joint_vel', 'joint_ang', 'force'])
    dataset.resample(bin_size_ms)

    # ---- Extract data ---- #
    if phase == 'val':
        train_split = 'train'
        eval_split = 'val'
    else:
        train_split = ['train', 'val']
        eval_split = 'test'
    train_dict = make_train_input_tensors(dataset, dataset_name, train_split, save_file=False)
    train_spikes_heldin = train_dict['train_spikes_heldin']
    train_spikes_heldout = train_dict['train_spikes_heldout']
    eval_dict = make_eval_input_tensors(dataset, dataset_name, eval_split, save_file=False)
    eval_spikes_heldin = eval_dict['eval_spikes_heldin']

    if phase == 'val':
        target_dict = make_eval_target_tensors(dataset, dataset_name=dataset_name, train_trial_split='train', eval_trial_split='val', include_psth=True, save_file=False)

    torch.save(train_dict, 'real_dataset_tensors/val_dmfc_rsg_train.pt')
    torch.save(eval_dict, 'real_dataset_tensors/val_dmfc_rsg_test.pt')
    torch.save(target_dict, 'real_dataset_tensors/val_dmfc_rsg_target.pt')
    # num trials x num_timesteps x num_neurons
    print(train_dict['train_spikes_heldin'].shape)
    print(train_dict['train_spikes_heldout'].shape)
    print(eval_dict['eval_spikes_heldin'].shape)

