#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Duong Nguyen
# Created Date: 2019/03/03
# =============================================================================
"""Dataset utils for DAODEN"""


import numpy as np
import os
import pickle
from torch.utils.data import Dataset
import daoden_utils

class DAODEN_Dataset(Dataset):
    """DAODEN dataset."""

    def __init__(self, config, m_obs, m_seqs, m_masks, v_mean=None, v_std=None, dtype=np.float32):
        """
        Args:
            - dataset_path (string): Path to the pickle file.
            - transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        n_seqs, seq_len, data_dim = m_seqs.shape
        self.dtype = dtype
        
        ## Masks
        self.masks = np.zeros(m_masks.shape)
        if config.dt_obs == 0.00:
            self.masks = m_masks
        else:
            obs_cycle = int(config.dt_obs/config.dt_integration)
            self.masks[:,::obs_cycle,:] = 1.0
            
        ## Normalization
        if config.normalize_data:
            assert (v_mean is not None) and (v_std is not None), "To normalize data, v_mean and v_std must be sepecified."
        self.normalize_data = config.normalize_data
        self.dt_integration = config.dt_integration
        if self.normalize_data:
            m_seqs      = (m_seqs-v_mean)/v_std
            m_obs       = (m_obs-v_mean)/v_std

        ## Interpolation
        self.interp_seq = np.zeros(m_seqs.shape)
        v_interp_idx = np.arange(seq_len)
        for idx_seq in range(n_seqs): 
            for d_idx in range(data_dim):
                v_idx = np.where(self.masks[idx_seq,:,d_idx])[0]
                self.interp_seq[idx_seq,:,d_idx] = np.interp(v_interp_idx,v_idx,m_obs[idx_seq,v_idx,d_idx])
        
        self.clean_seqs = m_seqs

    def __len__(self):
        return len(self.interp_seq)

    def __getitem__(self, idx):
        ## batch-major
        targets         = self.interp_seq[idx]
        clean_targets   = self.clean_seqs[idx]
        masks           = self.masks[idx]

        # Shift the inputs one step forward in time. Also remove the last
        # timestep so that targets and inputs are the same length.
        obs_dim = targets.shape[-1]
        inputs = np.concatenate([np.zeros((1,obs_dim)),targets[:-1]],axis=0)

        return (inputs.astype(self.dtype), targets.astype(self.dtype),
                    masks.astype(np.uint8), clean_targets.astype(self.dtype))

def generate_dataset(func,z0s,dt_integration,seq_len,args,
                  N_init = 500,
                  v_std=None,
                  noise_ratio=0.0,
                  obs_rate = 1/8
                 ):
    """Generate a system of ODEs.
    Args:
        func: the ODE function, dz/dt = func(z, t, ...).
        z0s: the initial conditions, shape = (n_ics,data_dim)
        dt_integration:
        seq_len: 
        args: arguments of the ODE functions
        N_init: the "wash-out" step, to make sure that the states are 
            on the attractor.
        v_std: std of the variables of the system.
        noise_ratio: std_noise/std_signal.
        obs_rate: observation rate.
    Returns:
        seqs: sequences of the true states of the system.
        obs: noisy observations.
        masks: masks[i,j,k] == 1: element k at timestep j of sequence i is observed.
    """
    n_ics,data_dim = z0s.shape
    seqs   = np.zeros((n_ics,seq_len,data_dim))
    obs    = np.zeros_like(seqs)
    masks  = np.zeros_like(seqs)
    
    
    print("Generating data...")
    for idx in tqdm(range(n_ics)):
        z0 = z0s[idx,:] # shape = (3,)
        # Solve the ODE
        traj = odeint(func,
                      z0,
                      np.arange(seq_len+N_init)*dt_integration,
                      args=args
                     );
        # Only use the part that are in the attractor.
        seqs[idx,:,:] = traj[N_init:]
        
        # generate noisy/partial observations (yo)
        eps = np.random.multivariate_normal(np.zeros(data_dim),
                                            cov=np.diag((noise_ratio*v_std)**2),
                                            size=seq_len);
        obs[idx,:,:] = seqs[idx,:,:]+eps
        
        # Mask
        for d_i in range(data_dim):
            masks[idx,np.random.choice(seq_len, int(seq_len*obs_rate),replace=False),d_i] = 1.0
            
    return seqs,obs,masks
