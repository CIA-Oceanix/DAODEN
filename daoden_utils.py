#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Duong Nguyen
# Created Date: 2019/03/03
# =============================================================================
"""Utils for DAODEN"""




import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
import torch
import torch.distributions as tdist
import pickle
import os



L63_MEAN   = np.array([ 0.6773526,  0.6965731, 23.556093 ]).astype(np.float32)
L63_STD    = np.array([7.890003, 8.986262, 8.539845]).astype(np.float32)
L63_RANGE  = np.array([40,48,40]).astype(np.float32) #approximate
L63LG_STD  = np.array([0.13544865, 0.114841  , 0.1136952 , 0.12601069, 0.14394593,
                       0.1625788 , 0.17958239, 0.19397355, 0.20539782, 0.21380037,
                       0.21927558, 0.22199508, 0.22217153, 0.22003974, 0.21584669,
                       0.20984665, 0.20229935, 0.19347054, 0.18363437, 0.17307774,
                       0.16210658, 0.15105412, 0.14029028, 0.13022988, 0.1213335 ,
                       0.11409049, 0.10897057, 0.10633986, 0.10636553, 0.10896185,
                       0.11381753, 0.12048768, 0.1284952 , 0.13739998, 0.14682946,
                       0.15648306, 0.16612436, 0.17556982, 0.18467813, 0.19334128,
                       0.20147778, 0.20902742, 0.21594745, 0.22220964, 0.2277981 ,
                       0.23270758, 0.2369422 , 0.24051437, 0.24344392, 0.24575739,
                       0.24748739, 0.24867202, 0.24935431, 0.24958166, 0.24940528,
                       0.24887962, 0.24806166, 0.24701024, 0.24578528, 0.24444686,
                       0.24305433, 0.24166521, 0.24033422, 0.23911209, 0.23804454,
                       0.23717127, 0.23652507, 0.23613116, 0.23600673, 0.23616081,
                       0.23659434, 0.23730067, 0.23826622, 0.23947147, 0.24089207,
                       0.24250005, 0.24426516, 0.24615614, 0.24814192, 0.25019284,
                       0.25228157, 0.25438406, 0.25648025, 0.25855462, 0.26059666,
                       0.26260105, 0.26456783, 0.26650225, 0.26841459, 0.27031971,
                       0.27223643, 0.27418683, 0.27619522, 0.27828711, 0.28048797,
                       0.28282191, 0.28531032, 0.2879706 , 0.29081486, 0.29384893,
                       0.29707155, 0.30047395, 0.30403981, 0.30774576, 0.31156239,
                       0.31545591, 0.3193905 , 0.3233314 , 0.32724895, 0.33112352,
                       0.3349517 , 0.33875374, 0.34258248, 0.34653389, 0.35075902,
                       0.35547716, 0.36098911, 0.36768917, 0.37607319, 0.38673962,
                       0.40038042, 0.41775969, 0.43968131, 0.46694997, 0.50033358,
                       0.54053512, 0.58817905, 0.64381274]).astype(np.float32)
L63LG_MEAN = np.array([ 0.30241911,  0.3389347 ,  0.36879565,  0.39245305,  0.41034171,
                        0.42288033,  0.43047184,  0.43350366,  0.43234789,  0.4273617 ,
                        0.41888747,  0.40725314,  0.39277243,  0.37574515,  0.35645741,
                        0.33518192,  0.31217825,  0.28769309,  0.26196053,  0.2352023 ,
                        0.20762805,  0.17943562,  0.15081129,  0.12193006,  0.09295592,
                        0.06404209,  0.03533129,  0.00695605, -0.02096109, -0.04830727,
                       -0.07497906, -0.10088221, -0.12593134, -0.15004973, -0.17316901,
                       -0.19522893, -0.21617705, -0.23596852, -0.25456582, -0.27193842,
                       -0.28806261, -0.30292118, -0.31650316, -0.32880355, -0.3398231 ,
                       -0.34956798, -0.35804955, -0.3652841 , -0.37129258, -0.3761003 ,
                       -0.37973673, -0.38223518, -0.38363255, -0.38396908, -0.38328808,
                       -0.38163562, -0.37906036, -0.37561317, -0.37134697, -0.3663164 ,
                       -0.36057755, -0.35418775, -0.34720525, -0.33968899, -0.33169832,
                       -0.32329272, -0.31453156, -0.30547384, -0.29617788, -0.28670112,
                       -0.27709978, -0.26742868, -0.25774089, -0.24808752, -0.23851743,
                       -0.229077  , -0.21980981, -0.21075641, -0.20195405, -0.19343642,
                       -0.18523336, -0.17737063, -0.16986961, -0.16274707, -0.15601487,
                       -0.14967972, -0.14374291, -0.13820002, -0.13304071, -0.12824838,
                       -0.12379998, -0.11966569, -0.11580868, -0.11218483, -0.1087425 ,
                       -0.1054222 , -0.1021564 , -0.09886922, -0.09547614, -0.09188382,
                       -0.08798975, -0.08368202, -0.07883906, -0.07332936, -0.06701121,
                       -0.05973246, -0.05133019, -0.04163053, -0.03044832, -0.01758688,
                       -0.00283775,  0.01401959,  0.03321798,  0.05500284,  0.07963245,
                        0.10737821,  0.13852487,  0.17337088,  0.21222855,  0.25542439,
                        0.30329935,  0.35620909,  0.41452424,  0.47863067,  0.54892974,
                        0.6258386 ,  0.70979041,  0.80123465]).astype(np.float32)
L96_MEAN  = np.array([2.34815256, 2.0384624 , 2.4663713 , 2.66959106, 2.17639819,
                      2.42087137, 2.86378758, 2.55585967, 2.28691033, 2.57201891,
                      2.45804551, 2.38859373, 2.62620961, 2.36231943, 2.06150043,
                      2.56835326, 2.24506002, 1.95576678, 2.460721  , 2.64145592,
                      2.13670113, 2.14523662, 2.45322255, 2.47925549, 1.88855192,
                      2.14653819, 2.55384809, 2.43078241, 2.49281286, 2.2659346 ,
                      1.92629562, 2.38643262, 2.34869809, 2.10803692, 2.34101252,
                      2.44915523, 2.15583007, 2.46901227, 2.37713098, 2.72269531]).astype(np.float32)
L96_STD = np.array([3.54657435, 3.49473548, 3.59356142, 3.83864436, 3.56977765,
                    3.56872145, 3.75123364, 3.73934381, 3.66047858, 3.6470688 ,
                    3.50937159, 3.63284937, 3.71613899, 3.64941213, 3.36386299,
                    3.67826983, 3.55529745, 3.65173323, 3.62342736, 3.8674828 ,
                    3.68018221, 3.8416176 , 3.51709517, 3.29575751, 3.43697217,
                    3.79181016, 3.72154855, 3.8230597 , 3.56158731, 3.5089469 ,
                    3.51099144, 3.54898955, 3.68074192, 3.39604403, 3.44098367,
                    3.81690635, 3.81933892, 4.00748224, 3.57008627, 3.79835542]).astype(np.float32)
L96_RANGE  = (np.ones(40)*1/20).astype(np.float32) #approximate



def plot_attractor(sequence,fig=None,title=None,**kwargs):
    """ Plot a L63 attractor in 3D space
    Args:
        sequence: a L63 sequence (seq_len,data_dim)
    Returns:
        None
    """
    if fig is None:
        FIG_DPI = 80
        fig=plt.figure(figsize=(1920/2/FIG_DPI, 1080/2/FIG_DPI), dpi=FIG_DPI)
    ax=fig.gca(projection='3d')
    line2, = ax.plot(sequence[:,0],sequence[:,1],sequence[:,2],**kwargs)
    #line3, = ax.plot(sequence[:10,0],sequence[:10,1],sequence[:10,2],'k')
    ax.set_xlabel('$x_1$');ax.set_ylabel('$x_2$');ax.set_zlabel('$x_3$');
    plt.title(title)
    
def plot_L96(gen_seq,true_seq, fig=None, dt_integration = 0.05):
    """Plot the Hovmöller diagram of the L96 sequences.
    Args:
        gen_seq: the generated sequence, a tensor of shape [seq_len,data_dim]
        true_seq: the true sequence, a tensor of shape [seq_len,data_dim]
    """
    lambda1=1.67
    if fig is None:
        FIG_DPI = 100
        plt.figure(figsize=(1920/2/FIG_DPI, 1080/2/FIG_DPI), dpi=FIG_DPI)
    
#    N = 100
#    gen_seq  = gen_seq[:N,:]
#    true_seq = true_seq[:N,:]
    N = true_seq.shape[0]
    v_time   = np.arange(N)*dt_integration*lambda1
    
    error = np.abs(true_seq - gen_seq)
    
    ## Hovmöller diagram
    [X,Y]=np.meshgrid(v_time,range(40))
    
    # True seq
    plt.subplot(3,1,1)
    plt.pcolor(X,Y,np.swapaxes(true_seq,0,1))
    plt.clim([-10,10])
    plt.colorbar()
    plt.ylabel('True')

    plt.subplot(3,1,2)
    plt.pcolor(X,Y,np.swapaxes(gen_seq,0,1))
    plt.clim([-10,10])
    plt.colorbar()
    plt.ylabel('Generated')

    plt.subplot(3,1,3)
    plt.pcolor(X,Y,np.swapaxes(error,0,1))
    plt.xlabel('Time (in Lyapynov unit)')
    plt.ylabel('Error') 
    plt.colorbar()
    
def generate_seq(model,x0,dt_integration,n_step):
    """ Generate a dynamical sequence
    Args:
        x0: the initial state.
        dt_integration: time step.
        model: a dynamical model, which take x0 and dt as args. 
        n_step: sequence length.
    Returns:
        seq: the generated sequence (seq_len,data_dim)
    """
    l_seq = []
    with torch.no_grad():
        inp = x0
        for d_i in range(n_step):
            pred,_ = model(inp,dt_integration)
            l_seq.append(pred.cpu().detach().numpy())
            inp = pred
    seq = np.concatenate(l_seq)
    return(seq)

def generate_seq_stoch(model,x0,dt,n_step):
    l_state_seq = []
    l_std_seq = []
    with torch.no_grad():
        x_prev = x0
        for d_i in range(n_step):
            mu_prior,var_prior = model.transition(x_prev)
            prior_dist = tdist.MultivariateNormal(mu_prior,
                                                  covariance_matrix=torch.diag_embed(var_prior))
            x_prev = prior_dist.sample()
            l_state_seq.append(x_prev.cpu().detach().numpy())
            l_std_seq.append(torch.sqrt(var_prior).cpu().detach().numpy())
    state_seq = np.concatenate(l_state_seq)
    var_seq   = np.concatenate(l_std_seq)
    return(state_seq,var_seq)

def compute_largest_lyap(trans_model,x0,dt_integration,d0,nb_iter,v_mean,v_std):
    """Estimate the largest Lyapunov exponent of a dynamical system.
    For details, see: A. Wolf, J. B. Swift, H. L. Swinney, and J. A. 
    Vastano, “Determining Lyapunov exponents from a time series,” 
    Physica D: Nonlinear Phenomena, vol. 16, no. 3, pp. 285–317, Jul. 1985.
    
    Args:
        trans_model: a dynamical model, which take x0 and dt as args. 
        x0: the initial states (nb_initial_conditions,data_dim).
        dt_integration: time step.
        d0: the perturbation.
        model: a dynamical model, which take x0 and dt as args. 
        nb_iter: number of iteration.
    Returns:
        largest_lyap: the largest Lyapunov exponents.
        forecasted_states: the forcasted sequences.
    """
    n_ics_test,data_dim = x0.shape
    device = trans_model.device
    with torch.no_grad():
        forecasted_states_noisy, forecasted_states_noisy_proj = [],[]
        forecasted_states = np.empty((0,n_ics_test,data_dim))
        log_pret = np.empty((0,n_ics_test))
        largest_lyap = np.empty((0,n_ics_test))
        tmp = np.reshape(x0,(n_ics_test,data_dim))
        tmp_init = tmp
        tmp_noisy = (tmp+d0*np.ones(data_dim)).astype(np.float32)
        for i in tqdm(range(nb_iter)):
            #1 - forecast the states :
            state_pred,_ = trans_model(torch.from_numpy((tmp_init-v_mean)/v_std).to(device),dt_integration)
            state_pred = state_pred.cpu().detach().numpy()*v_std + v_mean

            state_pred_perturbed,_ = trans_model(torch.from_numpy((tmp_noisy-v_mean)/v_std).to(device),dt_integration)
            state_pred_perturbed = state_pred_perturbed.cpu().detach().numpy()*v_std + v_mean

            #forecasted_states.append(state_pred[:,:].astype(np.float32))
            forecasted_states = np.concatenate((forecasted_states,np.expand_dims(state_pred,0)))
            forecasted_states_noisy.append(state_pred_perturbed[:,:].astype(np.float32))

            d1 = np.linalg.norm(forecasted_states_noisy[-1]-forecasted_states[-1],axis=1)
            #compute log(d1/d0)
            log_pret = np.concatenate((log_pret,np.expand_dims(np.log(d1/d0),0)))

            # readjusting orbits
            forecasted_states_noisy_proj.append(
forecasted_states[-1]+d0*(forecasted_states_noisy[-1]-forecasted_states[-1])/np.repeat(np.expand_dims(d1,1),data_dim,axis=1))

            tmp_init  = np.reshape(forecasted_states[-1],(n_ics_test,-1)).astype(np.float32)
            tmp_noisy = np.reshape(forecasted_states_noisy_proj[-1],(n_ics_test,-1)).astype(np.float32)
            #largest_lyap.append(np.mean(log_pret)/dt_integration)
            lambda1_tmp = np.mean(log_pret,axis=0,keepdims=True)/dt_integration
            largest_lyap = np.concatenate((largest_lyap,lambda1_tmp),axis=0)
    return largest_lyap, forecasted_states
