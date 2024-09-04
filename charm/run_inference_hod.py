import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import pickle as pk
import numpy as np
import torch
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)
dev = torch.device(device)
import torch.optim as optim
import sys, os
from tqdm import tqdm

import yaml
import matplotlib
import matplotlib.pyplot as pl
pl.rc('text', usetex=True)
# Palatino
pl.rc('font', family='DejaVu Sans')
import yaml
import matplotlib


import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sbi import utils
from sbi import analysis
from sbi import inference
from sbi.inference import SNPE, SNLE, SNRE, simulate_for_sbi, prepare_for_sbi
from tqdm import tqdm
import pickle as pk
import dill


pos = sys.argv[1]
pk_type = sys.argv[2]
inference = sys.argv[3]
try:
    data = sys.argv[4]
except:
    data = 'mock'
# try:
#     do_bk = sys.argv[5]
# except:
#     do_bk = 'bk1_bk2_bk3'
try:
    do_wavelets = sys.argv[5]
except:
    do_wavelets = 's1_s2'

try:
    hidden_features = int(sys.argv[6])
except:
    hidden_features = 30

try:
    num_transforms = int(sys.argv[7])
except:
    num_transforms = 3

ldir_stats = '/mnt/home/spandey/ceph/CHARM/data/summary_stats_galaxies_sigpos_8/'
pos = 'rsd'
isim_array = np.arange(0,1800)
x_all = []
x_all_truth = []
theta_all = []
for ji in tqdm(range(len(isim_array))):
    isim = isim_array[ji]
    try:
        saved_j = pk.load(open(ldir_stats + '/Pk_Bk_galaxy_LH_' + str(isim) + '.dill', 'rb'))    
        saved_w = pk.load(open(ldir_stats + '/wavelets_simbigsettings_galaxy_LH_' + str(isim) + '.dill', 'rb'))    
        
        for ihod in range(10):
            if pk_type == 'mono':
                Pk_mock_ds = saved_j[pos + f'_Pk_mock_{ihod}'][:,:1].T
                Pk_truth_ds = saved_j[pos + f'_Pk_truth_{ihod}'][:,:1].T
            
            if pk_type == 'quad':
                Pk_mock_ds = saved_j[pos + f'_Pk_mock_{ihod}'][:,:2].T
                Pk_truth_ds = saved_j[pos + f'_Pk_truth_{ihod}'][:,:2].T
            
            if pk_type == 'all':
                Pk_mock_ds = saved_j[pos + f'_Pk_mock_{ihod}'][:,:3].T
                Pk_truth_ds = saved_j[pos + f'_Pk_truth_{ihod}'][:,:3].T



            Bk_truth_k0p06 = saved_j[pos + f'_Bk_truth_0p08_{ihod}'][1:-1]
            Bk_mock_k0p06 = saved_j[pos + f'_Bk_mock_0p08_{ihod}'][1:-1]

            Bk_truth_k0p2 = saved_j[pos + f'_Bk_truth_0p16_{ihod}'][1:-1]
            Bk_mock_k0p2 = saved_j[pos + f'_Bk_mock_0p16_{ihod}'][1:-1]

            Bk_truth_k0p3 = saved_j[pos + f'_Bk_truth_0p32_{ihod}'][1:-1]
            Bk_mock_k0p3 = saved_j[pos + f'_Bk_mock_0p32_{ihod}'][1:-1]

            s0_mock = saved_w[pos + f'_s0_mock_{ihod}']
            s0_truth = saved_w[pos + f'_s0_truth_{ihod}']

            s1_mock = saved_w[pos + f'_s1_mock_{ihod}']
            s1_truth = saved_w[pos + f'_s1_truth_{ihod}']

            s2_mock = saved_w[pos + f'_s2_mock_{ihod}']
            s2_truth = saved_w[pos + f'_s2_truth_{ihod}']

            if do_wavelets == 's0_s1_s2':
                summary_concat_mock_all_weighted = np.concatenate((Pk_mock_ds.flatten(), Bk_mock_k0p06, Bk_mock_k0p2, Bk_mock_k0p3, s0_mock, s1_mock[::3], s2_mock[::6]))
                summary_concat_truth_all_weighted = np.concatenate((Pk_truth_ds.flatten(), Bk_truth_k0p06, Bk_truth_k0p2, Bk_truth_k0p3, s0_truth, s1_truth[::3], s2_truth[::6]))
            elif do_wavelets == 's1_s2':
                summary_concat_mock_all_weighted = np.concatenate((Pk_mock_ds.flatten(), Bk_mock_k0p06, Bk_mock_k0p2, Bk_mock_k0p3, s1_mock[::3], s2_mock[::6]))
                summary_concat_truth_all_weighted = np.concatenate((Pk_truth_ds.flatten(), Bk_truth_k0p06, Bk_truth_k0p2, Bk_truth_k0p3, s1_truth[::3], s2_truth[::6]))
            elif do_wavelets == 's1':
                summary_concat_mock_all_weighted = np.concatenate((Pk_mock_ds.flatten(), Bk_mock_k0p06, Bk_mock_k0p2, Bk_mock_k0p3, s1_mock[::3]))
                summary_concat_truth_all_weighted = np.concatenate((Pk_truth_ds.flatten(), Bk_truth_k0p06, Bk_truth_k0p2, Bk_truth_k0p3, s1_truth[::3]))
            else:
                summary_concat_mock_all_weighted = np.concatenate((Pk_mock_ds.flatten(), Bk_mock_k0p06, Bk_mock_k0p2, Bk_mock_k0p3))
                summary_concat_truth_all_weighted = np.concatenate((Pk_truth_ds.flatten(), Bk_truth_k0p06, Bk_truth_k0p2, Bk_truth_k0p3))

            theta_hod = list(saved_j[f'theta_hod_{ihod}'].values())
            theta_cosmo = list(saved_j[f'theta_cosmo_{ihod}'].values())[:-1]
            theta_comb = np.array(theta_cosmo + theta_hod)


            if len(x_all) == 0:
                x_all = summary_concat_mock_all_weighted[None,:]
                x_all_truth = summary_concat_truth_all_weighted[None,:]            
                theta_all = theta_comb[None,:]
            else:
                # x_new = saved_j['summary_concat_mock_all'][None,:]
                # x_new_truth = saved_j['summary_concat_truth_all'][None,:]      
                #       
                # x_new = saved_j[pos + '_summary_concat_mock_all_weighted'][None,:]
                # x_new_truth = saved_j[pos + '_summary_concat_truth_all_weighted'][None,:]         

                x_new = summary_concat_mock_all_weighted[None,:]
                x_new_truth = summary_concat_truth_all_weighted[None,:]         


                # x_all = np.concatenate((x_all, saved_j['summary_concat_mock_all'][None,:]), axis = 0)
                x_all = np.concatenate((x_all, x_new), axis = 0)       
                x_all_truth = np.concatenate((x_all_truth, x_new_truth), axis = 0)       
                theta_all = np.concatenate((theta_all, theta_comb[None,:]), axis = 0)
        
    except Exception as e:
        print(e)
        pass


if data == 'truth':
    x_all = x_all_truth

x_all_std = np.std(x_all, axis = 0)
x_all_mean = np.mean(x_all, axis = 0)
theta_all_std = np.std(theta_all, axis = 0)
theta_all_mean = np.mean(theta_all, axis = 0)

indsel_p = [0, 4]

# x_all = torch.tensor(x_all).float()
# theta_all = torch.tensor(theta_all).float()
x_all_norm = torch.Tensor((x_all - x_all_mean)/x_all_std).float()
# x_all = torch.Tensor((x_all)/x_all_std).float()
theta_all_norm = torch.Tensor((theta_all - theta_all_mean)/theta_all_std).float()[:, indsel_p]


theta_all_std = theta_all_std[indsel_p]
theta_all_mean = theta_all_mean[indsel_p]


prior = utils.BoxUniform(
    low=(torch.tensor((np.array([0.1, 0.03, 0.5, 0.8, 0.6])[indsel_p]  - theta_all_mean)/theta_all_std)), 
    high=torch.tensor((np.array([0.5, 0.07, 0.9, 1.2, 1.0])[indsel_p] - theta_all_mean)/theta_all_std)
)

if inference == 'SNPE':
    neural_posterior = utils.posterior_nn(model="maf", hidden_features=hidden_features, num_transforms=num_transforms)
    inferer = SNPE(prior=prior, density_estimator=neural_posterior)
    density_estimator = inferer.append_simulations(theta_all_norm, x_all_norm).train()
    posterior = inferer.build_posterior(density_estimator)

if inference == 'SNLE':
    inferer = SNLE(prior, density_estimator="maf")
    inferer = inferer.append_simulations(theta_all_norm, x_all_norm)
    likelihood_estimator = inferer.train()
    posterior = inferer.build_posterior(mcmc_method="slice_np_vectorized", 
                                        mcmc_parameters=dict(thin=1))

if inference == 'SNRE':
    inferer = SNRE(prior)
    inferer = inferer.append_simulations(theta_all_norm, x_all_norm)
    likelihood_estimator = inferer.train()
    posterior = inferer.build_posterior(mcmc_method="slice_np_vectorized", 
                                        mcmc_parameters=dict(thin=1))


isim_obs_array = np.arange(1800, 2000)  
# isim_obs_array = np.arange(1000, 1020)  
# isim_obs_array = np.arange(600, 630)
# isim_obs_array = np.arange(430, 460)
# isim_obs_array = np.arange(900, 925)

Om_true_all = np.zeros(len(isim_obs_array))
Om_mock_all_mean = np.zeros(len(isim_obs_array))
Om_mock_all_std = np.zeros(len(isim_obs_array))

sig8_true_all = np.zeros(len(isim_obs_array))
sig8_mock_all_mean = np.zeros(len(isim_obs_array))
sig8_mock_all_std = np.zeros(len(isim_obs_array))

samples_all_isims = np.zeros((len(isim_obs_array), 1000, len(indsel_p)))
true_all_isims = np.zeros((len(isim_obs_array), len(indsel_p)))


for ji in range(len(isim_obs_array)):
    # print(ji)
    isim_obs = isim_obs_array[ji]

    saved_j = pk.load(open(ldir_stats + '/Pk_Bk_galaxy_LH_' + str(isim_obs) + '.dill', 'rb'))
    saved_w = pk.load(open(ldir_stats + '/wavelets_simbigsettings_galaxy_LH_' + str(isim_obs) + '.dill', 'rb'))        
    # saved_j = pk.load(open(ldir_stats + '/summary_stats_weighted_rsd_' + str(isim) + '_lgMmin_13p0.pk', 'rb'))    

    # Pk_mock_ds = saved_j[pos + '_Pk_mock_weighted'][:,:3][:,None]
    # Pk_truth_ds = saved_j[pos + '_Pk_truth_weighted'][:,:3][:,None]
    
    ihod = 0
    if pk_type == 'mono':
        Pk_mock_ds = saved_j[pos + f'_Pk_mock_{ihod}'][:,:1].T
        Pk_truth_ds = saved_j[pos + f'_Pk_truth_{ihod}'][:,:1].T
    
    if pk_type == 'quad':
        Pk_mock_ds = saved_j[pos + f'_Pk_mock_{ihod}'][:,:2].T
        Pk_truth_ds = saved_j[pos + f'_Pk_truth_{ihod}'][:,:2].T
    
    if pk_type == 'all':
        Pk_mock_ds = saved_j[pos + f'_Pk_mock_{ihod}'][:,:3].T
        Pk_truth_ds = saved_j[pos + f'_Pk_truth_{ihod}'][:,:3].T

    Bk_truth_k0p06 = saved_j[pos + f'_Bk_truth_0p08_{ihod}'][1:-1]
    Bk_mock_k0p06 = saved_j[pos + f'_Bk_mock_0p08_{ihod}'][1:-1]

    Bk_truth_k0p2 = saved_j[pos + f'_Bk_truth_0p16_{ihod}'][1:-1]
    Bk_mock_k0p2 = saved_j[pos + f'_Bk_mock_0p16_{ihod}'][1:-1]

    Bk_truth_k0p3 = saved_j[pos + f'_Bk_truth_0p32_{ihod}'][1:-1]
    Bk_mock_k0p3 = saved_j[pos + f'_Bk_mock_0p32_{ihod}'][1:-1]

    s0_mock = saved_w[pos + f'_s0_mock_{ihod}']
    s0_truth = saved_w[pos + f'_s0_truth_{ihod}']

    s1_mock = saved_w[pos + f'_s1_mock_{ihod}']
    s1_truth = saved_w[pos + f'_s1_truth_{ihod}']

    s2_mock = saved_w[pos + f'_s2_mock_{ihod}']
    s2_truth = saved_w[pos + f'_s2_truth_{ihod}']


    # summary_concat_mock_all_weighted = np.concatenate((Pk_mock_ds.flatten(), Bk_mock_k0p06, Bk_mock_k0p2, Bk_mock_k0p3))
    # summary_concat_truth_all_weighted = np.concatenate((Pk_truth_ds.flatten(), Bk_truth_k0p06, Bk_truth_k0p2, Bk_truth_k0p3))

    theta_hod = list(saved_j[f'theta_hod_{ihod}'].values())
    theta_cosmo = list(saved_j[f'theta_cosmo_{ihod}'].values())[:-1]
    theta_comb = np.array(theta_cosmo + theta_hod)

    

    if do_wavelets == 's0_s1_s2':
        x_obs = np.concatenate((Pk_truth_ds.flatten(), Bk_truth_k0p06, Bk_truth_k0p2, Bk_truth_k0p3, s0_truth, s1_truth[::3], s2_truth[::6]))
    elif do_wavelets == 's1_s2':
        x_obs = np.concatenate((Pk_truth_ds.flatten(), Bk_truth_k0p06, Bk_truth_k0p2, Bk_truth_k0p3, s1_truth[::3], s2_truth[::6]))
    elif do_wavelets == 's1':
        x_obs = np.concatenate((Pk_truth_ds.flatten(), Bk_truth_k0p06, Bk_truth_k0p2, Bk_truth_k0p3, s1_truth[::3]))
    else:
        x_obs = np.concatenate((Pk_truth_ds.flatten(), Bk_truth_k0p06, Bk_truth_k0p2, Bk_truth_k0p3))


    # x_obs = saved_j[pos + '_Pk_truth_weighted'].flatten()[None,:]                
    theta_obs = theta_comb[None,indsel_p]

    x_obs = torch.tensor((x_obs - x_all_mean)/x_all_std).float()
    # x_obs = torch.tensor((x_obs)/x_all_std).float()    
    theta_obs = torch.tensor((theta_obs  - theta_all_mean)/theta_all_std).float()

    samples = posterior.set_default_x(x_obs).sample((1000,))

    samples_all = samples.cpu().numpy()

    samples_all_isims[ji] = samples_all
    true_all_isims[ji] = theta_obs.cpu().numpy()
    # print(theta_obs.cpu().numpy())

    theta_obs_transformed = ((theta_obs.cpu().numpy() * theta_all_std) + theta_all_mean)
    Om_true_all[ji] = theta_obs_transformed[0,0]
    # sig8_true_all[ji] = theta_obs_transformed[0,4]
    sig8_true_all[ji] = theta_obs_transformed[0,1]


    # samples_all_transformed = ((samples_all * theta_all_std) + theta_all_mean)
    samples_all_transformed = ((samples_all * theta_all_std) + theta_all_mean)

    Om_mock_all_mean[ji] = np.mean(samples_all_transformed[:,0])
    Om_mock_all_std[ji] = np.std(samples_all_transformed[:,0])

    # sig8_mock_all_mean[ji] = np.mean(samples_all_transformed[:,4])
    # sig8_mock_all_std[ji] = np.std(samples_all_transformed[:,4])

    sig8_mock_all_mean[ji] = np.mean(samples_all_transformed[:,1])
    sig8_mock_all_std[ji] = np.std(samples_all_transformed[:,1])


references = "random"
metric = 'euclidean'
bootstrap= True
norm = True
num_alpha_bins = None
num_bootstrap = 20

# samples_all_tarp = np.moveaxis(samples_all_isims, 0, 1)[:,:71, :]
# true_all_tarp = true_all_isims[:71,...]

samples_all_tarp = np.moveaxis(samples_all_isims, 0, 1)
true_all_tarp = true_all_isims

samples_all_transformed = ((samples_all_tarp * theta_all_std) + theta_all_mean)
true_all_transformed = ((true_all_tarp * theta_all_std) + theta_all_mean)
samples_all_tarp_mean = np.mean(samples_all_transformed, axis = 0)
samples_all_tarp_std = np.std(samples_all_transformed, axis = 0)

saved = {'samples_all_tarp': samples_all_tarp, 'true_all_tarp': true_all_tarp, 
        'theta_all_std': theta_all_std, 'theta_all_mean': theta_all_mean,
        'samples_all_tarp_mean': samples_all_tarp_mean, 'samples_all_tarp_std': samples_all_tarp_std,
        'samples_all_transformed': samples_all_transformed, 'true_all_transformed': true_all_transformed}
        
import pickle as pk
# pk.dump(saved, open(f'/mnt/home/spandey/ceph/CHARM/results/hod_inference/saved_GALAXIES_Pk_Bk_wavelets_tarp_{pos}_Pk_{pk_type}_wavelets_{do_wavelets}_{inference}_data_{data}.pk', 'wb'))
pk.dump(saved, open(f'/mnt/home/spandey/ceph/CHARM/results/hod_inference/saved_GALAXIES_Pk_Bk_wavelets_tarp_{pos}_Pk_{pk_type}_wavelets_{do_wavelets}_{inference}_nt_{num_transforms}_nf_{hidden_features}_data_{data}.pk', 'wb'))

import sys, os
sys.path.append('/mnt/home/spandey/ceph/tarp/src/tarp')
import drp

ecp, alpha = drp.get_tarp_coverage(
            samples_all_tarp, true_all_tarp,
            references=references, metric=metric,
            norm=norm, bootstrap=bootstrap,
            num_alpha_bins=num_alpha_bins,
            num_bootstrap=num_bootstrap
        )

sdirf = '/mnt/home/spandey/ceph/CHARM/results/hod_inference/'

Om_min = 0.12
Om_max = 0.48
sig8_min = 0.62
sig8_max = 0.98
indsel = np.where((Om_true_all > Om_min) & (Om_true_all < Om_max) & (sig8_true_all > sig8_min) & (sig8_true_all < sig8_max))[0]

fig, ax = pl.subplots(1, 3, figsize=(15, 4))
ax[0].errorbar(true_all_transformed[:,0], samples_all_tarp_mean[:,0], yerr = samples_all_tarp_std[:,0], fmt = 'o')
ax[0].plot([Om_min, Om_max], [Om_min, Om_max], 'k--')
ax[0].tick_params(axis='both', which='major', labelsize=15)
ax[0].tick_params(axis='both', which='minor', labelsize=15)
ax[0].set_xlabel('Om true', size=15)
ax[0].set_ylabel('Om inferred', size=15)

ax[1].errorbar(true_all_transformed[:, -1], samples_all_tarp_mean[:,-1], yerr = samples_all_tarp_std[:,-1], fmt = 'o')
ax[1].plot([sig8_min, sig8_max], [sig8_min, sig8_max], 'k--')
ax[1].set_xlabel('sig8 true', size=15)
ax[1].set_ylabel('sig8 inferred', size=15)
ax[1].tick_params(axis='both', which='major', labelsize=15)
ax[1].tick_params(axis='both', which='minor', labelsize=15)

ax = ax[2]
ax.plot([0, 1], [0, 1], ls='--', color='k')
if bootstrap:
    ecp_mean = np.mean(ecp, axis=0)
    ecp_std = np.std(ecp, axis=0)
    ax.plot(alpha, ecp_mean, label='TARP', color='b')
    ax.fill_between(alpha, ecp_mean - ecp_std, ecp_mean + ecp_std,
                    alpha=0.2, color='b')
    ax.fill_between(alpha, ecp_mean - 2 * ecp_std, ecp_mean + 2 * ecp_std,
                    alpha=0.2, color='b')
else:
    ax.plot(alpha, ecp, label='TARP')
# ax.legend()
pl.tick_params(axis='both', which='major', labelsize=15)
pl.tick_params(axis='both', which='minor', labelsize=15)
ax.set_ylabel("Expected Coverage", size=15)
ax.set_xlabel("Credibility Level", size=15)
pl.tight_layout()
pl.savefig(sdirf + f'Om_sig8_TARP_inferred_saved_GALAXIES_Pk_Bk_wavelets_{pos}_Pk_{pk_type}_wavelets_{do_wavelets}_{inference}_nt_{num_transforms}_nf_{hidden_features}_data_{data}.pdf')

# 
# pl.suptitle(r'Train:CHARM, Test:Quijote; Redshift space; Galaxies; $P_{\ell = 0} + P_{\ell = 1} + B_{\ell = 0}$; SNPE', size=17)
# pl.suptitle(r'Train:CHARM, Test:Qu

# fig, ax = plt.subplots(1, 1, figsize=(5, 4))
# ax.plot([0, 1], [0, 1], ls='--', color='k')
# if bootstrap:
#     ecp_mean = np.mean(ecp, axis=0)
#     ecp_std = np.std(ecp, axis=0)
#     ax.plot(alpha, ecp_mean, label='TARP', color='b')
#     ax.fill_between(alpha, ecp_mean - ecp_std, ecp_mean + ecp_std,
#                     alpha=0.2, color='b')
#     ax.fill_between(alpha, ecp_mean - 2 * ecp_std, ecp_mean + 2 * ecp_std,
#                     alpha=0.2, color='b')
# else:
#     ax.plot(alpha, ecp, label='TARP')
# # ax.legend()
# pl.tick_params(axis='both', which='major', labelsize=15)
# pl.tick_params(axis='both', which='minor', labelsize=15)
# ax.set_ylabel("Expected Coverage", size=15)
# ax.set_xlabel("Credibility Level", size=15)
# pl.savefig(sdirf + f'TARP_coverage_saved_GALAXIES_Pk_Bk_wavelets_{pos}_Pk_{pk_type}_wavelets_{do_wavelets}_{inference}_data_{data}.pdf', bbox_inches='tight')

# pl.figure()
# pl.errorbar(true_all_transformed[:,0], samples_all_tarp_mean[:,0], yerr = samples_all_tarp_std[:,0], fmt = 'o')
# pl.plot([0.1, 0.5], [0.1, 0.5], 'k--')
# pl.tick_params(axis='both', which='major', labelsize=15)
# pl.tick_params(axis='both', which='minor', labelsize=15)
# pl.xlabel('Om$ true', size=15)
# pl.ylabel('Om inferred', size=15)
# pl.savefig(sdirf + f'Om_inferred_saved_GALAXIES_Pk_Bk_wavelets_inferred_{pos}_Pk_{pk_type}_wavelets_{do_wavelets}_{inference}_data_{data}.pdf', bbox_inches='tight')

# pl.figure()
# pl.errorbar(true_all_transformed[:,-1], samples_all_tarp_mean[:,-1], yerr = samples_all_tarp_std[:,-1], fmt = 'o')
# pl.plot([0.55, 1.05], [0.55, 1.05], 'k--')
# pl.xlabel('sigma8 true', size=15)
# pl.ylabel('sigma8 inferred', size=15)
# pl.savefig(sdirf + f'sig8_inferred_saved_GALAXIES_Pk_Bk_wavelets_{pos}_Pk_{pk_type}_wavelets_{do_wavelets}_{inference}_data_{data}.pdf', bbox_inches='tight')

