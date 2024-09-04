import sys, os
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
pl.rc('text', usetex=False)
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

# pos = 'rsd'
# pk_type = 'mono'
# inference = 'SNPE'

pos = sys.argv[1]
pk_type = sys.argv[2]
inference = sys.argv[3]
try:
    data = sys.argv[4]
except:
    data = 'mock'

ldir_stats = '/mnt/home/spandey/ceph/CHARM/data/summary_stats_charm_truth_nsubv_vel_10k/'
isim_array = np.arange(0,1800)
for ji in tqdm(range(len(isim_array))):
    isim = isim_array[ji]
    # saved_j = pk.load(open(ldir_stats + '/saved_stats_halos_' + str(isim) + '_CMASS.pk', 'rb'))
    try:
        # saved_j = pk.load(open(ldir_stats + '/saved_stats_halos_with_MASS_WEIGHTED_' + str(isim) + '_CMASS.pk', 'rb'))    
        saved_j = pk.load(open(ldir_stats + '/summary_stats_weighted_rsd_' + str(isim) + '_nbar_4en4.pk', 'rb'))    

        # Pk_mock_ds = saved_j[pos + '_Pk_mock_weighted'][:,:3][:,None]
        # Pk_truth_ds = saved_j[pos + '_Pk_truth_weighted'][:,:3][:,None]

        if pk_type == 'mono':
            Pk_mock_ds = saved_j[pos + '_Pk_mock_weighted'][:,:1].T
            Pk_truth_ds = saved_j[pos + '_Pk_truth_weighted'][:,:1].T
        elif pk_type == 'quad':
            Pk_mock_ds = saved_j[pos + '_Pk_mock_weighted'][:,:2].T
            Pk_truth_ds = saved_j[pos + '_Pk_truth_weighted'][:,:2].T
        elif pk_type == 'all':
            Pk_mock_ds = saved_j[pos + '_Pk_mock_weighted'][:,:3].T
            Pk_truth_ds = saved_j[pos + '_Pk_truth_weighted'][:,:3].T



        Bk_truth_k0p06 = saved_j[pos + '_Bk_truth_k0p06_weighted'][1:-1]
        Bk_mock_k0p06 = saved_j[pos + '_Bk_mock_k0p06_weighted'][1:-1]

        Bk_truth_k0p2 = saved_j[pos + '_Bk_truth_k0p2_weighted'][1:-1]
        Bk_mock_k0p2 = saved_j[pos + '_Bk_mock_k0p2_weighted'][1:-1]

        Bk_truth_k0p3 = saved_j[pos + '_Bk_truth_k0p3_weighted'][1:-1]
        Bk_mock_k0p3 = saved_j[pos + '_Bk_mock_k0p3_weighted'][1:-1]

        s0_mock = saved_j[pos + '_s0_mock_weighted']
        s0_truth = saved_j[pos + 's0_truth_weighted']

        s1_mock = saved_j[pos + '_s1_mock_weighted']
        s1_truth = saved_j[pos + 's1_truth_weighted']

        s2_mock = saved_j[pos + '_s2_mock_weighted']
        s2_truth = saved_j[pos + 's2_truth_weighted']

        summary_concat_mock_all_weighted = np.concatenate((Pk_mock_ds.flatten(), Bk_mock_k0p06, Bk_mock_k0p2, Bk_mock_k0p3, s1_mock[::4], s2_mock[::6]))
        summary_concat_truth_all_weighted = np.concatenate((Pk_truth_ds.flatten(), Bk_truth_k0p06, Bk_truth_k0p2, Bk_truth_k0p3, s1_truth[::4], s2_truth[::6]))


        if ji == 0:

            x_all = summary_concat_mock_all_weighted[None,:]
            x_all_truth = summary_concat_truth_all_weighted[None,:]            

            theta_all = saved_j['cosmo'][None,:]
        else:

            x_new = summary_concat_mock_all_weighted[None,:]
            x_new_truth = summary_concat_truth_all_weighted[None,:]         
            x_all = np.concatenate((x_all, x_new), axis = 0)       
            x_all_truth = np.concatenate((x_all_truth, x_new_truth), axis = 0)       
            theta_all = np.concatenate((theta_all, saved_j['cosmo'][None,:]), axis = 0)
        
    except:
        pass


if data == 'truth':
    x_all = x_all_truth

x_all_std = np.std(x_all, axis = 0)
x_all_mean = np.mean(x_all, axis = 0)
theta_all_std = np.std(theta_all, axis = 0)
theta_all_mean = np.mean(theta_all, axis = 0)

indsel_p = [0, 4]

x_all = torch.Tensor((x_all - x_all_mean)/x_all_std).float()
theta_all = torch.Tensor((theta_all - theta_all_mean)/theta_all_std).float()[:, indsel_p]


theta_all_std = theta_all_std[indsel_p]
theta_all_mean = theta_all_mean[indsel_p]


prior = utils.BoxUniform(
    low=(torch.tensor((np.array([0.1, 0.03, 0.5, 0.8, 0.6])[indsel_p]  - theta_all_mean)/theta_all_std)), 
    high=torch.tensor((np.array([0.5, 0.07, 0.9, 1.2, 1.0])[indsel_p] - theta_all_mean)/theta_all_std)
)

if inference == 'SNPE':
    neural_posterior = utils.posterior_nn(model="maf", hidden_features=30, num_transforms=3)
    inferer = SNPE(prior=prior, density_estimator=neural_posterior)
    density_estimator = inferer.append_simulations(theta_all, x_all).train()
    posterior = inferer.build_posterior(density_estimator)

if inference == 'SNLE':
    inferer = SNLE(prior, density_estimator="maf")
    inferer = inferer.append_simulations(theta_all, x_all)
    likelihood_estimator = inferer.train()
    posterior = inferer.build_posterior(mcmc_method="slice_np_vectorized", 
                                        mcmc_parameters=dict(thin=1))



if inference == 'SNRE':
    inferer = SNRE(prior)
    inferer = inferer.append_simulations(theta_all, x_all)
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
    print(ji)
    isim_obs = isim_obs_array[ji]

    # saved_j = pk.load(open(ldir_stats + '/saved_stats_halos_' + str(isim_obs) + '_CMASS.pk', 'rb'))
    # x_obs = saved_j['summary_concat_truth_all'][None,:]
        
    # saved_j = pk.load(open(ldir_stats + '/saved_stats_halos/_with_MASS_WEIGHTED_' + str(isim_obs) + '_CMASS.pk', 'rb'))    
    saved_j = pk.load(open(ldir_stats + '/summary_stats_weighted_rsd_' + str(isim_obs) + '_nbar_4en4.pk', 'rb'))    

    # Pk_mock_ds = saved_j[pos + '_Pk_mock_weighted'][:,:1].T
    # Pk_truth_ds = saved_j[pos + '_Pk_truth_weighted'][:,:1].T
    if pk_type == 'mono':
        Pk_mock_ds = saved_j[pos + '_Pk_mock_weighted'][:,:1].T
        Pk_truth_ds = saved_j[pos + '_Pk_truth_weighted'][:,:1].T
    elif pk_type == 'quad':
        Pk_mock_ds = saved_j[pos + '_Pk_mock_weighted'][:,:2].T
        Pk_truth_ds = saved_j[pos + '_Pk_truth_weighted'][:,:2].T
    elif pk_type == 'all':
        Pk_mock_ds = saved_j[pos + '_Pk_mock_weighted'][:,:3].T
        Pk_truth_ds = saved_j[pos + '_Pk_truth_weighted'][:,:3].T


    Bk_truth_k0p06 = saved_j[pos + '_Bk_truth_k0p06_weighted'][1:-1]
    Bk_mock_k0p06 = saved_j[pos + '_Bk_mock_k0p06_weighted'][1:-1]

    Bk_truth_k0p2 = saved_j[pos + '_Bk_truth_k0p2_weighted'][1:-1]
    Bk_mock_k0p2 = saved_j[pos + '_Bk_mock_k0p2_weighted'][1:-1]

    Bk_truth_k0p3 = saved_j[pos + '_Bk_truth_k0p3_weighted'][1:-1]
    Bk_mock_k0p3 = saved_j[pos + '_Bk_mock_k0p3_weighted'][1:-1]

    s0_mock = saved_j[pos + '_s0_mock_weighted']
    s0_truth = saved_j[pos + 's0_truth_weighted']

    s1_mock = saved_j[pos + '_s1_mock_weighted']
    s1_truth = saved_j[pos + 's1_truth_weighted']

    s2_mock = saved_j[pos + '_s2_mock_weighted']
    s2_truth = saved_j[pos + 's2_truth_weighted']  

    # x_obs = saved_j[pos + '_summary_concat_truth_all_weighted'][None,:]            
    # x_obs = saved_j[pos + '_summary_concat_mock_all_weighted'][None,:]                

    x_obs = np.concatenate((Pk_truth_ds.flatten(), Bk_truth_k0p06, Bk_truth_k0p2, Bk_truth_k0p3, s1_truth[::4], s2_truth[::6]))
    # x_obs = np.concatenate((Pk_truth_ds.flatten(), s0_truth, s1_truth[::4]))    
    # x_obs = Pk_truth_ds.flatten()


    # x_obs = saved_j[pos + '_Pk_truth_weighted'].flatten()[None,:]                
    theta_obs = saved_j['cosmo'][None,indsel_p]

    x_obs = torch.tensor((x_obs - x_all_mean)/x_all_std).float()
    # x_obs = torch.tensor((x_obs)/x_all_std).float()    
    theta_obs = torch.tensor((theta_obs  - theta_all_mean)/theta_all_std).float()

    samples = posterior.set_default_x(x_obs).sample((1000,))

    samples_all = samples.cpu().numpy()

    samples_all_isims[ji] = samples_all
    true_all_isims[ji] = theta_obs.cpu().numpy()

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
pk.dump(saved, open(f'/mnt/home/spandey/ceph/CHARM/results/saved_nsubv_vel10k_try2_tarp_{pos}_Pk_{pk_type}_{inference}_data_{data}.pk', 'wb'))

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


sdirf = '/mnt/home/spandey/ceph/CHARM/results/'

fig, ax = plt.subplots(1, 1, figsize=(5, 4))
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
pl.savefig(sdirf + f'TARP_nsubv_vel10k_try2_coverage_{pos}_Pk_{pk_type}_{inference}_data_{data}.pdf', bbox_inches='tight')

# samples_all_tarp.shape, true_all_tarp.shape
# true_all_tarp_repeat = np.tile(true_all_tarp, (1000, 1, 1))



pl.figure()
pl.errorbar(true_all_transformed[:,0], samples_all_tarp_mean[:,0], yerr = samples_all_tarp_std[:,0], fmt = 'o')
pl.plot([0.1, 0.5], [0.1, 0.5], 'k--')
pl.tick_params(axis='both', which='major', labelsize=15)
pl.tick_params(axis='both', which='minor', labelsize=15)
pl.xlabel('Om$ true', size=15)
pl.ylabel('Om inferred', size=15)
pl.savefig(sdirf + f'Om_nsubv_vel10k_try2_inferred_{pos}_Pk_{pk_type}_{inference}_data_{data}.pdf', bbox_inches='tight')

pl.figure()
pl.errorbar(true_all_transformed[:,-1], samples_all_tarp_mean[:,-1], yerr = samples_all_tarp_std[:,-1], fmt = 'o')
pl.plot([0.55, 1.05], [0.55, 1.05], 'k--')
pl.xlabel('sigma8 true', size=15)
pl.ylabel('sigma8 inferred', size=15)
pl.savefig(sdirf + f'sig8_nsubv_vel10k_try2_inferred_{pos}_Pk_{pk_type}_{inference}_data_{data}.pdf', bbox_inches='tight')


