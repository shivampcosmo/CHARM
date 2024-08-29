import sys, os
import numpy as np
import torch
dev = torch.device("cuda")
import torch.optim as optim
root_dir = '/mnt/home/spandey/ceph/CHARM/'
os.chdir(root_dir)
# import colossus
import sys, os
# append the root_dir to the path
sys.path.append(root_dir)
from charm.utils_data_prep_cosmo_vel import *
from colossus.cosmology import cosmology
params = {'flat': True, 'H0': 67.11, 'Om0': 0.3175, 'Ob0': 0.049, 'sigma8': 0.834, 'ns': 0.9624}
cosmo = cosmology.setCosmology('myCosmo', **params)
# get halo mass function:
from colossus.lss import mass_function
from tqdm import tqdm
    
import yaml
import pickle as pk
# autoreload modules
import matplotlib
import matplotlib.pyplot as pl


# run_config_name = 'TRAIN_MASS_FREECOSMO_cond_fastpm_ns128_lresdata.yaml'
run_config_name = 'TRAIN_MASS_FREECOSMO_cond_fastpm_ns128_fof.yaml'
with open("/mnt/home/spandey/ceph/CHARM/run_configs/" + run_config_name,"r") as file_object:
    config=yaml.load(file_object,Loader=yaml.SafeLoader)


config_sims = config['sim_settings']
ji_array = np.arange(int(config_sims['nsims']))
nsubvol_per_ji = int(config_sims['nsubvol_per_ji'])
nsubvol_fid = int(config_sims['nsubvol_fid'])
subsel_criteria = config_sims['subsel_criteria']
num_cosmo_params = int(config_sims['num_cosmo_params'])
ns_d = config_sims['ns_d']
nb = config_sims['nb']
nax_d =  ns_d // nb
nf = config_sims['nf']
layers_types = config_sims['layers_types']
z_inference = config_sims['z_inference']
nc = 0
for jl in range(len(layers_types)):
    if layers_types[jl] == 'cnn':
        nc += 1
    elif layers_types[jl] == 'res':
        nc += 2
    else:
        raise ValueError("layer type not supported")

z_all = config_sims['z_all']
z_all_FP = config_sims['z_all_FP']
ns_h = config_sims['ns_h']
nax_h = ns_h // nb
cond_sim = config_sims['cond_sim']

nsims_per_batch = config_sims['nsims_per_batch']
nbatches_train = config_sims['nbatches_train']

mass_type = config_sims['mass_type']
lgMmin = config_sims['lgMmin']
lgMmax = config_sims['lgMmax']
stype = config_sims['stype']
rescale_sub = config_sims['rescale_sub']
lgMmincutstr = config_sims['lgMmincutstr']
# subsel_highM1 = config_sims['subsel_highM1']
# nsubsel = config_sims['nsubsel']
is_HR = config_sims['is_HR']

try:
    Nmax = config_sims['Nmax']
except:
    Nmax = 4

config_net = config['network_settings']
hidden_dim_MAF = config_net['hidden_dim_MAF']
learning_rate = config_net['learning_rate']
K_M1 = config_net['K_M1']
B_M1 = config_net['B_M1']
nflows_M1_NSF = config_net['nflows_M1_NSF']

K_Mdiff = config_net['K_Mdiff']
B_Mdiff = config_net['B_Mdiff']
nflows_Mdiff_NSF = config_net['nflows_Mdiff_NSF']

base_dist_Ntot = config_net['base_dist_Ntot']
if base_dist_Ntot == 'None':
    base_dist_Ntot = None
base_dist_M1 = config_net['base_dist_M1']
base_dist_Mdiff = config_net['base_dist_Mdiff']
ngauss_M1 = config_net['ngauss_M1']

changelr = config_net['changelr']
ksize = nf
nfeature_cnn = config_net['nfeature_cnn']
nout_cnn = 4 * nfeature_cnn
if cond_sim == 'fastpm':
    ninp = len(z_all_FP) + 2
    # if 'v' in z_all_FP
elif cond_sim == 'quijote':
    ninp = len(z_all)
else:
    raise ValueError("cond_sim not supported")



num_cond = nout_cnn + ninp + num_cosmo_params

import pickle as pk
import numpy as np

df_d_all_train, df_d_all_nsh_train, df_Mh_all_train, df_Nh_train, df_vh_train, ind_subsel_all_train, ind_subsel_fid_train, cosmo_val_all_train = load_density_halo_data_NGP(
    ji_array, ns_d, nb, nf, nc, z_all, ns_h,z_inference=z_inference,nsubvol_per_ji=nsubvol_per_ji,nsubvol_fid=nsubvol_fid,
    sdir_cosmo='/mnt/home/spandey/ceph/Quijote/data_NGP_self_LH',
    sdir_fid='/mnt/home/spandey/ceph/Quijote/data_NGP_self',  
    LH_cosmo_val_file='/mnt/home/spandey/ceph/Quijote/latin_hypercube_params.txt',  
    stype=stype, mass_type=mass_type, lgMmincutstr = lgMmincutstr, subsel_criteria=subsel_criteria, is_HR = is_HR,  vel_type='diff',
    get_density=False,
    get_halos=True
    )

import sparse
saved = {'df_d_all_train':df_d_all_train,
            'df_d_all_nsh_train':df_d_all_nsh_train,
            'df_Mh_all_train':sparse.COO(df_Mh_all_train),
            'df_Nh_train':sparse.COO(df_Nh_train),
            'df_vh_train':sparse.COO(df_vh_train),
            'ind_subsel_all_train':ind_subsel_all_train,
            'ind_subsel_fid_train':ind_subsel_fid_train,
            'cosmo_val_all_train':cosmo_val_all_train
            }

import pickle as pk
nsims = int(config_sims['nsims'])
pk.dump(saved, open('/mnt/home/spandey/ceph/CHARM/data/' + f'HALO_MASS_{mass_type}_VEL_varycosmo_subsel_random_nsims{nsims}_nspji{nsubvol_per_ji}_nfid{nsubvol_fid}' + '_train_data_QUIJOTE.pk', 'wb'))

# import pickle as pk
# nsims = int(config_sims['nsims'])
# df = pk.load(open('/mnt/home/spandey/ceph/CHARM/data/' + f'HALO_MASS_VEL_varycosmo_subsel_random_nsims{nsims}_nspji{nsubvol_per_ji}_nfid{nsubvol_fid}' + '_train_data_QUIJOTE.pk', 'rb'))
# cosmo_val_all_train = df['cosmo_val_all_train']
# ind_subsel_all_train = df['ind_subsel_all_train']
# ind_subsel_fid_train = df['ind_subsel_fid_train']


# df_d_all_train_FP, df_d_all_nsh_train_FP, df_Mh_all_train_FP, df_Nh_train_FP, df_vh_train_FP, ind_subsel_all_train_FP, ind_subsel_fid_train_FP, cosmo_val_all_train_FP = load_density_halo_data_NGP(
#     ji_array, ns_d, nb, nf, nc, z_all_FP, ns_h,z_inference=z_inference,nsubvol_per_ji=nsubvol_per_ji,nsubvol_fid=nsubvol_fid,
#     sdir_cosmo='/mnt/home/spandey/ceph/Quijote/data_NGP_self_fastpm_LH',
#     sdir_fid='/mnt/home/spandey/ceph/Quijote/data_NGP_self/fastpm',  
#     LH_cosmo_val_file='/mnt/home/spandey/ceph/Quijote/latin_hypercube_params.txt',  
#     stype=stype, mass_type=mass_type, lgMmincutstr = lgMmincutstr, subsel_criteria=subsel_criteria,
#     indsubsel_all_inp=ind_subsel_all_train,
#     indsubsel_fid_inp=ind_subsel_fid_train,
#     is_HR = is_HR,  vel_type='diff',
#     get_density=True,
#     get_halos=False
#     )
# if cosmo_val_all_train_FP is None:
#     cosmo_val_all_train_FP = cosmo_val_all_train



# saved = {'df_d_all_train':df_d_all_train_FP,
#             'df_d_all_nsh_train':df_d_all_nsh_train_FP,
#             'df_Mh_all_train':df_Mh_all_train_FP,
#             'df_Nh_train':df_Nh_train_FP,
#             'df_vh_train_FP':df_vh_train_FP,
#             'ind_subsel_all_train':ind_subsel_all_train_FP,
#             'ind_subsel_fid_train':ind_subsel_fid_train_FP,
#             'cosmo_val_all_train':cosmo_val_all_train_FP
#             }

# import pickle as pk
# nsims = int(config_sims['nsims'])
# pk.dump(saved, open('/mnt/home/spandey/ceph/CHARM/data/' + f'DENSITY_varycosmo_subsel_random_nsims{nsims}_nspji{nsubvol_per_ji}_nfid{nsubvol_fid}' + '_train_data_FASTPM.pk', 'wb'))



