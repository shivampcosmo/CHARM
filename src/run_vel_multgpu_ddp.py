import sys, os
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
import time
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
os.environ['NCCL_BLOCKING_WAIT'] = '0'
from datetime import timedelta
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.distributed as dist
import torch.optim as optim
import pickle as pk
from combined_models import *
from all_models import *
from utils_data_prep_cosmo_vel import *
from colossus.cosmology import cosmology
params = {'flat': True, 'H0': 67.11, 'Om0': 0.3175, 'Ob0': 0.049, 'sigma8': 0.834, 'ns': 0.9624}
cosmo = cosmology.setCosmology('myCosmo', **params)
from colossus.lss import mass_function
from tqdm import tqdm
import sparse
import numpy as np
import h5py as h5
# get directory of this file, absolute path:
import pathlib
curr_path = pathlib.Path().absolute()
# get absolute path of run configs which is one level up:
abs_path_config = os.path.abspath(curr_path / "../run_configs/") 
abs_path_data = os.path.abspath(curr_path / "../data/") 
abs_path_checkpoint = os.path.abspath(curr_path / "../model_checkpoints/") 


def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def get_data_split(dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed, delta_box_all_squeezed, n1_fac=0.8, n2_fac=1.0):
    n1 = int(n1_fac*len(dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed))
    n2 = int(n2_fac*len(dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed))
    train_data_halos = dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed[:n1]
    val_data_halos = dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed[n1:n2]

    train_data_dm = delta_box_all_squeezed[:n1]
    val_data_dm = delta_box_all_squeezed[n1:n2]

    x = torch.tensor(train_data_halos[:, :-1])
    y = torch.tensor(train_data_halos[:, 1:])
    dm = torch.tensor(train_data_dm)
    mask_train_orig = x != 1
    mask_train = torch.logical_not(mask_train_orig)
    masked_logits = torch.zeros(mask_train.shape)
    mask_train_final = masked_logits.masked_fill(mask_train, float('-inf'))
    mask_train = mask_train_final[:,None,:]
    x, y = torch.tensor(x), torch.tensor(y)
    x_train = x.long()
    y_train = y.long()
    dm_train = dm.bfloat16()
    mask_train = torch.tensor(mask_train).bfloat16()

    x = torch.tensor(val_data_halos[:, :-1])
    y = torch.tensor(val_data_halos[:, 1:])
    dm = torch.tensor(val_data_dm)
    mask_val_orig = x != 1
    mask_val = torch.logical_not(mask_val_orig)
    masked_logits = torch.zeros(mask_val.shape)
    mask_val_final = masked_logits.masked_fill(mask_val, float('-inf'))
    mask_val = mask_val_final[:,None,:]
    x, y = torch.tensor(x), torch.tensor(y)
    x_val = x.long()
    y_val = y.long()
    dm_val = dm.bfloat16()
    mask_val = torch.tensor(mask_val).bfloat16()

    return x_train, y_train, dm_train, mask_train, x_val, y_val, dm_val, mask_val

run_config_name = 'TRAIN_VEL_FREECOSMO_cond_fastpm_ns128.yaml'
with open(abs_path_config + '/' + run_config_name,"r") as file_object:
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
# K_M1 = config_net['K_M1']
# B_M1 = config_net['B_M1']
# nflows_M1_NSF = config_net['nflows_M1_NSF']

K_vel = config_net['K_vel']
B_vel = config_net['B_vel']
nflows_vel_NSF = config_net['nflows_vel_NSF']

# base_dist_Ntot = config_net['base_dist_Ntot']
# if base_dist_Ntot == 'None':
#     base_dist_Ntot = None
# base_dist_M1 = config_net['base_dist_M1']
base_dist_vel = config_net['base_dist_vel']

cond_Mass_for_vel = config_net['cond_Mass_for_vel']

# ngauss_M1 = config_net['ngauss_M1']

changelr = config_net['changelr']
ksize = nf
nfeature_cnn = config_net['nfeature_cnn']
nout_cnn = 4 * nfeature_cnn
if cond_sim == 'fastpm':
    if any('v' in str(string) for string in z_all_FP):
        ninp = len(z_all_FP) + 2
    else:
        ninp = len(z_all_FP)

elif cond_sim == 'quijote':
    ninp = len(z_all)
else:
    raise ValueError("cond_sim not supported")

num_cond = nout_cnn + ninp + num_cosmo_params

if __name__ == "__main__":

    import pickle as pk
    # df = pk.load(open('/mnt/home/spandey/ceph/CHARM/data/HALO_MASS_VEL_varycosmo_subsel_random_nsims1800_nspji16_nfid512_train_data_QUIJOTE.pk', 'rb'))
    df = pk.load(open(abs_path_data + '/' + 'HALO_MASS_test7328_QUIJOTE_test.pk', 'rb'))

    df_d_all_train = df['df_d_all_train']
    df_d_all_nsh_train = df['df_d_all_nsh_train']
    df_Mh_all_train = df['df_Mh_all_train'].todense()
    df_Nh_train = df['df_Nh_train'].todense()
    df_vh_train = df['df_vh_train'].todense()
    ind_subsel_all_train = df['ind_subsel_all_train']
    ind_subsel_fid_train = df['ind_subsel_fid_train']
    cosmo_val_all_train = df['cosmo_val_all_train']


    import pickle as pk
    # df = pk.load(open('/mnt/home/spandey/ceph/CHARM/data/DENSITY_varycosmo_subsel_random_nsims1800_nspji16_nfid512_train_data_FASTPM.pk', 'rb'))
    df = pk.load(open(abs_path_data + '/' + 'DENSITY_test7328_FASTPM_test.pk', 'rb'))
    df_d_all_train_FP = df['df_d_all_train']
    df_d_all_nsh_train_FP = df['df_d_all_nsh_train']
    df_Mh_all_train_FP = df['df_Mh_all_train']
    df_Nh_train_FP = df['df_Nh_train']
    try:
        df_vh_train_FP = df['df_vh_train']
    except:
        pass
    ind_subsel_all_train_FP = df['ind_subsel_all_train']
    ind_subsel_fid_train_FP = df['ind_subsel_fid_train']
    cosmo_val_all_train_FP = df['cosmo_val_all_train']

    # nsims_per_batch, nbatches_train = 250, 4
    return_dict_train = prep_density_halo_cats_batched(
        df_d_all_train, df_d_all_nsh_train, df_Mh_all_train, df_Nh_train,cosmo_val_all_train, df_v_inp=df_vh_train, nsims=nsims_per_batch,
        nbatches = nbatches_train, Mmin=lgMmin, Mmax=lgMmax, rescaleM_sub=rescale_sub, Nmax=Nmax, get_density=False, get_halos=True
        )


    # # Prepare the density and halo data
    return_dict_train_FP = prep_density_halo_cats_batched(
        df_d_all_train_FP, df_d_all_nsh_train_FP, df_Mh_all_train_FP, df_Nh_train_FP, cosmo_val_all_train_FP, df_v_inp=df_vh_train_FP, nsims=nsims_per_batch,
        nbatches = nbatches_train, Mmin=lgMmin, Mmax=lgMmax, rescaleM_sub=rescale_sub, get_density=True, get_halos=False
        )
    
    if return_dict_train_FP is not None:
        cond_tensor = torch.Tensor(np.array(return_dict_train_FP['df_d_all']))
        cond_nsh = np.moveaxis(np.array(return_dict_train_FP['df_d_all_nsh']), 2, 5)
        cond_tensor_nsh = torch.Tensor((cond_nsh)).reshape(-1, nsims_per_batch * (nax_h ** 3), ninp)
        
        cond_cosmo = torch.Tensor(np.array(return_dict_train_FP['cosmo_val_all']))
        cond_cosmo = cond_cosmo.reshape(-1, nsims_per_batch * (nax_h**3), cond_cosmo.shape[-1])
    else:
        cond_tensor = torch.Tensor(np.array(return_dict_train['df_d_all']))
        cond_nsh = np.moveaxis(np.array(return_dict_train['df_d_all_nsh']), 2, 5)
        cond_tensor_nsh = torch.Tensor((cond_nsh)).reshape(-1, nsims_per_batch * (nax_h ** 3), ninp)

        cond_cosmo = torch.Tensor(np.array(return_dict_train['cosmo_val_all']))
        cond_cosmo = cond_cosmo.reshape(-1, nsims_per_batch * (nax_h**3), cond_cosmo.shape[-1])


    ndim_mass = Nmax
    ndim_vel = 3*Nmax

    v_halos_all = return_dict_train['v_halos_all_sort_norm'].reshape(*return_dict_train['v_halos_all_sort_norm'].shape[:-2],-1)
    M_halos_all = return_dict_train['M_halos_all_sort_norm'].reshape(*return_dict_train['M_halos_all_sort_norm'].shape[:-2],-1)

    mask_vel = return_dict_train['mask_vel']
    mask_vel_repeat = np.repeat(mask_vel[..., None], 3, axis=-1)
    mask_vel_repeat = mask_vel_repeat.reshape(*mask_vel_repeat.shape[:-2],-1)
    mask_tensor_vel_train = torch.Tensor((mask_vel_repeat)).reshape(-1, nsims_per_batch * (nax_h**3), ndim_vel)


    X_vel = torch.Tensor(np.array(v_halos_all)).reshape(-1, nsims_per_batch * (nax_h**3),ndim_vel)
    Nhalos_truth_tensor = torch.Tensor(((np.array(return_dict_train['N_halos_all'])))).reshape(-1, nsims_per_batch * (nax_h**3), 1)
    Mhalos_truth_tensor = torch.Tensor(((np.array(M_halos_all)))).reshape(-1, nsims_per_batch * (nax_h**3), ndim_mass)

else:
    return_dict_train = None
    return_dict_train_FP = None

# ndim_diff = Nmax - 1

lgM_array = np.linspace(lgMmin, lgMmax, 50)
M_array = 10**lgM_array
if '200c' in mass_type:
    hmf = mass_function.massFunction(M_array, float(z_inference), mdef = '200c', model = 'tinker08', q_out = 'dndlnM')
if 'vir' in mass_type:
    hmf = mass_function.massFunction(M_array, float(z_inference), mdef = 'vir', model = 'tinker08', q_out = 'dndlnM')    
if 'fof' in mass_type:
    hmf = mass_function.massFunction(M_array, float(z_inference), mdef = 'fof', model = 'bhattacharya11', q_out = 'dndlnM')
lgM_rescaled = rescale_sub + (lgM_array - lgMmin)/(lgMmax-lgMmin)

int_val = sp.integrate.simps(hmf, lgM_rescaled)
hmf_pdf = hmf/int_val
# define the cdf of the halo mass function
hmf_cdf = np.zeros_like(hmf_pdf)
for i in range(len(hmf_cdf)):
    hmf_cdf[i] = sp.integrate.simps(hmf_pdf[:i+1], lgM_rescaled[:i+1])

config_train = config['train_settings']
batch_size = config_train['batch_size_DL']
all_gpu = config_train['all_gpu']

try:
    L2norm_Ntothist = config_train['L2norm_Ntothist']
except:
    L2norm_Ntothist = False

try:
    L2norm_M1hist = config_train['L2norm_M1hist']
except:
    L2norm_M1hist = False

nflows_train = config_train['nflows_train']

save_string = config_train['save_string']

sdir_model_checkpoint = abs_path_checkpoint + \
                        '/MULT_GPU_VEL_nsims_' + \
                            str(len(ji_array)) + \
                            '_mass_' + mass_type + \
                             '_Nmax' + str(Nmax) + save_string



# print(save_bestfit_model_dir, os.path.exists(save_bestfit_model_dir))
# # make directory if it doesn't exist
# import os
# if not os.path.exists(save_bestfit_model_dir):
#     os.makedirs(save_bestfit_model_dir)


nepochs_Ntot_only = config_train['nepochs_Ntot_only']
nepochs_Ntot_M1_only = config_train['nepochs_Ntot_M1_only']
nepochs_all = config_train['nepochs_all']


from dataclasses import dataclass
# if __name__ == '__main__':
    # hyperparameters
    # batch_size = 16 # how many independent sequences will we process in parallel?
    # block_size = 32 # what is the maximum context length for predictions?
# max_iters = 8000
eval_interval = 10
learning_rate = 2e-3
sdir_model_checkpoint = abs_path_checkpoint + '/test2_vel/'
# make directory if does not exist:
try:
    if not os.path.exists(sdir_model_checkpoint):
        os.makedirs(sdir_model_checkpoint)
except:
    pass
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
dropout = 0.2

batch_size = 2048

def run_func():

    device = 'cuda'
    compile = True # use PyTorch 2.0 to compile the model to be faster
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda'
    dtype = 'bfloat16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    dist.init_process_group("nccl", timeout=timedelta(seconds=7200000))
    rank = dist.get_rank()
    # print(f"Start running basic DDP example on rank {rank}.")

    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()

    dtype = 'bfloat16'

    len_batches = cond_tensor.shape[0]
    start = rank * (len_batches // torch.cuda.device_count())
    end = start + (len_batches // torch.cuda.device_count())
    
    cond_tensor_jb = cond_tensor[start:end,...].to(device_id, non_blocking=True)
    cond_tensor_nsh_jb = cond_tensor_nsh[start:end,...].to(device_id, non_blocking=True)
    cond_cosmo_jb = cond_cosmo[start:end,...].to(device_id, non_blocking=True)
    Nhalos_truth_tensor_jb = Nhalos_truth_tensor[start:end,...].to(device_id, non_blocking=True)
    if cond_Mass_for_vel:
        Mhalos_truth_tensor_jb = Mhalos_truth_tensor[start:end,...].to(device_id, non_blocking=True)
    else:
        Mhalos_truth_tensor_jb = None
    mask_tensor_vel_train_jb = mask_tensor_vel_train[start:end,...].to(device_id, non_blocking=True)

    X_vel_jb = X_vel[start:end,...].to(device_id, non_blocking=True)
    
    print(f"I am rank {rank} and will process train data from {start} to {end}.")
    if rank == 0: print(f"Transferred train data to GPU", flush=True)        
      
    # num_cond_Ntot = num_cond

    if cond_Mass_for_vel:
        num_cond_vel = num_cond + ndim_mass
    else:
        num_cond_vel = num_cond
    model_vel = NSF_Autoreg_CNNcond(
        dim=ndim_vel,
        K=K_vel,
        B=B_vel,
        hidden_dim=hidden_dim_MAF,
        num_cond=num_cond_vel,
        nflows=nflows_vel_NSF,
        base_dist=base_dist_vel,
        mu_pos=False
        )


    # ndim = ndim_diff + 1
    model = COMBINED_Model_vel_only(
        None,
        model_vel,
        ndim_vel,
        ksize,
        ns_d,
        ns_h,
        nb,
        ninp,
        nfeature_cnn,
        nout_cnn,
        layers_types=layers_types,
        act='tanh',
        padding='valid',
        ).to(device_id)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # try:
    #     checkpoint = torch.load(sdir_model_checkpoint + f'test_model_bestfit_650.pth', map_location=device_id)
    #     model.load_state_dict(checkpoint['state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    # except:
    #     pass

    model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    decay_lr = True # whether to decay the learning rate
    decay_lr_model = 'cosine'
    min_lr = 1e-4 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    iter_num = 0
    local_iter_num = 0 # number of iterations in the lifetime of this process
    # raw_model = model.module
    running_mfu = -1.0    
    best_val_loss = 1e20
    loss_min = 1e20
    nbatches = 10
    # max_iters = 1800
    eval_interval = 20
    save_separate_interval = 20

    nepochs = 8000
    warmup_iters = int(nepochs//15) # how many steps to warm up for
    lr_decay_iters = int(nepochs*1.5) # should be ~= max_iters per Chinchilla
    
    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it, model='cosine'):
        # 1) linear warmup for warmup_iters steps
        if model == 'cosine':
            if it < warmup_iters:
                return learning_rate * it / warmup_iters
            # 2) if it > lr_decay_iters, return min learning rate
            if it > lr_decay_iters:
                return min_lr
            # 3) in between, use cosine decay down to min learning rate
            decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
            return min_lr + coeff * (learning_rate - min_lr)
        
        elif model == 'linear':
            if it < warmup_iters:
                return learning_rate * it / warmup_iters
            else:
                return learning_rate - (it - warmup_iters) * (learning_rate - min_lr) / (lr_decay_iters - warmup_iters)

        elif model == 'constant':
            return learning_rate




    # t0 = time.time()
    for iter_num in (range(nepochs)):
        lr = get_lr(iter_num, model=decay_lr_model) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


        loss = model(
            X_vel_jb,
            cond_tensor_jb,
            cond_tensor_nsh_jb,
            cond_cosmo_jb,
            Nhalos_truth_tensor_jb,
            Mhalos_truth_tensor_jb,
            mask_tensor_vel_train_jb         
            )            

        scaler.scale(loss).backward()   
        if (iter_num % 10) == 0 and (rank == 0):
            print(f"iter {iter_num}, loss: {loss.item()}")                 


        if (np.mod(iter_num, int(nepochs / 80)) == 0) or (iter_num == nepochs - 1):
            if float(loss.mean().cpu().detach().numpy()) < loss_min:
                loss_min = float(loss.mean().cpu().detach().numpy())
                print(loss_min)

                state = {'loss_min': loss_min, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'loss':loss}

                save_bestfit_model_name = sdir_model_checkpoint + 'test_model_bestfit_' + str(iter_num) + '.pth'
                torch.save(
                    state, save_bestfit_model_name
                    )

        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)


    dist.destroy_process_group()

if __name__ == "__main__":
    run_func()