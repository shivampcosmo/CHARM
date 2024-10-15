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
from utils_data_prep_cosmo_vel_conc_peak import *
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

run_config_name = 'TRAIN_CONC_FREECOSMO_cond_fastpm_ns128_200c.yaml'
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
cmin = config_sims['cmin']
cmax = config_sims['cmax']
stype = config_sims['stype']
rescale_sub = config_sims['rescale_sub']
lgMmincutstr = config_sims['lgMmincutstr']
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

K_conc = config_net['K_conc']
B_conc = config_net['B_conc']
nflows_conc_NSF = config_net['nflows_conc_NSF']

# base_dist_Ntot = config_net['base_dist_Ntot']
# if base_dist_Ntot == 'None':
#     base_dist_Ntot = None
# base_dist_M1 = config_net['base_dist_M1']
base_dist_conc = config_net['base_dist_conc']

cond_Mass_for_conc = config_net['cond_Mass_for_conc']

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
    df = pk.load(open(abs_path_data + '/' + 'HALO_MASS_rockstar_200c_VEL_NU_CONC_varycosmo_subsel_random_nsims1800_nspji16_nfid512_train_data_QUIJOTE.pk', 'rb'))    

    df_d_all_train = df['df_d_all_train']
    df_d_all_nsh_train = df['df_d_all_nsh_train']
    df_Mh_all_train = df['df_Mh_all_train'].todense()
    df_Nh_train = df['df_Nh_train'].todense()
    # df_vh_train = df['df_vh_train'].todense()
    # df_nuh_train = df['df_nuh_train'].todense()
    df_ch_train = df['df_ch_train'].todense()
    ind_subsel_all_train = df['ind_subsel_all_train']
    ind_subsel_fid_train = df['ind_subsel_fid_train']
    cosmo_val_all_train = df['cosmo_val_all_train']

    import pickle as pk
    df = pk.load(open(abs_path_data + '/' + 'DENSITY_varycosmo_subsel_random_nsims1800_nspji16_nfid512_train_data_FASTPM.pk', 'rb'))
    df_d_all_train_FP = df['df_d_all_train']
    df_d_all_nsh_train_FP = df['df_d_all_nsh_train']
    df_Mh_all_train_FP = df['df_Mh_all_train']
    df_Nh_train_FP = df['df_Nh_train']
    try:
        df_vh_train_FP = df['df_vh_train']
    except:
        df_vh_train_FP = None
        pass
    ind_subsel_all_train_FP = df['ind_subsel_all_train']
    ind_subsel_fid_train_FP = df['ind_subsel_fid_train']
    cosmo_val_all_train_FP = df['cosmo_val_all_train']


    # nsims_per_batch, nbatches_train = 250, 4
    return_dict_train = prep_density_halo_cats_batched(
        df_d_all_train, df_d_all_nsh_train, df_Mh_all_train, df_Nh_train,cosmo_val_all_train, df_v_inp=None, df_c_inp=df_ch_train, nsims=nsims_per_batch,
        nbatches = nbatches_train, Mmin=lgMmin, Mmax=lgMmax, rescaleM_sub=rescale_sub, Nmax=Nmax, get_density=False, get_halos=True, cmin=cmin, cmax=cmax
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
    ndim_conc = Nmax

    c_halos_all = return_dict_train['c_halos_all_sort_norm'].reshape(*return_dict_train['c_halos_all_sort_norm'].shape[:-2],-1)
    M_halos_all = return_dict_train['M_halos_all_sort_norm'].reshape(*return_dict_train['M_halos_all_sort_norm'].shape[:-2],-1)

    mask_conc = return_dict_train['mask_vel']
    mask_tensor_conc_train = torch.Tensor((mask_conc)).reshape(-1, nsims_per_batch * (nax_h**3), ndim_conc)

    X_conc = torch.Tensor(np.array(c_halos_all)).reshape(-1, nsims_per_batch * (nax_h**3),ndim_conc)
    Nhalos_truth_tensor = torch.Tensor(((np.array(return_dict_train['N_halos_all'])))).reshape(-1, nsims_per_batch * (nax_h**3), 1)
    Mhalos_truth_tensor = torch.Tensor(((np.array(M_halos_all)))).reshape(-1, nsims_per_batch * (nax_h**3), ndim_mass)

    print(X_conc.shape, Nhalos_truth_tensor.shape, Mhalos_truth_tensor.shape, mask_tensor_conc_train.shape, cond_tensor.shape, cond_tensor_nsh.shape, cond_cosmo.shape)
else:
    return_dict_train = None
    return_dict_train_FP = None    


# config_train = config['train_settings']
# batch_size = config_train['batch_size_DL']
# all_gpu = config_train['all_gpu']

# try:
#     L2norm_Ntothist = config_train['L2norm_Ntothist']
# except:
#     L2norm_Ntothist = False

# try:
#     L2norm_M1hist = config_train['L2norm_M1hist']
# except:
#     L2norm_M1hist = False

# nflows_train = config_train['nflows_train']

# save_string = config_train['save_string']

# sdir_model_checkpoint = abs_path_checkpoint + \
#                         '/MULT_GPU_VEL_nsims_' + \
#                             str(len(ji_array)) + \
#                             '_mass_' + mass_type + \
#                              '_Nmax' + str(Nmax) + save_string


from dataclasses import dataclass
# if __name__ == '__main__':
    # hyperparameters
    # batch_size = 16 # how many independent sequences will we process in parallel?
    # block_size = 32 # what is the maximum context length for predictions?
# max_iters = 8000
eval_interval = 100
learning_rate = 2e-3
sdir_model_checkpoint = abs_path_checkpoint + '/test2_conc/'
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

print('Initial GPU memory usage, before training starts:')
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}:")
    print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
    print(f"  Reserved:  {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")



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
    if cond_Mass_for_conc:
        Mhalos_truth_tensor_jb = Mhalos_truth_tensor[start:end,...].to(device_id, non_blocking=True)
    else:
        Mhalos_truth_tensor_jb = None
    mask_tensor_conc_train_jb = mask_tensor_conc_train[start:end,...].to(device_id, non_blocking=True)

    X_conc_jb = X_conc[start:end,...].to(device_id, non_blocking=True)


    # num_cond_conc = num_cond
    if cond_Mass_for_conc:
        num_cond_conc = num_cond + ndim_mass
    else:
        num_cond_conc = num_cond
    model_conc = NSF_Autoreg_CNNcond(
        dim=ndim_conc,
        K=K_conc,
        B=B_conc,
        hidden_dim=hidden_dim_MAF,
        num_cond=num_cond_conc,
        nflows=nflows_conc_NSF,
        base_dist=base_dist_conc,
        mu_pos=True
        )


    # ndim = ndim_diff + 1
    model = COMBINED_Model_conc_only(
        None,
        model_conc,
        ndim_conc,
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

    model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    checkpoint = torch.load(sdir_model_checkpoint + f'test_model_conc_bestfit_2560.pth', map_location=f'cuda:{device_id}')            
    model.load_state_dict(checkpoint['state_dict'])


    decay_lr = True # whether to decay the learning rate
    decay_lr_model = 'cosine'
    min_lr = 1e-4 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    iter_num = 0
    local_iter_num = 0 # number of iterations in the lifetime of this process
    # raw_model = model.module
    running_mfu = -1.0    
    best_val_loss = 1e20
    loss_min = 1e20
    # nbatches = 10
    # max_iters = 1800
    eval_interval = 20
    save_separate_interval = 20

    nepochs = 4000
    warmup_iters = int(nepochs//15) # how many steps to warm up for
    lr_decay_iters = int(nepochs*1.5) # should be ~= max_iters per Chinchilla

    # sdir_model_checkpoint = '/mnt/home/spandey/ceph/CHARM/model_checkpoints/temp/'

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

    for iter_num in (range(nepochs)):
        lr = get_lr(iter_num, model=decay_lr_model) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if (rank == 0) and (iter_num == 0):
            print('GPU memory usage after calling model forward once:')
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}:")
                print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
                print(f"  Reserved:  {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")

        loss = model(
            X_conc_jb,
            cond_tensor_jb,
            cond_tensor_nsh_jb,
            cond_cosmo_jb,
            Nhalos_truth_tensor_jb,
            Mhalos_truth_tensor_jb,
            mask_tensor_conc_train_jb         
            )            

        if (rank == 0) and (iter_num == 0):
            print('GPU memory usage after calling model forward once:')
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}:")
                print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
                print(f"  Reserved:  {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")

        scaler.scale(loss).backward()   
        if (iter_num % 10) == 0 and (rank == 0):
            print(f"iter {iter_num}, loss: {loss.item()}")                          


        if (np.mod(iter_num, int(nepochs / 50)) == 0) or (iter_num == nepochs - 1):
            if float(loss.mean().cpu().detach().numpy()) < loss_min:
                loss_min = float(loss.mean().cpu().detach().numpy())
                print(loss_min)

                state = {'loss_min': loss_min, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'loss':loss}

                save_bestfit_model_name = sdir_model_checkpoint + 'test_model_conc_bestfit_' + str(2560 + iter_num) + '.pth'
                torch.save(
                    state, save_bestfit_model_name
                    )

        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        if (rank == 0) and (iter_num == 0):
            print('GPU memory usage after passing gradients backward once:')
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}:")
                print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
                print(f"  Reserved:  {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")



    dist.destroy_process_group()

if __name__ == "__main__":
    run_func()
