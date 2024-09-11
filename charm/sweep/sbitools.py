import numpy as np
import os
import pickle
import yaml
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from sbi.inference import SNPE, SNLE_A 
from sbi.neural_nets.embedding_nets import FCEmbedding
from sbi.utils.get_nn_models import posterior_nn, likelihood_nn
from sbi import utils
from collections import namedtuple
import torch.optim as optim
import yaml
from wandb import Api

api = Api()

class Objectify(object):
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

# def quijote_params():
#     params = np.load('/mnt/ceph/users/cmodi/Quijote/params_lh.npy')
#     params_fid = np.load('/mnt/ceph/users/cmodi/Quijote/params_fid.npy')
#     ndim = len(params_fid)
#     cosmonames = r'$\Omega_m$,$\Omega_b$,$h$,$n_s$,$\sigma_8$'.split(",")
#     return params, params_fid, cosmonames

###
def sbi_prior(params, offset=0.25, device='cpu', round=True):
    '''
    Generate priors for parameters of the simulation set with offset from min and max value
    '''
    if round:
        lower_bound, upper_bound = .1 * np.round(10 * params.min(0)) * (1-offset),\
                                      .1 * np.round(10 * params.max(0)) * (1+offset)
    else:
        lower_bound, upper_bound = params.min(axis=0) * (1-offset),\
                                      params.max(axis=0) * (1+offset)
    lower_bound, upper_bound = (torch.from_numpy(lower_bound.astype('float32')), 
                                torch.from_numpy(upper_bound.astype('float32')))
    prior = utils.BoxUniform(lower_bound, upper_bound, device=device)
    return prior

from typing import List
import torch
import torch.nn as nn
from typing import OrderedDict


class FCN(nn.Module):
    """Fully connected network to compress data.

    Args:
        n_hidden (List[int]): number of hidden units per layer
        act_fn (str):  activation function to use
    """

    def __init__(
        self, n_hidden: List[int], act_fn: str = "SiLU"
    ):
        super().__init__()
        self.act_fn = getattr(nn, act_fn)()
        self.n_layers = len(n_hidden)
        self.n_hidden = n_hidden

    def initalize_model(self, n_input: int):
        """Initialize network once the input dimensionality is known.

        Args:
            n_input (int): input dimensionality
        """
        model = []
        n_left = n_input
        for layer in range(self.n_layers):
            model.append((f"mlp{layer}", nn.Linear(
                n_left, self.n_hidden[layer])))
            model.append((f"act{layer}", self.act_fn))
            n_left = self.n_hidden[layer]
        model.pop()  # remove last activation
        self.mlp = nn.Sequential(OrderedDict(model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network, returns the compressed data
        vector.

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor: data
        """
        return self.mlp(x)

    
def test_train_split(x, y, train_size_frac=0.8, random_state=0, reshape=True, retindex=False):
    '''
    Split the data into test and training dataset
    '''

    # idxpath = '/mnt/ceph/users/cmodi/contrastive/analysis/test-train-splits/'
    n = x.shape[0]
    test_frac = 1 - train_size_frac
    train_id = np.arange(n)[:int(n*train_size_frac)]
    test_id = np.arange(n)[int(n*train_size_frac):]
    # fname = f"N{n}-f{test_frac:0.2f}-S{random_state}"

    # try:
    #     print(f"Loading test-train split index from {idxpath}train-{fname}.npy")
    #     train_id = np.load(f"{idxpath}train-{fname}.npy")
    #     test_id = np.load(f"{idxpath}test-{fname}.npy")
    #     print("Successfully loaded")
    #     if retindex: return train_id, test_id
        
    # except Exception as e:
    #     print("\nEXCEPTION occured in loading test_train_split")
    #     print(e)
    #     print("Generate splits now and save them")
    #     train_id, test_id = split_index(n, test_frac, random_state)
 
    data = namedtuple("data", ["trainx", "trainy", "testx", "testy"])
    data.tidx = [train_id, test_id]
    data.trainx = x[train_id]
    data.testx =  x[test_id]
    data.trainy = y[train_id]
    data.testy = y[test_id]    
    if reshape:
        if len(data.trainx.shape) > 2:
            nsim = data.trainx.shape[1] # assumes that features are on last axis
            nfeats, nparams = data.trainx.shape[-1], data.trainy.shape[-1]
            data.nsim, data.nfeatures, data.nparams = nsim, nfeats, nparams
            data.trainx = data.trainx.reshape(-1, nfeats)
            data.testx = data.testx.reshape(-1, nfeats)
            data.trainy = data.trainy.reshape(-1, nparams)
            data.testy = data.testy.reshape(-1, nparams)

    return data


###
def standardize(data, secondary=None, log_transform=True, scaler=None):
    '''
    Given a dataset, standardize by removing mean and scaling by standard deviation
    '''
    if log_transform:
        data = np.log10(data)
        if secondary is not None:
            secondary = np.log10(secondary)
    if scaler is None: 
        scaler = StandardScaler()
        data_s = scaler.fit_transform(data)
    else: 
        data_s = scaler.transform(data)
    if secondary is not None:
        secondary_s = scaler.transform(secondary)
        return data_s, secondary_s, scaler
    return data_s, scaler


###
def minmax(data, log_transform=True, scaler=None):
    '''
    Given a dataset, standardize by removing mean and scaling by standard deviation
    '''
    if log_transform:
        data = np.log10(data)
    if scaler is None: 
        scaler = MinMaxScaler()
        data_s = scaler.fit_transform(data)
    else: 
        data_s = scaler.transform(data)
    return data_s, scaler


###
def save_scaler(scaler, savepath):
    with open(savepath + "scaler.pkl", "wb") as handle:
        pickle.dump(scaler, handle)

def save_posterior(posterior, savepath):
    with open(savepath + "posterior.pkl", "wb") as handle:
        pickle.dump(posterior, handle)

def save_inference(inference, savepath):
    with open(savepath + "inference.pkl", "wb") as handle:
        pickle.dump(inference, handle)

def load_scaler(savepath, fname="scaler.pkl"):
    with open(savepath + fname, "rb") as handle:
        return pickle.load(handle)

def load_posterior(savepath):
    with open(savepath + "posterior.pkl", "rb") as handle:
        return pickle.load(handle)

def load_inference(savepath):
    with open(savepath + "inference.pkl", "rb") as handle:
        return pickle.load(handle)


###
def sbi(trainx, trainy, prior, alg, savepath=None, model='maf', 
        out_dim_embed=16,num_layers_embed=2, num_hidden_embed=16,
        nhidden=32, nlayers=5, nblocks=2,
        batch_size=128, lr=0.0005,
        validation_fraction=0.2, 
        retrain=False, summarize=False, verbose=True):

    if (savepath is not None) & (not retrain):
        try:
            print("Load an existing posterior model")
            posterior = load_posterior(savepath)
            inference = load_inference(savepath)
            if summarize:
                return posterior, inference, None
            else:
                return posterior

        except Exception as e:
            print("##Exception##\n", e)

    print("Training a new NF")

    model_embed = FCEmbedding(input_dim=trainx.shape[-1], output_dim=out_dim_embed, num_layers=num_layers_embed, num_hiddens=num_hidden_embed)
    # model_embed = FCN(n_hidden=[num_hidden_embed]*num_layers_embed)
    # model_embed.initalize_model(trainx.shape[-1])
    
    if alg == 'snpe':
        print("With algorithm SNPE")
        density_estimator_build_fun = posterior_nn(model=model, \
                                               hidden_features=nhidden, \
                                               num_transforms=nlayers,
                                               num_blocks=nblocks,
                                               embedding_net=model_embed)
        inference = SNPE(prior=prior, density_estimator=density_estimator_build_fun)    
        
    elif alg == 'snle':
        print("With algorithm SNLE")
        density_estimator_build_fun = likelihood_nn(model=model, \
                                               hidden_features=nhidden, \
                                               num_transforms=nlayers,
                                               num_blocks=nblocks,
                                               embedding_net=model_embed)
        inference = SNLE_A(prior=prior, density_estimator=density_estimator_build_fun)
    else:
        print('Algorithm should be either snpe or snle')
        raise NotImplementedError
    
    inference.append_simulations(
        x = torch.from_numpy(trainx.astype('float32')), 
        theta = torch.from_numpy(trainy.astype('float32')), 
        )
    
    density_estimator = inference.train(training_batch_size=batch_size, 
                                        validation_fraction=validation_fraction, 
                                        learning_rate=lr,
                                        show_train_summary=verbose)
    
    posterior = inference.build_posterior(density_estimator)
    
    if savepath is not None:
        save_posterior(posterior, savepath)
        save_inference(inference, savepath)

    if summarize:
        # Log summary
        summary = {"train_log_probs":[], "validation_log_probs":[]}
        for i in range(len(inference.summary['training_log_probs'])):
            summary['train_log_probs'].append(inference.summary['training_log_probs'][i])
            summary['validation_log_probs'].append(inference.summary['validation_log_probs'][i])
        summary["best_validation_log_prob"] = inference.summary['best_validation_log_prob'][0]
        if savepath is not None:
            np.save(savepath + 'train_log_probs', summary['training_log_probs'])
            np.save(savepath + 'validation_log_probs', summary['validation_log_probs'])
            np.save(savepath + 'best_validation_log_prob', summary['best_validation_log_prob'])
        return posterior, inference, summary
    else:
        return posterior


#############
def analysis(cfgm, features, params, verbose=True):
    data = test_train_split(features, params, train_size_frac=0.9)

    ### Standaradize
    scaler = None
    if cfgm.standardize:
        try:
            scaler = load_scaler(cfgm.analysis_path)
            data.trainx = standardize(data.trainx, scaler=scaler, log_transform=cfgm.logit)[0]
            data.testx = standardize(data.testx, scaler=scaler, log_transform=cfgm.logit)[0]
        except Exception as e:
            print("EXCEPTION occured in loading scaler", e)
            print("Fitting for the scaler and saving it")
            data.trainx, data.testx, scaler = standardize(data.trainx, secondary=data.testx, log_transform=cfgm.logit)
            with open(cfgm.analysis_path + "scaler.pkl", "wb") as handle:
                pickle.dump(scaler, handle)

    ### SBI
    prior = sbi_prior(params.reshape(-1, params.shape[-1]), offset=cfgm.prior_offset)
    
    print("trainx and trainy shape : ", data.trainx.shape, data.trainy.shape)
    posterior, inference, summary = sbi(data.trainx, data.trainy, prior, \
                                        alg = cfgm.alg,
                                        model=cfgm.model,
                                        out_dim_embed=cfgm.out_dim_embed,
                                        num_layers_embed=cfgm.num_layers_embed, 
                                        num_hidden_embed=cfgm.num_hidden_embed,
                                        nlayers=cfgm.ntransforms,
                                        nhidden=cfgm.nhidden,
                                        nblocks=cfgm.nblocks,
                                        batch_size=cfgm.batch,
                                        lr=cfgm.lr,
                                        validation_fraction=cfgm.validation_fraction,
                                        # savepath=cfgm.model_path,
                                        # retrain=bool(cfgm.retrain),
                                        summarize=True,
                                        verbose=verbose)

    return data, posterior, inference, summary

    




# def train(x, y, model, criterion, batch_size=32, niter=100, lr=1e-3, optimizer=None, nprint=20, scheduler=None):

#     if optimizer is None: optimizer = optim.Adam(model.parameters(), lr=lr)
#     if scheduler is not None: scheduler = scheduler(optimizer) 
#     # in your training loop:
#     losses = []
#     for j in range(niter+1):
#         optimizer.zero_grad()   # zero the gradient buffers
#         idx = np.random.randint(0, x.shape[0], batch_size)
#         inp = torch.tensor(x[idx], dtype=torch.float32)
#         target = torch.tensor(y[idx], dtype=torch.float32)  # a dummy target, for example
#         output = model(inp)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()    # Does the update
#         losses.append(loss.detach().numpy())
#         if (j*nprint)%niter == 0: print(j, losses[-1])
#         if (scheduler is not None) & ((j * batch_size)%x.shape[0] == 0) : 
#             #print('scheduel step ')
#             scheduler.step()

#     return losses, optimizer




# def embed_data(x, model, batch=256, device='cuda'):
#     em = []

#     for i in range(x.shape[0]//batch + 1):
#         em.append(model(torch.tensor(x[i*batch : (i+1)*batch], dtype=torch.float32).to(device)).detach().cpu().numpy())
        
#     em = np.concatenate(em, axis=0)
#     return em



# def insert_sweep_name(path):
#     if '%s' in path:
#         dirname = '/'.join(cfg_path.split('/')[:-2])
#     else:
#         dirname = path
#     print(f"In directory {dirname}")
#     for root, dirs, files in os.walk(dirname):
#         if len(dirs) > 1 :
#             print('More than 1 sweeps, abort!')
#             raise
#         break
#     print(f"Sweep found : {dirs[0]}")
#     if '%s' in path:
#         return path%dirs[0]
#     else:
#         return path + f'/{dirs[0]}/'


# def setup_sweepdict(analysis_path, verbose=False):

#     args = {}
#     args['analysis_path'] = analysis_path
#     cfg_path = insert_sweep_name(analysis_path)
#     # print(cfg_path)
#     cfg_dict = yaml.load(open(f'{cfg_path}/sweep_config.yaml'), Loader=yaml.Loader)
#     sweep_id = cfg_dict['sweep']['id']
#     for i in cfg_dict.keys():
#         args.update(**cfg_dict[i])
#     cfg = Objectify(**args)

#     sweep = api.sweep(f'modichirag92/hysbi/{sweep_id}')
#     #sort in the order of validation log prob
#     names, log_prob = [], []
#     for run in sweep.runs:
#         if run.state == 'finished':
#             # print(run.name, run.summary['best_validation_log_prob'])
#             try:
#                 model_path = run.summary['output_directory']
#                 names.append(run.name)
#                 log_prob.append(run.summary['best_validation_log_prob'])
#             except Exception as e:
#                 print('Exception in checking state of run : ', e)
#     idx = np.argsort(log_prob)[::-1]
#     names = np.array(names)[idx]

#     scaler = load_scaler(cfg.analysis_path)

#     toret = {'sweepid':sweep_id, 'cfg':cfg, 'idx':idx, 'scaler':scaler, 'names':names, 'cfg_dict':cfg_dict}
#     return toret