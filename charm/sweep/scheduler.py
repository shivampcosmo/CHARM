import numpy as np
import sys, os
import wandb
from ruamel import yaml
# sys.path.append('../../src/')
# import loader_pk, loader_pk_splits, loader_wv, loader_wv_splits
wandb.login()

config_data = sys.argv[1]
print("config file : ", config_data)
cfgd = yaml.load(open(f'{config_data}'), Loader=yaml.RoundTripLoader)

#initialize sweep
sweep_id = wandb.sweep(sweep=yaml.load(open(f'/mnt/home/spandey/ceph/CHARM/charm/wandb/config_wandb_snle.yaml'), Loader=yaml.Loader), \
                        project='charm', entity='shivampcosmo')
print("Schedule sweep with id : ", sweep_id)
cfgd['sweep'] = {'id' : sweep_id}
nmodels = 1

#save config file in sweep folder
#save config file in sweep folder
if 'pk' in config_data:
    if 'splits' in config_data: 
        loader = loader_pk_splits
    else:
        loader  = loader_pk
elif 'wv' in config_data:
    if 'splits' in config_data: 
        loader = loader_wv_splits
    else:
        loader  = loader_wv
else:
    print("Loader could not be determined. Exiting")
    sys.exit()

analysis_path = loader.folder_path(cfgd)
model_path = f'{analysis_path}/{sweep_id}/'
config_path = f'{model_path}/sweep_config.yaml'
command = f"time python -u run_wandb.py {config_path} {nmodels}"
 
os.makedirs(model_path, exist_ok=True)
with open(config_path, 'w') as outfile:
    yaml.dump(cfgd, outfile, Dumper=yaml.RoundTripDumper)

print(f"config path saved at:\n{config_path}\n")
    
#run once to initiate the sweep    
print(command)
# os.system(command)
