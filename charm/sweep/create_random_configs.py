import numpy as np
import yaml
import ast
import math
fname = '/mnt/home/spandey/ceph/CHARM/charm/sweep/config_wandb_snle.yaml'
config_data = yaml.safe_load(open(fname, 'r'))
params_all = config_data['parameters']
n_configs = config_data['num_configs']
for jc in range(n_configs):
    saved_config = {}
    saved_config['alg'] = 'snpe'
    saved_config['model'] = 'maf'
    saved_config['logit'] = 0
    saved_config['analysis_path'] = '/mnt/home/spandey/ceph/CHARM/charm/sweep/output/'
    saved_config['validation_fraction'] = 0.1
    saved_config['retrain'] = False
    saved_config['standardize'] = True
    saved_config['prior_offset'] = 0.25
    
    for param_name in list(params_all.keys()):
        param = params_all[param_name]
        distribution = param['distribution']
        if distribution == 'q_log_uniform_values':
            if param_name == 'lr':
                min_val = float(param['min'])
                max_val = float(param['max'])
                base_log = float(param['q'])
                param_rand_val = round(round(np.exp((np.random.rand())*(np.log(max_val) - np.log(min_val)) + np.log(min_val))/base_log)*base_log, 7)
            else:
                min_val = int(param['min'])
                max_val = int(param['max'])
                base_log = int(param['q'])
                param_rand_val = round(np.exp((np.random.rand())*(np.log(max_val) - np.log(min_val)) + np.log(min_val))/base_log)*base_log
        
        if distribution == 'int_uniform':
            min_val = int(param['min'])
            max_val = int(param['max'])
            param_rand_val = np.random.randint(min_val, max_val+1)
        
        saved_config[param_name] = param_rand_val
    
    config_fname = f'/mnt/home/spandey/ceph/CHARM/charm/sweep/snpe_configs/config_{jc}.yaml'
    with open(config_fname, 'w') as outfile:
        yaml.dump(saved_config, outfile, default_flow_style=False)

