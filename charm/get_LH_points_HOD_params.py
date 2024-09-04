import numpy as np
import matplotlib.pyplot as plt
from smt.sampling_methods import LHS
import copy
import numpy as np
import pathlib
curr_path = pathlib.Path().absolute()
data_path = os.path.abspath(curr_path / "../data/") 


param_name_all = ['logMmin','sigma_logM','logM0','logM1','alpha']
min_param_val_all = [-0.15, -0.1, -0.2, -0.3, -0.3]
max_param_val_all = [0.15, 0.1, 0.2, 0.3, 0.3]
    
nvar_all = len(param_name_all)
xlimits = np.zeros((nvar_all,2))
for jv in range((nvar_all)):
    xlimits[jv,0] = 0.0
    xlimits[jv,1] = 1.0


sampling = LHS(xlimits=xlimits,criterion='cm',random_state=int(0))

num_LHS = 20000
LHS_points = sampling(num_LHS)

LHS_points_final = np.zeros_like(LHS_points)
for jv in range(nvar_all):
    LHS_points_final[:,jv] = min_param_val_all[jv] + (max_param_val_all[jv] - min_param_val_all[jv]) * LHS_points[:,jv]

LH_cosmo_val_file='/mnt/home/spandey/ceph/Quijote/latin_hypercube_params.txt'
LH_cosmo_val_all = np.loadtxt(LH_cosmo_val_file)
id_LH_cosmo_all = np.arange(0, len(LH_cosmo_val_all))
LH_cosmo_val_all_wid = np.concatenate((LH_cosmo_val_all, id_LH_cosmo_all[:,None]), axis=1)
num = num_LHS // len(LH_cosmo_val_all)
LH_cosmo_val_all_wid_tile = np.tile(LH_cosmo_val_all_wid, (num,1))

all_params = np.concatenate((LHS_points_final, LH_cosmo_val_all_wid_tile), axis=1)

cosmo_params_names = ['Om','Ob','h','ns','s8', 'LH_cosmo_id']

all_params_names = param_name_all + cosmo_params_names

first_line = ''
for var in all_params_names:
    first_line += str(var) + '     '


np.savetxt(data_path + 'LH_points_HOD_cosmo_np_' + str(num_LHS) + '.txt',all_params,header=first_line)
