
import numpy as np
import sys, os
import sbitools
import yaml
import pickle as pk
import dill

pk_type = sys.argv[1]
do_wavelets = sys.argv[2]


def run_sbi(config_id, verbose=False, sel_n=None):
    # Load the config file
    import dill
    fname = f'/mnt/home/spandey/ceph/CHARM/charm/sweep/snpe_configs/config_{config_id}.yaml'
    cfgm = yaml.safe_load(open(fname, 'r'))
    cfgm = sbitools.Objectify(**cfgm)
    cfgm.retrain = True
    cfgm.logit = False
    cfgm.standardize = True
    cfgm.analysis_path = f'/mnt/home/spandey/ceph/CHARM/charm/sweep/output/trained_posteriors_pk_{pk_type}_bk_wavelets_{do_wavelets}/'

    # load data:
    df = pk.load(open(f'/mnt/home/spandey/ceph/CHARM/charm/sweep/data/saved_data_pk_{pk_type}_bk_wavelets_{do_wavelets}.pk','rb'))
    x_all = df['x_all']
    x_all_truth = df['x_all_truth']
    theta_all = df['theta_all']

    if sel_n is not None:
        x_all = x_all[:sel_n]
        theta_all = theta_all[:sel_n]

    data, posterior, inference, summary = sbitools.analysis(cfgm, x_all, theta_all, verbose=verbose)

    # Make the loss and optimizer
    for i in range(len(summary['train_log_probs'])):
        metrics = {"train_log_probs": summary['train_log_probs'][i],
                "validation_log_probs": summary['validation_log_probs'][i]}
    # wandb.run.summary["best_validation_log_prob"] = summary['best_validation_log_prob']
    print(summary['best_validation_log_prob'])

    saved = {'data': data, 'posterior': posterior, 'inference': inference, 'summary': summary}
    with open(f'/mnt/home/spandey/ceph/CHARM/charm/sweep/output/trained_posteriors_pk_{pk_type}_bk_wavelets_{do_wavelets}/saved_{config_id}.pkl', 'wb') as f:
        dill.dump(saved, f)
    
    with open(f'/mnt/home/spandey/ceph/CHARM/charm/sweep/output/trained_summary_pk_{pk_type}_bk_wavelets_{do_wavelets}/saved_{config_id}.pkl', 'wb') as f:
        dill.dump(summary, f)
    
    print(f"Saved trained posterior and summary for config {config_id} with best validation log prob : {summary['best_validation_log_prob']}")


from mpi4py import MPI
if __name__ == '__main__':
    run_count = 0
    n_jobs = 256

    while run_count < n_jobs:
        comm = MPI.COMM_WORLD
        print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
        if (run_count + comm.rank) < n_jobs:
            run_sbi(comm.rank)
        run_count += comm.size
        comm.bcast(run_count, root=0)
        comm.Barrier()