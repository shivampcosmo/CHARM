import sys, os
import numpy as np
import sys, os
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as pl
import yaml
import readfof
import sys, os
import numpy as np
import pickle as pk 
import h5py as h5
import numpy as np
import Pk_library as PKL
import MAS_library as MASL
import yaml
import galactic_wavelets as gw
import torch

def setup_hod(hmass, nbar=4e-4, satfrac=0.2, bs=1000, alpha_fid=0.76):
    numdhalos = len(hmass)/bs**3
    numhalos_nbarf = int(nbar * bs**3 * (1-satfrac))
    mcut = hmass[:numhalos_nbarf][-1]
    nsat = satfrac * nbar * bs**3
    mdiff = (hmass - mcut + mcut*1e-3)[:numhalos_nbarf] ** alpha_fid
    msum = mdiff.sum()/nsat
    m1 = msum**(1/alpha_fid)
    return mcut, m1



def calc_summary(isim, cat_type='constant_nbar',nbar=4e-4, lgMmin=13.0):
    try:
        ldir = '/mnt/home/spandey/ceph/CHARM/data/halo_cats_charm_truth_nsubv_vel_10k/'
        LH_cosmo_val_file = '/mnt/home/spandey/ceph/Quijote/latin_hypercube_params.txt'

        if cat_type == 'constant_nbar':
            sdir_stats = '/mnt/home/spandey/ceph/CHARM/data/test_mp/'
            if nbar == 4e-4:
                suffix = '4en4'
            if nbar == 1e-4:
                suffix = '1en4'
            savefname = sdir_stats + '/summary_stats_weighted_rsd_' + str(isim) + '_nbar_' + suffix + '.pk'
        elif cat_type == 'constant_Mmin':
            sdir_stats = '/mnt/home/spandey/ceph/CHARM/data/test_mp/'
            if lgMmin == 13.0:
                suffix = '13p0'
            if lgMmin == 12.5:
                suffix = '12p5'
            savefname = sdir_stats + '/summary_stats_weighted_rsd_' + str(isim) + '_lgMmin_' + suffix + '.pk'
        else:
            raise ValueError(f"Unknown cat_type: {cat_type}")

        if os.path.exists(savefname):
            print(f"File exists: {savefname}")
            return
        else:

            df = pk.load(open(ldir + f'halo_cat_pos_vel_LH_{isim}.pk', 'rb'))

            lgmass = df['lgmass_mock']
            mass_argsort = np.flip(np.argsort(lgmass))
            lgmass_sort = lgmass[mass_argsort]

            sigma_lgM = 0.2
            alpha_sat = 0.76
            # nbar = 4e-4
            sat_frac = 0.2

            ns_h = 128
            grid = ns_h    #the 3D field will have grid x grid x grid voxels
            BoxSize = 1000.0 #Mpc/h ; size of box

            kmax = 0.32
            nk = 16
            threads = 10

            dx = BoxSize/grid # Mpc/h
            import torch
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            J = 3
            Q = 4
            kc = 2*np.pi/3 # Cutoff frequency of the mother wavelet (in px^-1 units)
            moments = [1/2, 1, 2]
            MAS     = 'NGP'  #mass-assigment scheme
            verbose = False   #print information on progress


            if cat_type == 'constant_nbar':
                mcut, m1 = setup_hod(10**lgmass_sort, nbar=nbar, satfrac=sat_frac, bs=BoxSize, alpha_fid=alpha_sat)
                lgMmin_cut = np.log10(mcut) - sigma_lgM/2.
                Mmin_Pk, Mmax_Pk = lgMmin_cut, 16.0
            
            elif cat_type == 'constant_Mmin':
                Mmin_Pk, Mmax_Pk = lgMmin, 16.0
                m1 = 1e14
            
            else:
                raise ValueError(f"Unknown cat_type: {cat_type}")

            saved = pk.load(open(ldir + f'halo_cat_pos_vel_LH_{isim}.pk', 'rb'))
            # saved = {'pos_h_truth': pos_h_truth, 'lgMass_truth': lgMass_truth, 'pos_h_mock': pos_h_mock, 'lgMass_mock': lgMass_mock, 'cosmo':cosmo_val_all_test[0,0,0,0,:]}
            pos_h_truth = saved['pos_truth']
            lgMass_truth = saved['lgmass_truth']
            pos_h_mock = saved['pos_mock']
            lgMass_mock = saved['lgmass_mock']

            pos_h_rsd_mock = saved['pos_rsdx_mock']
            pos_h_rsd_truth = saved['pos_rsdx_truth']        

            LH_cosmo_val_all = np.loadtxt(LH_cosmo_val_file)
            cosmo = LH_cosmo_val_all[isim]

            saved_j = {}
            # Mmin_Pk, Mmax_Pk = Mmin_Pk_all[j], Mmax_Pk_all[j]

            M1 = np.log10(m1)
            alpha = alpha_sat

            pos_type = ['real', 'rsd']
            for pos in pos_type:
                if pos == 'real':
                    pos_mock_here = pos_h_mock
                    pos_truth_here = pos_h_truth
                elif pos == 'rsd':
                    pos_mock_here = pos_h_rsd_mock
                    pos_truth_here = pos_h_rsd_truth
                else:
                    raise ValueError(f"Unknown pos type: {pos}")
                indsel_Pk_truth = np.where((lgMass_truth > Mmin_Pk) & (lgMass_truth < Mmax_Pk))[0]
                mesh_truth2 = np.zeros((grid, grid, grid), dtype=np.float32)
                pos_truth2 = (pos_truth_here[indsel_Pk_truth,...]).astype(np.float32)
                W_truth = (10**lgMass_truth[indsel_Pk_truth]/10**M1)**(alpha)
                # MASL.MA(pos_truth2, mesh_truth2, BoxSize, MAS, W=10**((lgMass_truth[indsel_Pk_truth]- Mmax_Pk)))
                MASL.MA(pos_truth2, mesh_truth2, BoxSize, MAS, W=W_truth)
                mesh_truth2 /= np.mean(mesh_truth2, dtype=np.float64);  mesh_truth2 -= 1.0
                Pk_truth2 = PKL.Pk(mesh_truth2, BoxSize, axis=0, MAS=None, threads=threads)

                indsel_Pk_mock = np.where((lgMass_mock > Mmin_Pk) & (lgMass_mock < Mmax_Pk))[0]
                mesh_mock = np.zeros((grid, grid, grid), dtype=np.float32)
                pos_mock = (pos_mock_here[indsel_Pk_mock,...]).astype(np.float32)
                W_mock = (10**lgMass_mock[indsel_Pk_mock]/10**M1)**(alpha)
                # MASL.MA(pos_mock, mesh_mock, BoxSize, MAS, W=10**((lgMass_mock[indsel_Pk_mock]- Mmax_Pk)))
                MASL.MA(pos_mock, mesh_mock, BoxSize, MAS, W=W_mock)
                mesh_mock /= np.mean(mesh_mock, dtype=np.float64);  mesh_mock -= 1.0
                Pk_mock = PKL.Pk(mesh_mock, BoxSize, axis=0, MAS=None, threads=threads)

                indk_sel = np.where((Pk_truth2.k3D >= 0.01) & (Pk_truth2.k3D <= kmax))[0]
                Pkmock_sel = Pk_mock.Pk[indk_sel,:]
                Pktruth_sel = Pk_truth2.Pk[indk_sel,:]

                k_Pk = Pk_truth2.k3D[indk_sel]

                len_k_sel = len(indk_sel)
                ds_fac = len_k_sel//nk
                Pk_mock_ds = Pkmock_sel[::ds_fac]
                Pk_truth_ds = Pktruth_sel[::ds_fac]
                k_Pk_ds = k_Pk[::ds_fac]

                saved_j[pos + '_Pk_truth_weighted'] = Pk_truth_ds
                saved_j[pos + '_Pk_mock_weighted'] = Pk_mock_ds
                saved_j[pos + '_k_Pk_weighted'] = k_Pk_ds

                BoxSize = 1000.0 #Size of the density field in Mpc/h
                # threads = 1
                theta   = np.linspace(0.1, np.pi-0.1, 8) #array with the angles between k1 and k2
                saved_j['theta'] = theta

                k1 = 0.08
                k2 = 0.08
                BBk = PKL.Bk(mesh_truth2, BoxSize, k1, k2, theta, None, threads)
                Bk_truth_k0p06  = BBk.B     #bispectrum
                Qk_truth_k0p06  = BBk.Q     #reduced bispectrum
                BBk = PKL.Bk(mesh_mock, BoxSize, k1, k2, theta, None, threads)
                Bk_mock_k0p06  = BBk.B     #bispectrum
                Qk_mock_k0p06  = BBk.Q     #reduced bispectrum

                saved_j[pos + '_Bk_truth_k0p06_weighted'] = Bk_truth_k0p06
                saved_j[pos + '_Qk_truth_k0p06_weighted'] = Qk_truth_k0p06
                saved_j[pos + '_Bk_mock_k0p06_weighted'] = Bk_mock_k0p06
                saved_j[pos + '_Qk_mock_k0p06_weighted'] = Qk_mock_k0p06

                k1 = 0.16
                k2 = 0.16
                BBk = PKL.Bk(mesh_truth2, BoxSize, k1, k2, theta, None, threads)
                Bk_truth_k0p2  = BBk.B     #bispectrum
                Qk_truth_k0p2  = BBk.Q     #reduced bispectrum
                BBk = PKL.Bk(mesh_mock, BoxSize, k1, k2, theta, None, threads)
                Bk_mock_k0p2  = BBk.B     #bispectrum
                Qk_mock_k0p2  = BBk.Q     #reduced bispectrum

                saved_j[pos + '_Bk_truth_k0p2_weighted'] = Bk_truth_k0p2
                saved_j[pos + '_Qk_truth_k0p2_weighted'] = Qk_truth_k0p2
                saved_j[pos + '_Bk_mock_k0p2_weighted'] = Bk_mock_k0p2
                saved_j[pos + '_Qk_mock_k0p2_weighted'] = Qk_mock_k0p2


                k1 = 0.32
                k2 = 0.32
                BBk = PKL.Bk(mesh_truth2, BoxSize, k1, k2, theta, None, threads)
                Bk_truth_k0p3  = BBk.B     #bispectrum
                Qk_truth_k0p3  = BBk.Q     #reduced bispectrum
                BBk = PKL.Bk(mesh_mock, BoxSize, k1, k2, theta, None, threads)
                Bk_mock_k0p3  = BBk.B     #bispectrum
                Qk_mock_k0p3  = BBk.Q     #reduced bispectrum

                saved_j[pos + '_Bk_truth_k0p3_weighted'] = Bk_truth_k0p3
                saved_j[pos + '_Qk_truth_k0p3_weighted'] = Qk_truth_k0p3
                saved_j[pos + '_Bk_mock_k0p3_weighted'] = Bk_mock_k0p3
                saved_j[pos + '_Qk_mock_k0p3_weighted'] = Qk_mock_k0p3


                df = torch.from_numpy(mesh_truth2).to(device)
                df_shape = df.shape
                N = df_shape[0]
                wst_op = gw.ScatteringOp(df_shape, J, Q,
                                        moments=moments,
                                        kc=kc,
                                        scattering=True)
                s0_truth, s1_truth, s2_truth = wst_op(df)

                s0_truth = s0_truth.cpu().numpy()
                s1_truth = s1_truth.cpu().numpy().flatten()
                s2_truth = s2_truth.cpu().numpy().flatten()

                saved_j[pos + 's0_truth_weighted'] = s0_truth
                saved_j[pos + 's1_truth_weighted'] = s1_truth
                saved_j[pos + 's2_truth_weighted'] = s2_truth


                df = torch.from_numpy(mesh_mock).to(device)
                df_shape = df.shape
                N = df_shape[0]
                wst_op = gw.ScatteringOp(df_shape, J, Q,
                                        moments=moments,
                                        kc=kc,
                                        scattering=True
                                        )
                s0_mock, s1_mock, s2_mock = wst_op(df)

                s0_mock = s0_mock.cpu().numpy()
                s1_mock = s1_mock.cpu().numpy().flatten()
                s2_mock = s2_mock.cpu().numpy().flatten()

                saved_j[pos + '_s0_mock_weighted'] = s0_mock
                saved_j[pos + '_s1_mock_weighted'] = s1_mock
                saved_j[pos + '_s2_mock_weighted'] = s2_mock

                summary_concat_mock_all_weighted = np.concatenate((Pk_mock_ds.flatten(), Bk_mock_k0p06, Bk_mock_k0p2, Bk_mock_k0p3, s0_mock, s1_mock[::4], s2_mock[::6]))
                summary_concat_truth_all_weighted = np.concatenate((Pk_truth_ds.flatten(), Bk_truth_k0p06, Bk_truth_k0p2, Bk_truth_k0p3, s0_truth, s1_truth[::4], s2_truth[::6]))

                saved_j[pos + '_summary_concat_mock_all_weighted'] = summary_concat_mock_all_weighted
                saved_j[pos + '_summary_concat_truth_all_weighted'] = summary_concat_truth_all_weighted

            saved_j['cosmo'] = cosmo        

            pk.dump(saved_j, open(savefname, 'wb'))
            print('saved_with_weighting')
            return

    except Exception as e:
        print(f"Error in calc_summary: {e}")
        return

# if __name__ == "__main__":
#     n1 = int(sys.argv[1])
#     n2 = int(sys.argv[2])
#     for test_id in (range(n1, n2)):
#         calc_summary(test_id)        
#         # calc_summary(test_id, cat_type='constant_Mmin', lgMmin=13.0)
#     # # test_id = int(sys.argv[1])
#     # # save_cats(test_id)
#     # for test_id in (range(361, 362)):
#     #     save_cats(test_id, verbose=True)


# import multiprocessing as mp
# if __name__ == '__main__':
#     # n_sims = 1100
#     # n_sims_offset = 1100
#     # n_sims = 900
#     n1 = int(sys.argv[1])
#     n2 = int(sys.argv[2])
#     n_sims = n2 - n1
#     n_sims_offset = n1
#     n_cores = mp.cpu_count()
#     print(n_cores)

#     # Create a pool of worker processes
#     pool = mp.Pool(processes=n_cores)

#     # Distribute the simulations across the available cores
#     sims_per_core = n_sims // n_cores
#     sim_ranges = [(n_sims_offset + i * sims_per_core, n_sims_offset + (i + 1) * sims_per_core) for i in range(n_cores)]

#     # Handle any remaining simulations
#     remaining_sims = n_sims % n_cores
#     if remaining_sims > 0:
#         sim_ranges[-1] = (sim_ranges[-1][0], sim_ranges[-1][1] + remaining_sims)

#     # Run save_cic_densities function for each simulation range in parallel
#     results = [pool.apply_async(calc_summary, args=(ji,)) for sim_range in sim_ranges for ji in range(*sim_range)]

#     # Wait for all tasks to complete
#     [result.get() for result in results]

#     # Close the pool and wait for tasks to finish
#     pool.close()
#     pool.join()

import torch.multiprocessing as mp
if __name__ == "__main__":
    # Get input arguments for simulations range
    n1 = int(sys.argv[1])
    n2 = int(sys.argv[2])
    n_sims = n2 - n1
    n_sims_offset = n1

    # Get the number of available CPU cores
    n_cores = mp.cpu_count()
    print(f"Number of CPU cores available: {n_cores}")

    # Set up multiprocessing pool with torch.multiprocessing
    mp.set_start_method('spawn')  # 'spawn' is safer for PyTorch

    # Create a pool of worker processes
    pool = mp.Pool(processes=n_cores)

    # Distribute the simulations across the available cores
    sims_per_core = n_sims // n_cores
    sim_ranges = [(n_sims_offset + i * sims_per_core, n_sims_offset + (i + 1) * sims_per_core) for i in range(n_cores)]

    # Handle any remaining simulations
    remaining_sims = n_sims % n_cores
    if remaining_sims > 0:
        sim_ranges[-1] = (sim_ranges[-1][0], sim_ranges[-1][1] + remaining_sims)

    # Run calc_summary function for each simulation range in parallel
    results = [pool.apply_async(calc_summary, args=(ji,)) for sim_range in sim_ranges for ji in range(*sim_range)]

    # Wait for all tasks to complete
    [result.get() for result in results]

    # Close the pool and wait for tasks to finish
    pool.close()
    pool.join()