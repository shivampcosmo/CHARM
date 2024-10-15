import readgadget
import readfof
import bigfile
import nbodykit
import sys, os
import nbodykit.lab as nb
import readgadget
import numpy as np
import matplotlib.pyplot as pl
from ngp_mass import NGP_mass
import dill
from tqdm import tqdm
import sparse
import MAS_library as MASL
import colossus
from colossus.cosmology import cosmology
from colossus.halo import mass_so
from colossus.lss import peaks
from colossus.halo import concentration


mass_type = sys.argv[1]
snapnum = int(sys.argv[2])
def save_halo_cats(isim, mass_type=mass_type, isfid=False, 
                z_dict = {4:0.0, 3:0.5, 2:1.0, 1:2.0, 0:3.0, -1: 127}, snapnum = snapnum,
                grid = 128, BoxSize = 1000.,Mmin_cut = 5e12,Mmin_cut_str = '5e12',
                save_subvol = True, save_full = True, save_vel=True,
                LH_cosmo_val_file='/mnt/home/spandey/ceph/Quijote/latin_hypercube_params.txt'):
    

    redshift = z_dict[snapnum]


    if isfid:
        root_out = '/mnt/home/spandey/ceph/Quijote/data_NGP_self'
    else:
        root_out = '/mnt/home/spandey/ceph/Quijote/data_NGP_self_LH'
    folder_out = '%s/%d'%(root_out,isim)
    # create output folder if it does not exists
    if not(os.path.exists(folder_out)):
        os.system('mkdir %s'%folder_out)

    savefname_halos_subvol = '%s/SPARSEMATS_halos_HR_%s_lgMmincut_%s_subvol_res_%d_z=%s.pk'%(folder_out,mass_type,Mmin_cut_str,grid,redshift)
    savefname_halos_full = '%s/SPARSEMATS_halos_HR_%s_lgMmincut_%s_full_res_%d_z=%s.pk'%(folder_out,mass_type,Mmin_cut_str,grid,redshift)       
    try:
        saved_halos_full = dill.load(open(savefname_halos_full, 'rb'))
        c_halos_diff_combined = saved_halos_full['c_halos_diff_combined']
        nu_halos_combined = saved_halos_full['nu_halos_combined']
    except:
        if isfid:
            cosmo_val_all = np.array([0.3175, 0.049, 0.6711, 0.9624, 0.834])
        else:
            cosmo_val_all = np.loadtxt(LH_cosmo_val_file)[isim]

        Om0 = cosmo_val_all[0]
        Ob0 = cosmo_val_all[1]
        h0 = cosmo_val_all[2]
        ns = cosmo_val_all[3]
        sigma8 = cosmo_val_all[4]
        params = {'flat': True, 'H0': h0*100, 'Om0': Om0, 'Ob0': Ob0, 'sigma8': sigma8, 'ns': ns}
        cosmo = cosmology.setCosmology('myCosmo', **params)

        if 'rockstar' in mass_type:
            if isfid:
                snap_dir_base = '/mnt/home/fvillaescusa/ceph/Quijote/Halos/Rockstar/fiducial_HR'        
            else:            
                snap_dir_base = '/mnt/home/fvillaescusa/ceph/Quijote/Halos/Rockstar/latin_hypercube_HR'
            verbose = False   #print information on progress
            snapdir = snap_dir_base + '/' + str(isim)  #folder hosting the catalogue
            rockstar = np.loadtxt(snapdir + '/out_' + str(snapnum) + '_pid.list')
            with open(snapdir + '/out_' + str(snapnum) + '_pid.list', 'r') as f:
                lines = f.readlines()
            header = lines[0].split()
            # get the properties of the halos
            pos_h_truth = rockstar[:,header.index('X'):header.index('Z')+1]
            if '200c' in mass_type:
                index_M = header.index('M200c')                    
                mass_truth = rockstar[:,index_M]  #Halo masses in Msun/h
                Rhalo = (1+redshift)*mass_so.M_to_R(mass_truth, redshift, '200c')        
            
            if 'vir' in mass_type:
                index_M = header.index('Mvir')
                mass_truth = rockstar[:,index_M]
                index_R = header.index('Rvir')
                Rhalo = rockstar[:,index_R]

            Rs = rockstar[:,header.index('Rs')]
            conc_sim = Rhalo/Rs
            
            lgMass_truth = np.log10(mass_truth).astype(np.float32)
            vel_h_truth = rockstar[:,header.index('VX'):header.index('VZ')+1]

        if 'fof' in mass_type:
            if isfid:
                snap_dir_base = '/mnt/home/fvillaescusa/ceph/Quijote/Halos/FoF/fiducial_HR' 
            else:
                snap_dir_base = '/mnt/home/fvillaescusa/ceph/Quijote/Halos/FoF/latin_hypercube_HR'
            snapdir = snap_dir_base + '/' + str(isim)  #folder hosting the catalogue
            FoF = readfof.FoF_catalog(snapdir, snapnum, long_ids=False, swap=False, SFR=False, read_IDs=False)
            # get the properties of the halos
            pos_h_truth = FoF.GroupPos / 1e3  #Halo positions in Mpc/h
            mass_truth = FoF.GroupMass * 1e10  #Halo masses in Msun/h
            lgMass_truth = np.log10(mass_truth).astype(np.float32)
            vel_h_truth = FoF.GroupVel*(1.0+redshift) #Halo peculiar velocities in km/s

            conc = np.zeros_like(mass_truth)

        

        
        
        n_batch = 8
            

        indsel = np.where(mass_truth > Mmin_cut)[0]
        pos_h_truth = pos_h_truth[indsel]
        mass_truth = mass_truth[indsel]
        lgMass_truth = lgMass_truth[indsel]
        vel_h_truth = vel_h_truth[indsel]
        nu = peaks.peakHeight(mass_truth, redshift)
        conc_sim = conc_sim[indsel]
        conc_func = concentration.concentration(mass_truth, '200c', redshift, model = 'diemer19')
        delta_conc = conc_sim - conc_func 

        from scipy.interpolate import RegularGridInterpolator
        xall = (np.linspace(0, BoxSize, grid + 1))
        xarray = 0.5 * (xall[1:] + xall[:-1])
        yarray = np.copy(xarray)
        zarray = np.copy(xarray)

        if save_vel:
            import pickle as pk
            if isfid:
                vel_load_dir = '/mnt/home/spandey/ceph/Quijote/data_NGP_self/fastpm/'
            else:
                vel_load_dir = '/mnt/home/spandey/ceph/Quijote/data_NGP_self_fastpm_LH/'
            df = pk.load(open(vel_load_dir + f'{isim}/velocity_HR_full_m_res_128_z={redshift}_nbatch_8_nfilter_3_ncnn_0.pk', 'rb'))['velocity_cic_unpad_combined']
            vx_mesh_load = 1000.*df[0,...]
            vy_mesh_load = 1000.*df[1,...]
            vz_mesh_load = 1000.*df[2,...]

            vx_all_3D_interp_l = RegularGridInterpolator((xarray, yarray, zarray), vx_mesh_load, bounds_error=False, fill_value=None)
            vy_all_3D_interp_l = RegularGridInterpolator((xarray, yarray, zarray), vy_mesh_load, bounds_error=False, fill_value=None)
            vz_all_3D_interp_l = RegularGridInterpolator((xarray, yarray, zarray), vz_mesh_load, bounds_error=False, fill_value=None)

            vx_eval_interp_l = vx_all_3D_interp_l(pos_h_truth)
            vy_eval_interp_l = vy_all_3D_interp_l(pos_h_truth)
            vz_eval_interp_l = vz_all_3D_interp_l(pos_h_truth)

            vx_diff = vel_h_truth[:,0] - vx_eval_interp_l
            vy_diff = vel_h_truth[:,1] - vy_eval_interp_l
            vz_diff = vel_h_truth[:,2] - vz_eval_interp_l


        Nhalos = np.float32(np.zeros((grid, grid, grid)))
        MASL.NGP(np.float32(pos_h_truth), Nhalos, BoxSize)
        print('max number of halos:', np.amax(Nhalos))

        if grid == 64:
            nMax_h = 30  # maximum number of halos expected in a cell
        elif grid == 128:
            nMax_h = 20
        elif grid == 256:
            nMax_h = 8
        elif grid == 512:
            nMax_h = 3
        elif grid == 1024:
            nMax_h = 2
        else:
            print('nside not supported')
            sys.exit()

        dfhalo_ngp_wmass = np.float32(np.zeros((grid, grid, grid, nMax_h)))
        NGP_mass(np.float32(pos_h_truth), np.float32(lgMass_truth), dfhalo_ngp_wmass, BoxSize)

        dfhalo_ngp_wnu = np.float32(np.zeros((grid, grid, grid, nMax_h)))
        NGP_mass(np.float32(pos_h_truth), np.float32(nu), dfhalo_ngp_wnu, BoxSize)    

        dfhalo_ngp_wconc_sim = np.float32(np.zeros((grid, grid, grid, nMax_h)))
        NGP_mass(np.float32(pos_h_truth), np.float32(conc_sim), dfhalo_ngp_wconc_sim, BoxSize)        

        dfhalo_ngp_wconc_func = np.float32(np.zeros((grid, grid, grid, nMax_h)))
        NGP_mass(np.float32(pos_h_truth), np.float32(conc_func), dfhalo_ngp_wconc_func, BoxSize)            

        dfhalo_ngp_wconc_diff = np.float32(np.zeros((grid, grid, grid, nMax_h)))
        NGP_mass(np.float32(pos_h_truth), np.float32(delta_conc), dfhalo_ngp_wconc_diff, BoxSize)

        dfhalo_ngp_wvx_true = np.float32(np.zeros((grid, grid, grid, nMax_h)))
        dfhalo_ngp_wvy_true = np.float32(np.zeros((grid, grid, grid, nMax_h)))
        dfhalo_ngp_wvz_true = np.float32(np.zeros((grid, grid, grid, nMax_h)))
        dfhalo_ngp_wvx_pred = np.float32(np.zeros((grid, grid, grid, nMax_h)))
        dfhalo_ngp_wvy_pred = np.float32(np.zeros((grid, grid, grid, nMax_h)))
        dfhalo_ngp_wvz_pred = np.float32(np.zeros((grid, grid, grid, nMax_h)))

        if save_vel:
            NGP_mass(np.float32(pos_h_truth), np.float32(vel_h_truth[:,0]), dfhalo_ngp_wvx_true, BoxSize)
            NGP_mass(np.float32(pos_h_truth), np.float32(vel_h_truth[:,1]), dfhalo_ngp_wvy_true, BoxSize)
            NGP_mass(np.float32(pos_h_truth), np.float32(vel_h_truth[:,2]), dfhalo_ngp_wvz_true, BoxSize)
            NGP_mass(np.float32(pos_h_truth), np.float32(vx_eval_interp_l), dfhalo_ngp_wvx_pred, BoxSize)
            NGP_mass(np.float32(pos_h_truth), np.float32(vy_eval_interp_l), dfhalo_ngp_wvy_pred, BoxSize)
            NGP_mass(np.float32(pos_h_truth), np.float32(vz_eval_interp_l), dfhalo_ngp_wvz_pred, BoxSize)

        argsort_M = np.flip(np.argsort(dfhalo_ngp_wmass, axis=-1), axis=-1)

        M_halos = np.take_along_axis(dfhalo_ngp_wmass, argsort_M, axis=-1)
        nu_halos = np.take_along_axis(dfhalo_ngp_wnu, argsort_M, axis=-1)
        conc_sim_halos = np.take_along_axis(dfhalo_ngp_wconc_sim, argsort_M, axis=-1)
        conc_func_halos = np.take_along_axis(dfhalo_ngp_wconc_func, argsort_M, axis=-1)
        conc_diff_halos = np.take_along_axis(dfhalo_ngp_wconc_diff, argsort_M, axis=-1)
        dfhalo_ngp_wvx_true = np.take_along_axis(dfhalo_ngp_wvx_true, argsort_M, axis=-1)
        dfhalo_ngp_wvy_true = np.take_along_axis(dfhalo_ngp_wvy_true, argsort_M, axis=-1)
        dfhalo_ngp_wvz_true = np.take_along_axis(dfhalo_ngp_wvz_true, argsort_M, axis=-1)
        dfhalo_ngp_wvx_pred = np.take_along_axis(dfhalo_ngp_wvx_pred, argsort_M, axis=-1)
        dfhalo_ngp_wvy_pred = np.take_along_axis(dfhalo_ngp_wvy_pred, argsort_M, axis=-1)
        dfhalo_ngp_wvz_pred = np.take_along_axis(dfhalo_ngp_wvz_pred, argsort_M, axis=-1)


        dfhalo_ngp_wvall_true = np.moveaxis(np.array([dfhalo_ngp_wvx_true, dfhalo_ngp_wvy_true, dfhalo_ngp_wvz_true]), 0, -1)
        dfhalo_ngp_wvall_pred = np.moveaxis(np.array([dfhalo_ngp_wvx_pred, dfhalo_ngp_wvy_pred, dfhalo_ngp_wvz_pred]), 0, -1)
        dfhalo_ngp_wvall_diff =  dfhalo_ngp_wvall_pred - dfhalo_ngp_wvall_true


        if save_subvol:
            # now split it into nbatches each side
            subvol_size = grid // n_batch
            nsubvol = n_batch**3
            save_subvol_Nhalo = np.zeros((nsubvol, subvol_size, subvol_size, subvol_size))
            save_subvol_Mhalo = np.zeros((nsubvol, subvol_size, subvol_size, subvol_size, nMax_h))
            save_subvol_nuhalo = np.zeros((nsubvol, subvol_size, subvol_size, subvol_size, nMax_h))
            save_subvol_concsimhalos = np.zeros((nsubvol, subvol_size, subvol_size, subvol_size, nMax_h))
            save_subvol_concfunchalos = np.zeros((nsubvol, subvol_size, subvol_size, subvol_size, nMax_h))
            save_subvol_concdiffhalos = np.zeros((nsubvol, subvol_size, subvol_size, subvol_size, nMax_h))
            save_subvol_vtrue = np.zeros((nsubvol, subvol_size, subvol_size, subvol_size, nMax_h, 3))
            save_subvol_vpred = np.zeros((nsubvol, subvol_size, subvol_size, subvol_size, nMax_h, 3))
            save_subvol_vdiff = np.zeros((nsubvol, subvol_size, subvol_size, subvol_size, nMax_h, 3))

            jc = 0
            from tqdm import tqdm
            for jx in (range(n_batch)):
                for jy in range(n_batch):
                    for jz in range(n_batch):
                        # get the sub-cube
                        save_subvol_Nhalo[jc] = Nhalos[jx * subvol_size:(jx + 1) * subvol_size,
                                                    jy * subvol_size:(jy + 1) * subvol_size,
                                                    jz * subvol_size:(jz + 1) * subvol_size]
                        save_subvol_Mhalo[jc] = M_halos[jx * subvol_size:(jx + 1) * subvol_size,
                                                        jy * subvol_size:(jy + 1) * subvol_size,
                                                        jz * subvol_size:(jz + 1) * subvol_size]

                        save_subvol_nuhalo[jc] = nu_halos[jx * subvol_size:(jx + 1) * subvol_size,
                                                        jy * subvol_size:(jy + 1) * subvol_size,
                                                        jz * subvol_size:(jz + 1) * subvol_size]
                        
                        save_subvol_concsimhalos[jc] = conc_sim_halos[jx * subvol_size:(jx + 1) * subvol_size,
                                                        jy * subvol_size:(jy + 1) * subvol_size,
                                                        jz * subvol_size:(jz + 1) * subvol_size]

                        save_subvol_concfunchalos[jc] = conc_func_halos[jx * subvol_size:(jx + 1) * subvol_size,
                                                        jy * subvol_size:(jy + 1) * subvol_size,
                                                        jz * subvol_size:(jz + 1) * subvol_size]
                        save_subvol_concdiffhalos[jc] = conc_diff_halos[jx * subvol_size:(jx + 1) * subvol_size,
                                                        jy * subvol_size:(jy + 1) * subvol_size,
                                                        jz * subvol_size:(jz + 1) * subvol_size]

                        save_subvol_vtrue[jc] = dfhalo_ngp_wvall_true[jx * subvol_size:(jx + 1) * subvol_size,
                                                        jy * subvol_size:(jy + 1) * subvol_size,
                                                        jz * subvol_size:(jz + 1) * subvol_size]

                        save_subvol_vpred[jc] = dfhalo_ngp_wvall_pred[jx * subvol_size:(jx + 1) * subvol_size,
                                                        jy * subvol_size:(jy + 1) * subvol_size,
                                                        jz * subvol_size:(jz + 1) * subvol_size]

                        save_subvol_vdiff[jc] = dfhalo_ngp_wvall_diff[jx * subvol_size:(jx + 1) * subvol_size,
                                                        jy * subvol_size:(jy + 1) * subvol_size,
                                                        jz * subvol_size:(jz + 1) * subvol_size]


                        jc += 1

            save_subvol_Nhalo = sparse.COO(save_subvol_Nhalo)
            save_subvol_Mhalo = sparse.COO(save_subvol_Mhalo)
            save_subvol_nuhalo = sparse.COO(save_subvol_nuhalo)
            save_subvol_concsimhalos = sparse.COO(save_subvol_concsimhalos)
            save_subvol_concfunchalos = sparse.COO(save_subvol_concfunchalos)
            save_subvol_concdiffhalos = sparse.COO(save_subvol_concdiffhalos)
            save_subvol_vtrue = sparse.COO(save_subvol_vtrue)
            save_subvol_vpred = sparse.COO(save_subvol_vpred)
            save_subvol_vdiff = sparse.COO(save_subvol_vdiff)
            saved_halos_subvol = {
                'N_halos': save_subvol_Nhalo,
                'M_halos': save_subvol_Mhalo,
                'nu_halos': save_subvol_nuhalo,
                'c_halos_sim': save_subvol_concsimhalos,
                'c_halos_func': save_subvol_concfunchalos,
                'c_halos_diff': save_subvol_concdiffhalos,
                'v_halos_true': save_subvol_vtrue,
                'v_halos_pred': save_subvol_vpred,
                'v_halos_diff': save_subvol_vdiff
                }    
            dill.dump(saved_halos_subvol, open(savefname_halos_subvol, 'wb'))

        if save_full:
            Nhalos = sparse.COO(Nhalos)
            M_halos = sparse.COO(M_halos)
            nu_halos = sparse.COO(nu_halos)
            conc_sim_halos = sparse.COO(conc_sim_halos)
            conc_func_halos = sparse.COO(conc_func_halos)
            conc_diff_halos = sparse.COO(conc_diff_halos)
            dfhalo_ngp_wvall_true = sparse.COO(dfhalo_ngp_wvall_true)
            dfhalo_ngp_wvall_pred = sparse.COO(dfhalo_ngp_wvall_pred)
            dfhalo_ngp_wvall_diff = sparse.COO(dfhalo_ngp_wvall_diff)
            saved_halos_full = {
                'N_halos_combined': Nhalos,
                'M_halos_combined': M_halos,
                'nu_halos_combined': nu_halos,
                'c_halos_sim_combined': conc_sim_halos,
                'c_halos_func_combined': conc_func_halos,
                'c_halos_diff_combined': conc_diff_halos,
                'v_halos_true_combined': dfhalo_ngp_wvall_true,
                'v_halos_pred_combined': dfhalo_ngp_wvall_pred,
                'v_halos_diff_combined': dfhalo_ngp_wvall_diff
                }    
            dill.dump(saved_halos_full, open(savefname_halos_full, 'wb'))




import multiprocessing as mp
do_mp = 0
isfid = 1
if __name__ == '__main__':
    n_sims_offset = 0
    n_sims = 2

    if do_mp:
        n_cores = mp.cpu_count()
        print(n_cores)

        # Create a pool of worker processes
        pool = mp.Pool(processes=n_cores)

        # Distribute the simulations across the available cores
        sims_per_core = n_sims // n_cores
        sim_ranges = [(n_sims_offset + i * sims_per_core, n_sims_offset + (i + 1) * sims_per_core) for i in range(n_cores)]

        # Handle any remaining simulations
        remaining_sims = n_sims % n_cores
        if remaining_sims > 0:
            sim_ranges[-1] = (sim_ranges[-1][0], sim_ranges[-1][1] + remaining_sims)

        # Run save_cic_densities function for each simulation range in parallel
        results = [pool.apply_async(save_halo_cats, args=(ji,)) for sim_range in sim_ranges for ji in range(*sim_range)]

        # Wait for all tasks to complete
        [result.get() for result in results]

        # Close the pool and wait for tasks to finish
        pool.close()
        pool.join()
    else:
        for ji in range(n_sims_offset, n_sims_offset + n_sims):
            save_halo_cats(ji, mass_type=mass_type, isfid=isfid, snapnum=snapnum, save_subvol=1, save_full=1, save_vel=0)
