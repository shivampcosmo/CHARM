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

def save_halo_cats(isim, mass_type='fof'):
    snapnum = 3
    grid = 128
    BoxSize = 1000.
    z_dict = {4:0.0, 3:0.5, 2:1.0, 1:2.0, 0:3.0}
    redshift = z_dict[snapnum]
    # mass_type = 'rockstar_200c'


    if 'rockstar' in mass_type:
        # snap_dir_base = '/mnt/home/fvillaescusa/ceph/Quijote/Halos/Rockstar/latin_hypercube_HR'
        snap_dir_base = '/mnt/home/fvillaescusa/ceph/Quijote/Halos/Rockstar/fiducial_HR'        
        verbose = False   #print information on progress
        snapdir = snap_dir_base + '/' + str(isim)  #folder hosting the catalogue
        rockstar = np.loadtxt(snapdir + '/out_' + str(snapnum) + '_pid.list')
        with open(snapdir + '/out_' + str(snapnum) + '_pid.list', 'r') as f:
            lines = f.readlines()
        header = lines[0].split()
        # get the properties of the halos
        pos_h_truth = rockstar[:,header.index('X'):header.index('Z')+1]
        index_M = header.index('M200c')                    
        mass_truth = rockstar[:,index_M]  #Halo masses in Msun/h
        lgMass_truth = np.log10(mass_truth).astype(np.float32)
        vel_h_truth = rockstar[:,header.index('VX'):header.index('VZ')+1]

    if 'fof' in mass_type:
        # snap_dir_base = '/mnt/home/fvillaescusa/ceph/Quijote/Halos/FoF/latin_hypercube_HR'
        snap_dir_base = '/mnt/home/fvillaescusa/ceph/Quijote/Halos/FoF/fiducial_HR'        
        snapdir = snap_dir_base + '/' + str(isim)  #folder hosting the catalogue
        FoF = readfof.FoF_catalog(snapdir, snapnum, long_ids=False, swap=False, SFR=False, read_IDs=False)
        # get the properties of the halos
        pos_h_truth = FoF.GroupPos / 1e3  #Halo positions in Mpc/h
        mass_truth = FoF.GroupMass * 1e10  #Halo masses in Msun/h
        lgMass_truth = np.log10(mass_truth).astype(np.float32)
        vel_h_truth = FoF.GroupVel*(1.0+redshift) #Halo peculiar velocities in km/s

    Mmin_cut = 5e12
    Mmin_cut_str = '5e12'
    n_batch = 8

    # root_out = '/mnt/home/spandey/ceph/Quijote/data_NGP_self_LH'
    root_out = '/mnt/home/spandey/ceph/Quijote/data_NGP_self'    
    folder_out = '%s/%d'%(root_out,isim)
    # create output folder if it does not exists
    if not(os.path.exists(folder_out)):
        os.system('mkdir %s'%folder_out)

    z = {4:0, 3:0.5, 2:1, 1:2, 0:3, -1: 127}[snapnum]
    savefname_halos_subvol = '%s/SPARSEMATS_halos_HR_%s_lgMmincut_%s_subvol_res_%d_z=%s.pk'%(folder_out,mass_type,Mmin_cut_str,grid,z)
    savefname_halos_full = '%s/SPARSEMATS_halos_HR_%s_lgMmincut_%s_full_res_%d_z=%s.pk'%(folder_out,mass_type,Mmin_cut_str,grid,z)            


    indsel = np.where(mass_truth > Mmin_cut)[0]
    pos_h_truth = pos_h_truth[indsel]
    mass_truth = mass_truth[indsel]
    lgMass_truth = lgMass_truth[indsel]
    vel_h_truth = vel_h_truth[indsel]

    from scipy.interpolate import RegularGridInterpolator
    xall = (np.linspace(0, BoxSize, grid + 1))
    xarray = 0.5 * (xall[1:] + xall[:-1])
    yarray = np.copy(xarray)
    zarray = np.copy(xarray)

    import pickle as pk
    # vel_load_dir = '/mnt/home/spandey/ceph/Quijote/data_NGP_self_fastpm_LH/'
    vel_load_dir = '/mnt/home/spandey/ceph/Quijote/data_NGP_self/fastpm/'    
    df = pk.load(open(vel_load_dir + f'{isim}/velocity_HR_full_m_res_128_z=0.5_nbatch_8_nfilter_3_ncnn_0.pk', 'rb'))['velocity_cic_unpad_combined']
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

    dfhalo_ngp_wvx_true = np.float32(np.zeros((grid, grid, grid, nMax_h)))
    NGP_mass(np.float32(pos_h_truth), np.float32(vel_h_truth[:,0]), dfhalo_ngp_wvx_true, BoxSize)

    dfhalo_ngp_wvy_true = np.float32(np.zeros((grid, grid, grid, nMax_h)))
    NGP_mass(np.float32(pos_h_truth), np.float32(vel_h_truth[:,1]), dfhalo_ngp_wvy_true, BoxSize)

    dfhalo_ngp_wvz_true = np.float32(np.zeros((grid, grid, grid, nMax_h)))
    NGP_mass(np.float32(pos_h_truth), np.float32(vel_h_truth[:,2]), dfhalo_ngp_wvz_true, BoxSize)


    dfhalo_ngp_wvx_pred = np.float32(np.zeros((grid, grid, grid, nMax_h)))
    NGP_mass(np.float32(pos_h_truth), np.float32(vx_eval_interp_l), dfhalo_ngp_wvx_pred, BoxSize)

    dfhalo_ngp_wvy_pred = np.float32(np.zeros((grid, grid, grid, nMax_h)))
    NGP_mass(np.float32(pos_h_truth), np.float32(vy_eval_interp_l), dfhalo_ngp_wvy_pred, BoxSize)

    dfhalo_ngp_wvz_pred = np.float32(np.zeros((grid, grid, grid, nMax_h)))
    NGP_mass(np.float32(pos_h_truth), np.float32(vz_eval_interp_l), dfhalo_ngp_wvz_pred, BoxSize)

    argsort_M = np.flip(np.argsort(dfhalo_ngp_wmass, axis=-1), axis=-1)

    # M_halos = np.flip(np.sort(dfhalo_ngp_wmass, axis=-1), axis=-1)
    M_halos = np.take_along_axis(dfhalo_ngp_wmass, argsort_M, axis=-1)
    dfhalo_ngp_wvx_true = np.take_along_axis(dfhalo_ngp_wvx_true, argsort_M, axis=-1)
    dfhalo_ngp_wvy_true = np.take_along_axis(dfhalo_ngp_wvy_true, argsort_M, axis=-1)
    dfhalo_ngp_wvz_true = np.take_along_axis(dfhalo_ngp_wvz_true, argsort_M, axis=-1)
    dfhalo_ngp_wvx_pred = np.take_along_axis(dfhalo_ngp_wvx_pred, argsort_M, axis=-1)
    dfhalo_ngp_wvy_pred = np.take_along_axis(dfhalo_ngp_wvy_pred, argsort_M, axis=-1)
    dfhalo_ngp_wvz_pred = np.take_along_axis(dfhalo_ngp_wvz_pred, argsort_M, axis=-1)


    dfhalo_ngp_wvall_true = np.moveaxis(np.array([dfhalo_ngp_wvx_true, dfhalo_ngp_wvy_true, dfhalo_ngp_wvz_true]), 0, -1)
    dfhalo_ngp_wvall_pred = np.moveaxis(np.array([dfhalo_ngp_wvx_pred, dfhalo_ngp_wvy_pred, dfhalo_ngp_wvz_pred]), 0, -1)
    dfhalo_ngp_wvall_diff =  dfhalo_ngp_wvall_pred - dfhalo_ngp_wvall_true

    # now split it into nbatches each side

    subvol_size = grid // n_batch
    nsubvol = n_batch**3
    save_subvol_Nhalo = np.zeros((nsubvol, subvol_size, subvol_size, subvol_size))
    save_subvol_Mhalo = np.zeros((nsubvol, subvol_size, subvol_size, subvol_size, nMax_h))
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
    save_subvol_vtrue = sparse.COO(save_subvol_vtrue)
    save_subvol_vpred = sparse.COO(save_subvol_vpred)
    save_subvol_vdiff = sparse.COO(save_subvol_vdiff)
    saved_halos_subvol = {
        'N_halos': save_subvol_Nhalo,
        'M_halos': save_subvol_Mhalo,
        'v_halos_true': save_subvol_vtrue,
        'v_halos_pred': save_subvol_vpred,
        'v_halos_diff': save_subvol_vdiff
        }    
    dill.dump(saved_halos_subvol, open(savefname_halos_subvol, 'wb'))

    Nhalos = sparse.COO(Nhalos)
    M_halos = sparse.COO(M_halos)
    dfhalo_ngp_wvall_true = sparse.COO(dfhalo_ngp_wvall_true)
    dfhalo_ngp_wvall_pred = sparse.COO(dfhalo_ngp_wvall_pred)
    dfhalo_ngp_wvall_diff = sparse.COO(dfhalo_ngp_wvall_diff)
    saved_halos_full = {
        'N_halos_combined': Nhalos,
        'M_halos_combined': M_halos,
        'v_halos_true_combined': dfhalo_ngp_wvall_true,
        'v_halos_pred_combined': dfhalo_ngp_wvall_pred,
        'v_halos_diff_combined': dfhalo_ngp_wvall_diff
        }    
    dill.dump(saved_halos_full, open(savefname_halos_full, 'wb'))



import multiprocessing as mp
if __name__ == '__main__':
    for ji in range(10):
        save_halo_cats(ji)
