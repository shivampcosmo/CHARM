import h5py as h5
import dill
import sys, os 
import numpy as np 
import nbodykit.lab as NBlab
from astropy.utils.misc import NumpyRNGContext
from nbodykit.hod import Zheng07Model, HODModel
import pickle as pk
from pmesh.pm import ParticleMesh, RealField
from nbodykit.lab import FFTPower, ProjectedFFTPower, ArrayMesh
import Pk_library as PKL
import MAS_library as MASL
import galactic_wavelets as gw
import torch
import warnings
warnings.filterwarnings("ignore")

def setup_hod(hmass, nbar=4e-4, satfrac=0.2, bs=1000, alpha_fid=0.7):
    numdhalos = len(hmass)/bs**3
    numhalos_nbarf = int(nbar * bs**3 * (1-satfrac))
    mcut = hmass[:numhalos_nbarf][-1]
    nsat = satfrac * nbar * bs**3
    mdiff = (hmass - mcut + mcut*1e-3)[:numhalos_nbarf] ** alpha_fid
    msum = mdiff.sum()/nsat
    m1 = msum**(1/alpha_fid)
    mcut = 10**(np.log10(mcut) + 0.1) 
    return mcut, m1

def sample_conditional_HOD(mcut, m1=None, seed=0): 
    ''' sample HOD value based on priors set by Parejko+(2013)
    centrals: 0.5*[1+erf((\log M_h - \log M_cut)/\sigma)]
    satellites: ((M_h - M_0)/M_1)**\alpha
    '''
    np.random.seed(seed)
    m0 = mcut
    if m1 is None: m1 = mcut + 0.5
    hod = np.array([mcut, 0.4, m0, m1, 0.7])
    dhod = np.array([0.15, 0.1, 0.2, 0.3, 0.3])
    #dhod = np.array([0.2, 0.1, 0.5, 0.4, 0.4, 0.5, 0.5])
    _hod = hod + dhod * np.random.uniform(-1, 1, size=(5))
    theta_hod = {'logMmin': _hod[0], 'sigma_logM': _hod[1], 'logM0': _hod[2], 'logM1': _hod[3], 'alpha': _hod[4]}
    return theta_hod


def get_halo_cats(isim, z=0.5, boxsize=1000., ldir = '/mnt/home/spandey/ceph/CHARM/data/halo_cats_charm_truth_nsubv_vel_10k/', LH_cosmo_val_file='/mnt/home/spandey/ceph/Quijote/latin_hypercube_params.txt'):
    
    LH_cosmo_val_all = np.loadtxt(LH_cosmo_val_file)

    cosmo_params = LH_cosmo_val_all[isim]
    Om, Ob, h, ns, s8 = cosmo_params
    params = {'flat': True, 'H0': 100*h, 'Om0': Om, 'Ob0': Ob, 'sigma8': s8, 'ns': ns}
    cosmo_nb = NBlab.cosmology.Planck15.clone(
                h=params['H0']/100., 
                Omega0_b=params['Ob0'], 
                Omega0_cdm=params['Om0']-params['Ob0'],
                n_s=params['ns']) 
    Ol = 1 - Om
    Hz = 100.0 * np.sqrt(Om * (1. + z)**3 + Ol) # km/s/(Mpc/h)


    saved = pk.load(open(ldir + f'halo_cat_pos_vel_LH_{isim}.pk', 'rb'))
    pos_h_truth = saved['pos_truth']
    lgMass_truth = saved['lgmass_truth']
    vel_h_truth = saved['vel_truth']
    pos_h_mock = saved['pos_mock']
    lgMass_mock = saved['lgmass_mock']
    vel_h_mock = saved['vel_mock']

    pos_types = ['mock', 'truth']
    halos_cats = []
    for pos in pos_types:
        if pos == 'mock':
            real_pos = pos_h_mock
            lgmass = lgMass_mock
            halo_vel = vel_h_mock
        else:
            real_pos = pos_h_truth
            lgmass = lgMass_truth
            halo_vel = vel_h_truth
        group_data = {}  
        group_data['Length']    = np.ones(len(real_pos)) * len(real_pos)
        group_data['Position']  = real_pos
        group_data['Velocity']  = halo_vel
        group_data['Mass']      = 10**lgmass
        # calculate velocity offset
        rsd_factor = (1. + z) / Hz
        group_data['VelocityOffset'] = halo_vel * rsd_factor
        # save to ArryCatalog for consistency
        cat = NBlab.ArrayCatalog(group_data, BoxSize=np.array([boxsize, boxsize, boxsize])) 
        cat['Length'] = len(cat)
        cat.attrs['rsd_factor'] = rsd_factor 

        cat.attrs['Om'] = Om
        cat.attrs['Ob'] = Ob
        cat.attrs['Ol'] = Ol
        cat.attrs['h'] = h 
        cat.attrs['ns'] = ns
        cat.attrs['s8'] = s8
        cat.attrs['Hz'] = Hz # km/s/(Mpc/h)   
        halos = NBlab.HaloCatalog(cat, cosmo=cosmo_nb, redshift=0.5, mdef='vir')     
        halos_cats.append(halos)
    return halos_cats[0], halos_cats[1]

def get_theta_hod(halos, isim, ihod, seed=0, nbar=4e-4, sat_frac=0.2, alpha_sat_fid=0.7, BoxSize=1000.0, LH_file_all = '/mnt/home/spandey/ceph/CHARM/data/LH_points_HOD_cosmo_np_20000.txt'):
    hmass = halos['Mass']
    hmass_sort = np.flip(np.sort(hmass))
    M_min, M1 = setup_hod(hmass_sort, nbar=nbar, satfrac=sat_frac, bs=BoxSize, alpha_fid=alpha_sat_fid)
    M0 = M_min    
    LH_points = np.loadtxt(LH_file_all)
    delta_hod = LH_points[isim + 2000*ihod][:5]
    cosmo_params = LH_points[isim + 2000*ihod][5:]
    DlogMmin, Dsigma_logM, DlogM0, DlogM1, Dalpha = delta_hod
    theta_hod = {'logMmin': np.log10(M_min) + DlogMmin, 'sigma_logM': 0.4 + Dsigma_logM, 'logM0': np.log10(M0) + DlogM0, 
                'logM1': np.log10(M1) + DlogM1, 'alpha': alpha_sat_fid + Dalpha}
    Om, Ob, h, ns, s8, LH_id_cosmo = cosmo_params
    theta_cosmo = {'Om': Om, 'Ob': Ob, 'h': h, 'ns': ns, 'sigma8': s8, 'LH_id_cosmo': LH_id_cosmo}
    return theta_hod, theta_cosmo

def get_gal_cats(halos, theta_hod, boxsize=1000., seed=0):
    hod = halos.populate(Zheng07Model, seed=seed, **theta_hod)
    gtype = hod['gal_type'].compute()
    galsum = {}
    galsum['total'], galsum['number density'] = gtype.size, gtype.size/boxsize**3
    galsum['centrals'], galsum['satellites'] = np.unique(gtype, return_counts=True)[1]
    galsum['fsat'] = galsum['satellites']/galsum['total']
    return hod, galsum

def get_gal_mesh_Pk(gal_hod=None, mesh=None, pos_type='rsd', sigma_pos=12.0, boxsize=1000., los=[1,0,0], grid=128, MAS='NGP', compensated=False, return_Pk=True):
    if mesh is None:
        if pos_type == 'rsd':
            pos = gal_hod['Position'] + gal_hod['VelocityOffset']*los
        else:
            pos = gal_hod['Position']

        pos = (pos.compute()).astype(np.float32)

        if sigma_pos > 0.0:
            sigma_1d = sigma_pos/np.sqrt(3)
            randg = np.random.randn(len(pos),3)*sigma_1d
            pos += randg

        pos = pos % boxsize

        mesh = np.zeros((grid, grid, grid), dtype=np.float32)
        MASL.MA(pos, mesh, boxsize, MAS)
        mesh /= np.mean(mesh, dtype=np.float32)
        mesh -= 1.0
    if return_Pk:
        if compensated:
            Pk = PKL.Pk(mesh, boxsize, axis=0, MAS=MAS)
        else:
            Pk = PKL.Pk(mesh, boxsize, axis=0, MAS=None)
        return Pk, mesh
    else:
        return mesh

def get_wavelets_all_hods(isim, nhod_LH_samp=10, sigma_pos=12.0, nk=12, kmax=0.32, grid=128):
    saved_j = {'isim': isim, 'sigma_pos': sigma_pos, 'nk': nk, 'kmax': kmax}
    halos_mock, halos_truth = get_halo_cats(isim)

    # mesh_mock_all = np.zeros((nhod_LH_samp, grid, grid, grid), dtype=np.float32)
    # mesh_truth_all = np.zeros((nhod_LH_samp, grid, grid, grid), dtype=np.float32)

    # device = "cpu"      
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    # J = 3
    # Q = 4
    # kc = 2*np.pi/3 # Cutoff frequency of the mother wavelet (in px^-1 units)
    # moments = [1/2, 1, 2]

    J = 3
    Q = 2
    kc = 2*np.pi*kmax/3 # Cutoff frequency of the mother wavelet (in px^-1 units)
    moments = [1/2, 1, 2]


    # df = torch.from_numpy(mesh_truth).to(device)
    # df_shape = df.shape
    # N = df_shape[0]
    wst_op = gw.ScatteringOp(torch.Size([grid, grid, grid]), J, Q,
                            moments=moments,
                            kc=kc,
                            scattering=True,
                            device=device,
                            use_mp=False)

    for ihod in range(nhod_LH_samp):
        theta_hod, theta_cosmo = get_theta_hod(halos_mock, isim, ihod=ihod)
        saved_j[f'theta_hod_{ihod}'] = theta_hod
        saved_j[f'theta_cosmo_{ihod}'] = theta_cosmo
        hod_mock, galsum_mock = get_gal_cats(halos_mock, theta_hod)
        saved_j[f'galsum_mock_{ihod}'] = galsum_mock
        hod_truth, galsum_truth = get_gal_cats(halos_truth, theta_hod)
        saved_j[f'galsum_truth_{ihod}'] = galsum_truth


        # Pk_mock, mesh_mock = get_gal_Pk(gal_hod=hod_mock, sigma_pos=sigma_pos)
        # Pk_truth, mesh_truth = get_gal_Pk(gal_hod=hod_truth, sigma_pos=sigma_pos)
        mesh_mock = get_gal_mesh_Pk(gal_hod=hod_mock, grid=grid, sigma_pos=sigma_pos, return_Pk=False)
        mesh_truth = get_gal_mesh_Pk(gal_hod=hod_truth, grid=grid, sigma_pos=sigma_pos, return_Pk=False)

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        df = torch.from_numpy(mesh_truth).to(device)
        s0_truth, s1_truth, s2_truth = wst_op(df)

        s0_truth = s0_truth.cpu().numpy()
        s1_truth = s1_truth.cpu().numpy().flatten()
        s2_truth = s2_truth.cpu().numpy().flatten()

        saved_j[f'rsd_s0_truth_{ihod}'] = s0_truth
        saved_j[f'rsd_s1_truth_{ihod}'] = s1_truth
        saved_j[f'rsd_s2_truth_{ihod}'] = s2_truth


        df = torch.from_numpy(mesh_mock).to(device)
        # df_shape = df.shape
        # N = df_shape[0]
        # wst_op = gw.ScatteringOp(df_shape, J, Q,
        #                         moments=moments,
        #                         kc=kc,
        #                         scattering=True,
        #                         device=device)
        s0_mock, s1_mock, s2_mock = wst_op(df)

        s0_mock = s0_mock.cpu().numpy()
        s1_mock = s1_mock.cpu().numpy().flatten()
        s2_mock = s2_mock.cpu().numpy().flatten()

        saved_j[f'rsd_s0_mock_{ihod}'] = s0_mock
        saved_j[f'rsd_s1_mock_{ihod}'] = s1_mock
        saved_j[f'rsd_s2_mock_{ihod}'] = s2_mock

    dill.dump(saved_j, open(f'/mnt/home/spandey/ceph/CHARM/data/summary_stats_galaxies_sigpos_{int(sigma_pos)}/wavelets_simbigsettings_galaxy_LH_{isim}.dill', 'wb'))
    # dill.dump(saved_j, open(f'/mnt/home/spandey/ceph/CHARM/data/temp/wavelets_simbigsettings_galaxy_LH_{isim}.dill', 'wb'))    

    return 




import multiprocessing as mp
if __name__ == '__main__':
    n1 = int(sys.argv[1])
    n2 = int(sys.argv[2])
    nprocs = int(sys.argv[3])

    # for ji in range(n1, n2):
    #     get_wavelets_all_hods(ji)

    n_sims = n2 - n1
    n_sims_offset = n1
    # n_cores = mp.cpu_count()
    n_cores = nprocs
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
    results = [pool.apply_async(get_wavelets_all_hods, args=(ji,)) for sim_range in sim_ranges for ji in range(*sim_range)]

    # Wait for all tasks to complete
    [result.get() for result in results]

    # Close the pool and wait for tasks to finish
    pool.close()
    pool.join()