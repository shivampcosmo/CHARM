boxsize = 2000
nc = 1024
B = 2
T = 20
random_seed = 1000
prefix = "/mnt/ceph/users/cmodi/fastpm-shivam/fid_2Gpc/0/"
read_lineark = prefix .. "/linear/"


----------------------------------------
--- This file needs to be concatenated with parameters from run.py ---
----------------------------------------
-------- Time Sequence -------- 
-- linspace: Uniform time steps in a
-- time_step = linspace(0.025, 1.0, 39)
-- logspace: Uniform time steps in loga

time_step = linspace(0.01, 1.0, T)
output_redshifts= {0.5, 99.0}  -- redshifts of output


----------------------------------------
-------- Cosmology -------- 
Omega_m = 0.31750
h = 0.67110
-- h         = 0.6711

-- Start with a power spectrum file
-- Initial power spectrum: k P(k) in Mpc/h units
-- Must be compatible with the Cosmology parameter

-- read_powerspectrum= prefix .. "/Pk_mm_z=0.000.txt"
read_powerspectrum= "/mnt/home/spandey/ceph/fastpm/fiducial_HR/CAMB_TABLES/CAMB_matterpow_0.dat"
linear_density_redshift = 0.0 -- the redshift of the linear density field.

-- remove_cosmic_variance = true


----------------------------------------
-------- Approximation Method -------- 

force_mode = "fastpm"
pm_nc_factor = B            -- Particle Mesh grid pm_nc_factor*nc per dimension in the beginning
np_alloc_factor= 2.2      -- Amount of memory allocated for particle


----------------------------------------
-------- Output -------- 

-- prefix = "/"
-- Dark matter particle outputs (all particles)
write_snapshot= prefix .. "/fastpm_B2"
write_lineark= prefix .. "/linear_B2"
particle_fraction = 1.00

write_fof     = prefix .. "/fof_B2"
fof_linkinglength = 0.200
fof_nmin = 16

-- 1d power spectrum (raw), without shotnoise correction
write_powerspectrum = prefix .. '/powerspec'

