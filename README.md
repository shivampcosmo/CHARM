# CHARM
Creating Halos with Auto-Regressive Multi-stage networks. 

## Description
The code uses multiple auto-regressive networks to jointly infer N-body Rockstar-like halo counts, masses and their velocities when conditioned on matter density and velocity fields from a particle mesh simulation. 

## To install
```
git clone https://github.com/shivampcosmo/CHARM.git
cd CHARM
pip install charm
```

## To infer the halo catalog
Follow the notebook where we assume that we have access to 3D matter density and velocity fields. Note that the resolution of the fields are fixed to 8Mpc/h, or $128^3$ for 
