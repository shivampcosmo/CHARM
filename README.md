# CHARM
Creating Halos with Auto-Regressive Multi-stage networks. 

## Description
The code uses multiple auto-regressive networks to jointly infer N-body Rockstar-like halo counts, masses and their velocities when conditioned on matter density and velocity fields from a particle mesh simulation. 

## To install
```
git clone https://github.com/shivampcosmo/CHARM.git
cd CHARM
python setup.py install
```

## To process the matter fields
To pre-process your matter fields to predict the halo catalogs, one can use the code in ```CHARM/prep_data/process_density_NGP_fastpm.py``` and ```CHARM/prep_data/process_velocity_NGP_fastpm.py``` to process the matter density and velocity fields (and save 3D cubes for CNN to extract the features). This is all you need to call a trained CHARM model (provided with the repo at ```CHARM/charm/trained_models```) that will output the halo positions, masses and velocities. A sample data from one of the Quijote LH set (for LH_id=3) is provided at https://www.dropbox.com/scl/fo/bncvxvm1sc5zo5klgzmb0/AAKEPut9c-t7X8rbABH1Glw?rlkey=ilpo3rftdy9ekn06namqc5be9&st=pmqyzd2d&dl=0 . 

## To infer the halo catalog
Follow the notebook at ```CHARM/notebooks/example_get_mock_halos.ipynb``` to get the halo catalogs from trained CHARM model, where we assume that we have access to processed matter density and velocity fields from PM simulation. 


