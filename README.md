# KN1D
KN1D is a 1D-space, 2-D velocity neutral kinetic code developed by B. LaBombard (MIT).
This repo contains a modified version of the Python translation KN1DPy designed to be fully automatically differentiable with PyTorch.
Contact: njbrown@wm.edu

## NOTE
This progam is still in development and not fully working.

## Requirements
This translation was developed in python 3.12.3. 

All dependencies are located in requirements.txt. To install, run the following in the terminal:
```
pip install -r requirements.txt
```


## Configuration File
The file config.json is used to handle several settings

### Kinetic_H

- mesh_size - sets the size of the mesh generated for the kinetic_h calculations
- ion_rate - sets the method with which kinetic_h will perform ionization rate calculation
    - 'collrad' to use collrad ionization
    - 'jh' to use johnson-hinov ionization
    - 'janev' to use janev coefficients
    - KN1DPy will throw an exception if this value is not set to one of these three


### Kinetic_H2

- mesh_size - sets the size of the mesh generated for the kinetic_h2 calculations


### Collisions

- H2_H2_EL	- if set, then include H2 -> H2 elastic self collisions
- H2_P_EL	- if set, then include H2 -> H(+) elastic collisions
- H2_H_EL	- if set, then include H2 <-> H elastic collisions
- H2_HP_CX	- if set, then include H2 -> H2(+) charge exchange collisions
- H_H_EL	- if set, then include H -> H elastic self collisions
- H_P_CX	- if set, then include H -> H(+) charge exchange collisions
- H_P_EL	- if set, then include H -> H(+) elastic collisions
- SIMPLE_CX	- if set, then use CX source option (B): Neutrals are born
              in velocity with a distribution proportional to the local
              ion distribution function.