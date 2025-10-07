# KN1D
KN1D is a 1D-space, 2-D velocity neutral kinetic code developed by B. LaBombard (MIT).
This repo contains the updated python version, KN1DPy, of the original KN1D code.
Contact: njbrown@wm.edu

## NOTE
This translation is still in development and not fully tested. Certain functionality and accuracy may still be missing.

## Requirements
All dependencies are located in requirements.txt. To install, run the following in the terminal:
```
pip install -r requirements.txt
```

## Limitations
Currently, anything using the Johnson-Hinov Tables are not working.
This includes Lyman_Alpha and Balmer Alpha, which will return 0 for the moment.
As such, the default choice for ionization coefficients has been set to Collrad Ionization.

There are also various other features that are currently not implemented.
These may be added later once the core program is completed.

## Configuration File
The file config.json is used to handle several settings

### Kinetic_H

- mesh_size - sets the size of the mesh generated for the kinetic_h calculations
- ion_rate - sets the method with which kinetic_h will perform ionization rate calculation
    - 'collrad' for collrad ionization
    - 'jh' for johnson-hinov ionization
    - otherwise KN1DPy will use the Janev coefficents


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
