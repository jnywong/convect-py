[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# convect-py

Welcome! Here is my Python code to simulate 2D convection in a Cartesian box. Theory and methods used can be found in [Introduction to Modeling Convection in Planets and Stars: Magnetic Field, Density Stratification, Rotation](https://press.princeton.edu/books/hardcover/9780691141725/introduction-to-modeling-convection-in-planets-and-stars). 

## Getting Started

### Pre-requisites
- [Python](https://www.python.org/)

## Package structure
```
convect-py
    docs/
    scripts/
      convect_nonlinear.py
    src/
        data_utils.py
        postprocess.py
        routines.py
    LICENSE.md
    README.md
```

## Examples

### Nonlinear convection

1. Open `scripts/convect_nonlinear.py`

2. Specify input parameters
   
```
# Inputs
nz = 101 # no. of vertical gridpoints
nn = 50 # no. of Fourier modes (excluding zeroth mode)
a = 3 # L/D aspect ratio
Ra = 1e6 # Rayleigh number
Pr = 0.5 # Prandtl number
dt = 3e-6 # timestep size
nt = 1e4 # no. of timesteps
nout = 1e2 # save output every nout timesteps
initOn = 1 # initialise run, otherwise load existing data
saveDir = "/Users/wongj/Documents/convect-out/2021-09-03" # save directory
```
3. Run script from terminal using `python <working directory>/convect-jl/src/convect_nonlinear.py` (or from IPython using `%run <working directory>/convect-jl/src/convect_nonlinear.py)` 

## Authors

* [**Jenny Wong**](https://jnywong.github.io/) - *University of Birmingham*
  

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

:tada:
