
from src.routines import zdomain, preallocate_spec, nonlinear_solver

from datetime import datetime
date_today = datetime.today().strftime('%Y-%m-%d')
startTime = datetime.now()

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
saveDir = "/Users/wongj/Documents/work/convect-py-out/"+date_today # save directory

# Vertical domain
z, dz = zdomain(nz)

# Pre-allocate arrays for spectral coefficients
psi, tem, omg = preallocate_spec(nz,nn)

# Time integration
dtemdt, domgdt, tem, omg, psi = nonlinear_solver(z, dz, nz, nn, nt, nout, dt, a, Ra, Pr, psi, tem, omg, initOn, saveDir)

print("Code took {} to execute".format(datetime.now() - startTime))
