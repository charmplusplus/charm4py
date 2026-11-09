import numpy as np

pi        = 3.14159265358979323846264338327;   #Pi
grav      = 9.8;                               #Gravitational acceleration (m / s^2)
cp        = 1004.;                             #Specific heat of dry air at constant pressure
cv        = 717.;                              #Specific heat of dry air at constant volume
rd        = 287.;                              #Dry air constant for equation of state (P=rho*rd*T)
p0        = 1.e5;                              #Standard pressure at the surface in Pascals
C0        = 27.5629410929725921310572974482;   #Constant to translate potential temperature into pressure (P=C0*(rho*theta)**gamma)
gamm      = 1.40027894002789400278940027894;   #gamma=cp/Rd , have to call this gamm because "gamma" is taken (I hate C so much)
#Define domain and stability-related constants
xlen      = 2.e4;    #Length of the domain in the x-direction (meters)
zlen      = 1.e4;    #Length of the domain in the z-direction (meters)
hv_beta   = 0.25;     #How strong to diffuse the solution: hv_beta \in [0:1]
cfl       = 1.50;    #"Courant, Friedrichs, Lewy" number (for numerical stability)
max_speed = 450;        #Assumed maximum wave speed during the simulation (speed of sound + speed of wind) (meter / sec)
hs        = 2;          #"Halo" size: number of cells beyond the MPI tasks's domain needed for a full "stencil" of information for reconstruction
sten_size = 4;          #Size of the stencil used for interpolation

# Parameters for indexing and flags
NUM_VARS = 4;           #Number of fluid state variables
ID_DENS  = 0;           #index for density ("rho")
ID_UMOM  = 1;           #index for momentum in the x-direction ("rho * u")
ID_WMOM  = 2;           #index for momentum in the z-direction ("rho * w")
ID_RHOT  = 3;           #index for density * potential temperature ("rho * theta")
DIR_X = 1;              #Integer constant to express that this operation is in the x-direction
DIR_Z = 2;              #Integer constant to express that this operation is in the z-direction
DATA_SPEC_COLLISION       = 1;
DATA_SPEC_THERMAL         = 2;
DATA_SPEC_MOUNTAIN        = 3;
DATA_SPEC_TURBULENCE      = 4;
DATA_SPEC_DENSITY_CURRENT = 5;
DATA_SPEC_INJECTION       = 6;

nqpoints = 3;
qpoints = np.array([0.112701665379258311482073460022E0 , 0.500000000000000000000000000000E0 , 0.887298334620741688517926539980E0], dtype=np.float64)
qweights = np.array([0.277777777777777777777777777779E0 , 0.444444444444444444444444444444E0 , 0.277777777777777777777777777779E0], dtype=np.float64)