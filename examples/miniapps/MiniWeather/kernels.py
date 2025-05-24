from numba import cuda
import numba
from constants import *
import numpy as np
import math


@numba.jit(nopython=True)
def hydro_const_theta(z):
    """
    Establish hydrostatic balance using constant potential temperature (thermally neutral atmosphere)
    z is the input coordinate
    Returns r and t, the background hydrostatic density and potential temperature
    """
    theta0 = 300.  # Background potential temperature
    exner0 = 1.    # Surface-level Exner pressure
    # Establish hydrostatic balance first using Exner pressure
    t = theta0                                  # Potential Temperature at z
    exner = exner0 - grav * z / (cp * theta0)   # Exner pressure at z
    p = p0 * (exner**(cp/rd))                 # Pressure at z
    rt = (p / C0)**(1. / gamm)             # rho*theta at z
    r = rt / t                                  # Density at z
    return r, t

@numba.jit(nopython=True)
def hydro_const_bvfreq(z, bv_freq0):
    """
    Establish hydrostatic balance using constant Brunt-Vaisala frequency
    z is the input coordinate
    bv_freq0 is the constant Brunt-Vaisala frequency
    Returns r and t, the background hydrostatic density and potential temperature
    """
    theta0 = 300.  # Background potential temperature
    exner0 = 1.    # Surface-level Exner pressure
    t = theta0 * np.exp( bv_freq0*bv_freq0 / grav * z )                                    # Pot temp at z
    exner = exner0 - grav*grav / (cp * bv_freq0*bv_freq0) * (t - theta0) / (t * theta0) # Exner pressure at z
    p = p0 * (exner**(cp/rd))                                                         # Pressure at z
    rt = (p / C0)**(1. / gamm)                                                  # rho*theta at z
    r = rt / t                                                                          # Density at z
    return r, t

@numba.jit(nopython=True)
def sample_ellipse_cosine(x, z, amp, x0, z0, xrad, zrad):
    """
    Sample from an ellipse of a specified center, radius, and amplitude at a specified location
    x and z are input coordinates
    amp,x0,z0,xrad,zrad are input amplitude, center, and radius of the ellipse
    Returns a double.
    """
    # Compute distance from bubble center
    dist = np.sqrt( ((x-x0)/xrad)**2 + ((z-z0)/zrad)**2 ) * math.pi / 2.
    if dist <= math.pi / 2.:
        return amp * (np.cos(dist)**2.)
    else:
        return 0.

@numba.jit(nopython=True)
def injection(x, z):
    """
    This test case is initially balanced but injects fast, cold air from the left boundary near the model top
    x and z are input coordinates at which to sample
    Returns r,u,w,t (density, u-wind, w-wind, potential temperature) and hr,ht (background hydrostatic density and potential temperature)
    """
    hr, ht = hydro_const_theta(z)
    r = 0.
    t = 0.
    u = 0.
    w = 0.
    return r, u, w, t, hr, ht

@numba.jit(nopython=True)
def density_current(x, z):
    """
    Initialize a density current (falling cold thermal that propagates along the model bottom)
    x and z are input coordinates at which to sample
    Returns r,u,w,t (density, u-wind, w-wind, potential temperature) and hr,ht (background hydrostatic density and potential temperature)
    """
    hr, ht = hydro_const_theta(z)
    r = 0.
    t = 0.
    u = 0.
    w = 0.
    t = t + sample_ellipse_cosine(x,z,-20. ,xlen/2,5000.,4000.,2000.)
    return r, u, w, t, hr, ht

@numba.jit(nopython=True)
def turbulence(x, z):
    """
    x and z are input coordinates at which to sample
    Returns r,u,w,t (density, u-wind, w-wind, potential temperature) and hr,ht (background hydrostatic density and potential temperature)
    """
    hr, ht = hydro_const_theta(z)
    r = 0.
    t = 0.
    u = 0.
    w = 0.
    # call random_number(u);
    # call random_number(w)
    # u = (u_rand - 0.5) * 20.
    # w = (w_rand - 0.5) * 20.
    return r, u, w, t, hr, ht

@numba.jit(nopython=True)
def mountain_waves(x, z):
    """
    x and z are input coordinates at which to sample
    Returns r,u,w,t (density, u-wind, w-wind, potential temperature) and hr,ht (background hydrostatic density and potential temperature)
    """
    hr, ht = hydro_const_bvfreq(z,0.02)
    r = 0.
    t = 0.
    u = 15.
    w = 0.
    return r, u, w, t, hr, ht

@numba.jit(nopython=True)
def thermal(x, z):
    """
    Rising thermal
    x and z are input coordinates at which to sample
    Returns r,u,w,t (density, u-wind, w-wind, potential temperature) and hr,ht (background hydrostatic density and potential temperature)
    """
    hr, ht = hydro_const_theta(z)
    r = 0.
    t = 0.
    u = 0.
    w = 0.
    t = t + sample_ellipse_cosine(x,z, 3. ,xlen/2,2000.,2000.,2000.)
    return r, u, w, t, hr, ht

@numba.jit(nopython=True)
def collision(x, z):
    """
    Colliding thermals
    x and z are input coordinates at which to sample
    Returns r,u,w,t (density, u-wind, w-wind, potential temperature) and hr,ht (background hydrostatic density and potential temperature)
    """
    hr, ht = hydro_const_theta(z)
    r = 0.
    t = 0.
    u = 0.
    w = 0.
    t = t + sample_ellipse_cosine(x,z, 20.,xlen/2,2000.,2000.,2000.)
    t = t + sample_ellipse_cosine(x,z,-20.,xlen/2,8000.,2000.,2000.)
    return r, u, w, t, hr, ht

# End of CPU JIT functions

####################################################################################
# CUDA GPU KERNELS
####################################################################################

@cuda.jit
def compute_flux_x_kernel(state, flux, hy_dens_cell, hy_dens_theta_cell, hv_coef, nx, nz, hs):
    k_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    i_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    stencil = cuda.local.array(sten_size, dtype=numba.float64)
    d3_vals = cuda.local.array(NUM_VARS, dtype=numba.float64)
    vals = cuda.local.array(NUM_VARS, dtype=numba.float64)

    if i_idx < nx + 1 and k_idx < nz:
        for ll in range(NUM_VARS):
            for s in range(sten_size):
                stencil[s] = state[ll, k_idx + hs, i_idx + s]
            
            vals[ll] = -stencil[0]/12 + 7*stencil[1]/12 + 7*stencil[2]/12 - stencil[3]/12
            d3_vals[ll] = -stencil[0] + 3*stencil[1] - 3*stencil[2] + stencil[3]

        r_val = vals[ID_DENS] + hy_dens_cell[k_idx + hs]
        u_val = vals[ID_UMOM] / r_val
        w_val = vals[ID_WMOM] / r_val
        t_val = (vals[ID_RHOT] + hy_dens_theta_cell[k_idx + hs]) / r_val
        p_val = C0 * (r_val * t_val)**gamm

        flux[ID_DENS, k_idx, i_idx] = r_val * u_val - hv_coef * d3_vals[ID_DENS]
        flux[ID_UMOM, k_idx, i_idx] = r_val * u_val * u_val + p_val - hv_coef * d3_vals[ID_UMOM]
        flux[ID_WMOM, k_idx, i_idx] = r_val * u_val * w_val - hv_coef * d3_vals[ID_WMOM]
        flux[ID_RHOT, k_idx, i_idx] = r_val * u_val * t_val - hv_coef * d3_vals[ID_RHOT]

@cuda.jit
def compute_tend_x_kernel(flux, tend, nx, nz, grid_dx):
    ll = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    k_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    i_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if i_idx < nx and k_idx < nz and ll < NUM_VARS:
        tend[ll, k_idx, i_idx] = -(flux[ll, k_idx, i_idx + 1] - flux[ll, k_idx, i_idx]) / grid_dx

@cuda.jit
def compute_flux_z_kernel(state, flux, hy_dens_int, hy_pressure_int, hy_dens_theta_int, hv_coef, nx, nz, hs, k_beg_global, nz_global):
    k_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    i_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    stencil = cuda.local.array(sten_size, dtype=numba.float64)
    d3_vals = cuda.local.array(NUM_VARS, dtype=numba.float64)
    vals = cuda.local.array(NUM_VARS, dtype=numba.float64)

    if i_idx < nx and k_idx < nz + 1:
        for ll in range(NUM_VARS):
            for s in range(sten_size):
                stencil[s] = state[ll, k_idx + s, i_idx + hs]
            
            vals[ll] = -stencil[0]/12 + 7*stencil[1]/12 + 7*stencil[2]/12 - stencil[3]/12
            d3_vals[ll] = -stencil[0] + 3*stencil[1] - 3*stencil[2] + stencil[3]

        r_val = vals[ID_DENS] + hy_dens_int[k_idx]
        u_val = vals[ID_UMOM] / r_val
        w_val = vals[ID_WMOM] / r_val
        t_val = (vals[ID_RHOT] + hy_dens_theta_int[k_idx]) / r_val
        p_val = C0 * (r_val * t_val)**gamm - hy_pressure_int[k_idx]
        
        # Boundary conditions for w and density flux at global boundaries only
        actual_w_val = w_val
        actual_d3_dens = d3_vals[ID_DENS]

        # Check if at global boundaries
        global_k_idx = k_beg_global + k_idx
        if global_k_idx == 0 or global_k_idx == nz_global:
            actual_w_val = 0.0
            actual_d3_dens = 0.0

        flux[ID_DENS, k_idx, i_idx] = r_val * actual_w_val - hv_coef * actual_d3_dens
        flux[ID_UMOM, k_idx, i_idx] = r_val * actual_w_val * u_val - hv_coef * d3_vals[ID_UMOM]
        flux[ID_WMOM, k_idx, i_idx] = r_val * actual_w_val * actual_w_val + p_val - hv_coef * d3_vals[ID_WMOM]
        flux[ID_RHOT, k_idx, i_idx] = r_val * actual_w_val * t_val - hv_coef * d3_vals[ID_RHOT]

@cuda.jit
def compute_tend_z_kernel(state, flux, tend, nx, nz, hs, grid_dz):
    ll = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    k_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    i_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if i_idx < nx and k_idx < nz and ll < NUM_VARS:
        tend[ll, k_idx, i_idx] = -(flux[ll, k_idx + 1, i_idx] - flux[ll, k_idx, i_idx]) / grid_dz
        if ll == ID_WMOM:
            tend[ll, k_idx, i_idx] -= state[ID_DENS, k_idx + hs, i_idx + hs] * grav

@cuda.jit
def pack_send_buf_kernel(state, sendbuf_l, sendbuf_r, nx, nz, hs):
    ll = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    k_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    s_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if s_idx < hs and k_idx < nz and ll < NUM_VARS:
        sendbuf_l[ll, k_idx, s_idx] = state[ll, k_idx + hs, hs + s_idx]
        sendbuf_r[ll, k_idx, s_idx] = state[ll, k_idx + hs, nx + s_idx]

@cuda.jit
def unpack_recv_buf_kernel(state, recvbuf_l, recvbuf_r, nx, nz, hs):
    ll = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    k_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    s_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if s_idx < hs and k_idx < nz and ll < NUM_VARS:
        state[ll, k_idx + hs, s_idx] = recvbuf_l[ll, k_idx, s_idx]
        state[ll, k_idx + hs, nx + hs + s_idx] = recvbuf_r[ll, k_idx, s_idx]

@cuda.jit
def update_state_x_kernel(state, hy_dens_cell, hy_dens_theta_cell, nx, nz, hs, k_beg, grid_dz):
    k_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    i_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if i_idx < hs and k_idx < nz:
        z = (k_beg + k_idx + 0.5) * grid_dz
        if math.fabs(z - 3 * zlen / 4) <= zlen / 16:
            r_plus_hr = state[ID_DENS, k_idx + hs, i_idx] + hy_dens_cell[k_idx + hs]
            state[ID_UMOM, k_idx + hs, i_idx] = r_plus_hr * 50.0
            state[ID_RHOT, k_idx + hs, i_idx] = r_plus_hr * 298.0 - hy_dens_theta_cell[k_idx + hs]

@cuda.jit
def update_state_z_kernel(state, data_spec_int, i_beg, nx, nz, hs, grid_dx, mnt_width, k_beg_global, nz_global):
    ll = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    i_glob_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if i_glob_idx < nx + 2 * hs and ll < NUM_VARS:
        if k_beg_global == 0:
            if ll == ID_WMOM:
                state[ID_WMOM, 0, i_glob_idx] = 0.0
                state[ID_WMOM, 1, i_glob_idx] = 0.0
                
                if data_spec_int == DATA_SPEC_MOUNTAIN:
                    x = (i_beg + i_glob_idx - hs + 0.5) * grid_dx
                    if math.fabs(x - xlen / 4.0) < mnt_width:
                        xloc = (x - (xlen / 4.0)) / mnt_width
                        mnt_deriv = -pi * math.cos(pi * xloc / 2.0) * math.sin(pi * xloc / 2.0) * 10.0 / grid_dx 
                        state[ID_WMOM, 0, i_glob_idx] = mnt_deriv * state[ID_UMOM, hs, i_glob_idx]
                        state[ID_WMOM, 1, i_glob_idx] = mnt_deriv * state[ID_UMOM, hs, i_glob_idx]
            else:
                state[ll, 0, i_glob_idx] = state[ll, hs, i_glob_idx]
                state[ll, 1, i_glob_idx] = state[ll, hs, i_glob_idx]
                
        if k_beg_global + nz == nz_global:
            if ll == ID_WMOM:
                state[ID_WMOM, nz + hs, i_glob_idx] = 0.0
                state[ID_WMOM, nz + hs + 1, i_glob_idx] = 0.0
            else:
                state[ll, nz + hs, i_glob_idx] = state[ll, nz + hs - 1, i_glob_idx]
                state[ll, nz + hs + 1, i_glob_idx] = state[ll, nz + hs - 1, i_glob_idx]

@cuda.jit
def acc_mass_te_kernel(mass_arr, te_arr, state, hy_dens_cell, hy_dens_theta_cell, nx, nz, hs, grid_dx, grid_dz):
    k_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    i_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if k_idx < nz and i_idx < nx:
        r_pert  = state[ID_DENS, k_idx + hs, i_idx + hs]
        u_mom = state[ID_UMOM, k_idx + hs, i_idx + hs]
        w_mom = state[ID_WMOM, k_idx + hs, i_idx + hs]
        rhot_pert = state[ID_RHOT, k_idx + hs, i_idx + hs]

        r_full = r_pert + hy_dens_cell[hs + k_idx]
        u_vel = u_mom / r_full
        w_vel = w_mom / r_full
        th_full = (rhot_pert + hy_dens_theta_cell[hs + k_idx]) / r_full
        
        p_full = C0 * (r_full * th_full)**gamm
        t_abs = th_full / ((p0 / p_full)**(rd / cp))
        
        ke = 0.5 * r_full * (u_vel**2 + w_vel**2)
        ie = r_full * cv * t_abs

        cuda.atomic.add(mass_arr, 0, r_full * grid_dx * grid_dz)
        cuda.atomic.add(te_arr, 0, (ke + ie) * grid_dx * grid_dz)

@cuda.jit
def update_fluid_state_kernel(state_init, state_out, tend, nx, nz, hs, dt_arg):
    ll = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    k_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    i_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if i_idx < nx and k_idx < nz and ll < NUM_VARS:
        state_idx_k = k_idx + hs
        state_idx_i = i_idx + hs
        state_out[ll, state_idx_k, state_idx_i] = state_init[ll, state_idx_k, state_idx_i] + dt_arg * tend[ll, k_idx, i_idx]

@cuda.jit
def pack_send_buf_z_kernel(state, sendbuf_b, sendbuf_t, nx, nz, hs):
    ll = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    i_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    s_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if s_idx < hs and i_idx < nx and ll < NUM_VARS:
        sendbuf_b[ll, s_idx, i_idx] = state[ll, hs + s_idx, i_idx + hs]
        sendbuf_t[ll, s_idx, i_idx] = state[ll, nz + s_idx, i_idx + hs]

@cuda.jit
def unpack_recv_buf_z_kernel(state, recvbuf_b, recvbuf_t, nx, nz, hs):
    ll = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    i_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    s_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if s_idx < hs and i_idx < nx and ll < NUM_VARS:
        state[ll, s_idx, i_idx + hs] = recvbuf_b[ll, s_idx, i_idx]
        state[ll, nz + hs + s_idx, i_idx + hs] = recvbuf_t[ll, s_idx, i_idx]

@cuda.jit
def unpack_recv_buf_z_bottom_kernel(state, recvbuf_b, nx, nz, hs):
    ll = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    i_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    s_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if s_idx < hs and i_idx < nx and ll < NUM_VARS:
        state[ll, s_idx, i_idx + hs] = recvbuf_b[ll, s_idx, i_idx]

@cuda.jit
def unpack_recv_buf_z_top_kernel(state, recvbuf_t, nx, nz, hs):
    ll = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    i_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    s_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if s_idx < hs and i_idx < nx and ll < NUM_VARS:
        state[ll, nz + hs + s_idx, i_idx + hs] = recvbuf_t[ll, s_idx, i_idx]

