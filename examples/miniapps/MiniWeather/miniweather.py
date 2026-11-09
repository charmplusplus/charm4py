import argparse
import numpy as np
import math
import time
import os
from numba import cuda
from charm4py import charm, Chare, Array, Future, Reducer, coro, Channel

from constants import (
    pi, grav, cp, cv, rd, p0, C0, gamm, xlen, zlen, hv_beta, cfl, max_speed, hs,
    sten_size, NUM_VARS, ID_DENS, ID_UMOM, ID_WMOM, ID_RHOT, DIR_X, DIR_Z,
    DATA_SPEC_COLLISION, DATA_SPEC_THERMAL, DATA_SPEC_MOUNTAIN,
    DATA_SPEC_TURBULENCE, DATA_SPEC_DENSITY_CURRENT, DATA_SPEC_INJECTION,
    nqpoints, qpoints, qweights
)
from kernels import (
    hydro_const_theta, hydro_const_bvfreq, sample_ellipse_cosine,
    collision as collision_init, thermal as thermal_init, mountain_waves as mountain_waves_init, 
    turbulence as turbulence_init, density_current as density_current_init, injection as injection_init,
    compute_flux_x_kernel, compute_tend_x_kernel,
    compute_flux_z_kernel, compute_tend_z_kernel,
    pack_send_buf_kernel, unpack_recv_buf_kernel,
    pack_send_buf_z_kernel, unpack_recv_buf_z_kernel,
    unpack_recv_buf_z_bottom_kernel, unpack_recv_buf_z_top_kernel,
    update_state_x_kernel, update_state_z_kernel,
    acc_mass_te_kernel, update_fluid_state_kernel
)

# Helper for domain decomposition
def calculate_domain_decomposition_x(chare_idx, num_chares_x, nx_glob):
    nx_local_base = nx_glob // num_chares_x
    remainder = nx_glob % num_chares_x
    if remainder:
        raise ValueError("nx_glob must be divisible by num_chares_x")
    return nx_local_base, chare_idx * nx_local_base

def calculate_domain_decomposition_z(chare_idx, num_chares_z, nz_glob):
    nz_local_base = nz_glob // num_chares_z
    remainder = nz_glob % num_chares_z
    if remainder:
        raise ValueError("nz_glob must be divisible by num_chares_z")
    return nz_local_base, chare_idx * nz_local_base

class MiniWeatherChare(Chare):
    def __init__(self, args):
        args_dict               = args[0]
        num_chares_x_in         = args[1]
        num_chares_z_in         = args[2]
        global_nx_in            = args[3]
        global_nz_in            = args[4]
        data_spec_int_in        = args[5]
        dt_in                   = args[6]
        initial_etime_in        = args[7]

        self.args = argparse.Namespace(**args_dict)
        # For 2D chare array, thisIndex is a tuple (i, j)
        self.chare_idx_x = self.thisIndex[0]
        self.chare_idx_z = self.thisIndex[1]
        self.num_chares_x = num_chares_x_in
        self.num_chares_z = num_chares_z_in
        
        self.nx_glob = global_nx_in
        self.nz_glob = global_nz_in
        self.data_spec_int = data_spec_int_in
        self.dt = dt_in 
        self.etime = initial_etime_in

        self.grid_dx = xlen / self.nx_glob
        self.grid_dz = zlen / self.nz_glob

        # These will be set in setup_chare_domain
        self.nx = 0
        self.nz = 0
        self.i_beg_global_idx = 0
        self.k_beg_global_idx = 0

        self.state_host = None
        self.hy_dens_cell_host = None
        self.hy_dens_theta_cell_host = None
        self.hy_dens_int_host = None
        self.hy_dens_theta_int_host = None
        self.hy_pressure_int_host = None

        self.d_state = None
        self.d_state_tmp = None
        self.d_flux = None
        self.d_tend = None
        self.d_hy_dens_cell = None
        self.d_hy_dens_theta_cell = None
        self.d_hy_dens_int = None
        self.d_hy_dens_theta_int = None
        self.d_hy_pressure_int = None
        self.d_sendbuf_l = None
        self.d_sendbuf_r = None
        self.d_recvbuf_l = None
        self.d_recvbuf_r = None
        self.d_sendbuf_b = None
        self.d_sendbuf_t = None
        self.d_recvbuf_b = None
        self.d_recvbuf_t = None
        
        self._direction_switch = True

        # Channel-based communication attributes
        self.left_channel = None
        self.right_channel = None
        self.bottom_channel = None
        self.top_channel = None

        # For initial reductions
        self.initial_mass_val = 0.0
        self.initial_te_val = 0.0

        self.setup_channels()
        
        if charm.myPe() == 0 and self.chare_idx_x == 0 and self.chare_idx_z == 0:
            print(f"Chare {self.chare_idx_x}, {self.chare_idx_z} initialized on PE {charm.myPe()}")

    def setup_channels(self):
        left_proxy_idx_x = (self.chare_idx_x - 1 + self.num_chares_x) % self.num_chares_x
        left_neighbor_proxy = self.thisProxy[left_proxy_idx_x, self.chare_idx_z]
        self.left_channel = Channel(self, remote=left_neighbor_proxy)

        right_proxy_idx_x = (self.chare_idx_x + 1) % self.num_chares_x
        right_neighbor_proxy = self.thisProxy[right_proxy_idx_x, self.chare_idx_z]
        self.right_channel = Channel(self, remote=right_neighbor_proxy)

        if self.chare_idx_z > 0:
            bottom_proxy_idx_z = self.chare_idx_z - 1
            bottom_neighbor_proxy = self.thisProxy[self.chare_idx_x, bottom_proxy_idx_z]
            self.bottom_channel = Channel(self, remote=bottom_neighbor_proxy)
        else:
            self.bottom_channel = None

        if self.chare_idx_z < self.num_chares_z - 1:
            top_proxy_idx_z = self.chare_idx_z + 1
            top_neighbor_proxy = self.thisProxy[self.chare_idx_x, top_proxy_idx_z]
            self.top_channel = Channel(self, remote=top_neighbor_proxy)
        else:
            self.top_channel = None

    def setup_chare_domain(self, local_nx, i_beg_global, local_nz, k_beg_global, setup_done_future):
        self.nx = local_nx
        self.nz = local_nz
        self.i_beg_global_idx = i_beg_global
        self.k_beg_global_idx = k_beg_global

        self.state_host = np.zeros((NUM_VARS, self.nz + 2 * hs, self.nx + 2 * hs), dtype=np.float64)
        self.hy_dens_cell_host = np.zeros(self.nz + 2 * hs, dtype=np.float64)
        self.hy_dens_theta_cell_host = np.zeros(self.nz + 2 * hs, dtype=np.float64)
        self.hy_dens_int_host = np.zeros(self.nz + 1, dtype=np.float64)
        self.hy_dens_theta_int_host = np.zeros(self.nz + 1, dtype=np.float64)
        self.hy_pressure_int_host = np.zeros(self.nz + 1, dtype=np.float64)

        problem_init_map = {
            DATA_SPEC_COLLISION: collision_init, DATA_SPEC_THERMAL: thermal_init,
            DATA_SPEC_MOUNTAIN: mountain_waves_init, DATA_SPEC_TURBULENCE: turbulence_init,
            DATA_SPEC_DENSITY_CURRENT: density_current_init, DATA_SPEC_INJECTION: injection_init,
        }
        init_routine = problem_init_map[self.data_spec_int]

        for k_loop_idx in range(self.nz + 2 * hs):
            for i_loop_idx in range(self.nx + 2 * hs):
                for kk_quad in range(nqpoints):
                    for ii_quad in range(nqpoints):
                        x_glob = (self.i_beg_global_idx + i_loop_idx - hs + 0.5) * self.grid_dx + (qpoints[ii_quad] - 0.5) * self.grid_dx
                        z_glob = (self.k_beg_global_idx + k_loop_idx - hs + 0.5) * self.grid_dz + (qpoints[kk_quad] - 0.5) * self.grid_dz
                        
                        r, u, w, t, hr, ht = init_routine(x_glob, z_glob)

                        self.state_host[ID_DENS, k_loop_idx, i_loop_idx] += r * qweights[ii_quad] * qweights[kk_quad]
                        self.state_host[ID_UMOM, k_loop_idx, i_loop_idx] += (r + hr) * u * qweights[ii_quad] * qweights[kk_quad]
                        self.state_host[ID_WMOM, k_loop_idx, i_loop_idx] += (r + hr) * w * qweights[ii_quad] * qweights[kk_quad]
                        self.state_host[ID_RHOT, k_loop_idx, i_loop_idx] += ((r + hr) * (t + ht) - hr * ht) * qweights[ii_quad] * qweights[kk_quad]
        
        for k_loop_idx in range(self.nz + 2 * hs):
            for kk_quad in range(nqpoints):
                z_quad_hydro = (self.k_beg_global_idx + k_loop_idx - hs + 0.5) * self.grid_dz + (qpoints[kk_quad] - 0.5) * self.grid_dz
                _r, _u, _w, _t, hr, ht = init_routine(0.0, z_quad_hydro)
                self.hy_dens_cell_host[k_loop_idx] += hr * qweights[kk_quad]
                self.hy_dens_theta_cell_host[k_loop_idx] += hr * ht * qweights[kk_quad]

        for k_loop_idx in range(self.nz + 1):
            z_interface = (self.k_beg_global_idx + k_loop_idx) * self.grid_dz
            _r, _u, _w, _t, hr, ht = init_routine(0.0, z_interface)
            self.hy_dens_int_host[k_loop_idx] = hr
            self.hy_dens_theta_int_host[k_loop_idx] = hr * ht
            self.hy_pressure_int_host[k_loop_idx] = C0 * ((hr * ht)**gamm)

        self.d_state = cuda.to_device(self.state_host)
        self.d_state_tmp = cuda.to_device(self.state_host)
        self.d_hy_dens_cell = cuda.to_device(self.hy_dens_cell_host)
        self.d_hy_dens_theta_cell = cuda.to_device(self.hy_dens_theta_cell_host)
        self.d_hy_dens_int = cuda.to_device(self.hy_dens_int_host)
        self.d_hy_dens_theta_int = cuda.to_device(self.hy_dens_theta_int_host)
        self.d_hy_pressure_int = cuda.to_device(self.hy_pressure_int_host)

        flux_shape = (NUM_VARS, self.nz + 1, self.nx + 1)
        tend_shape = (NUM_VARS, self.nz, self.nx)
        self.d_flux = cuda.device_array(shape=flux_shape, dtype=np.float64)
        self.d_tend = cuda.device_array(shape=tend_shape, dtype=np.float64)

        sendrecv_shape = (NUM_VARS, self.nz, hs) 
        self.d_sendbuf_l = cuda.device_array(shape=sendrecv_shape, dtype=np.float64)
        self.d_sendbuf_r = cuda.device_array(shape=sendrecv_shape, dtype=np.float64)
        self.d_recvbuf_l = cuda.device_array(shape=sendrecv_shape, dtype=np.float64)
        self.d_recvbuf_r = cuda.device_array(shape=sendrecv_shape, dtype=np.float64)

        sendrecv_shape_z = (NUM_VARS, hs, self.nx)
        self.d_sendbuf_b = cuda.device_array(shape=sendrecv_shape_z, dtype=np.float64)
        self.d_sendbuf_t = cuda.device_array(shape=sendrecv_shape_z, dtype=np.float64)
        self.d_recvbuf_b = cuda.device_array(shape=sendrecv_shape_z, dtype=np.float64)
        self.d_recvbuf_t = cuda.device_array(shape=sendrecv_shape_z, dtype=np.float64)

        local_mass, local_te = self._reductions()
        self.initial_mass_val = local_mass
        self.initial_te_val = local_te
        
        self.reduce(setup_done_future, [local_mass, local_te], Reducer.sum)

    def _reductions(self):
        d_mass_val = cuda.to_device(np.zeros(1, dtype=np.float64))
        d_te_val = cuda.to_device(np.zeros(1, dtype=np.float64))

        threadsperblock = (16, 16, 1)
        blockspergrid = (math.ceil(self.nx / threadsperblock[0]),
                         math.ceil(self.nz / threadsperblock[1]), 
                         1)

        acc_mass_te_kernel[blockspergrid, threadsperblock](
            d_mass_val, d_te_val, self.d_state, self.d_hy_dens_cell, self.d_hy_dens_theta_cell,
            self.nx, self.nz, hs, self.grid_dx, self.grid_dz
        )
        mass_host = d_mass_val.copy_to_host()
        te_host = d_te_val.copy_to_host()
        return mass_host[0], te_host[0]

    @coro
    def _set_halo_values_x(self, d_state_forcing):
        threadsperblock_buffer = (16, 16, 1) 
        blockspergrid_buffer = (math.ceil(hs / threadsperblock_buffer[0]),
                                math.ceil(self.nz / threadsperblock_buffer[1]),
                                NUM_VARS)
        pack_send_buf_kernel[blockspergrid_buffer, threadsperblock_buffer](
            d_state_forcing, self.d_sendbuf_l, self.d_sendbuf_r, self.nx, self.nz, hs
        )
        cuda.synchronize()

        data_to_send_left = self.d_sendbuf_l.copy_to_host()
        data_to_send_right = self.d_sendbuf_r.copy_to_host()

        self.left_channel.send(data_to_send_left)
        self.right_channel.send(data_to_send_right)
        
        data_for_my_d_recvbuf_l = self.left_channel.recv()
        data_for_my_d_recvbuf_r = self.right_channel.recv()
        
        self.d_recvbuf_l.copy_to_device(data_for_my_d_recvbuf_l)
        self.d_recvbuf_r.copy_to_device(data_for_my_d_recvbuf_r)

        unpack_recv_buf_kernel[blockspergrid_buffer, threadsperblock_buffer](
            d_state_forcing, self.d_recvbuf_l, self.d_recvbuf_r, self.nx, self.nz, hs
        )

        if self.data_spec_int == DATA_SPEC_INJECTION and self.chare_idx_x == 0:
            threadsperblock_inj = (16, 16, 1)
            blockspergrid_inj = (math.ceil(hs / threadsperblock_inj[0]),
                                 math.ceil(self.nz / threadsperblock_inj[1]),
                                 1)
            update_state_x_kernel[blockspergrid_inj, threadsperblock_inj](
                self.d_state, self.d_hy_dens_cell, self.d_hy_dens_theta_cell, 
                self.nx, self.nz, hs, self.k_beg_global_idx, self.grid_dz
            )
        cuda.synchronize()

    @coro
    def _set_halo_values_z(self, d_state_forcing):
        threadsperblock_buffer = (16, 16, 1) 
        blockspergrid_buffer = (math.ceil(self.nx / threadsperblock_buffer[0]),
                                math.ceil(hs / threadsperblock_buffer[1]),
                                NUM_VARS)
        pack_send_buf_z_kernel[blockspergrid_buffer, threadsperblock_buffer](
            d_state_forcing, self.d_sendbuf_b, self.d_sendbuf_t, self.nx, self.nz, hs
        )
        cuda.synchronize()

        if self.k_beg_global_idx == 0 or self.k_beg_global_idx + self.nz == self.nz_glob:
            mnt_width = xlen / 8.0
            threadsperblock_update_z = (16, 16, 1) 
            blockspergrid_x = math.ceil((self.nx + 2 * hs) / threadsperblock_update_z[0]) 
            blockspergrid_y = math.ceil(NUM_VARS / threadsperblock_update_z[1])
            blockspergrid_update_z = (blockspergrid_x, blockspergrid_y, 1)

            update_state_z_kernel[blockspergrid_update_z, threadsperblock_update_z](
                d_state_forcing, self.data_spec_int, 
                self.i_beg_global_idx, self.nx, self.nz, hs,
                self.grid_dx, mnt_width, self.k_beg_global_idx, self.nz_glob
            )
        cuda.synchronize()

        if self.bottom_channel is not None:
            data_to_send_bottom = self.d_sendbuf_b.copy_to_host()
            self.bottom_channel.send(data_to_send_bottom)
        
        if self.top_channel is not None:
            data_to_send_top = self.d_sendbuf_t.copy_to_host()
            self.top_channel.send(data_to_send_top)
        
        if self.bottom_channel is not None:
            data_for_my_d_recvbuf_b = self.bottom_channel.recv()
            self.d_recvbuf_b.copy_to_device(data_for_my_d_recvbuf_b)
        
        if self.top_channel is not None:
            data_for_my_d_recvbuf_t = self.top_channel.recv()
            self.d_recvbuf_t.copy_to_device(data_for_my_d_recvbuf_t)

        if self.bottom_channel is not None and self.top_channel is not None:
            unpack_recv_buf_z_kernel[blockspergrid_buffer, threadsperblock_buffer](
                d_state_forcing, self.d_recvbuf_b, self.d_recvbuf_t, self.nx, self.nz, hs
            )
        elif self.bottom_channel is not None:
            unpack_recv_buf_z_bottom_kernel[blockspergrid_buffer, threadsperblock_buffer](
                d_state_forcing, self.d_recvbuf_b, self.nx, self.nz, hs
            )
        elif self.top_channel is not None:
            unpack_recv_buf_z_top_kernel[blockspergrid_buffer, threadsperblock_buffer](
                d_state_forcing, self.d_recvbuf_t, self.nx, self.nz, hs
            )

        cuda.synchronize()

    def _compute_tendencies_x(self, dt_arg_for_hv_coef, d_state_forcing):
        threadsperblock_flux = (16, 16, 1)
        blockspergrid_flux_x = (math.ceil((self.nx + 1) / threadsperblock_flux[0]), 
                                  math.ceil(self.nz / threadsperblock_flux[1]), 1)
        hv_coef = -hv_beta * self.grid_dx / (16.0 * dt_arg_for_hv_coef)
        compute_flux_x_kernel[blockspergrid_flux_x, threadsperblock_flux](
            d_state_forcing, self.d_flux, self.d_hy_dens_cell, self.d_hy_dens_theta_cell,
            hv_coef, self.nx, self.nz, hs
        )

        threadsperblock_tend = (16, 16, 1) 
        blockspergrid_tend_x = (math.ceil(self.nx / threadsperblock_tend[0]),
                                  math.ceil(self.nz / threadsperblock_tend[1]), NUM_VARS)
        compute_tend_x_kernel[blockspergrid_tend_x, threadsperblock_tend](
            self.d_flux, self.d_tend, self.nx, self.nz, self.grid_dx
        )
        cuda.synchronize()

    def _compute_tendencies_z(self, dt_arg_for_hv_coef, d_state_forcing):
        hv_coef = -hv_beta * self.grid_dz / (16.0 * dt_arg_for_hv_coef)
        threadsperblock_flux = (16, 16, 1)
        blockspergrid_flux_z = (math.ceil(self.nx / threadsperblock_flux[0]),
                                  math.ceil((self.nz + 1) / threadsperblock_flux[1]), 1)
        compute_flux_z_kernel[blockspergrid_flux_z, threadsperblock_flux](
            d_state_forcing, self.d_flux, self.d_hy_dens_int, self.d_hy_pressure_int, self.d_hy_dens_theta_int,
            hv_coef, self.nx, self.nz, hs, self.k_beg_global_idx, self.nz_glob
        )

        threadsperblock_tend = (16, 16, 1)
        blockspergrid_tend_z = (math.ceil(self.nx / threadsperblock_tend[0]),
                                  math.ceil(self.nz / threadsperblock_tend[1]), NUM_VARS)
        compute_tend_z_kernel[blockspergrid_tend_z, threadsperblock_tend](
            d_state_forcing, self.d_flux, self.d_tend, self.nx, self.nz, hs, self.grid_dz
        )
        cuda.synchronize()

    @coro
    def _semi_discrete_step(self, dt_arg, current_dir, d_state_init, d_state_forcing, d_state_out):
        if current_dir == DIR_X:
            self._set_halo_values_x(d_state_forcing)
            self._compute_tendencies_x(dt_arg, d_state_forcing)
        elif current_dir == DIR_Z:
            self._set_halo_values_z(d_state_forcing)
            self._compute_tendencies_z(dt_arg, d_state_forcing)
        
        threadsperblock_update = (16, 16, 1)
        blockspergrid_update = (math.ceil(self.nx / threadsperblock_update[0]),
                                  math.ceil(self.nz / threadsperblock_update[1]), NUM_VARS)
        update_fluid_state_kernel[blockspergrid_update, threadsperblock_update](
            d_state_init, d_state_out, self.d_tend, self.nx, self.nz, hs, dt_arg
        )
        cuda.synchronize()

    @coro
    def _perform_timestep(self, dt_full_step):
        dt_rk_stage1 = dt_full_step / 3.0
        dt_rk_stage2 = dt_full_step / 2.0
        dt_rk_stage3 = dt_full_step / 1.0

        if self._direction_switch:
            self._semi_discrete_step(dt_rk_stage1, DIR_X, self.d_state, self.d_state,     self.d_state_tmp)
            self._semi_discrete_step(dt_rk_stage2, DIR_X, self.d_state, self.d_state_tmp, self.d_state_tmp)
            self._semi_discrete_step(dt_rk_stage3, DIR_X, self.d_state, self.d_state_tmp, self.d_state)
            self._semi_discrete_step(dt_rk_stage1, DIR_Z, self.d_state, self.d_state,     self.d_state_tmp)
            self._semi_discrete_step(dt_rk_stage2, DIR_Z, self.d_state, self.d_state_tmp, self.d_state_tmp)
            self._semi_discrete_step(dt_rk_stage3, DIR_Z, self.d_state, self.d_state_tmp, self.d_state)
        else:
            self._semi_discrete_step(dt_rk_stage1, DIR_Z, self.d_state, self.d_state,     self.d_state_tmp)
            self._semi_discrete_step(dt_rk_stage2, DIR_Z, self.d_state, self.d_state_tmp, self.d_state_tmp)
            self._semi_discrete_step(dt_rk_stage3, DIR_Z, self.d_state, self.d_state_tmp, self.d_state)
            self._semi_discrete_step(dt_rk_stage1, DIR_X, self.d_state, self.d_state,     self.d_state_tmp)
            self._semi_discrete_step(dt_rk_stage2, DIR_X, self.d_state, self.d_state_tmp, self.d_state_tmp)
            self._semi_discrete_step(dt_rk_stage3, DIR_X, self.d_state, self.d_state_tmp, self.d_state)

        self._direction_switch = not self._direction_switch

    @coro
    # Main simulation loop
    def start_main_loop(self, all_chares_done_future):
        current_sim_time_target = self.args.sim_time
        
        chare_loop_start_time = time.time()
        
        n_iters = 0
        while self.etime < current_sim_time_target and n_iters < self.args.max_iters:
            actual_dt = self.dt
            if self.etime + self.dt > current_sim_time_target:
                actual_dt = current_sim_time_target - self.etime
            
            self._perform_timestep(actual_dt)
            
            self.etime += actual_dt
            n_iters += 1

            if self.args.output_freq > 0 and n_iters % self.args.output_freq == 0:
                cuda.synchronize() 
                state_host_output_local = self.d_state.copy_to_host()
                
                if hs > 0:
                    state_ext = state_host_output_local[:, hs:-hs, hs:-hs]
                else:
                    state_ext = state_host_output_local

                if hs > 0:
                    hy_dens_cell_local = self.hy_dens_cell_host[hs:-hs]
                    hy_dens_theta_cell_local = self.hy_dens_theta_cell_host[hs:-hs]
                else:
                    hy_dens_cell_local = self.hy_dens_cell_host
                    hy_dens_theta_cell_local = self.hy_dens_theta_cell_host

                dens = state_ext[ID_DENS, :, :]
                denom = hy_dens_cell_local[:, None] + dens
                uwnd = state_ext[ID_UMOM, :, :] / denom
                wwnd = state_ext[ID_WMOM, :, :] / denom
                theta = (state_ext[ID_RHOT, :, :] + hy_dens_theta_cell_local[:, None]) / denom \
                        - (hy_dens_theta_cell_local / hy_dens_cell_local)[:, None]

                norm_state = np.stack([dens, uwnd, wwnd, theta], axis=0)
                output_filename = os.path.join(self.args.output_dir, f"data_iter_{n_iters:06d}_chare_{self.chare_idx_x}_{self.chare_idx_z}.npz")
                np.savez(output_filename,
                         state=norm_state,
                         etime=self.etime,
                         chare_nx=self.nx,
                         chare_i_beg=self.i_beg_global_idx,
                         chare_nz=self.nz,
                         chare_k_beg=self.k_beg_global_idx)
                if self.chare_idx_x == 0 and self.chare_idx_z == 0:
                    print(f"Iter: {n_iters}, Chare 0,0 output data to {output_filename} pattern at SimTime: {self.etime:.4f}s")
            
            if self.chare_idx_x == 0 and self.chare_idx_z == 0 and (n_iters % 10 == 0 or n_iters == 1 or (self.etime >= current_sim_time_target) or (n_iters == self.args.max_iters)):
                print(f"Chare 0,0 - Iter: {n_iters:5d}, Sim Time: {self.etime:8.4f}s / {current_sim_time_target:.2f}s, Step dt: {actual_dt:.6f}s")

        chare_loop_end_time = time.time()
        cuda.synchronize()
        
        if self.chare_idx_x == 0 and self.chare_idx_z == 0:
             print(f"\nChare 0,0 finished main loop after {n_iters} iterations. Local loop wall time: {chare_loop_end_time - chare_loop_start_time:.3f} s.")
             print(f"Chare 0,0 final simulation time: {self.etime:.4f}s")

        final_mass_local, final_te_local = self._reductions()
        
        self.reduce(all_chares_done_future, [final_mass_local, final_te_local, self.etime, float(n_iters)], 
                    Reducer.gather)

def main_charm_wrapper(charm_args_list):
    parser = argparse.ArgumentParser(description="MiniWeather Python Numba CUDA Simulation (Charm4Py)")
    parser.add_argument("--nx_glob", type=int, default=200, help="Number of global cells in x-direction (default: 200)")
    parser.add_argument("--nz_glob", type=int, default=100, help="Number of global cells in z-direction (default: 100)")
    parser.add_argument("--sim_time", type=float, default=1.0, help="How many seconds to run the simulation (default: 1.0s)")
    parser.add_argument("--max_iters", type=int, default=10000, help="Maximum number of iterations (default: 10000)")
    parser.add_argument("--output_freq", type=int, default=0, help="Frequency of outputting data in iterations (0 for no output, default: 0)")
    parser.add_argument("--output_dir", type=str, default="output_data_charm", help="Directory to save output files (default: output_data_charm)")
    
    data_spec_choices_map = {
        DATA_SPEC_COLLISION: "collision", DATA_SPEC_THERMAL: "thermal",
        DATA_SPEC_MOUNTAIN: "mountain_waves", DATA_SPEC_TURBULENCE: "turbulence",
        DATA_SPEC_DENSITY_CURRENT: "density_current", DATA_SPEC_INJECTION: "injection"
    }
    default_data_spec_name = data_spec_choices_map.get(DATA_SPEC_THERMAL, str(DATA_SPEC_THERMAL))
    parser.add_argument("--data_spec", type=str, default=default_data_spec_name,
                        choices=list(data_spec_choices_map.values()),
                        help=f"Data specification name (default: {default_data_spec_name})")
    
    parser.add_argument("--num_chares_x", type=int, default=1, help="Number of chares in X-direction for domain decomposition (default: 1)")
    parser.add_argument("--num_chares_z", type=int, default=1, help="Number of chares in Z-direction for domain decomposition (default: 1)")
    
    args = parser.parse_args(charm_args_list[1:])

    data_spec_int = None
    for val, name in data_spec_choices_map.items():
        if name == args.data_spec:
            data_spec_int = val
            break
    
    if charm.myPe() == 0:
        print(f"Running MiniWeather (Charm4Py) with: "
              f"nx_glob={args.nx_glob}, nz_glob={args.nz_glob}, num_chares_x={args.num_chares_x}, "
              f"num_chares_z={args.num_chares_z}, data_spec='{args.data_spec}' (ID: {data_spec_int}), sim_time={args.sim_time:.2f}s, "
              f"max_iters={args.max_iters}, output_freq={args.output_freq}, output_dir='{args.output_dir}'")

        if args.output_freq > 0:
            if not os.path.exists(args.output_dir):
                try:
                    os.makedirs(args.output_dir)
                    print(f"Created output directory: {args.output_dir}")
                except FileExistsError:
                    print(f"Output directory already exists or was just created: {args.output_dir}")
            else:
                print(f"Output directory already exists: {args.output_dir}")

    grid_dx = xlen / args.nx_glob
    grid_dz = zlen / args.nz_glob
    initial_dt = min(grid_dx, grid_dz) / max_speed * cfl
    initial_etime = 0.0
    
    num_chares_x = args.num_chares_x
    if num_chares_x > args.nx_glob:
        if charm.myPe() == 0:
            print(f"Warning: num_chares_x ({num_chares_x}) > nx_glob ({args.nx_glob}). Setting num_chares_x = nx_glob.")
        num_chares_x = args.nx_glob
        args.num_chares_x = num_chares_x

    num_chares_z = args.num_chares_z
    if num_chares_z > args.nz_glob:
        if charm.myPe() == 0:
            print(f"Warning: num_chares_z ({num_chares_z}) > nz_glob ({args.nz_glob}). Setting num_chares_z = nz_glob.")
        num_chares_z = args.nz_glob
        args.num_chares_z = num_chares_z

    chare_constructor_args = (vars(args), num_chares_x, num_chares_z, args.nx_glob, args.nz_glob, data_spec_int, initial_dt, initial_etime)
    
    chares = Array(MiniWeatherChare, dims=(num_chares_x, num_chares_z), args=[chare_constructor_args])
    
    setup_completion_future = Future()
    
    for i in range(num_chares_x):
        for j in range(num_chares_z):
            local_nx, i_beg_global = calculate_domain_decomposition_x(i, num_chares_x, args.nx_glob)
            local_nz, k_beg_global = calculate_domain_decomposition_z(j, num_chares_z, args.nz_glob)
            chares[i, j].setup_chare_domain(local_nx, i_beg_global, local_nz, k_beg_global, setup_completion_future)
    
    initial_reductions_sum = setup_completion_future.get()
    mass0_sum = initial_reductions_sum[0]
    te0_sum = initial_reductions_sum[1]

    if charm.myPe() == 0:
        print(f"Initial Global Mass: {mass0_sum:.6e}, Initial Global Total Energy: {te0_sum:.6e}")
        print("\nCUDA device array setup and initial reductions complete for all chares.")
        print(f"Starting main simulation loop up to sim_time: {args.sim_time:.2f}s or max_iters: {args.max_iters}")

    main_loop_done_future = Future()

    chares.start_main_loop(main_loop_done_future)

    gathered_results = main_loop_done_future.get()
    
    total_final_mass = sum(res[0] for res in gathered_results)
    total_final_te = sum(res[1] for res in gathered_results)
    max_etime = 0.0
    max_niters = 0
    if gathered_results:
        max_etime = max(res[2] for res in gathered_results)
        max_niters = max(int(res[3]) for res in gathered_results)


    if charm.myPe() == 0:
        main_loop_wall_time = -1
        print(f"\nAll chares finished main simulation loop (max_iters {max_niters}).")
        print(f"Max final simulation time reached: {max_etime:.4f}s")

        print(f"Final Global Mass:   {total_final_mass:.6e}, Final Global Total Energy:   {total_final_te:.6e}")
        if abs(mass0_sum) > 1e-12:
            print(f"Relative mass change: {(total_final_mass - mass0_sum) / mass0_sum:.6e}")
        else:
            print(f"Relative mass change: (initial mass was near zero)")
        if abs(te0_sum) > 1e-12:
            print(f"Relative TE change:   {(total_final_te - te0_sum) / te0_sum:.6e}")
        else:
            print(f"Relative TE change:   (initial TE was near zero)")

        print("\nMiniWeather Charm4Py Numba CUDA simulation finished.")
    
    charm.exit()

charm.start(main_charm_wrapper)