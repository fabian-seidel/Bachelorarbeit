import dill
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from gpe_numeric import itime_ev_2nd_2D, calculate_mu, plot_density, psi_time_ev_2nd_2D, extend_time_ev, make_animation
from utils import plot_by_name, execute_for_param_cases
import inspect


def itime_ev_2nd_vortex_dipole(g, d, tau_max=2, dtau=0.001, x_max=6, dx=0.025, y_max=6, dy=0.025, paths=None):
    paths = paths or {}
    z1_0 = d + 0j
    z2_0 = -d + 0j

    num_x = int(2 * x_max / dx)
    num_y = int(2 * y_max / dy)
    x, dx = cp.linspace(-x_max, x_max, num_x, endpoint=False, retstep=True)
    y, dy = cp.linspace(-y_max, y_max, num_y, endpoint=False, retstep=True)
    x2d, y2d = cp.meshgrid(x, y)
    z = x2d + 1j * y2d
    vortex_phase = (z - z1_0) * (cp.conj(z) - cp.conj(z2_0)) / (
                cp.abs((z - z1_0) * (cp.conj(z) - cp.conj(z2_0))) + 1e-14)

    k_x = 2 * cp.pi * cp.fft.fftfreq(num_x, d=dx)
    k_y = 2 * cp.pi * cp.fft.fftfreq(num_y, d=dy)
    k_x2d, k_y2d = cp.meshgrid(k_x, k_y)
    k_sq = k_x2d ** 2 + k_y2d ** 2
    pot = 1 / 2 * (x2d ** 2 + y2d ** 2)
    psi = cp.ones((num_y, num_x))
    psi = cp.fft.fft2(psi)
    psi = cp.exp(-dtau * k_sq / 4) * psi
    tau = 0
    while tau < tau_max:
        psi = cp.fft.ifft2(psi)
        psi = cp.abs(psi) * vortex_phase
        Mb = cp.exp(dtau * (-pot - g * cp.abs(psi) ** 2))
        psi *= Mb
        psi /= cp.sqrt(cp.sum(cp.abs(psi) ** 2 * dx * dy))
        psi = cp.fft.fft2(psi)
        psi *= cp.exp(-dtau * k_sq / 2)
        tau += dtau
        if (int(round(tau / dtau)) % 50) == 0:
            print(f"Imaginary time evolution: {tau:.3f} of {tau_max}")

    psi = cp.fft.ifft2(psi)
    psi = cp.abs(psi) * vortex_phase
    Mb = cp.exp(dtau * (-pot - g * cp.abs(psi) ** 2))
    psi *= Mb
    psi /= cp.sqrt(cp.sum(cp.abs(psi) ** 2 * dx * dy))
    psi = cp.fft.fft2(psi)
    psi *= cp.exp(-dtau * k_sq / 4)
    psi = cp.fft.ifft2(psi)
    psi = cp.abs(psi) * vortex_phase
    psi /= cp.sqrt(cp.sum(cp.abs(psi) ** 2 * dx * dy))
    if 'savepath' in paths:
        np.savez(paths['savepath'], x2d=cp.asnumpy(x2d), y2d=cp.asnumpy(y2d), psi=cp.asnumpy(psi))
    return cp.asnumpy(x2d), cp.asnumpy(y2d), cp.asnumpy(psi)


def angle_diff(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def find_vortex(radius_threshold=4, t=None, x2d=None, y2d=None, psi=None, paths=None):
    print(f"Finding vortices.")
    paths = paths or None
    if (x2d is None or y2d is None or psi is None) and 'psi_loadpath' in paths:
        data = np.load(paths['psi_loadpath'])
        t = data['t'] if t is None else t
        x2d = data['x2d'] if x2d is None else x2d
        y2d = data['y2d'] if y2d is None else y2d
        psi = data['psi'] if psi is None else psi
    if any(v is None for v in (x2d, y2d, psi)):
        raise ValueError("Missing required arguments: x2d, y2d, or psi.")
    Nx = x2d.shape[1] - 1
    Ny = y2d.shape[0] - 1
    Nt = psi.shape[0]
    cw_vortex_map = np.zeros((Nt, Ny, Nx), dtype=bool)
    ccw_vortex_map = np.zeros((Nt, Ny, Nx), dtype=bool)
    radius = np.sqrt(x2d ** 2 + y2d ** 2)[1:, 1:]
    radius_mask = radius < radius_threshold
    for i in range(Nt):
        if i % 10 == 0:
            print(f"Finding vortices, step {i} of {Nt}")
        psi_i = psi[i]
        psi_i_angle = np.angle(psi_i)
        dl = angle_diff(psi_i_angle[:, :-1], psi_i_angle[:, 1:])
        du = angle_diff(psi_i_angle[:-1, :], psi_i_angle[1:, :])
        total_phase = dl[:-1, :] + du[:, 1:] - dl[1:, :] - du[:, :-1]
        vortex_winding_num = np.round(total_phase / (2 * np.pi))
        cw_vortex_map[i] = (vortex_winding_num == 1) & radius_mask
        ccw_vortex_map[i] = (vortex_winding_num == -1) & radius_mask
    cw_vortex_idx = cw_vortex_map.nonzero()
    ccw_vortex_idx = ccw_vortex_map.nonzero()
    if 'savepath' in paths:
        with open(paths['savepath'], 'wb') as file:
            dill.dump({'t': t, 'x2d': x2d, 'y2d': y2d, 'cw_vortex_idx': cw_vortex_idx, 'ccw_vortex_idx': ccw_vortex_idx}, file)
    return t, x2d, y2d, cw_vortex_idx, ccw_vortex_idx


def plot_vortex_trajectories(plot_array_shape, cases=None, data_cases=None, plot_type='scatter', figsize=(6.4, 4.8), paths=None):
    paths = paths or {}
    if data_cases is None:
        if 'vortex_idx_loadpath' in paths:
            data_cases = []
            for vals in zip(*cases.values()):
                single_case = dict(zip(cases.keys(), vals))
                loadpath = paths['vortex_idx_loadpath'].format(**single_case)
                with open(loadpath, 'rb') as file:
                    data_cases.append(dill.load(file))
        else:
            raise ValueError("Missing required data, either from data_cases or 'vortex_idx_loadpath' in paths.")
    fig, axes = plt.subplots(*plot_array_shape, sharex=True, sharey=True, squeeze=False, figsize=figsize)
    for row in range(plot_array_shape[0]):
        axes[row][0].set_ylabel('y')
    for column in range(plot_array_shape[1]):
        axes[plot_array_shape[0] - 1][column].set_xlabel('x')
    axes = axes.flatten()
    for idx, data in enumerate(data_cases):
        x2d, y2d, cw_vortex_idx, ccw_vortex_idx = data['x2d'], data['y2d'], data['cw_vortex_idx'], data[
            'ccw_vortex_idx']
        x, y = x2d[0, :], y2d[:, 0]
        if plot_type == 'line':
            axes[idx].plot(x[cw_vortex_idx[2]], y[cw_vortex_idx[1]])
            axes[idx].plot(x[ccw_vortex_idx[2]], y[ccw_vortex_idx[1]])
        elif plot_type == 'scatter':
            axes[idx].scatter(x[cw_vortex_idx[2]], y[cw_vortex_idx[1]], s=1)
            axes[idx].scatter(x[ccw_vortex_idx[2]], y[ccw_vortex_idx[1]], s=1)
        else:
            raise ValueError("Plot type must be either 'line' or 'scatter'.")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # cases = [(g_1, d_1, gamma_1), (g_2, d_2, gamma_2), ...]
    # cases = [(150, 0.8, 0), (150, 1.1, 0), (150, 1.3, 0), (150, 1.5, 0), (10, 0.8, 0), (50, 0.8, 0)]
    cases = {'g': (2000,), 'd': (1.3,), 'gamma': (1e-2,)}
    gs_no_vortex_path = 'saves/vortex_dipole/numeric/g{g}_init_no_vortex.npz'
    mu_path = 'saves/vortex_dipole/numeric/g{g}_mu.npy'
    psi_0_vortex_dp_path = 'saves/vortex_dipole/numeric/g{g}_d{d}_init.npz'
    psi_time_ev_path = 'saves/vortex_dipole/numeric/g{g}_d{d}_gamma{gamma}_time_ev.npz'
    animation_path = 'videos/vortex_dipole/g{g}_d{d}_gamma{gamma}_animation.mp4'
    vortex_idx_path = 'saves/vortex_dipole/numeric/g{g}_d{d:.1f}_gamma{gamma}_vortex_idx.dill'
    x_max = 20
    dx = 0.05
    y_max = 20
    dy = 0.05
    t_max = 20
    dt = 1e-3

    # execute_for_param_cases(itime_ev_2nd_2D, cases=cases, x_max=x_max, y_max=y_max, dx=dx, dy=dy, paths={'savepath':gs_no_vortex_path})
    # execute_for_param_cases(calculate_mu, cases=cases, paths={'psi_loadpath':gs_no_vortex_path,'savepath':mu_path})
    # execute_for_param_cases(itime_ev_2nd_vortex_dipole, cases=cases, x_max=x_max, y_max=y_max, dx=dx, dy=dy, paths={'savepath':psi_0_vortex_dp_path})
    # execute_for_param_cases(plot_density, cases=cases, paths={'psi_loadpath':psi_0_vortex_dp_path})
    # execute_for_param_cases(psi_time_ev_2nd_2D, cases=cases, dt=dt, t_max=t_max, paths={'psi_0_loadpath':psi_0_vortex_dp_path, 'mu_loadpath':mu_path, 'savepath':psi_time_ev_path})
    # execute_for_param_cases(extend_time_ev, cases=cases, scheme='2nd', dt=dt, t_stop=60, paths={'prev_time_ev_loadpath':psi_time_ev_path, 'mu_loadpath':mu_path, 'savepath':psi_time_ev_path})
    # execute_for_param_cases(make_animation, cases=cases, paths={'psi_loadpath':psi_time_ev_path, 'savepath':animation_path})
    # execute_for_param_cases(find_vortex, cases=cases, paths={'psi_loadpath':psi_time_ev_path, 'savepath':vortex_idx_path})
    # plot_vortex_trajectories((1, 1), cases=cases, paths={'vortex_idx_loadpath': vortex_idx_path})
    # plot_vortex_trajectories(cases[0][0], cases[0][1], gamma=cases[0][2])

    cases_g150_no_diss = {'g':(150,)*4, 'd':(0.8, 1.1, 1.3, 1.5)}
    cases_lowg_no_diss = {'g':(10, 50), 'd':(0.8,)*2}
    cases_d1p3_diss = {'g':(500,)*2, 'd':(1.3,)*2, 'gamma':(0.03, 0.3)}

    plot_dict = {'Numerical vortex trajectories for g=150; d=0.8,1.1,1.3,1.5; gamma=0': (plot_vortex_trajectories, {'cases':cases_g150_no_diss, 'plot_array_shape':(2,2), 'plot_type':'line', 'paths':{'vortex_idx_loadpath':vortex_idx_path}}),
                 'Numerical vortex trajectories for g=10,50; d=0.8; gamma=0': (plot_vortex_trajectories, {'cases':cases_lowg_no_diss, 'plot_array_shape':(1,2), 'plot_type':'scatter','paths':{'vortex_idx_loadpath':vortex_idx_path}}),
                 'Numerical vortex trajectories for g=500; d=1.3; gamma=0.03,0.3': (plot_vortex_trajectories, {'cases':cases_d1p3_diss, 'plot_array_shape':(1,2), 'plot_type':'line', 'figsize':(10, 5), 'paths':{'vortex_idx_loadpath':vortex_idx_path}})}
    plot_by_name(plot_dict, 'Numerical vortex trajectories for g=500; d=1.3; gamma=0.03,0.3')

