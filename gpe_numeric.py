import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def itime_ev_2nd_2D(g, tau_max=2, dtau=0.001, x_max=6, dx=0.025, y_max=6, dy=0.025, N0=1, paths=None):
    paths = paths or {}
    num_x = int(2 * x_max / dx)
    num_y = int(2 * y_max / dy)
    x, dx = cp.linspace(-x_max, x_max, num_x, endpoint=False, retstep=True)
    y, dy = cp.linspace(-y_max, y_max, num_y, endpoint=False, retstep=True)
    x2d, y2d = cp.meshgrid(x, y)
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
        Mb = cp.exp(dtau * (-pot - g * cp.abs(psi) ** 2))
        psi *= Mb
        psi /= cp.sqrt(N0*cp.sum(cp.abs(psi) ** 2 * dx * dy))
        psi = cp.fft.fft2(psi)
        psi *= cp.exp(-dtau * k_sq / 2)
        tau += dtau
        if (int(round(tau / dtau)) % 50) == 0:
            print(f"Imaginary time evolution (no vortex): {tau:.3f} of {tau_max}")
    psi = cp.fft.ifft2(psi)
    Mb = cp.exp(dtau * (-pot - g * cp.abs(psi) ** 2))
    psi *= Mb
    psi /= cp.sqrt(N0*cp.sum(cp.abs(psi) ** 2 * dx * dy))
    psi = cp.fft.fft2(psi)
    psi *= cp.exp(-dtau * k_sq / 4)
    psi = cp.fft.ifft2(psi)
    psi /= cp.sqrt(N0*cp.sum(cp.abs(psi) ** 2 * dx * dy))
    if 'savepath' in paths:
        np.savez(paths['savepath'], x2d=cp.asnumpy(x2d), y2d=cp.asnumpy(y2d), psi=cp.asnumpy(psi))
    return cp.asnumpy(x2d), cp.asnumpy(y2d), cp.asnumpy(psi)


def psi_time_ev_4th_2D(g, gamma=0, mu=0, t_max=20, dt=0.001, x2d=None, y2d=None, psi_0=None, save_step=100, paths=None):
    paths = paths or {}
    if (x2d is None or y2d is None or psi_0 is None) and 'psi_0_loadpath' in paths:
        groundstate_data = np.load(paths['psi_0_loadpath'])
        x2d = groundstate_data['x2d'] if x2d is None else x2d
        y2d = groundstate_data['y2d'] if y2d is None else y2d
        psi_0 = groundstate_data['psi'] if psi_0 is None else psi_0
    if (mu == 0) and 'mu_loadpath' in paths:
        mu = np.load(paths['mu_loadpath'])
    if any(v is None for v in (x2d, y2d, psi_0)):
        raise ValueError("Missing required arguments: x2d, y2d, or psi_0.")
    x2d = cp.asarray(x2d)
    y2d = cp.asarray(y2d)
    psi_0 = cp.asarray(psi_0)
    t, dt = cp.linspace(0, t_max, int(t_max/dt), retstep=True)
    mu = cp.asarray(mu)
    num_x = x2d.shape[1]
    num_y = x2d.shape[0]
    dx = x2d[0, 1] - x2d[0, 0]
    dy = y2d[1, 0] - y2d[0, 0]
    k_x = 2 * cp.pi * cp.fft.fftfreq(num_x, d=dx)
    k_y = 2 * cp.pi * cp.fft.fftfreq(num_y, d=dy)
    k_x2d, k_y2d = cp.meshgrid(k_x, k_y)
    k_sq = k_x2d ** 2 + k_y2d ** 2
    pot = 1 / 2 * (x2d ** 2 + y2d ** 2)
    c0 = -2 ** (1 / 3) / (2 - 2 ** (1 / 3))
    c1 = 1 / (2 - 2 ** (1 / 3))
    a = cp.array([c1 / 2, (c0 + c1) / 2, (c0 + c1) / 2, c1 / 2])
    b = cp.array([c1, c0, c1])
    Ma = cp.exp(-1j * dt * (1 - gamma * 1j) * cp.multiply.outer(a, k_sq) / 2)
    psi = cp.zeros((int((len(t) - 1) / save_step) + 1, num_y, num_x), dtype=cp.complex128)
    psi_i = psi_0.copy()
    psi_i = cp.fft.fft2(psi_i)
    for idx in range(len(t)):
        if idx % save_step == 0:
            psi_i = cp.fft.ifft2(psi_i)
            psi[int(idx / save_step)] = psi_i
            current_norm = cp.sum(cp.abs(psi_i) ** 2) * dx * dy
            psi_i = cp.fft.fft2(psi_i)
            print(f"Time evolution: {t[idx]:.3f} of {t[-1]:.3f}, current norm: {current_norm}.")
        for factor_num in range(3, 0, -1):
            psi_i *= Ma[factor_num]
            psi_i = cp.fft.ifft2(psi_i)
            psi_i *= cp.exp(b[factor_num - 1] * 1j * dt * (1 - gamma * 1j) * (-pot - g * cp.abs(psi_i) ** 2 + mu))
            psi_i = cp.fft.fft2(psi_i)
        psi_i *= Ma[0]
    t_np, x2d_np, y2d_np, psi_np = cp.asnumpy(t[::save_step]), cp.asnumpy(x2d), cp.asnumpy(y2d), cp.asnumpy(psi)
    if 'savepath' in paths:
        np.savez(paths['savepath'], t=t_np, x2d=x2d_np, y2d=y2d_np, psi=psi_np)
    return t_np, x2d_np, y2d_np, psi_np


def psi_time_ev_2nd_2D(g, gamma=0, mu=0, t=None, t_max=20, dt=1e-3, x2d=None, y2d=None, psi_0=None, save_step=100, paths=None):
    paths = paths or {}
    if (x2d is None or y2d is None or psi_0 is None) and 'psi_0_loadpath' in paths:
        groundstate_data = np.load(paths['psi_0_loadpath'])
        x2d = groundstate_data['x2d'] if x2d is None else x2d
        y2d = groundstate_data['y2d'] if y2d is None else y2d
        psi_0 = groundstate_data['psi'] if psi_0 is None else psi_0
    if (mu == 0) and 'mu_loadpath' in paths:
        mu = np.load(paths['mu_loadpath'])
    if any(v is None for v in (x2d, y2d, psi_0)):
        raise ValueError("Missing required arguments: x2d, y2d, or psi_0.")
    x2d = cp.asarray(x2d)
    y2d = cp.asarray(y2d)
    psi_0 = cp.asarray(psi_0)
    if t is None:
        t, dt = cp.linspace(0, t_max, int(t_max/dt), retstep=True)
    else:
        t = cp.asarray(t)
    mu = cp.asarray(mu)
    num_x = x2d.shape[1]
    num_y = x2d.shape[0]
    dx = x2d[0, 1] - x2d[0, 0]
    dy = y2d[1, 0] - y2d[0, 0]
    k_x = 2 * cp.pi * cp.fft.fftfreq(num_x, d=dx)
    k_y = 2 * cp.pi * cp.fft.fftfreq(num_y, d=dy)
    k_x2d, k_y2d = cp.meshgrid(k_x, k_y)
    k_sq = k_x2d ** 2 + k_y2d ** 2
    pot = 1 / 2 * (x2d ** 2 + y2d ** 2)
    # k_max = min(cp.max(cp.abs(k_x)), cp.max(cp.abs(k_y)))
    # mask = k_sq < ((2 / 3) * k_max) ** 2
    Ma = cp.exp(-1j * dt * (1 - gamma * 1j) * k_sq / 2)
    psi = cp.zeros((int((len(t) - 1) / save_step) + 1, num_y, num_x), dtype=cp.complex128)
    psi_i = psi_0.copy()
    for idx in range(len(t)):
        if idx % save_step == 0:
            psi[int(idx / save_step)] = psi_i
            current_norm = cp.sum(cp.abs(psi_i) ** 2) * dx * dy
            print(f"Time evolution: {t[idx]:.3f} of {t[-1]:.3f}, current Norm: {current_norm}")
        psi_i *= cp.exp(-1j * (dt / 2) * (1 - gamma * 1j) * (pot - mu + g * cp.abs(psi_i) ** 2))
        psi_i = cp.fft.fft2(psi_i)
        psi_i *= Ma
        # psi_i *= mask
        psi_i = cp.fft.ifft2(psi_i)
        psi_i *= cp.exp(-1j * (dt / 2) * (1 - gamma * 1j) * (pot - mu + g * cp.abs(psi_i) ** 2))
    t_np, x2d_np, y2d_np, psi_np = cp.asnumpy(t[::save_step]), cp.asnumpy(x2d), cp.asnumpy(y2d), cp.asnumpy(psi)
    if 'savepath' in paths:
        np.savez(paths['savepath'], t=t_np, x2d=x2d_np, y2d=y2d_np, psi=psi_np)
    return t_np, x2d_np, y2d_np, psi_np


def extend_time_ev(g, gamma=0, mu=0, t_stop=20, dt=2e-4, t_prev=None, x2d=None, y2d=None, psi_prev=None, scheme='4th', paths=None):
    paths = paths or {}
    if (x2d is None or y2d is None or psi_prev is None) and 'prev_time_ev_loadpath' in paths:
        prev_data = np.load(paths['prev_time_ev_loadpath'])
        t_prev = prev_data['t'] if t_prev is None else t_prev
        x2d = prev_data['x2d'] if x2d is None else x2d
        y2d = prev_data['y2d'] if y2d is None else y2d
        psi_prev = prev_data['psi'] if psi_prev is None else psi_prev
    if (mu == 0) and 'mu_loadpath' in paths:
        mu = np.load(paths['mu_loadpath'])
    if any(v is None for v in (x2d, y2d, psi_prev)):
        raise ValueError("Missing required arguments: x2d, y2d, or psi_prev.")
    psi_0 = psi_prev[-1]
    t_new = t_prev[-1] + np.arange(1, int(t_stop / dt) + 1) * dt
    t_for_method = np.concatenate(([t_prev[-1]], t_new))
    if scheme == '4th':
        time_ev_method = psi_time_ev_4th_2D
    elif scheme == '2nd':
        time_ev_method = psi_time_ev_2nd_2D
    else:
        raise ValueError("'scheme' has to be either '4th' or '2nd'")
    t_new_ret, x2d, y2d, psi_new = time_ev_method(g, x2d=x2d, y2d=y2d, psi_0=psi_0, t=t_for_method, gamma=gamma, mu=mu)
    psi = np.concatenate([psi_prev, psi_new[1:]], axis=0)
    t_combined = np.concatenate([t_prev, t_new_ret[1:]])
    if 'savepath' in paths:
        np.savez(paths['savepath'], t=t_combined, x2d=x2d, y2d=y2d, psi=psi)
    return t_combined, x2d, y2d, psi

def calculate_mu(g, x2d=None, y2d=None, psi=None, paths=None):
    paths = paths or {}
    print(f"Calculating mu")
    if (x2d is None or y2d is None or psi is None) and 'psi_loadpath' in paths:
        data = np.load(paths['psi_loadpath'])
        x2d= data['x2d'] if x2d is None else x2d
        y2d= data['y2d'] if y2d is None else y2d
        psi= data['psi'] if psi is None else psi
    if any(v is None for v in (x2d, y2d, psi)):
        raise ValueError("Missing required arguments: x2d, y2d, or psi.")
    dx = x2d[0, 1] - x2d[0, 0]
    dy = y2d[1, 0] - y2d[0, 0]
    num_x = x2d.shape[1]
    num_y = y2d.shape[0]
    pot = 1 / 2 * (x2d ** 2 + y2d ** 2)
    k_x = 2 * np.pi * np.fft.fftfreq(num_x, d=dx)
    k_y = 2 * np.pi * np.fft.fftfreq(num_y, d=dy)
    k_x2d, k_y2d = np.meshgrid(k_x, k_y)
    k_sq = k_x2d ** 2 + k_y2d ** 2
    N = np.sum(np.abs(psi) ** 2) * dx * dy
    mu_spacial = np.sum(np.abs(psi) ** 2 * pot + g * np.abs(psi) ** 4) * dx * dy
    psi_fft = np.fft.fft2(psi)
    mu_kin = np.sum(1 / 2 * k_sq * np.abs(psi_fft) ** 2) * dx * dy / (num_x * num_y)
    mu = (mu_spacial + mu_kin) / N
    if 'savepath' in paths:
        np.save(paths['savepath'], mu)
    return mu

def plot_density(x2d=None, y2d=None, psi=None, paths=None):
    paths = paths or {}
    if (x2d is None or y2d is None or psi is None) and 'psi_loadpath' in paths:
        data = np.load(paths['psi_loadpath'])
        x2d = data['x2d'] if x2d is None else x2d
        y2d = data['y2d'] if y2d is None else y2d
        psi = data['psi'] if psi is None else psi
    if any(v is None for v in (x2d, y2d, psi)):
        raise ValueError("Missing required arguments: x2d, y2d, or psi.")
    plt.imshow(np.abs(psi) ** 2, extent=[np.min(y2d), np.max(y2d), np.min(x2d), np.max(x2d)])
    plt.show()

def make_animation(x2d=None, y2d=None, psi=None, paths=None):
    print(f"Making animation.")
    paths = paths or {}
    if (x2d is None or y2d is None or psi is None) and 'psi_loadpath' in paths:
        data = np.load(paths['psi_loadpath'])
        x2d = data['x2d'] if x2d is None else x2d
        y2d = data['y2d'] if y2d is None else y2d
        psi = data['psi'] if psi is None else psi
    psi_sq = np.abs(psi) ** 2
    fig, ax = plt.subplots()
    image = ax.imshow(psi_sq[0], interpolation='none', extent=[np.min(y2d), np.max(y2d), np.min(x2d), np.max(x2d)],
                      animated=True)
    def update(frame):
        image.set_data(psi_sq[frame])
        return [image]
    ani = animation.FuncAnimation(fig=fig, func=update, frames=psi.shape[0], interval=20, blit=True)
    if 'savepath' in paths:
        ani.save(paths['savepath'])
    return ani