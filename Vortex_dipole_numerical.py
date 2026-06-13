import dill
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from plot_utils import plot_by_name


def itime_ev_2nd(x_max=6, y_max=6, tau_max = 2, dtau = 0.001, dx=0.025, dy=0.025, g=80, d=0, gamma=0):
    num_x = int(2 * x_max / dx)
    num_y = int(2 * y_max / dy)
    x, dx = cp.linspace(-x_max, x_max, num_x, endpoint=False, retstep=True)
    y, dy = cp.linspace(-y_max, y_max, num_y, endpoint=False, retstep=True)
    x2d, y2d = cp.meshgrid(x, y)

    k_x = 2*cp.pi*cp.fft.fftfreq(num_x, d=dx)
    k_y = 2*cp.pi*cp.fft.fftfreq(num_y, d=dy)
    k_x2d, k_y2d = cp.meshgrid(k_x, k_y)
    k_sq = k_x2d**2 + k_y2d**2
    pot = 1/2*(x2d**2 + y2d**2)
    psi = cp.ones((num_x,num_y))
    psi = cp.fft.fft2(psi)
    psi = cp.exp(-dtau * k_sq / 4) * psi
    tau = 0
    while tau < tau_max:
        psi = cp.fft.ifft2(psi)
        Mb = cp.exp(dtau * (-pot-g*cp.abs(psi)**2))
        psi *= Mb
        psi /= cp.sqrt(cp.sum(cp.abs(psi)**2*dx*dy))
        psi = cp.fft.fft2(psi)
        psi *= cp.exp(-dtau * k_sq / 2)
        tau += dtau
        if (tau//dtau % 50) == 0:
            print(f"Imaginary time evolution (no vortex): {tau:.3f} of {tau_max}")
    psi = cp.fft.ifft2(psi)
    Mb = cp.exp(dtau * (-pot - g*cp.abs(psi) ** 2))
    psi *= Mb
    psi /= cp.sqrt(cp.sum(cp.abs(psi)**2*dx*dy))
    psi = cp.fft.fft2(psi)
    psi *= cp.exp(-dtau * k_sq / 4)
    psi = cp.fft.ifft2(psi)
    psi /= cp.sqrt(cp.sum(cp.abs(psi)**2*dx*dy))
    np.savez(f'numerical_saves/Vortex_dipole/g{g}_init_no_vortex', x2d=x2d, y2d=y2d, psi=psi)
    return cp.asnumpy(x2d), cp.asnumpy(y2d), cp.asnumpy(psi)

def calculate_mu(g, gamma=0, d=0):
    print(f"Calculating mu for the no-vortex groundstate for g={g}.")
    data = np.load(f'numerical_saves/Vortex_dipole/g{g}_init_no_vortex.npz')
    x2d, y2d, psi = data['x2d'], data['y2d'], data['psi']
    dx = x2d[0, 1] - x2d[0, 0]
    dy = y2d[1, 0] - y2d[0, 0]
    num_x = x2d.shape[1]
    num_y = y2d.shape[0]
    pot = 1 / 2 * (x2d ** 2 + y2d ** 2)
    k_x = 2 * np.pi * np.fft.fftfreq(num_x, d=dx)
    k_y = 2 * np.pi * np.fft.fftfreq(num_y, d=dy)
    k_x2d, k_y2d = np.meshgrid(k_x, k_y)
    k_sq = k_x2d ** 2 + k_y2d ** 2

    N = np.sum(np.abs(psi)**2)*dx*dy
    mu_spacial = np.sum(np.abs(psi)**2 * pot + g*np.abs(psi)**4)*dx*dy
    psi_fft = np.fft.fft2(psi)
    mu_kin = np.sum(1/2*k_sq*np.abs(psi_fft)**2) * dx * dy / (num_x * num_y)
    mu = (mu_spacial + mu_kin)/N
    np.save(f'numerical_saves/Vortex_dipole/g{g}_mu', mu)
    return mu

def itime_ev_2nd_vortex_dipole(x_max=6, y_max=6, tau_max = 2, dtau = 0.001, dx=0.025, dy=0.025, g=80, d=0.8, gamma=0):
    z1_0 = d + 0j
    z2_0 = -d + 0j

    num_x = int(2 * x_max / dx)
    num_y = int(2 * y_max / dy)
    x, dx = cp.linspace(-x_max, x_max, num_x, endpoint=False, retstep=True)
    y, dy = cp.linspace(-y_max, y_max, num_y, endpoint=False, retstep=True)
    x2d, y2d = cp.meshgrid(x, y)
    z = x2d + 1j*y2d
    vortex_phase = (z - z1_0) * (cp.conj(z) - cp.conj(z2_0)) / (cp.abs((z - z1_0) * (cp.conj(z) - cp.conj(z2_0)))+1e-14)

    k_x = 2*cp.pi*cp.fft.fftfreq(num_x, d=dx)
    k_y = 2*cp.pi*cp.fft.fftfreq(num_y, d=dy)
    k_x2d, k_y2d = cp.meshgrid(k_x, k_y)
    k_sq = k_x2d**2 + k_y2d**2
    pot = 1/2*(x2d**2 + y2d**2)
    psi = cp.ones((num_x,num_y))
    psi = cp.fft.fft2(psi)
    psi = cp.exp(-dtau * k_sq / 4) * psi
    tau = 0
    while tau < tau_max:
        psi = cp.fft.ifft2(psi)
        psi = cp.abs(psi) * vortex_phase
        Mb = cp.exp(dtau * (-pot-g*cp.abs(psi)**2))
        psi *= Mb
        psi /= cp.sqrt(cp.sum(cp.abs(psi)**2*dx*dy))
        psi = cp.fft.fft2(psi)
        psi *= cp.exp(-dtau * k_sq / 2)
        tau += dtau
        if (tau//dtau % 50) == 0:
            print(f"Imaginary time evolution: {tau:.3f} of {tau_max}")
    psi = cp.fft.ifft2(psi)
    psi = cp.abs(psi) * vortex_phase
    Mb = cp.exp(dtau * (-pot - g*cp.abs(psi) ** 2))
    psi *= Mb
    psi /= cp.sqrt(cp.sum(cp.abs(psi)**2*dx*dy))
    psi = cp.fft.fft2(psi)
    psi *= cp.exp(-dtau * k_sq / 4)
    psi = cp.fft.ifft2(psi)
    psi = cp.abs(psi) * vortex_phase
    psi /= cp.sqrt(cp.sum(cp.abs(psi)**2*dx*dy))
    np.savez(f'numerical_saves/Vortex_dipole/g{g}_d{d:.1f}_init', x2d=x2d, y2d=y2d, psi=psi)
    return cp.asnumpy(x2d), cp.asnumpy(y2d), cp.asnumpy(psi)

def psi_time_ev_4th(x2d, y2d, psi0, t, g, gamma=0, mu=0):
    x2d = cp.asarray(x2d)
    y2d = cp.asarray(y2d)
    psi0 = cp.asarray(psi0)
    t = cp.asarray(t)

    num_x = x2d.shape[1]
    num_y = x2d.shape[0]
    dx = x2d[0, 1] - x2d[0, 0]
    dy = y2d[1, 0] - y2d[0, 0]

    k_x = 2*cp.pi*cp.fft.fftfreq(num_x, d=dx)
    k_y = 2*cp.pi*cp.fft.fftfreq(num_y, d=dy)
    k_x2d, k_y2d = cp.meshgrid(k_x, k_y)
    k_sq = k_x2d ** 2 + k_y2d ** 2
    pot = 1 / 2 * (x2d ** 2 + y2d ** 2)
    dt = t[1] - t[0]

    c0 = -2 ** (1 / 3) / (2 - 2 ** (1 / 3))
    c1 = 1 / (2 - 2 ** (1 / 3))
    a = cp.array([c1 / 2, (c0 + c1) / 2, (c0 + c1) / 2, c1 / 2])
    b = cp.array([c1, c0, c1])
    Ma = cp.exp(-1j*dt*(1-gamma*1j)*cp.multiply.outer(a,k_sq)/2)

    psi = cp.zeros((int((len(t)-1)/100) + 1, num_y, num_x), dtype=cp.complex128)
    psi_i = psi0.copy()
    psi_i = cp.fft.fft2(psi_i)
    for idx in range(len(t)):
        if idx % 100 == 0:
            psi_i = cp.fft.ifft2(psi_i)
            psi[int(idx/100)] = psi_i
            psi_i = cp.fft.fft2(psi_i)
            print(f"Time evolution: {t[idx]:.3f} of {t[-1]}")
        for factor_num in range(3,0,-1):
            psi_i *= Ma[factor_num]
            psi_i = cp.fft.ifft2(psi_i)
            psi_i *= cp.exp(b[factor_num-1]*1j*dt*(1-gamma*1j)*(-pot-g*cp.abs(psi_i)**2+mu))
            psi_i = cp.fft.fft2(psi_i)
        psi_i *= Ma[0]
    return cp.asnumpy(t), cp.asnumpy(x2d), cp.asnumpy(y2d), cp.asnumpy(psi)


def psi_time_ev_2nd(x2d, y2d, psi0, t, g, gamma=0.1, mu=0):
    x2d = cp.asarray(x2d)
    y2d = cp.asarray(y2d)
    psi0 = cp.asarray(psi0)
    t = cp.asarray(t)
    mu = cp.asarray(mu)

    num_x = x2d.shape[1]
    num_y = x2d.shape[0]
    dx = x2d[0, 1] - x2d[0, 0]
    dy = y2d[1, 0] - y2d[0, 0]
    dt = t[1] - t[0]

    k_x = 2 * cp.pi * cp.fft.fftfreq(num_x, d=dx)
    k_y = 2 * cp.pi * cp.fft.fftfreq(num_y, d=dy)
    k_x2d, k_y2d = cp.meshgrid(k_x, k_y)
    k_sq = k_x2d ** 2 + k_y2d ** 2
    pot = 1 / 2 * (x2d ** 2 + y2d ** 2)

    # --- THE 2/3 RULE (Anti-Aliasing Mask) ---
    k_max = min(cp.max(cp.abs(k_x)), cp.max(cp.abs(k_y)))
    mask = k_sq < ((2 / 3) * k_max) ** 2

    # Kinetic Propagator (Full step)
    Ma = cp.exp(-1j * dt * (1 - gamma * 1j) * k_sq / 2)

    psi = cp.zeros((int((len(t) - 1) / 100) + 1, num_y, num_x), dtype=cp.complex128)
    psi_i = psi0.copy()
    current_norm = 1
    for idx in range(len(t)):
        if idx % 100 == 0:
            psi[int(idx / 100)] = psi_i
            print(f"Time evolution: {t[idx]:.3f} of {t[-1]:.3f}, current Norm: {current_norm}")

        psi_i *= cp.exp(-1j * (dt / 2) * (1 - gamma * 1j) * (pot - mu + g * cp.abs(psi_i) ** 2))

        psi_i = cp.fft.fft2(psi_i)
        psi_i *= Ma
        psi_i *= mask
        psi_i = cp.fft.ifft2(psi_i)

        psi_i *= cp.exp(-1j * (dt / 2) * (1 - gamma * 1j) * (pot - mu + g * cp.abs(psi_i) ** 2))
        current_norm = cp.sum(np.abs(psi_i)**2)*dx*dy

    return cp.asnumpy(t), cp.asnumpy(x2d), cp.asnumpy(y2d), cp.asnumpy(psi)

def plot_groundstate_density(g, d, gamma=0):
    groundstate_data = np.load(f'numerical_saves/Vortex_dipole/g{g}_d{d:.1f}_init.npz')
    x2d, y2d, psi0 = groundstate_data['x2d'], groundstate_data['y2d'], groundstate_data['psi']
    plt.imshow(np.abs(psi0)**2, extent=[np.min(y2d),np.max(y2d),np.min(x2d),np.max(x2d)])
    plt.show()

def psi_time_ev_from_groundstate(g, d, t_stop=20, dt=2e-4, gamma=0, scheme='4th'):
    groundstate_data = np.load(f'numerical_saves/Vortex_dipole/g{g}_d{d:.1f}_init.npz')
    x2d, y2d, psi0 = groundstate_data['x2d'], groundstate_data['y2d'], groundstate_data['psi']
    t = np.linspace(0, 20, int(t_stop/dt))
    mu = np.load(f'numerical_saves/Vortex_dipole/g{g}_mu.npy')
    if scheme == '4th':
        time_ev_method = psi_time_ev_4th
    elif scheme == '2nd':
        time_ev_method = psi_time_ev_2nd
    else:
        print("'scheme' has to be either '4th' or '2nd'")
        return
    t, x2d, y2d, psi = time_ev_method(x2d, y2d, psi0, t, g, gamma=gamma, mu=mu)
    np.savez(f'numerical_saves/Vortex_dipole/g{g}_d{d}_gamma{gamma}_time_ev.npz', t=t, x2d=x2d, y2d=y2d, psi=psi)
    return t, x2d, y2d, psi0

def make_animation(g, d, gamma):
    print(f"Making animation for dynamics with g={g}, d={d}, gamma={gamma}")
    time_ev_data = np.load(f'numerical_saves/Vortex_dipole/g{g}_d{d:.1f}_gamma{gamma}_time_ev.npz')
    t, x2d, y2d, psi = time_ev_data['t'], time_ev_data['x2d'], time_ev_data['y2d'], time_ev_data['psi']
    psi_sq = np.abs(psi)**2
    fig, ax = plt.subplots()
    image = ax.imshow(psi_sq[0], interpolation='none', extent=[np.min(y2d),np.max(y2d),np.min(x2d),np.max(x2d)], animated=True)

    def update(frame):
        image.set_data(psi_sq[frame])
        return [image]

    ani = animation.FuncAnimation(fig=fig, func=update, frames=psi.shape[0], interval=20, blit=True)
    ani.save(f'videos/Vortex_dipole/g{g}_d{d:.1f}_gamma{gamma}_animation.mp4')

def angle_diff(a, b):
    return (a-b+np.pi) % (2*np.pi) - np.pi

def find_vortex(g, d, gamma=0, radius_threshold=4):
    print(f"Finding vortices for dynamics with g={g}, d={d}, gamma={gamma}")
    data = np.load(f'numerical_saves/Vortex_dipole/g{g}_d{d:.1f}_gamma{gamma}_time_ev.npz')
    t, x2d, y2d, psi = data['t'], data['x2d'], data['y2d'], data['psi']
    Nx = x2d.shape[1]-1
    Ny = y2d.shape[0]-1
    Nt = psi.shape[0]

    cw_vortex_map = np.zeros((Nt, Nx, Ny), dtype=bool)
    ccw_vortex_map = np.zeros((Nt, Nx, Ny), dtype=bool)
    radius = np.sqrt(x2d ** 2 + y2d ** 2)[1:, 1:]
    radius_mask = radius < radius_threshold
    for i in range(Nt):
        if i%10 == 0:
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
    with open(f'numerical_saves/Vortex_dipole/g{g}_d{d:.1f}_gamma{gamma}_vortex_idx.dill', 'wb') as file:
        dill.dump({'x2d':x2d, 'y2d':y2d, 'cw_vortex_idx':cw_vortex_idx, 'ccw_vortex_idx':ccw_vortex_idx}, file)
    return t, x2d, y2d, cw_vortex_map, ccw_vortex_map

def plot_vortex_trajectory(g, d, gamma=0):
    with open(f'numerical_saves/Vortex_dipole/g{g}_d{d:.1f}_gamma{gamma}_vortex_idx.dill', 'rb') as file:
        data = dill.load(file)
    x2d, y2d, cw_vortex_idx, ccw_vortex_idx = data['x2d'], data['y2d'],data['cw_vortex_idx'], data['ccw_vortex_idx']
    x, y = x2d[0,:], y2d[:,0]
    plt.scatter(x[cw_vortex_idx[2]], y[cw_vortex_idx[1]], s=1)
    plt.scatter(x[ccw_vortex_idx[2]], y[ccw_vortex_idx[1]], s=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def plot_vortex_trajectories(g, d, gamma, plot_array_shape, plot_type='scatter'):
    fig, axes = plt.subplots(*plot_array_shape, sharex=True, sharey=True, squeeze=False)
    for row in range(plot_array_shape[0]):
        axes[row][0].set_ylabel('y')
    for column in range(plot_array_shape[1]):
        axes[plot_array_shape[0]-1][column].set_xlabel('x')
    axes = axes.flatten()
    for idx, (g_i, d_i, gamma_i) in enumerate(zip(g, d, gamma)):
        data = np.load(f'numerical_saves/Vortex_dipole/g{g_i}_d{d_i}_gamma{gamma_i}_vortex_maps.npz')
        x2d, y2d, cw_vortex_map, ccw_vortex_map = data['x2d'], data['y2d'], data['cw_vortex_map'], data[
            'ccw_vortex_map'] 
        cw_vortex_idx = cw_vortex_map.nonzero()
        ccw_vortex_idx = ccw_vortex_map.nonzero()
        x, y = x2d[0,:], y2d[:,0]
        if plot_type == 'line':
            axes[idx].plot(x[cw_vortex_idx[2]], y[cw_vortex_idx[1]])
            axes[idx].plot(x[ccw_vortex_idx[2]], y[ccw_vortex_idx[1]])
        elif plot_type == 'scatter':
            axes[idx].scatter(x[cw_vortex_idx[2]], y[cw_vortex_idx[1]], s=1)
            axes[idx].scatter(x[ccw_vortex_idx[2]], y[ccw_vortex_idx[1]], s=1)
        else:
            print("Plot type must be either 'line' or 'scatter'.")
    plt.show()

def execute_for_g_d_gamma_cases(func, cases=None, **kwargs):
    if cases is None:
        cases = [(150, 0.8, 0), (150, 1.1, 0), (150, 1.3, 0), (150, 1.5, 0), (10, 0.8, 0), (50, 0.8, 0)]
    for g, d, gamma in cases:
        print(f'Executing case g={g}, d={d:.1f}, gamma={gamma}')
        func(g=g, d=d, gamma=gamma, **kwargs)

def attach_x2d_y2d(g, d, x_max=6, y_max=6, dx=0.025, dy=0.025):
    num_x = int(2 * x_max / dx)
    num_y = int(2 * y_max / dy)
    x, dx = cp.linspace(-x_max, x_max, num_x, endpoint=False, retstep=True)
    y, dy = cp.linspace(-y_max, y_max, num_y, endpoint=False, retstep=True)
    x2d, y2d = cp.meshgrid(x, y)
    t = np.load('times.npy')
    data = np.load(f'numerical_saves/Vortex_dipole/g{g}_d{d:.1f}_vortex_maps.npz')
    cw_vortex_map, ccw_vortex_map = data['cw_vortex_map'], data['ccw_vortex_map']
    np.savez(f'numerical_saves/Vortex_dipole/g{g}_d{d:.1f}_vortex_maps.npz', t=t, x2d=x2d, y2d=y2d, cw_vortex_map=cw_vortex_map, ccw_vortex_map=ccw_vortex_map)

if __name__ == '__main__':
    cases = [(500, 0.5, 0.1)]
    # execute_for_g_d_gamma_cases(itime_ev_2nd, cases=cases, x_max=12, y_max=12, dx=0.025, dy=0.025)
    # execute_for_g_d_gamma_cases(calculate_mu, cases=cases)
    # execute_for_g_d_gamma_cases(itime_ev_2nd_vortex_dipole, cases=cases, x_max=12, y_max=12, dx=0.025, dy=0.025)
    # execute_for_g_d_gamma_cases(plot_groundstate_density, cases=cases)
    # execute_for_g_d_gamma_cases(psi_time_ev_from_groundstate, cases=cases, scheme='2nd_mu')
    # execute_for_g_d_gamma_cases(make_animation, cases=cases)
    execute_for_g_d_gamma_cases(find_vortex, cases=cases)
    # plot_vortex_trajectory(cases[0][0], cases[0][1], gamma=cases[0][2])

    g_g150 = (150,)*4
    d_g150 = (0.8, 1.1, 1.3, 1.5)
    gamma_g150 = (0,)*4
    g_low_g = (10, 50)
    d_low_g = (0.8,)*2
    gamma_low_g = (0,)*2

    plot_dict = {'Numerical vortex trajectories for gamma=0, g=150 and d=0.8, 1.1, 1.3, 1.5': (plot_vortex_trajectories, {'g':g_g150, 'd':d_g150, 'gamma':gamma_g150, 'plot_array_shape':(2,2), 'plot_type':'line'}),
                 'Numerical vortex trajectories for gamma=0, d=0.8 and g=10, 50': (plot_vortex_trajectories, {'g':g_low_g, 'd':d_low_g, 'gamma':gamma_low_g, 'plot_array_shape':(1,2), 'plot_type':'scatter'})}
    # plot_by_name(plot_dict, 'Numerical vortex trajectories for gamma=0, g=150 and d=0.8, 1.1, 1.3, 1.5')

