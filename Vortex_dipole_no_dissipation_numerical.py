import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def itime_ev_2nd_vortex_dipole(x_max=6, y_max=6, tau_max = 2, dtau = 0.001, dx=0.025, dy=0.025, g=80, x1_0=0.8):
    z1_0 = x1_0 + 0j
    z2_0 = -x1_0 + 0j

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
            print(tau)
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
    np.savez(f'vortex_dipole_numerical_saves/g{g}_d{x1_0:.1f}_init', x2d=x2d, y2d=y2d, psi=psi)
    return cp.asnumpy(x2d), cp.asnumpy(y2d), cp.asnumpy(psi)

def psi_time_ev_4th(x2d, y2d, psi0, t, g):
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
    Ma = cp.exp(-1j*dt*cp.multiply.outer(a,k_sq)/2)

    psi = cp.zeros((int(len(t)/100), num_y, num_x), dtype=cp.complex128)
    psi[0] = psi0.copy()
    psi_i = cp.fft.fft2(psi0)
    for idx in range(1, len(t)):
        for factor_num in range(3,0,-1):
            psi_i *= Ma[factor_num]
            psi_i = cp.fft.ifft2(psi_i)
            psi_i /= cp.sqrt(cp.sum(cp.abs(psi_i)**2)*dx*dy)
            Mb = cp.exp(b[factor_num-1]*1j*dt*(-pot-g*cp.abs(psi_i)**2))
            psi_i *= Mb
            psi_i = cp.fft.fft2(psi_i)
        psi_i *= Ma[0]
        psi_i = cp.fft.ifft2(psi_i)
        psi_i /= cp.sqrt(cp.sum(cp.abs(psi_i)**2)*dx*dy)
        if idx % 100 == 0:
            psi[int(idx/100)] = psi_i
            print(t[idx])
        psi_i = cp.fft.fft2(psi_i)
    return cp.asnumpy(t), cp.asnumpy(psi)

def plot_groundstate_density(g, x1_0):
    data = np.load(f'vortex_dipole_numerical_saves/g{g}_d{x1_0:.1f}_init.npz')
    plt.imshow(np.abs(data['psi'])**2)
    plt.show()

def psi_time_ev_from_groundstate(g, x1_0):
    groundstate_data = np.load(f'vortex_dipole_numerical_saves/g{g}_d{x1_0:.1f}_init.npz')
    x2d, y2d, psi0 = groundstate_data['x2d'], groundstate_data['y2d'], groundstate_data['psi']
    t = np.linspace(0, 20, 100000)
    t, psi = psi_time_ev_4th(x2d, y2d, psi0, t, g)
    np.savez(f'vortex_dipole_numerical_saves/g{g}_d{x1_0:.1f}_time_ev.npz', t=t, psi=psi)

def make_animation(g, x1_0):
    time_ev_data = np.load(f'vortex_dipole_numerical_saves/g{g}_d{x1_0:.1f}_time_ev.npz')
    t, psi = time_ev_data['t'], time_ev_data['psi']
    psi_sq = np.abs(psi)**2
    fig, ax = plt.subplots()
    image = ax.imshow(psi_sq[0], interpolation='none', extent=[-6,6,-6,6], animated=True)

    def update(frame):
        image.set_data(psi_sq[frame])
        return [image]

    ani = animation.FuncAnimation(fig=fig, func=update, frames=psi.shape[0], interval=20, blit=True)
    ani.save(f'vortex_dipole_numerical_saves/g{g}_d{x1_0:.1f}_animation.mp4')

def angle_diff(a, b):
    return (a-b+np.pi) % (2*np.pi) - np.pi

def find_vortex(g, x1_0, amp_threshold=0.01):
    psi = np.load(f'vortex_dipole_numerical_saves/g{g}_d{x1_0:.1f}_time_ev.npz')['psi']
    psi_angle = np.angle(psi)
    dl = angle_diff(psi_angle[:, :, :-1], psi_angle[:, :, 1:])
    du = angle_diff(psi_angle[:, :-1, :], psi_angle[:, 1:, :])
    dr = -dl
    dd = -du
    total_phase = dl[:, :-1, :] + du[:, :, 1:] + dr[:, 1:, :] + dd[:, :, :-1]
    amplitude = np.abs(psi[:, :-1, :-1])
    threshold = amp_threshold * np.max(amplitude)
    density_mask = amplitude > threshold
    vortex_number = np.round(total_phase/(2*np.pi))
    cw_vortex_map = (vortex_number == 1) & density_mask
    ccw_vortex_map = (vortex_number == -1) & density_mask
    np.savez(f'vortex_dipole_numerical_saves/g{g}_d{x1_0:.1f}_vortex_maps', cw_vortex_map=cw_vortex_map, ccw_vortex_map=ccw_vortex_map)
    return cw_vortex_map, ccw_vortex_map

def plot_vortex_trajectory(g, x1_0):
    data = np.load(f'vortex_dipole_numerical_saves/g{g}_d{x1_0:.1f}_vortex_maps.npz')
    cw_vortex_map, ccw_vortex_map = data['cw_vortex_map'], data['ccw_vortex_map']
    cw_vortex_idx = cw_vortex_map.nonzero()
    ccw_vortex_idx = ccw_vortex_map.nonzero()

    x_max, y_max = 6, 6
    x = np.linspace(-x_max, x_max, cw_vortex_map.shape[2])
    y = np.linspace(-y_max, y_max, cw_vortex_map.shape[1])
    plt.scatter(x[cw_vortex_idx[2]], y[cw_vortex_idx[1]], s=1)
    plt.scatter(x[ccw_vortex_idx[2]], y[ccw_vortex_idx[1]], s=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def plot_vortex_trajectories_g150():
    x_max, y_max = 6, 6

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    axes[0][0].set_ylabel('y')
    axes[1][0].set_ylabel('y')
    axes[1][0].set_xlabel('x')
    axes[1][1].set_xlabel('x')
    axes = axes.flatten()
    for idx, x1_0 in enumerate([0.8, 1.1, 1.3, 1.5]):
        data = np.load(f'vortex_dipole_numerical_saves/g150_d{x1_0:.1f}_vortex_maps.npz')
        cw_vortex_map, ccw_vortex_map = data['cw_vortex_map'], data['ccw_vortex_map']
        cw_vortex_idx = cw_vortex_map.nonzero()
        ccw_vortex_idx = ccw_vortex_map.nonzero()
        x = np.linspace(-x_max, x_max, cw_vortex_map.shape[2])
        y = np.linspace(-y_max, y_max, cw_vortex_map.shape[1])
        axes[idx].plot(x[cw_vortex_idx[2]], y[cw_vortex_idx[1]])
        axes[idx].plot(x[ccw_vortex_idx[2]], y[ccw_vortex_idx[1]])
    plt.show()

def plot_vortex_trajectories_low_g():
    x_max, y_max = 6, 6

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    axes[0].set_ylabel('y')
    axes[0].set_xlabel('x')
    axes[1].set_xlabel('x')
    axes = axes.flatten()
    for idx, g in enumerate([10, 50]):
        data = np.load(f'vortex_dipole_numerical_saves/g{g}_d0.8_vortex_maps.npz')
        cw_vortex_map, ccw_vortex_map = data['cw_vortex_map'], data['ccw_vortex_map']
        cw_vortex_idx = cw_vortex_map.nonzero()
        ccw_vortex_idx = ccw_vortex_map.nonzero()
        x = np.linspace(-x_max, x_max, cw_vortex_map.shape[2])
        y = np.linspace(-y_max, y_max, cw_vortex_map.shape[1])
        axes[idx].scatter(x[cw_vortex_idx[2]], y[cw_vortex_idx[1]], s=1)
        axes[idx].scatter(x[ccw_vortex_idx[2]], y[ccw_vortex_idx[1]], s=1)
    plt.show()

def execute_for_g_x1_0_cases(func, cases=None):
    if cases is None:
        cases = [(150, 0.8), (150, 1.1), (150, 1.3), (150, 1.5), (10, 0.8), (50, 0.8)]
    for g, x1_0 in cases:
        print(f'Calulating case g={g}, x1_0={x1_0:.1f}')
        func(g=g, x1_0=x1_0)

# calculate_groundstates()
# plot_groundstate_density(10, 0.8)
# psi_time_ev_from_groundstate(150, 0.8)
# make_animation(150, 0.8)
# plot_vortex_trajectory(150, 0.8)
# execute_for_g_x1_0_cases(plot_vortex_trajectory)
# find_vortex(150, 0.8)
# plot_vortex_trajectories_g150()
plot_vortex_trajectories_low_g()


