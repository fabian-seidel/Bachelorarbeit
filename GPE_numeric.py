import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import pyfftw

def ddsigma(sigma, Omega, U):
    D = len(sigma)
    return (U/((2*np.pi)**(D/2)*np.prod(sigma))+1/sigma**2-Omega**2*sigma**2)/sigma

def imaginary_time_ev_direct(U = 10, x_max = 8, tau_max = 10, dtau = 0.0005, dx= 0.025):
    num_x = int(2 * x_max / dx)
    psi = np.ones(num_x)/num_x
    tau = 0
    H_kin = - 2 * np.diag(np.ones(num_x), 0) + np.diag(np.ones(num_x - 1), 1) + np.diag(np.ones(num_x - 1), -1)
    H_kin[0, -1] = H_kin[-1, 0] = 1
    H_kin = -0.5 * (H_kin) / (dx ** 2)
    x = np.linspace(-x_max, x_max, num_x, endpoint=False)
    pot = x ** 2 / 2
    while tau<tau_max:
        H_psi = H_kin @ psi + (pot + U*np.abs(psi)**2)*psi
        psi = psi - dtau * H_psi
        psi /= np.sqrt(np.trapezoid(np.abs(psi) ** 2, x))
        tau+=dtau
    return x, psi

def imaginary_time_ev_4th(U = 10, x_max = 8, tau_max = 10, dtau = 0.0001, dx= 0.025):
    num_x = int(2*x_max/dx)
    c0 = -2**(1/3)/(2-2**(1/3))
    c1 = 1/(2-2**(1/3))
    a = np.array([c1/2,(c0+c1)/2,(c0+c1)/2,c1/2])
    b = np.array([c1,c0,c1])
    k_sq = (2*np.pi*np.fft.fftfreq(num_x, d=dx))**2
    Ma = np.exp(-dtau*np.outer(a,k_sq)/2)
    x = np.linspace(-x_max,x_max,num_x,endpoint=False)
    pot = x**2/2
    # psi = dx*np.exp(-x**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    psi = np.ones(num_x)/num_x
    psi = np.fft.fft(psi)
    tau = 0
    while tau < tau_max:
        for factor_num in range(3,0,-1):
            psi *= Ma[factor_num]
            psi = np.fft.ifft(psi, n = num_x)
            psi /= np.sqrt(np.trapezoid(np.abs(psi)**2, x))
            Mb = np.exp(b[factor_num-1]*dtau*(-pot-U*np.abs(psi)**2))
            psi *= Mb
            psi = np.fft.fft(psi)
        psi *= Ma[0]
        tau+=dtau
    psi = np.fft.ifft(psi, n = num_x)
    psi /= np.sqrt(np.trapezoid(np.abs(psi)**2, x))
    return x, psi

def imaginary_time_ev_2nd(U = 25, x_max = 8, tau_max = 10, dtau = 0.001, dx= 0.025):
    num_x = int(2 * x_max / dx)
    x = np.linspace(-x_max, x_max, num_x, endpoint=False)
    k_sq = (2 * np.pi * np.fft.fftfreq(num_x, d=dx)) ** 2
    pot = x ** 2 / 2
    psi = np.ones(num_x) / num_x
    psi = np.fft.fft(psi)
    psi = np.exp(-dtau * k_sq / 4) * psi
    tau = 0
    while tau < tau_max:
        psi = np.fft.ifft(psi)
        Mb = np.exp(dtau * (-pot-U*np.abs(psi)**2))
        psi *= Mb
        psi /= np.sqrt(np.trapezoid(np.abs(psi)**2, x))
        psi = np.fft.fft(psi)
        psi *= np.exp(-dtau * k_sq / 2)
        tau += dtau
    psi = np.fft.ifft(psi)
    Mb = np.exp(dtau * (-pot - U*np.abs(psi) ** 2))
    psi *= Mb
    psi /= np.sqrt(np.trapezoid(np.abs(psi) ** 2, x))
    psi = np.fft.fft(psi)
    psi *= np.exp(-dtau * k_sq / 4)
    psi = np.fft.ifft(psi)
    psi /= np.sqrt(np.trapezoid(np.abs(psi) ** 2, x))
    return x, psi

def psisq_x_plot(x, psi, U):
    num_x = len(x)
    plot_range = np.s_[num_x // 2:int(num_x * 0.75)]
    plt.plot(x[plot_range], np.abs(psi)[plot_range] ** 2, label = 'Imaginary Time Evolution')
    D = 1
    sigma = scipy.optimize.root(ddsigma, [2] * D, args=(1, U)).x
    plt.plot(x[plot_range], np.exp(-x[plot_range] ** 2 / (sigma ** 2)) / (sigma * np.sqrt(np.pi)), label = 'Gaussian')
    rho = (3*U/2)**(1/3)
    plt.plot(x[plot_range], np.maximum(0,(3*(rho**2 - x[plot_range]**2)/(4*rho**3))), label='Thomas-Fermi')
    plt.title(f'Imaginary time evolution result for the GPE ground state, U = {U}')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel(r'$|\psi|^2$')
    plt.show()


def radius_U_plot(U_max = 25, x_max = 14, tau_max = 10, dtau = 0.001, dx= 0.025):
    Us = np.linspace(0,U_max,10)
    radius = []
    for idx, U in enumerate(Us):
        x, psi = imaginary_time_ev_2nd(U=U, x_max = x_max, tau_max = tau_max, dtau = dtau, dx= dx)
        r = np.sqrt(2*x**2 @ np.abs(psi)**2 * dx)
        radius.append(r)
    plt.scatter(Us, radius, label=r'$r_{GP}$')
    radius = np.linspace(1,2.2,100)
    Us = np.sqrt(2*np.pi)*(radius**3-1/radius)
    plt.plot(Us, radius, label=r'$r_G$')
    Us = np.linspace(0,U_max,100)
    radius = np.sqrt(2/5)*(3*Us/2)**(1/3)
    plt.plot(Us, radius, label=r'$r_F$')
    plt.title('Width of the 1D GPE ground state for different approximations')
    plt.legend()
    plt.xlabel('U')
    plt.ylabel('width')
    plt.show()

def energy_U_plot(U_max = 25, x_max = 14, tau_max = 10, dtau = 0.001, dx= 0.025):
    Us = np.linspace(0, U_max, 10)
    energy = []
    for idx, U in enumerate(Us):
        x, psi = imaginary_time_ev_2nd(U=U, x_max=x_max, tau_max=tau_max, dtau=dtau, dx=dx)
        energy_density = np.abs(np.pad(np.diff(psi)/dx,(0,1)))**2/2 + (x**2/2)*np.abs(psi)**2+U*np.abs(psi)**4/2
        energy.append(np.trapezoid(energy_density,x))
    plt.scatter(Us, energy, label=r'$E_{GP}$')
    Us = np.linspace(1, U_max, 100)
    rho = (3*Us/2)**(1/3)
    energy = rho**2/10 + 3*Us/(10*rho)
    plt.plot(Us, energy, label=r'$E_{TF}$')
    radius = np.linspace(1, 2.2, 100)
    Us = np.sqrt(2 * np.pi) * (radius ** 3 - 1 / radius)
    energy = (1/radius**2 + radius**2)/4 + Us/(2*radius*np.sqrt(2*np.pi))
    plt.plot(Us, energy, label=r'$E_G$')
    plt.xlabel('U')
    plt.ylabel('E')
    plt.title('Energy of the 1D GPE ground state for different approximations')
    plt.legend()
    plt.show()

def psi_time_ev_4th(x, psi0, ts, Us):
    num_x = len(x)
    dx = x[1]-x[0]
    c0 = -2**(1/3)/(2-2**(1/3))
    c1 = 1/(2-2**(1/3))
    a = np.array([c1/2,(c0+c1)/2,(c0+c1)/2,c1/2])
    b = np.array([c1,c0,c1])
    k_sq = (2*np.pi*np.fft.fftfreq(num_x, d=dx))**2
    dt = ts[1] - ts[0]
    Ma = np.exp(-1j*dt*np.outer(a,k_sq)/2)
    pot = x**2/2
    # psi = dx*np.exp(-x**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    psi = psi0
    psi = np.fft.fft(psi)
    psis = np.zeros((len(ts),num_x), dtype=np.complex128)
    for idx, t in enumerate(ts):
        for factor_num in range(3,0,-1):
            psi *= Ma[factor_num]
            psi = np.fft.ifft(psi, n = num_x)
            psi /= np.sqrt(np.trapezoid(np.abs(psi)**2, x))
            Mb = np.exp(b[factor_num-1]*1j*dt*(-pot-Us[idx]*np.abs(psi)**2))
            psi *= Mb
            psi = np.fft.fft(psi)
        psi *= Ma[0]
        psi = np.fft.ifft(psi, n=num_x)
        psi /= np.sqrt(np.trapezoid(np.abs(psi) ** 2, x))
        psis[idx] = psi
        psi = np.fft.fft(psi)
    # np.save(f'tspsis_U{np.max(Us)}',np.array((ts,psis)))
    return ts, psis

def radius_time_ev_4th(x, psi0, ts, Us):
    dx = x[1] - x[0]
    ts, psis = psi_time_ev_4th(x, psi0, ts, Us)
    radius = np.sqrt(2*np.matmul(np.abs(psis) ** 2 * dx,x ** 2))
    return ts, radius

def pos_time_ev_4th(x, psi0, ts, Us):
    dx = x[1] - x[0]
    ts, psis = psi_time_ev_4th(x, psi0, ts, Us)
    pos = np.matmul(np.abs(psis)**2*dx,x)
    return ts, pos

def Uquench_time_ev(U_0 = 1, U_max = 2, t_max = 19, num_t = 100000):
    x, psi0 = imaginary_time_ev_2nd(U = U_0)
    ts = np.linspace(-1,t_max,num_t)
    Us = np.minimum(U_max*np.ones_like(ts), U_max+(U_max-U_0)*ts)
    ts, radius = radius_time_ev_4th(x, psi0, ts, Us)
    np.save(f'ts_radius_U{U_max}_t{t_max}.npy',(ts, radius))
    # return ts, radius
    plt.plot(ts, radius)
    plt.xlabel('t')
    plt.ylabel('width')
    plt.title(f'Width time evolution after initial U ramp from {U_0} to {U_max}')
    plt.show()

def plot_fft():
    ts, radius = np.load('ts_radius_U2_t20.npy')
    ts, radius = ts[ts > 0], radius[ts > 0]
    radius_centered = radius - np.mean(radius)
    radius_fft = np.abs(np.fft.fft(radius_centered))
    omega = 2 * np.pi * np.fft.fftfreq(len(radius), ts[1]-ts[0])
    plot_range = (omega>0) & (omega<4)
    omega_max = omega[np.argmax(radius_fft[plot_range])+1]
    plt.axvline(omega_max)
    plt.scatter(omega[plot_range],radius_fft[plot_range], s=1)
    plt.xlabel(r'$\omega$')
    plt.yscale('log')
    plt.title(r'FFT of GPE radius dynamics after U quench from 1 to 20, $\omega_{max}$ =' f' {omega_max:.2f}')
    plt.show()

def omega_U_plot():
    Us = np.linspace(0,20,15)
    omegas = []
    for U in Us:
        # ts, radius = Uquench_time_ev(U_0 = U*0.9, U_max = U)
        ts, radius = np.load(f'ts_radius_U{U}_t19.npy')
        ts, radius = ts[ts>0], radius[ts>0]
        radius_centered = radius - np.mean(radius)
        crossings = (radius_centered[:-1] > 0) & (radius_centered[1:] < 0)
        crossings_idx = np.where(crossings)[0]
        omegas.append(2 * np.pi * (np.sum(crossings)-1) / ((crossings_idx[-1] - crossings_idx[0]) * (ts[1]-ts[0])))
    plt.scatter(Us, omegas, marker='.', label='numerical GPE')
    Us = np.linspace(0,20,100)
    omegas = []
    for U in Us:
        omega_0 = scipy.optimize.root(ddsigma, [2], args=(1, U)).x
        omegas.append(np.sqrt(1+3/omega_0**4+2*U/(np.sqrt(2*np.pi)*omega_0**3)))
    plt.plot(Us, omegas, label='Gaussian')
    plt.title(r'$\omega$ of the breathing mode for different U with D = 1')
    plt.ylabel(r'$\omega$')
    plt.xlabel('U')
    plt.axhline(np.sqrt(3), color='red', label='Thomas-Fermi')
    plt.legend()
    plt.show()

def dipoleomega_U_plot():
    Us = np.linspace(0, 20, 10)
    omegas = []
    x_max = 14
    for U in Us:
        x, psi0 = imaginary_time_ev_2nd(U=U, x_max=x_max)
        x += 1
        ts = np.linspace(0,20,10000)
        U_ts = np.ones_like(ts)*U
        ts, pos = pos_time_ev_4th(x, psi0, ts, U_ts)
        crossings = (pos[:-1] > 0) & (pos[1:] < 0)
        crossings_idx = np.where(crossings)[0]
        omegas.append(2 * np.pi * (np.sum(crossings) - 1) / ((crossings_idx[-1] - crossings_idx[0]) * (ts[1] - ts[0])))
    plt.scatter(Us, omegas, marker='.')
    plt.title(r'dipole frequency $\omega$ for different U with D = 1')
    plt.ylabel(r'$\omega$')
    plt.xlabel('U')
    plt.show()

# U = 25
# x, psi = imaginary_time_ev_2nd(U = U)
# psisq_x_plot(x, psi, U)
# radius_U_plot()
# energy_U_plot()
# Uquench_time_ev()
# plot_fft()
# omega_U_plot()
dipoleomega_U_plot()