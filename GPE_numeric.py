import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

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
# U = 25
# x, psi = imaginary_time_ev_2nd(U = U)
# psisq_x_plot(x, psi, U)
# radius_U_plot()
energy_U_plot()