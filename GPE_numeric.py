import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

def ddsigma(sigma, Omega, U):
    D = len(sigma)
    return (U/((2*np.pi)**(D/2)*np.prod(sigma))+1/sigma**2-Omega**2*sigma**2)/sigma

def imaginary_time_ev_4th(U = 10, x_max = 8, tau_max = 10, dtau = 0.1, dx= 0.025):
    # Doesn't work properly
    D=1
    sigma=scipy.optimize.root(ddsigma, [2]*D, args=(1,U)).x
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
    plt.plot(x, np.abs(psi)**2)
    plt.plot(x, np.exp(-x**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi)))
    plt.show()

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
    x_plot = x[num_x // 2:int(num_x * 0.75)]
    plt.plot(x_plot, np.abs(psi)[num_x//2:int(num_x * 0.75)] ** 2, label = 'Imaginary Time Evolution')
    D = 1
    sigma = scipy.optimize.root(ddsigma, [2] * D, args=(1, U)).x
    plt.plot(x_plot, np.exp(-x_plot ** 2 / (sigma ** 2)) / (sigma * np.sqrt(np.pi)), label = 'Gaussian')
    rho = (3*U/2)**(1/3)
    plt.plot(x_plot, np.maximum(0,(3*(rho**2 - x_plot**2)/(4*rho**3))), label='Thomas-Fermi')
    plt.title(f'Imaginary time evolution result for the GPE ground state, U = {U}')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel(r'$|\psi|^2$')
    plt.show()

imaginary_time_ev_2nd()