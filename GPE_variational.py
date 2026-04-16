import numpy as np
import scipy.integrate, scipy.fft, scipy.optimize
import matplotlib.pyplot as plt

def ddsigma(sigma, Omega, U):
    D = len(sigma)
    return (U/((2*np.pi)**(D/2)*np.prod(sigma))+1/sigma**2-Omega**2*sigma**2)/sigma

def diff(t, sigmadsigma, Omega, U):
    D = int(len(sigmadsigma) / 2)
    sigma=sigmadsigma[:D]
    dsigma=sigmadsigma[D:]
    dsigma_diff=ddsigma(sigma,Omega, U)
    sigma_diff=dsigma
    # print(dsigma_diff, sigma_diff)
    return np.concatenate([sigma_diff, dsigma_diff])

def diff_mod(t, sigmadsigma, Omega, p, q, Omega_mod):
    D = int(len(sigmadsigma) / 2)
    sigma=sigmadsigma[:D]
    dsigma=sigmadsigma[D:]
    dsigma_diff=((p+q*np.cos(Omega_mod*t))/((2*np.pi)**(D/2)*np.prod(sigma))+1/sigma**2-Omega**2*sigma**2)/sigma
    sigma_diff=dsigma
    # print(dsigma_diff, sigma_diff)
    return np.concatenate([sigma_diff, dsigma_diff])

def sigma_t_plot(N = 1000, T = 50, U=6):
    D = 3
    sigma_0 = scipy.optimize.root(ddsigma, [2] * D, args=(1, U)).x
    sigma_init = 1.01*sigma_0
    sigma_init[-1] = 0.99*sigma_0[-1]
    sigmadsigma_init = np.concatenate([sigma_init, np.zeros(D)])
    t_eval = np.linspace(0, T, N)
    sol = scipy.integrate.solve_ivp(diff, (0, T), sigmadsigma_init, args=(1, U), t_eval=t_eval)
    sigma_x = sol.y[0, :]
    sigma_z = sol.y[2, :]
    plt.plot(t_eval,sigma_x)
    plt.plot(t_eval,sigma_z)
    plt.show()

def sigma_t_plot_mod(N = 10000, T = 600, p=4, q=3, Omega_mod=2.01):
    D = 3
    sigma_0 = scipy.optimize.root(ddsigma, [2] * D, args=(1, p)).x
    sigma_init = 1.01*sigma_0
    # sigma_init[-1] = 0.99*sigma_0[-1]
    sigmadsigma_init = np.concatenate([sigma_init, np.zeros(D)])
    t_eval = np.linspace(0, T, N)
    sol = scipy.integrate.solve_ivp(diff_mod, (0, T), sigmadsigma_init, args=(1, p, q, Omega_mod), t_eval=t_eval, method='Radau')
    sigma_x = sol.y[0, :]
    # sigma_z = sol.y[2, :]
    plt.plot(sol.t,sigma_x)
    # plt.plot(t_eval,sigma_z)
    plt.show()

def resonanceamplitude_plot(N = 10000, T = 600, p=4, q=3):
    D = 3
    sigma_0 = scipy.optimize.root(ddsigma, [2] * D, args=(1, p)).x
    sigma_init = sigma_0
    # sigma_init = 1.01*sigma_0
    # sigma_init[-1] = 0.99*sigma_0[-1]
    sigmadsigma_init = np.concatenate([sigma_init, np.zeros(D)])
    t_eval = np.linspace(0, T, N)
    ampl = []
    Omega_mods = np.linspace(1, 5, 100)
    for Omega_mod in Omega_mods:
        sol = scipy.integrate.solve_ivp(diff_mod, (0, T), sigmadsigma_init, args=(1, p, q, Omega_mod), t_eval=t_eval,
                                        method='Radau')
        sigma_x = sol.y[0, :]
        ampl.append((np.max(sigma_x) - np.min(sigma_x)) / 2)
    # sigma_z = sol.y[2, :]
    plt.scatter(Omega_mods, ampl, marker = '.')
    # plt.plot(t_eval,sigma_z)
    plt.show()


def fft_plot(N = 10000, T = 500, U=25):
    dt = T / N
    t_eval=np.linspace(0,T,N)
    D = 3
    sigma_0 = scipy.optimize.root(ddsigma, [2] * D, args=(1, U)).x
    sigma_init = 1.01 * sigma_0
    sigma_init[-1] = 0.99 * sigma_0[-1]
    sigmadsigma_init = np.concatenate([sigma_init, np.zeros(D)])
    sol = scipy.integrate.solve_ivp(diff, (0,T), sigmadsigma_init, args=(1,U), t_eval=t_eval)
    sigma_x=sol.y[0,:]
    sigma_x_centered = sigma_x - np.mean(sigma_x)
    sigma_x_f = scipy.fft.fft(sigma_x_centered)
    freq = scipy.fft.fftfreq(N, dt)
    # plt.plot(sol.t,sigma_x)
    plt.plot(2*np.pi*freq[:N//20],np.abs(sigma_x_f[:N//20]))
    plt.show()


def freq_U_plot(N=10000, T=500, U_max=25):
    Us = []
    omegas = []
    D = 1
    for U in range(U_max):
        Us.append(U)
        t_eval=np.linspace(0,T,N)
        sigma_0=scipy.optimize.root(ddsigma, [2]*D, args=(1,U)).x
        sigma_init=sigma_0*1.01
        sigma_init[-1]=sigma_0[-1]*0.99
        sigmadsigma_init=np.pad(sigma_init,((0,D),))
        sol = scipy.integrate.solve_ivp(diff, (0,T), sigmadsigma_init, args=(1,U), t_eval=t_eval)
        sigma_x = sol.y[0,:]
        sigma_x_centered = sigma_x - np.mean(sigma_x)
        crossings = (sigma_x_centered[:-1]>0)&(sigma_x_centered[1:]<0)
        crossings_idx = np.where(crossings)[0]
        omegas.append(2*np.pi*np.sum(crossings)/((crossings_idx[-1]-crossings_idx[0])*T/N))
    plt.scatter(Us,omegas,marker='.')
    plt.show()

# sigma_t_plot_mod()
freq_U_plot()
# sinfittest()
# fft_plot()
# resonanceamplitude_plot()