import sympy as sp
import numpy as np
import scipy
import matplotlib.pyplot as plt

def derive_L_gaussian():
    r, t, mu, g = sp.symbols('r t mu g', real=True, positive=True)
    N0 = sp.Function('N0', real=True, positive=True)(t)
    sigma = sp.Function('sigma', real=True, positive=True)(t)
    beta = sp.Function('beta', real=True)(t)
    theta = sp.Function('theta', real=True)(t)

    psi = sp.sqrt(N0 / (sp.pi * sigma ** 2)) * sp.exp(-r ** 2 / (2 * sigma ** 2) + sp.I * beta * r ** 2 + sp.I * theta)
    psi_star = sp.conjugate(psi)

    T1_integrand = sp.I / 2 * (psi_star * sp.diff(psi, t) - psi * sp.diff(psi_star, t)) * 2 * sp.pi * r
    T2_integrand = 1 / 2 * psi_star * ((1 / r) * sp.diff(r * sp.diff(psi, r), r)) * 2 * sp.pi * r
    T3_integrand = (-1 / 2 * r ** 2 + mu) * (psi_star * psi) * 2 * sp.pi * r
    T4_integrand = -1 / 2 * g * (psi_star * psi) ** 2 * 2 * sp.pi * r

    T1 = sp.integrate(sp.simplify(T1_integrand), (r, 0, sp.oo))
    T2 = sp.integrate(sp.simplify(T2_integrand), (r, 0, sp.oo))
    T3 = sp.integrate(sp.simplify(T3_integrand), (r, 0, sp.oo))
    T4 = sp.integrate(sp.simplify(T4_integrand), (r, 0, sp.oo))

    L = sp.simplify(T1 + T2 + T3 + T4)
    return L

def derive_L_thomasfermi():
    r, t, mu, g = sp.symbols('r t mu g', real=True, positive=True)
    N0 = sp.Function('N0', real=True, positive=True)(t)
    rho = sp.Function('rho', real=True, positive=True)(t)
    beta = sp.Function('beta', real=True)(t)
    theta = sp.Function('theta', real=True)(t)

    f = sp.sqrt(2 * N0 / sp.pi * (rho**2 - r**2) / rho**4)
    v = sp.exp(sp.I * beta * r ** 2 + sp.I * theta)
    v_star = sp.conjugate(v)
    psi = f * v
    psi_star = f * v_star

    T1_integrand = sp.I / 2 * (psi_star * sp.diff(psi, t) - psi * sp.diff(psi_star, t)) * 2 * sp.pi * r
    T2_integrand = - f**2 * sp.diff(v, r) * sp.diff(v_star, r) / 2 * 2 * sp.pi * r
    T3_integrand = (-r ** 2 / 2 + mu) * (psi_star * psi) * 2 * sp.pi * r
    T4_integrand = - g * (psi_star * psi) ** 2 / 2 * 2 * sp.pi * r

    T1 = sp.integrate(sp.simplify(T1_integrand), (r, 0, rho))
    T2 = sp.integrate(sp.simplify(T2_integrand), (r, 0, rho))
    T3 = sp.integrate(sp.simplify(T3_integrand), (r, 0, rho))
    T4 = sp.integrate(sp.simplify(T4_integrand), (r, 0, rho))

    L = sp.simplify(T1 + T2 + T3 + T4)
    return L

def derive_R_gaussian():
    r, t, mu, g, gamma = sp.symbols('r t mu g gamma', real=True, positive=True)
    N0 = sp.Function('N0', real=True, positive=True)(t)
    sigma = sp.Function('sigma', real=True, positive=True)(t)
    beta = sp.Function('beta', real=True)(t)
    theta = sp.Function('theta', real=True)(t)

    psi = sp.sqrt(N0 / (sp.pi * sigma ** 2)) * sp.exp(-r ** 2 / (2 * sigma ** 2) + sp.I * beta * r ** 2 + sp.I * theta)
    psi_star = sp.conjugate(psi)

    T2_integrand = - 1 / 2 * sp.diff(psi_star, t) * ((1 / r) * sp.diff(r * sp.diff(psi, r), r)) * 2 * sp.pi * r
    T3_integrand = (1 / 2 * r ** 2 - mu) * (sp.diff(psi_star, t) * psi) * 2 * sp.pi * r
    T4_integrand = g * (sp.diff(psi_star, t) * psi * psi_star * psi) * 2 * sp.pi * r

    T2 = sp.integrate(sp.simplify(T2_integrand), (r, 0, sp.oo))
    T3 = sp.integrate(sp.simplify(T3_integrand), (r, 0, sp.oo))
    T4 = sp.integrate(sp.simplify(T4_integrand), (r, 0, sp.oo))

    T = T2 + T3 + T4
    R = - sp.I * gamma * (T - sp.conjugate(T))
    R = sp.simplify(R)
    return R


def derive_R_thomasfermi():
    r, t, mu, g, gamma = sp.symbols('r t mu g gamma', real=True, positive=True)
    N0 = sp.Function('N0', real=True)(t)
    rho = sp.Function('rho', real=True)(t)
    beta = sp.Function('beta', real=True)(t)
    theta = sp.Function('theta', real=True)(t)

    f = sp.sqrt(2 * N0 / sp.pi * (rho**2 - r**2) / rho**4)
    v = sp.exp(sp.I * (beta * r ** 2 + theta))
    v_star = sp.exp(-sp.I * (beta * r ** 2 + theta))

    psi = f * v
    psi_star = f * v_star
    dot_psi = sp.diff(psi, t)
    dot_psi_star = sp.diff(psi_star, t)
    grad_psi_TF = f * sp.diff(v, r)
    grad_psi_star_TF = f * sp.diff(v_star, r)
    grad_dot_psi_TF = sp.diff(grad_psi_TF, t)
    grad_dot_psi_star_TF = sp.diff(grad_psi_star_TF, t)

    T2_integrand = (grad_dot_psi_TF * grad_psi_star_TF - grad_dot_psi_star_TF * grad_psi_TF) / 2
    T3_integrand = (r ** 2 / 2 - mu) * (dot_psi * psi_star - dot_psi_star * psi)
    T4_integrand = g * (dot_psi * psi_star - dot_psi_star * psi) * (psi * psi_star)

    R_integrand = sp.I * gamma * (T2_integrand + T3_integrand + T4_integrand) * 2 * sp.pi * r
    R_integrand = sp.simplify(R_integrand)

    R = sp.integrate(R_integrand, (r, 0, rho))

    return sp.simplify(R)

def dparameters_dt_gaussian(t, N0sigmabeta, g, mu, gamma):
    N0 = N0sigmabeta[0]
    sigma = N0sigmabeta[1]
    beta = N0sigmabeta[2]
    N0_dot = -2*gamma*N0*((1/2 + g*N0/(2*np.pi))/sigma**2 + (2*beta**2+1/2)*sigma**2 - mu)
    sigma_dot = -gamma*sigma*(- (1/2 + g*N0/(4*np.pi))/sigma**2 + (2*beta**2+1/2)*sigma**2)+2*sigma*beta
    beta_dot = +(1+g*N0/(2*np.pi))/(2*sigma**4)-2*beta**2-1/2-2*gamma*beta/sigma**2
    return [N0_dot, sigma_dot, beta_dot]

def dparameters_dt_thomasfermi(t, N0rhobeta, g, mu, gamma):
    N0 = N0rhobeta[0]
    rho = N0rhobeta[1]
    beta = N0rhobeta[2]
    N0_dot = gamma * N0 * (6*mu*rho**2 - 8*g*N0/np.pi - (4*beta**2+1)*rho**4) / (3*rho**2)
    rho_dot = 2*beta*rho + gamma * (4*g*N0/np.pi - (4*beta**2 + 1)*rho**4) / (12*rho)
    beta_dot = 2*g*N0/(np.pi*rho**4) - 2*beta**2 - 1/2
    return [N0_dot, rho_dot, beta_dot]


def radius_t_plot_variational(t_num=1000, T=30, g_before=0.0103, g=0.01, N0_eq=1000, gamma=0.04, ansatz='gaussian'):
    t_eval = np.linspace(0, T, t_num)

    if ansatz=='gaussian':
        sigma_gs = (1 + g * N0_eq / (2 * np.pi)) ** (1 / 4)
        sigma_0 = (1 + g_before * N0_eq / (2 * np.pi)) ** (1 / 4)
        params_init = [N0_eq, sigma_0, 0]
        mu = 1 / (2 * sigma_gs ** 2) + sigma_gs ** 2 / 2 + (g * N0_eq) / (2 * np.pi * sigma_gs ** 2)
        diff_function = dparameters_dt_gaussian
    elif ansatz=='thomas-fermi':
        rho_gs = (4 * g * N0_eq / np.pi) ** (1 / 4)
        rho_0 = (4 * g_before * N0_eq / np.pi) ** (1 / 4)
        params_init = [N0_eq, rho_0, 0]
        mu = rho_gs**2/6 + 4 * g * N0_eq / (3 * np.pi * rho_gs**2)
        diff_function = dparameters_dt_thomasfermi
    else:
        print("Only ansatz='gaussian' or 'thomas-fermi' possible")

    sol = scipy.integrate.solve_ivp(diff_function, (0, T), params_init, args=(g, mu, gamma), t_eval=t_eval)
    N0 = sol.y[0]
    if ansatz=='gaussian':
        width = sol.y[1]
    elif ansatz=='thomas-fermi':
        width = sol.y[1] / np.sqrt(3)

    fig, ax1 = plt.subplots()
    ax1.plot(t_eval, width, label='width')
    ax1.set_xlabel('t')
    ax1.set_ylabel('width')
    ax2 = ax1.twinx()
    ax2.set_ylabel(r'$N_0$')
    ax2.plot(t_eval, N0, color='red', label=r'$N_0$')
    # fig.legend()
    fig.tight_layout()

    plt.show()

# print(sp.latex(derive_R_thomasfermi()))
radius_t_plot_variational(ansatz='thomas-fermi')