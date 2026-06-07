import sympy as sp
import numpy as np
from sympy.utilities.lambdify import lambdify
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from plot_utils import plot_by_name
import dill

def get_lagrangian_gaussian():
    g = sp.symbols('g', real=True)
    z1, z2, z1s, z2s = sp.symbols('z1 z2 z1s z2s')
    z1_dot, z2_dot, z1s_dot, z2s_dot = sp.symbols('z1_dot z2_dot z1s_dot z2s_dot')
    A = sp.sqrt(1 / (sp.pi * (2 + (z1 + z2) * (z1s + z2s) + z1 * z1s * z2 * z2s)))
    L1 = sp.I * sp.pi / 2 * A ** 2 * ((z1s + z2s + z1s * z2 * z2s) * z1_dot + (z1 + z2 + z2 * z1 * z1s) * z2s_dot - (
                z1 + z2 + z1 * z2 * z2s) * z1s_dot - (z1s + z2s + z2s * z1 * z1s) * z2_dot)
    L21 = - sp.pi * A ** 2 * (4 + z1 * z1s + z2 * z2s + (z1 + z2) * (z1s + z2s) + z1 * z1s * z2 * z2s)
    L22 = - sp.pi * g * A ** 4 / 8 * (
                3 + 6 * (z1 + z2) * (z1s + z2s) + 3 * (z1 + z2) ** 2 * (z1s + z2s) ** 2 / 2 - (z1 - z2) ** 2 * (
                    z1s - z2s) ** 2 / 2 + 12 * z1 * z1s * z2 * z2s + 4 * (z1 + z2) * (
                            z1s + z2s) * z1 * z1s * z2 * z2s + 2 * z1 ** 2 * z1s ** 2 * z2 ** 2 * z2s ** 2)
    L = L1 + L21 + L22
    return L

def derive_lagrangian_gaussian():
    g = sp.symbols('g', real=True)

    z1, z2, z1s, z2s = sp.symbols('z1 z2 z1s z2s')
    z1_dot, z2_dot, z1s_dot, z2s_dot = sp.symbols('z1_dot z2_dot z1s_dot z2s_dot')

    x, y = sp.symbols('x y', real=True)
    z = x + sp.I * y
    zs = x - sp.I * y

    A2 = 1 / (sp.pi * (2 + (z1 + z2) * (z1s + z2s) + z1 * z1s * z2 * z2s))

    f = sp.exp(-z * zs / 2)

    u = (z - z1) * (zs - z2s) * f
    u_dot = -(z1_dot * (zs - z2s) * f + z2s_dot * (z - z1) * f)

    u_star = (zs - z1s) * (z - z2) * f
    u_dot_star = -(z1s_dot * (z - z2) * f + z2_dot * (zs - z1s) * f)

    laplace_u = sp.diff(u, x, 2) + sp.diff(u, y, 2)

    print("Beginning calculation of L1")
    L1_integrand = sp.expand(u_star * u_dot)
    L1_int = sp.integrate(L1_integrand, (x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo))
    L1_part = sp.I / 2 * A2 * L1_int

    L1_cc_integrand = sp.expand(u * u_dot_star)
    L1_cc_int = sp.integrate(L1_cc_integrand, (x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo))
    L1_cc = - sp.I / 2 * A2 * L1_cc_int

    L1 = L1_part + L1_cc

    print("Beginning calculation of L2, t1")
    t1_integrand = sp.expand(u_star * laplace_u)
    t1_int = sp.integrate(t1_integrand, (x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo))

    print("Beginning calculation of L2, t2")
    t2_integrand = sp.expand((x ** 2 + y ** 2) * u_star * u)
    t2_int = sp.integrate(t2_integrand, (x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo))

    print("Beginning calculation of L2, t3")
    t3_integrand = sp.expand((u_star * u) ** 2)
    t3_int = sp.integrate(t3_integrand, (x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo))

    L2 = 1 / 2 * (A2 * t1_int - A2 * t2_int - g * A2 ** 2 * t3_int)

    L = L1 + L2
    print("Finished L calculation, starting simplification")
    L = sp.simplify(L)
    print("Finished L simplification")
    with open("numerical_saves/Vortex_dipole/lagrangian_gaussian.dill", "wb") as f:
        dill.dump(L, f)
    return L

def load_lagrangian_gaussian():
    with open("numerical_saves/Vortex_dipole/lagrangian_gaussian.dill", "rb") as f:
        L = dill.load(f)
    return L

def get_lagrangian_thomasfermi():
    g, mu = sp.symbols('g mu', real=True)
    mu = sp.sqrt(g / sp.pi)
    z1, z2, z1s, z2s = sp.symbols('z1 z2 z1s z2s')
    z1_dot, z2_dot, z1s_dot, z2s_dot = sp.symbols('z1_dot z2_dot z1s_dot z2s_dot')
    c1 = (z1+z2)*(z1s+z2s)
    c2 = z1*z1s+z2*z2s
    c3 = z1*z1s*z2*z2s
    A = sp.sqrt(3 / (2 * mu ** 2 + 2 * mu * c1 + 3 * c3))
    L1 = sp.I*sp.pi/(2*g)*A**2*((2/3*(z1s+z2s)*mu**3 + z1s*z2*z2s*mu**2) * z1_dot
                                +(2/3*(z1+z2)*mu**3 + z2*z1*z1s*mu**2) * z2s_dot
                                -(2/3*(z1+z2)*mu**3 + z1*z2*z2s*mu**2) * z1s_dot
                                -(2/3*(z1s+z2s)*mu**3 + z2s*z1*z1s*mu**2) * z2_dot)
    L21 = - A**2*(4/3 * mu + c2)
    L22 = - A**2*(2/5*mu**3 + 1/3*mu**2*c1 + 1/3*mu*c3)
    L23 = - A**4*(16/105*mu**5 + 8/15*mu**4*c1 + 2/15*mu**3*(c1**2+12*c3+2*c2*(z1s*z2+z1*z2s)) + 2/3*mu**2*c1*c3 + 1/3*mu*c3**2)
    L = L1 + L21 + L22 + L23
    return L


def integrate_complex_disc(integrand, z, zs, mu):
    expanded = sp.expand(integrand)
    if isinstance(expanded, sp.Add):
        terms = expanded.args
    else:
        terms = [expanded]

    total_integral = 0
    for term in terms:
        coeff, rest = term.as_independent(z, zs, as_Add=False)
        degree_z = sp.degree(rest, z)
        degree_zs = sp.degree(rest, zs)

        if degree_z == degree_zs:
            n = degree_z
            radial_part = (2 ** n * mu ** (n + 1)) / (n + 1)
            total_integral += coeff * 2 * sp.pi * radial_part

    return total_integral

def derive_lagrangian_thomasfermi():
    g, mu = sp.symbols('g mu', real=True)
    z1, z2, z1s, z2s = sp.symbols('z1 z2 z1s z2s')
    z1_dot, z2_dot, z1s_dot, z2s_dot = sp.symbols('z1_dot z2_dot z1s_dot z2s_dot')
    z, zs = sp.symbols('z zs')

    f2 = (mu - z * zs / 2) / g
    v = (z - z1) * (zs - z2s)
    vs = (zs - z1s) * (z - z2)

    v_dot = -z1_dot * (zs - z2s) - z2s_dot * (z - z1)
    vs_dot = -z1s_dot * (z - z2) - z2_dot * (zs - z1s)

    dv_dz = zs - z2s
    dv_dzs = z - z1

    R1_integrand = (sp.I / 2) * (vs * v_dot - v * vs_dot) * f2
    L2_1_integrand = 2 * f2 * vs + vs * (sp.diff(f2, z) * dv_dzs + sp.diff(f2, zs) * dv_dz)
    L2_2_integrand = - (z * zs) * (v * vs) * f2 / 2
    L2_3_integrand = - g * (v * vs) ** 2 * f2 ** 2 / 2

    L1_int = integrate_complex_disc(R1_integrand, z, zs, mu)
    L2_1_int = integrate_complex_disc(L2_1_integrand, z, zs, mu)
    L2_2_int = integrate_complex_disc(L2_2_integrand, z, zs, mu)
    L2_3_int = integrate_complex_disc(L2_3_integrand, z, zs, mu)

    A2 = 3 / (2 * mu ** 2 + 2 * mu * (z1 + z2) * (z1s + z2s) + 3 * z1 * z1s * z2 * z2s)
    L = A2 * (L1_int + L2_1_int + L2_2_int) + (A2 ** 2) * L2_3_int

    L = L.subs(mu, sp.sqrt(g / sp.pi))

    with open("numerical_saves/Vortex_dipole/lagrangian_thomasfermi.dill", "wb") as f:
        dill.dump(L, f)

    return L

def load_lagrangian_thomasfermi():
    with open("numerical_saves/Vortex_dipole/lagrangian_thomasfermi.dill", "rb") as f:
        L = dill.load(f)
    return L

def derive_R_gaussian():
    g, gamma = sp.symbols('g gamma', real=True)

    z1, z2, z1s, z2s = sp.symbols('z1 z2 z1s z2s')
    z1_dot, z2_dot, z1s_dot, z2s_dot = sp.symbols('z1_dot z2_dot z1s_dot z2s_dot')

    x, y = sp.symbols('x y', real=True)
    z = x + sp.I * y
    zs = x - sp.I * y

    A2 = 1 / (sp.pi * (2 + (z1 + z2) * (z1s + z2s) + z1 * z1s * z2 * z2s))

    f = sp.exp(-z * zs / 2)

    u = (z - z1) * (zs - z2s) * f
    u_dot = -(z1_dot * (zs - z2s) * f + z2s_dot * (z - z1) * f)

    u_star = (zs - z1s) * (z - z2) * f
    u_dot_star = -(z1s_dot * (z - z2) * f + z2_dot * (zs - z1s) * f)

    laplace_u = sp.diff(u, x, 2) + sp.diff(u, y, 2)
    laplace_u_star = sp.diff(u_star, x, 2) + sp.diff(u_star, y, 2)

    print("Beginning calculation of R1")
    R1_integrand = sp.expand(u_dot * laplace_u_star)
    R1_int = sp.integrate(R1_integrand, (x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo))
    R1_cc_integrand = sp.expand(u_dot_star * laplace_u)
    R1_cc_int = sp.integrate(R1_cc_integrand, (x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo))

    print("Beginning calculation of R2")
    R2_integrand = sp.expand(u_dot * z * zs * u_star)
    R2_int = sp.integrate(R2_integrand, (x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo))
    R2_cc_integrand = sp.expand(u_dot_star * z * zs * u)
    R2_cc_int = sp.integrate(R2_integrand, (x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo))

    print("Beginning calculation of R3")
    R3_integrand = sp.expand(u_dot * u * u_star**2)
    R3_int = sp.integrate(R3_integrand, (x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo))
    R3_cc_integrand = sp.expand(u_dot_star * u**2 * u_star)
    R3_cc_int = sp.integrate(R3_cc_integrand, (x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo))

    R = -A2*(R1_int-R1_cc_int)/2 + A2*(R2_int-R2_cc_int)/2 + A2**2*g*(R3_int-R3_cc_int)
    R *= sp.I * gamma
    with open("numerical_saves/Vortex_dipole/dissipation_func_gaussian.dill", "wb") as f:
        dill.dump(R, f)
    print("Finished calculating R")
    return R

def load_R_gaussian():
    with open("numerical_saves/Vortex_dipole/dissipation_func_gaussian.dill", "rb") as f:
        R = dill.load(f)
    return R

def get_EL_SOE_matrix(L, R = None, gamma_val=0.1):
    g = sp.symbols('g', real=True)
    z1, z2, z1s, z2s = sp.symbols('z1 z2 z1s z2s')
    z1_dot, z2_dot, z1s_dot, z2s_dot = sp.symbols('z1_dot z2_dot z1s_dot z2s_dot')
    dL_dz1 = sp.diff(L, z1)
    dL_dz2 = sp.diff(L, z2)
    dL_dz1_dot = sp.diff(L, z1_dot)
    dL_dz2_dot = sp.diff(L, z2_dot)
    ddt_dL_z1_dot = sp.diff(dL_dz1_dot, z1)*z1_dot + sp.diff(dL_dz1_dot, z2)*z2_dot + sp.diff(dL_dz1_dot, z1s)*z1s_dot + sp.diff(dL_dz1_dot, z2s)*z2s_dot
    ddt_dL_z2_dot = sp.diff(dL_dz2_dot, z1)*z1_dot + sp.diff(dL_dz2_dot, z2)*z2_dot + sp.diff(dL_dz2_dot, z1s)*z1s_dot + sp.diff(dL_dz2_dot, z2s)*z2s_dot
    EL_z1 = ddt_dL_z1_dot - dL_dz1
    EL_z2 = ddt_dL_z2_dot - dL_dz2

    M11 = sp.diff(EL_z1, z1_dot)
    M12 = sp.diff(EL_z1, z2_dot)
    M13 = sp.diff(EL_z1, z1s_dot)
    M14 = sp.diff(EL_z1, z2s_dot)
    F1 = -EL_z1.subs({z1_dot:0, z2_dot:0, z1s_dot:0, z2s_dot:0})
    M21 = sp.diff(EL_z2, z1_dot)
    M22 = sp.diff(EL_z2, z2_dot)
    M23 = sp.diff(EL_z2, z1s_dot)
    M24 = sp.diff(EL_z2, z2s_dot)
    F2 = -EL_z2.subs({z1_dot:0, z2_dot:0, z1s_dot:0, z2s_dot:0})
    M31 = sp.conjugate(M13)
    M32 = sp.conjugate(M14)
    M33 = sp.conjugate(M11)
    M34 = sp.conjugate(M12)
    M41 = sp.conjugate(M23)
    M42 = sp.conjugate(M24)
    M43 = sp.conjugate(M21)
    M44 = sp.conjugate(M22)
    M = sp.Matrix([[M11, M12, M13, M14], [M21, M22, M23, M24], [M31, M32, M33, M34], [M41, M42, M43, M44]])
    F = sp.Matrix([F1, F2, sp.conjugate(F1), sp.conjugate(F2)])

    if R:
        gamma = sp.symbols('gamma', real=True)
        FR1 = sp.diff(R, z1_dot)
        FR2 = sp.diff(R, z2_dot)
        FR3 = sp.diff(R, z1s_dot)
        FR4 = sp.diff(R, z2s_dot)
        FR = sp.Matrix([FR1, FR2, FR3, FR4])
        FR = FR.subs(gamma, gamma_val)
        F = F - FR

    vars_list = [z1, z2, z1s, z2s, g]

    M_func = lambdify(vars_list, M, 'numpy')
    F_func = lambdify(vars_list, F, 'numpy')

    return M_func, F_func


def z_derivatives(t, state, g_val, M_func, F_func):
    z1_val, z2_val, z1s_val, z2s_val = state
    z1s_val = np.conjugate(z1_val)
    z2s_val = np.conjugate(z2_val)
    args = (z1_val, z2_val, z1s_val, z2s_val, g_val)

    M_num = np.asarray(M_func(*args), dtype=complex)
    F_num = np.asarray(F_func(*args), dtype=complex)

    dz = np.linalg.solve(M_num, F_num).flatten()
    return dz

def solve_dynamics(x1_0, g, M_func, F_func, T = 20):
    res = solve_ivp(z_derivatives, [0, T], [x1_0 + 0j, -x1_0 + 0j, x1_0 + 0j, -x1_0 + 0j], rtol=1e-10, atol=1e-10,
                    args=(g, M_func, F_func), method='BDF')
    return res.y

def plot_vortex_antivortex_trajectories(M_func, F_func):
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    axes[1, 0].set_xlabel('x')
    axes[1, 1].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[1, 0].set_ylabel('y')
    axes = axes.flatten()
    g = [6, 40]
    for idx, x1_0 in enumerate([0.3, 0.6, 0.9, 1.2]):
        print(f"solving dynamics for x1_0 = {x1_0}, g = {g[0]}")
        res_g0 = solve_dynamics(x1_0, g[0], M_func, F_func)
        print(f"solving dynamics for x1_0 = {x1_0}, g = {g[1]}")
        res_g1 = solve_dynamics(x1_0, g[1], M_func, F_func)
        axes[idx].plot(np.real(res_g0[0,:]),np.imag(res_g0[0,:]), color='blue')
        axes[idx].plot(np.real(res_g0[1, :]), np.imag(res_g0[1, :]), color='blue')
        axes[idx].plot(np.real(res_g1[0, :]), np.imag(res_g1[0, :]),  color='red')
        axes[idx].plot(np.real(res_g1[1, :]), np.imag(res_g1[1, :]),  color='red')
    plt.show()

def plot_vortex_antivortex_trajectory(M_func, F_func, g = 40, x1_0 = 1.3, T = 50):
    fig, ax = plt.subplots()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    res = solve_dynamics(x1_0, g, M_func, F_func, T=T)
    print(res.shape)
    ax.plot(np.real(res[0,:]),np.imag(res[0,:]), color='blue')
    ax.plot(np.real(res[1, :]), np.imag(res[1, :]), color='blue')
    plt.show()

if __name__ == '__main__':
    L_gaussian = load_lagrangian_gaussian()
    L_thomasfermi = load_lagrangian_thomasfermi()
    R_gaussian = load_R_gaussian()
    M_gaussian, F_gaussian = get_EL_SOE_matrix(L_gaussian)
    M_gaussian_diss, F_gaussian_diss = get_EL_SOE_matrix(L_gaussian, R=R_gaussian, gamma_val=0.1)
    M_thomasfermi, F_thomasfermi = get_EL_SOE_matrix(L_thomasfermi)

    plot_dict = {'Vortex-Antivortex Trajectories for Gaussian ansatz': (plot_vortex_antivortex_trajectories, {'M_func':M_gaussian, 'F_func':F_gaussian}),
                 'Vortex-Antivortex Trajectories for Thomas-Fermi ansatz': (plot_vortex_antivortex_trajectories, {'M_func':M_thomasfermi, 'F_func':F_thomasfermi}),
                 'Vortex-Antivortex Trajectories for Gaussian ansatz with dissipation': (plot_vortex_antivortex_trajectories, {'M_func':M_gaussian_diss, 'F_func':F_gaussian_diss})}
    plot_by_name(plot_dict, 'Vortex-Antivortex Trajectories for Gaussian ansatz with dissipation')
    # plot_vortex_antivortex_trajectory(M_func_gaussian, F_func_gaussian_diss)