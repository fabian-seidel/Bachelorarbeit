import sympy as sp
import numpy as np
from sympy.utilities.lambdify import lambdify
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from plot_utils import plot_by_name

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

def get_EL_SOE_matrix(L):
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

    vars_list = [z1, z2, z1s, z2s, g]
    M_func = lambdify(vars_list, M, 'numpy')
    F_func = lambdify(vars_list, F, 'numpy')

    return M_func, F_func

def z_derivatives(t, state, g_val, M_func, F_func):
    z1_val, z2_val, z1s_val, z2s_val = state
    z1s_val = np.conjugate(z1_val)
    z2s_val = np.conjugate(z2_val)
    args = (z1_val, z2_val, z1s_val, z2s_val, g_val)
    M_num = M_func(*args)
    F_num = F_func(*args)
    dz = np.linalg.solve(M_num, F_num).flatten()
    return dz

def solve_dynamics(x1_0, g, M_func, F_func):
    res = solve_ivp(z_derivatives, [0, 20], [x1_0 + 0j, -x1_0 + 0j, x1_0 + 0j, -x1_0 + 0j], rtol=1e-10, atol=1e-10,
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
        res_g0 = solve_dynamics(x1_0, g[0], M_func, F_func)
        res_g1 = solve_dynamics(x1_0, g[1], M_func, F_func)
        axes[idx].plot(np.real(res_g0[0,:]),np.imag(res_g0[0,:]), color='blue')
        axes[idx].plot(np.real(res_g0[1, :]), np.imag(res_g0[1, :]), color='blue')
        axes[idx].plot(np.real(res_g1[0, :]), np.imag(res_g1[0, :]),  color='red')
        axes[idx].plot(np.real(res_g1[1, :]), np.imag(res_g1[1, :]),  color='red')
    plt.show()

if __name__ == '__main__':
    L_gaussian = get_lagrangian_gaussian()
    L_thomasfermi = get_lagrangian_thomasfermi()
    M_func_gaussian, F_func_gaussian = get_EL_SOE_matrix(L_gaussian)
    M_func_thomasfermi, F_func_thomasfermi = get_EL_SOE_matrix(L_thomasfermi)
    plot_dict = {'Vortex-Antivortex Trajectories for Gaussian ansatz': (plot_vortex_antivortex_trajectories, {'M_func':M_func_gaussian, 'F_func':F_func_gaussian}),
                 'Vortex-Antivortex Trajectories for Thomas-Fermi ansatz': (plot_vortex_antivortex_trajectories, {'M_func':M_func_thomasfermi, 'F_func':F_func_thomasfermi})}
    plot_by_name(plot_dict, 'Vortex-Antivortex Trajectories for Thomas-Fermi ansatz')