import sympy as sp
import numpy as np
from sympy.utilities.lambdify import lambdify
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from plot_utils import plot_by_name

def get_M_func_F_func_gaussian():
    g = sp.symbols('g', real=True)
    z1, z2, z1s, z2s = sp.symbols('z1 z2 z1s z2s')
    z1_dot, z2_dot, z1s_dot, z2s_dot = sp.symbols('z1_dot z2_dot z1s_dot z2s_dot')
    A_G = sp.sqrt(1 / (sp.pi*(2 + (z1 + z2)*(z1s + z2s) + z1*z1s*z2*z2s)))
    L1 = sp.I*sp.pi/2 * A_G**2 * ((z1s + z2s + z1s*z2*z2s)*z1_dot+(z1 + z2 + z2*z1*z1s)*z2s_dot-(z1 + z2 + z1*z2*z2s)*z1s_dot-(z1s + z2s + z2s*z1*z1s)*z2_dot)
    L21 = - sp.pi * A_G**2 * (4 + z1*z1s + z2*z2s + (z1 + z2)*(z1s + z2s) + z1*z1s*z2*z2s)
    L22 = - sp.pi * g * A_G**4 / 8 * (3 + 6*(z1+z2)*(z1s+z2s) + 3*(z1+z2)**2*(z1s+z2s)**2/2 - (z1-z2)**2*(z1s-z2s)**2/2 + 12*z1*z1s*z2*z2s + 4*(z1+z2)*(z1s+z2s)*z1*z1s*z2*z2s + 2*z1**2*z1s**2*z2**2*z2s**2)
    L = L1 + L21 + L22
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

def z_derivatives_gaussian(t, state, g_val):
    z1_val, z2_val, z1s_val, z2s_val = state
    z1s_val = np.conjugate(z1_val)
    z2s_val = np.conjugate(z2_val)
    args = (z1_val, z2_val, z1s_val, z2s_val, g_val)
    M_num = M_func(*args)
    F_num = F_func(*args)
    dz = np.linalg.solve(M_num, F_num).flatten()
    return dz

def solve_dynamics_gaussian(x1_0, g):
    res = solve_ivp(z_derivatives_gaussian, [0, 10], [x1_0 + 0j, -x1_0 + 0j, x1_0 + 0j, -x1_0 + 0j], rtol=1e-10, atol=1e-10,
                    args=(g,), method='BDF')
    return res.y

def plot_vortex_antivortex_trajectories(ansatz='Gaussian'):
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    axes[1, 0].set_xlabel('x')
    axes[1, 1].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[1, 0].set_ylabel('y')
    axes = axes.flatten()
    g = [6, 40]
    for idx, x1_0 in enumerate([0.3, 0.6, 0.9, 1.2]):
        if ansatz == 'Gaussian':
            res_g0 = solve_dynamics_gaussian(x1_0, g[0])
            res_g1 = solve_dynamics_gaussian(x1_0, g[1])
        axes[idx].plot(np.real(res_g0[0,:]),np.imag(res_g0[0,:]), color='blue')
        axes[idx].plot(np.real(res_g0[1, :]), np.imag(res_g0[1, :]), color='blue')
        axes[idx].plot(np.real(res_g1[0, :]), np.imag(res_g1[0, :]),  color='red')
        axes[idx].plot(np.real(res_g1[1, :]), np.imag(res_g1[1, :]),  color='red')
    plt.show()

if __name__ == '__main__':
    M_func, F_func = get_M_func_F_func_gaussian()
    plot_dict = {'Vortex-Antivortex Trajectories for Gaussian ansatz': (plot_vortex_antivortex_trajectories, {'ansatz':'Gaussian'})}
    plot_by_name(plot_dict, 'Vortex-Antivortex Trajectories for Gaussian ansatz')