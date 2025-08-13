import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import numba

from functools import lru_cache

import gammas as ga

def solve(Na, gp_ratio=0.95, gd_ratio=0, ti=1e-4, tf=1e4, ntimes=200, p_init=1):
    '''
    '''
    gm = 1
    td = 1/(Na*gm)
    gm = 1*td        # gamma-
    gp = gp_ratio*td # gamma+
    gd = gd_ratio*td # gamma_phi

    t_span = (ti, tf)
    t_eval = np.geomspace(*t_span, ntimes)
    
    jz0 = 0
    jj0 = 0.5 * Na + 0.25 * Na**2 * p_init**2
    jjz0 = (Na / 4) 
    
    def system(t, y):
        jz, jj, jjz = y
        djz_dt = -gm * (jj - jjz + jz) + gp * (jj - jjz - jz)
        djj_dt = -gd * (jj - jjz - 0.5 * Na)
        djjz_dt = (
            gm * (jj + jz - 3 * jjz + 2 * jz * jjz - 2 * jz * jj) +
            gp * (jj - jz - 3 * jjz - 2 * jz * jjz + 2 * jz * jj)
        )
        return [djz_dt, djj_dt, djjz_dt]

    # Solve the first system
    y0 = [jz0, jj0, jjz0]
    sol = solve_ivp(system, (t_eval[0], t_eval[-1]), y0, t_eval=t_eval, method='BDF', rtol=1e-10, atol=1e-12, dense_output=True)

    t = sol.t
    jz = sol.y[0]
    jj = sol.y[1]
    jz2 = sol.y[2]
    sz = np.sqrt(jz2-jz**2)
    return t, jz, sz

def solvex(Na, gp_ratio=0.95, gd_ratio=0, ti=1e-4, tf=1e4, ntimes=200, p_init=1):
    '''
    '''
    gm = 1
    td = 1/(Na*gm)
    gm = 1*td        # gamma-
    gp = gp_ratio*td # gamma+
    gd = gd_ratio*td # gamma_phi

    t_span = (ti, tf)
    t_eval = np.geomspace(*t_span, ntimes)
    
    jz0 = -0.5 * Na * p_init
    jj0 = 0.5 * Na + 0.25 * Na**2 * p_init**2
    jjz0 = (Na**2 / 4) *p_init**2
    
    def system1(t, y):
        jz, jj, jjz = y
        djz_dt = -gm * (jj - jjz + jz) + gp * (jj - jjz - jz)
        djj_dt = -gd * (jj - jjz - 0.5 * Na)
        djjz_dt = (
            gm * (jj + jz - 3 * jjz + 2 * jz * jjz - 2 * jz * jj) +
            gp * (jj - jz - 3 * jjz - 2 * jz * jjz + 2 * jz * jj)
        )
        return [djz_dt, djj_dt, djjz_dt]

    def system2(t, y):
        jp, jm = y
        jz = jz_t(t)
        djp_dt = -0.5 * gd * jp + gm * jp * jz - gp * jm * jz
        djm_dt = -0.5 * gd * jm + gm * jz * jm - gp * jz * jp
        return [djp_dt, djm_dt]

    # Solve the first system
    y0_1 = [jz0, jj0, jjz0]
    sol1 = solve_ivp(system1, (t_eval[0], t_eval[-1]), y0_1, t_eval=t_eval, method='BDF', rtol=1e-10, atol=1e-12, dense_output=True)

    # Interpolate solutions
    jz_t  = interp1d(sol1.t, sol1.y[0], kind='cubic', fill_value="extrapolate")
    jj_t  = interp1d(sol1.t, sol1.y[1], kind='cubic', fill_value="extrapolate")
    jjz_t = interp1d(sol1.t, sol1.y[2], kind='cubic', fill_value="extrapolate")

    # Initial conditions for second system
    y0_2 = [0, 0]
    sol2 = solve_ivp(system2, (t_eval[0], t_eval[-1]), y0_2, t_eval=t_eval, method='BDF', rtol=1e-10, atol=1e-12, dense_output=True)

    t = sol2.t
    jpm = sol2.y[0]
    jmp = sol2.y[1]
    jz = sol1.y[0]
    jj = sol1.y[1]
    jz2 = sol1.y[2]
    jx2 = (jpm*jmp + jpm*jmp + 2*jj - 2*jz2)/4
    return t, jx2, jz, jz2

       


def find_delta(R, mnu, p_init=1, sampf=14.3e3, B=0.1, T2=1, ns=1.35e22, Nshots=100, 
                 seed=42, d_init=1e5, d_fin=1e20, ndelta=100,
                 chi2_crit=2.7, squid_noise_ratio=0.0, ncode=1e9, A=129, Z=54, gy=11.78e6, mode='m1'):
    '''
    Chi-squared analysis on normalized ⟨J_z⟩, starting from
    equatorial product state assuming T1 >> T2.
    Returns upper limit on delta at specified confidence level.
    '''
    eVHz   = 1 / 6.58e-16                   # eV/Hz conversion
    w0     = 2 * np.pi * gy * B / eVHz      # eV
    knu    = 1 / 0.037                      # cm^-1
    N      = ns * 4 * np.pi / 3 * R**3      # number of spins
    fsup   = 4*(knu * R)**3                   # coherent suppression factor
    
    # --- Time grid ---
    tf       = T2
    ti       = 1 / sampf
    n_times  = int((tf - ti) * sampf)
    t_exp    = np.geomspace(ti, tf, n_times)
    
    # --- Gamma ratios ---
    gratio, gm = ga.compute_ratio_v2(mnu, w0, A=A, Z=Z, mode=mode)

    np.random.seed(seed)
    jz_true_mean = 0.0
    jz_true_std = np.sqrt(N / 4)
    
    Jz_samples = np.random.normal(jz_true_mean, jz_true_std, size=(n_times, Nshots))
    Jz_mean_exp = np.mean(Jz_samples, axis=1) / (N / 2)

    # --- Memoized model prediction for normalized ⟨J_z⟩ ---
    @lru_cache(maxsize=64)
    def get_model_jz(delta):
        Ncode = int(ncode)
        # tmin_code = min(t_exp) * N * gm / (1-gratio) * delta / fsup
        # tmax_code = max(t_exp) * N * gm / (1-gratio) * delta / fsup
        tmin_code = min(t_exp) * N * gm * delta / fsup
        tmax_code = max(t_exp) * N * gm * delta / fsup
        t, jz, sz = solve(
            Ncode,
            gp_ratio=gratio,
            gd_ratio=Ncode,
            p_init=p_init,
            ti=tmin_code,
            tf=tmax_code,
            ntimes=n_times
        )
        jz_norm = jz / (Ncode / 2)
        sz_norm = sz/ ( Ncode**(1/2) / 2)
        return t, np.abs(jz_norm), sz_norm
    
    delta_list = np.geomspace(d_init, d_fin, ndelta)
    chi2_min = np.inf
    delta_best = None
    delta_crit = None
    
    chi2l = []
    
    for delta in delta_list:
        _, jz_pred, sz_pred = get_model_jz(delta)
        # Old line
        #sigma_jz = np.sqrt((N/4) / Nshots + squid_noise_ratio * (N / 4)) / (N / 2)
        sigma_jz = np.sqrt(sz_pred**2/Nshots + squid_noise_ratio/Nshots)/np.sqrt(N)*2
        chi2 = np.sum(((Jz_mean_exp - jz_pred) / sigma_jz) ** 2)
        chi2l.append(chi2)
    
        if chi2 < chi2_min:
            chi2_min = chi2
            delta_best = delta
    
        if delta_crit is None and chi2 - chi2_min > chi2_crit:
            delta_crit = delta
            break
    
    if delta_crit is None:
       print("No delta found within scan range for J_z")

    return delta_crit, chi2l, chi2-chi2_min



def get_deltax(R, mnu, p_init=1, sampf=14.3e3, B=0.1, T2=1, Nshots=100, squid_noise_ratio=0, 
              tf_ratio=0.2, seed=42, d_init=1e15, d_fin=1e30, ndelta=100, chi2_crit=2.7):
    '''
    '''
    # --- Constants ---
    ns     = 1.35e22                          # spin density (cm^-3)
    gy     = 11.78e6                          # Hz/T2
    eVHz   = 1 / 6.58e-16
    w0     = 2 * np.pi * gy * B / eVHz        # eV
    knu    = 1 / 0.037                        # cm^-1
    N      = ns * 4 * np.pi / 3 * R**3        # number of spins
    fsup   = 4 * (knu * R)**2                 # coherent suppression factor
    
    # --- Time grid ---
    tf       = T2
    ti       = 1 / sampf
    n_times  = int((tf - ti) * sampf)
    t_exp    = np.geomspace(ti, tf, n_times)
    
    
    # --- gammas ---
    # ga.compute_ratio(mnu, w0, A=1, Z=1) to recover nucleon case
    gratio, gm = ga.compute_ratio(mnu, w0, A=A, Z=Z)
    
    # --- Generate J_x data ---
    np.random.seed(seed)
    true_var = N / 4 * (1+squid_noise_ratio)
    Jx_samples = np.random.normal(0, np.sqrt(true_var), size=(n_times, Nshots))
    Jx2_exp = np.var(Jx_samples, axis=1, ddof=1)
    
    # --- Error on the sample variance ---
    sigma2 = (2 * true_var**2) / (Nshots - 1)
    sigma = np.sqrt(sigma2) * np.ones(n_times)
    
    # --- Normalize data ---
    jx2_exp = Jx2_exp / N * 4
    sigma   = sigma / N * 4
    
    # --- Memoized Solver Wrapper ---
    @lru_cache(maxsize=64)
    def get_model_jx2(delta):
        Ncode = int(1e8)
        tmin_code = min(t_exp) * N * gm * delta / fsup
        tmax_code = max(t_exp) * N * gm * delta / fsup * tf_ratio
        t, _, _, jx2, _ = solve2nd(
            Ncode,
            gp_ratio=gratio,
            gd_ratio=Ncode*tf_ratio,
            p_init=p_init,
            ti=tmin_code,
            tf=tmax_code,
            ntimes=n_times
        )
        return jx2 / Ncode * 4

    # --- Chi-squared scan over delta ---
    delta_list = np.geomspace(d_init, d_fin, ndelta)
    chi2_vals = []
    chi2_min = np.inf
    delta_best = None
    delta_crit = None
    
    for delta in delta_list:
        jx2_pred = get_model_jx2(delta)
        chi2 = np.sum(((jx2_exp - jx2_pred) / sigma) ** 2)
        chi2_vals.append(chi2)
    
        # Track best
        if chi2 < chi2_min:
            chi2_min = chi2
            delta_best = delta
    
        # Confidence level threshold
        if delta_crit is None and chi2 - chi2_min > chi2_crit:
            delta_crit = delta
            break

    if delta_crit is None:
        print("No delta found within scan range")
    return delta_crit, chi2 - chi2_min

# def solve2nd(Na, state='G', gp_ratio=0.95, gd_ratio=1, ti=1e-4, tf=1e4, ntimes=200, p_init=1.0,):
#     '''
#     '''
#     gm = 1
#     td = 1/(Na*gm)
#     gm = 1*td        # gamma-
#     gp = gp_ratio*td # gamma+
#     gd = gd_ratio*td # gamma_phi

#     t_span = (ti, tf)
#     t_eval = np.geomspace(*t_span, ntimes)

#     # Initial conditions
#     # For product styate (coherent spin in equatorial plane)
#     if state == 'P':
#         jz0 = 0
#         jj0 = 0.5*Na + 0.25*Na**2*p_init**2 + 0.25*Na*(1-p_init**2)
#         jjz0 = Na**2 / 4 * p_init**2 + Na/4 * (1-p_init**2)

#     # Defaulting to Ground state
#     else:
#         jj0 = 0.5*Na + 0.25*Na**2*p_init**2 + 0.25*Na*(1-p_init**2)
#         jz0 = -0.5 * Na * p_init
#         jjz0 = Na**2 / 4 * p_init**2 + Na/4 * (1-p_init**2)

#     def system1(t, y):
#         jz, jj, jjz = y
#         djz_dt = -gm * (jj - jjz + jz) + gp * (jj - jjz - jz)
#         djj_dt = -gd * (jj - jjz - 0.5 * Na)
#         djjz_dt = (
#             gm * (jj + jz - 3 * jjz + 2 * jz * jjz - 2 * jz * jj) +
#             gp * (jj - jz - 3 * jjz - 2 * jz * jjz + 2 * jz * jj)
#         )
#         return [djz_dt, djj_dt, djjz_dt]

#     def system2(t, y):
#         jp, jm = y
#         jz = jz_t(t)
#         djp_dt = -0.5 * gd * jp + gm * jp * jz - gp * jm * jz
#         djm_dt = -0.5 * gd * jm + gm * jz * jm - gp * jz * jp
#         return [djp_dt, djm_dt]

#     # Solve the first system
#     y0_1 = [jz0, jj0, jjz0]
#     sol1 = solve_ivp(system1, (t_eval[0], t_eval[-1]), y0_1, t_eval=t_eval, method='BDF', rtol=1e-10, atol=1e-12, dense_output=True)

#     # Interpolate solutions
#     jz_t  = interp1d(sol1.t, sol1.y[0], kind='cubic', fill_value="extrapolate")
#     jj_t  = interp1d(sol1.t, sol1.y[1], kind='cubic', fill_value="extrapolate")
#     jjz_t = interp1d(sol1.t, sol1.y[2], kind='cubic', fill_value="extrapolate")

#     # Initial conditions for second system
#     y0_2 = [0, 0]
#     sol2 = solve_ivp(system2, (t_eval[0], t_eval[-1]), y0_2, t_eval=t_eval, method='BDF', rtol=1e-10, atol=1e-12, dense_output=True)

#     t = sol2.t
#     jpm = sol2.y[0]
#     jmp = sol2.y[1]
#     jz = sol1.y[0]
#     jj = sol1.y[1]
#     jz2 = sol1.y[2]
#     jx2 = (jpm*jmp + jpm*jmp + 2*jj - 2*jz2)/4
#     return t, jz, jz2, jx2, jj


# def get_delta_jz(R, mnu, p_init=1, sampf=14.3e3, B=0.1, T2=1, ns=1.35e22, Nshots=100, 
#                  seed=42, d_init=1e15, d_fin=1e30, ndelta=100,
#                  chi2_crit=2.7, squid_noise_ratio=0.0, ncode=1e8, A=129, Z=54, gy=11.78e6):
#     '''
#     Chi-squared analysis on normalized ⟨J_z⟩, starting from
#     equatorial product state assuming T1 >> T2.
#     Returns upper limit on delta at specified confidence level.
#     '''

#     # --- Constants ---
#     # gy     = 11.78e6                        # Hz/T
#     eVHz   = 1 / 6.58e-16                   # eV/Hz conversion
#     w0     = 2 * np.pi * gy * B / eVHz      # eV
#     knu    = 1 / 0.037                      # cm^-1
#     N      = ns * 4 * np.pi / 3 * R**3      # number of spins
#     fsup   = 4*(knu * R)**3                 # coherent suppression factor

#     # --- Time grid ---
#     tf       = T2
#     ti       = 1 / sampf
#     n_times  = int((tf - ti) * sampf)
#     t_exp    = np.geomspace(ti, tf, n_times)

#     # --- Gamma ratios ---
#     gratio, gm = ga.compute_ratio(mnu, w0, A=A, Z=Z)

#     # Simulate raw samples
#     np.random.seed(seed)
#     jz_true_mean = 0.0
#     jz_true_std = np.sqrt(N / 4)

#     Jz_samples = np.random.normal(jz_true_mean, jz_true_std, size=(n_times, Nshots))
#     Jz_mean_exp = np.mean(Jz_samples, axis=1) / (N / 2)

#     # Total std dev normalized (quantum + SQUID noise)
#     sigma_jz = np.sqrt((N / 4) / Nshots + squid_noise_ratio * (N / 4)) / (N / 2) * np.ones(n_times)

#     # --- Memoized model prediction for normalized ⟨J_z⟩ ---
#     @lru_cache(maxsize=64)
#     def get_model_jz(delta):
#         Ncode = int(ncode)
#         tmin_code = min(t_exp) * N * gm * delta / fsup
#         tmax_code = max(t_exp) * N * gm * delta / fsup
#         t, jz, _, _, _ = solve2nd(
#             Ncode,
#             state='P',
#             gp_ratio=gratio,
#             gd_ratio=Ncode,
#             p_init=p_init,
#             ti=tmin_code,
#             tf=tmax_code,
#             ntimes=n_times
#         )
#         return t, np.abs(jz) / (Ncode / 2) 

#     # --- Chi2 scan over delta ---
#     delta_list = np.geomspace(d_init, d_fin, ndelta)
#     chi2_min = np.inf
#     delta_best = None
#     delta_crit = None

#     chi2l = []

#     for delta in delta_list:
#         jz_pred = get_model_jz(delta)
#         chi2 = np.sum(((Jz_mean_exp - jz_pred) / sigma_jz) ** 2)
#         chi2l.append(chi2)

#         if chi2 < chi2_min:
#             chi2_min = chi2
#             delta_best = delta

#         if delta_crit is None and chi2 - chi2_min > chi2_crit:
#             delta_crit = delta
#             break

#     if delta_crit is None:
#         print("No delta found within scan range for J_z")

#     return delta_crit, chi2 - chi2_min, chi2l
