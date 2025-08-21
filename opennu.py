import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.integrate import dblquad
from mpmath import mp, quad, sqrt, exp, inf, fabs, findroot
from functools import lru_cache


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
    sol = solve_ivp(system, (t_eval[0], t_eval[-1]), y0, t_eval=t_eval, method='BDF',
                             rtol=1e-10, atol=1e-12, dense_output=True)

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
    sol1 = solve_ivp(system1, (t_eval[0], t_eval[-1]), y0_1, t_eval=t_eval, method='BDF',
                               rtol=1e-10, atol=1e-12, dense_output=True)

    # Interpolate solutions
    jz_t  = interp1d(sol1.t, sol1.y[0], kind='cubic', fill_value="extrapolate")
    jj_t  = interp1d(sol1.t, sol1.y[1], kind='cubic', fill_value="extrapolate")
    jjz_t = interp1d(sol1.t, sol1.y[2], kind='cubic', fill_value="extrapolate")

    # Initial conditions for second system
    y0_2 = [0, 0]
    sol2 = solve_ivp(system2, (t_eval[0], t_eval[-1]), y0_2, t_eval=t_eval, method='BDF',
                               rtol=1e-10, atol=1e-12, dense_output=True)

    t = sol2.t
    jpm = sol2.y[0]
    jmp = sol2.y[1]
    jz = sol1.y[0]
    jj = sol1.y[1]
    jz2 = sol1.y[2]
    jx2 = (jpm*jmp + jpm*jmp + 2*jj - 2*jz2)/4
    return t, jx2, jz, jz2


def compute_gav_matrix(A, Z):
    '''
    '''
    ga = np.array([-A / 2 + Z, -A / 2 + Z, -A / 2 + Z])

    # PMNS matrix parameters
    theta12 = np.deg2rad(33.82)
    theta13 = np.deg2rad(8.62)
    theta23 = np.deg2rad(45.0)
    delta = np.deg2rad(197)

    # PMNS matrix
    U = np.array([
        [np.cos(theta12) * np.cos(theta13), np.sin(theta12) * np.cos(theta13), np.sin(theta13) * np.exp(-1j * delta)],
        [-np.sin(theta12) * np.cos(theta23) - np.cos(theta12) * np.sin(theta23) * np.sin(theta13) * np.exp(1j * delta),
         np.cos(theta12) * np.cos(theta23) - np.sin(theta12) * np.sin(theta23) * np.sin(theta13) * np.exp(1j * delta),
         np.sin(theta23) * np.cos(theta13)],
        [np.sin(theta12) * np.sin(theta23) - np.cos(theta12) * np.cos(theta23) * np.sin(theta13) * np.exp(1j * delta),
         -np.cos(theta12) * np.sin(theta23) - np.sin(theta12) * np.cos(theta23) * np.sin(theta13) * np.exp(1j * delta),
         np.cos(theta23) * np.cos(theta13)]
    ])

    gav = np.zeros((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            gav[i, j] = sum(ga[k] * np.conj(U[k, i]) * U[k, j] for k in range(3))
    return gav



def compute_gammas(w_val, m_vals, A, Z, majorana=False):
    eV_2_Hz = 1/6.58e-16
    mp.dps = 70  # High precision

    T = mp.mpf("0.00016809")  # Temperature 1.95*8.62e-5
    Gf = mp.mpf("1.17e-23")   # Fermi constant (example)
    w = mp.mpf(w_val)
    m = [mp.mpf(mval) for mval in m_vals]

    gav = compute_gav_matrix(A, Z)

    def integrand_p(p, m1, m2, w):
        E1 = sqrt(p**2 + m1**2)
        arg = (E1 - w)**2 - m2**2
        if arg < 0:
            return mp.mpf(0)
        val = 2 * (1 / (exp(p / T) + 1)) * p**2 * sqrt(arg) * (E1 - w)
        return val

    def integrand_m(p, m1, m2, w):
        E1 = sqrt(p**2 + m1**2)
        arg = (E1 + w)**2 - m2**2
        if arg < 0:
            return mp.mpf(0)
        val = 2 * (1 / (exp(p / T) + 1)) * p**2 * sqrt(arg) * (E1 + w)
        return val

    ina = [[mp.mpf(0) for _ in range(3)] for _ in range(3)]
    inam = [[mp.mpf(0) for _ in range(3)] for _ in range(3)]

    for i in range(3):
        for j in range(3):
            # ---- gammap ----
            delta_p = (w + m[j])**2 - m[i]**2
            if delta_p > 0:
                limit_p = sqrt(delta_p)
                result_p = quad(lambda p: integrand_p(p, m[i], m[j], w), [limit_p, inf])
                ina[i][j] = fabs(gav[i][j] * gav[j][i]) * result_p
            else:
                ina[i][j] = mp.mpf(0)

            # ---- gammam ----
            # For E1 + w: just integrate from p=0 safely, arg is checked inside the integrand
            result_m = quad(lambda p: integrand_m(p, m[i], m[j], w), [0, inf])
            inam[i][j] = fabs(gav[i][j] * gav[j][i]) * result_m

    if majorana:
        pref = 2
    else:
        pref = 1
    prefactor = pref * (4 * Gf**2) / (2 * mp.pi)**3
    gammap = prefactor * sum(sum(row) for row in ina)
    gammam = prefactor * sum(sum(row) for row in inam)

    return float(gammap), float(gammam), float(gammap*eV_2_Hz), float(gammam*eV_2_Hz)


def compute_ratio(m, w, A=129, Z=54, mode="sum", majorana=False):
    mnu = mp.mpf(m)
    w0 = mp.mpf(w)

    dm21 = mp.mpf("7.4e-5")
    dm31 = mp.mpf("2.5e-3")

    def get_masses_from_sum(m_sum):
        min_sum = sqrt(dm21) + sqrt(dm31)
        if m_sum < min_sum:
            raise ValueError(f"Mass sum {m_sum} eV is below physical minimum (~0.059 eV).")
        def mass_sum_difference(m1_guess):
            m1 = mp.mpf(m1_guess)
            m2 = sqrt(m1**2 + dm21)
            m3 = sqrt(m1**2 + dm31)
            return m1 + m2 + m3 - m_sum
        m1 = findroot(mass_sum_difference, 0.001)
        m2 = sqrt(m1**2 + dm21)
        m3 = sqrt(m1**2 + dm31)
        return [m1, m2, m3]

    def get_masses_from_m1(m1):
        m1 = mp.mpf(m1)
        m2 = sqrt(m1**2 + dm21)
        m3 = sqrt(m1**2 + dm31)
        return [m1, m2, m3]

    if mode == "sum":
        mm = get_masses_from_sum(mnu)
    elif mode == "m1":
        mm = get_masses_from_m1(mnu)
    else:
        raise ValueError("mode must be 'sum' or 'm1'")

    gp_eV, gm_eV, gp, gm = compute_gammas(w0, mm, A, Z, majorana=majorana)
    return gp / gm, gm, mm


def find_delta(
        R,
        mnu,
        p_init=1,
        sampf=14.3e3,
        B=0.1,
        T2=1,
        Nshots=100,
        seed=42,
        d_init=1e5,
        d_fin=1e20,
        ndelta=100,
        chi2_crit=2.7,
        squid_noise_ratio=0.0,
        ncode=1e9,
        Bmax=12,
        mode='m1',
        sample='Xe',
        squeeze=1,
        opt=True,
        sigma_spn=False,
        verb=False
    ):
    '''
    Chi-squared analysis on normalized ⟨J_z⟩, starting from
    equatorial product state assuming T1 >> T2.
    Returns upper limit on delta at specified confidence level.
    '''

    if sample == 'Xe':
        ns = 1.35e22
        A  = 129
        Z  = 54
        gy = 11.78e6
    elif sample == 'He':
        ns = 3e22
        A  = 3
        Z  = 2
        gy = 32.43e6
    elif sample == 'H':
        ns = 3e22
        A  = 1
        Z  = 1
        gy = 42.58e6
    else:
        raise ValueError("Sample not valid, choose 'Xe', 'He' or 'H'!")

    eVHz   = 1 / 6.58e-16                   # eV/Hz conversion
    w0     = 2 * np.pi * gy * B / eVHz      # eV
    knu    = 1 / 0.037                      # cm^-1
    N      = ns * 4 * np.pi / 3 * R**3      # number of spins
    fsup   = max(1, 4*(knu * R)**2)         # coherent suppression factor
    w0_i   = w0

    # --- Time grid ---
    tf       = T2
    ti       = 1 / sampf
    n_times  = int((tf - ti) * sampf)
    t_exp    = np.geomspace(ti, tf, n_times)

    # --- Gamma ratios ---
    gratio, gm, mm = compute_ratio(mnu, w0, A=A, Z=Z, mode=mode)

    if verb:
        print("At B=%.2f, I get w=%.3e"%(B, w0))
        print("g+/g- = %.7f"%gratio)


    # --- Optimal splitting and B-field limitations
    if opt:
        knu = 5.3e-4
        m1 = mm[0]
        w0_opt = knu/m1/R/8065
        if w0_opt < w0:
            if verb:
                print("Warning: splitting too large, adjusting to optimal value!")
        w0 = w0_opt
        B_opt = w0_opt * eVHz / (2*np.pi * gy)
        if B_opt > Bmax:
            if verb:
                print("Optimal splitting needs too large B field, adjusting to Bmax!")
            B = Bmax
            w0 = 2 * np.pi * gy * B / eVHz

        gratio, gm, mm = compute_ratio(mnu, w0, A=A, Z=Z, mode=mode)
        if verb:
            print("Passed w=%.3e, optimal w=%.3e,  used w=%.3e"%(w0_i, w0_opt, w0))
            print("g+/g- = %.7f"%gratio)

    # -- Data generation -------
    np.random.seed(seed)
    jz_true_mean = 0.0
    jz_true_std = np.sqrt(N / 4)

    Jz_samples = np.random.normal(jz_true_mean, jz_true_std, size=(n_times, Nshots))
    Jz_mean_exp = np.mean(Jz_samples, axis=1) / (N / 2)

    # --- Memoized model prediction for normalized ⟨J_z⟩ ---
    @lru_cache(maxsize=64)
    def get_model_jz(delta):
        Ncode = int(ncode)
        tmin_code = min(t_exp) * N * gm/fsup * delta
        tmax_code = max(t_exp) * N * gm/fsup * delta
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
        if sigma_spn:
            sigma_jz = np.sqrt(1/Nshots + squid_noise_ratio/Nshots) / np.sqrt(N/4)
        else:
            sigma_jz = np.sqrt(sz_pred**2/Nshots + squid_noise_ratio/Nshots) / np.sqrt(N/4)

        sigma_jz = sigma_jz/squeeze
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

    return delta_crit, chi2l, chi2-chi2_min, w0


def find_deltax(
        R,
        mnu,
        p_init=1,
        sampf=14.3e3,
        B=0.1,
        T2=1,
        Nshots=100,
        seed=42,
        d_init=1e5,
        d_fin=1e20,
        ndelta=100,
        chi2_crit=2.7,
        squid_noise_ratio=0.0,
        ncode=1e9,
        Bmax=12,
        mode='m1',
        sample='Xe',
        squeeze=1,
        opt=True,
        verb=False
    ):
    '''
    Chi-squared analysis on normalized ⟨J^2_x⟩, starting from
    equatorial product state assuming T1 >> T2.
    Returns upper limit on delta at specified confidence level.
    '''
    if sample == 'Xe':
        ns = 1.35e22
        A  = 129
        Z  = 54
        gy = 11.78e6
    elif sample == 'He':
        ns = 3e22
        A  = 3
        Z  = 2
        gy = 32.43e6
    elif sample == 'H':
        ns = 3e22
        A  = 1
        Z  = 1
        gy = 42.58e6
    else:
        raise ValueError("Sample not valid, choose 'Xe', 'He' or 'H'!")

    eVHz   = 1 / 6.58e-16                   # eV/Hz conversion
    w0     = 2 * np.pi * gy * B / eVHz      # eV
    knu    = 1 / 0.037                      # cm^-1
    N      = ns * 4 * np.pi / 3 * R**3      # number of spins
    fsup   = max(1, 4*(knu * R)**2)         # coherent suppression factor
    w0_i   = w0

    # --- Time grid ---
    tf       = T2
    ti       = 1 / sampf
    n_times  = int((tf - ti) * sampf)
    t_exp    = np.geomspace(ti, tf, n_times)

    # --- Gamma ratios ---
    gratio, gm, mm = compute_ratio(mnu, w0, A=A, Z=Z, mode=mode)

    if verb:
        print("At B=%.2f, I get w=%.3e"%(B, w0))
        print("g+/g- = %.7f"%gratio)


    # --- Optimal splitting and B-field limitations
    if opt:
        knu = 5.3e-4
        m1 = mm[0]
        w0_opt = knu/m1/R/8065
        if w0_opt < w0:
            if verb:
                print("Warning: splitting too large, adjusting to optimal value!")
        w0 = w0_opt
        B_opt = w0_opt * eVHz / (2*np.pi * gy)
        if B_opt > Bmax:
            if verb:
                print("Optimal splitting needs too large B field, adjusting to Bmax!")
            B = Bmax
            w0 = 2 * np.pi * gy * B / eVHz

        gratio, gm, mm = compute_ratio(mnu, w0, A=A, Z=Z, mode=mode)
        if verb:
            print("Passed w=%.3e, optimal w=%.3e,  used w=%.3e"%(w0_i, w0_opt, w0))
            print("g+/g- = %.7f"%gratio)


    # -- Data generation -------
    np.random.seed(seed)
    jx_true_mean = 0.0
    true_var = N/4*(1+squid_noise_ratio)
    jx_true_std = np.sqrt(true_var)

    Jx_samples = np.random.normal(jx_true_mean, jx_true_std, size=(n_times, Nshots))
    Jx2_exp = np.var(Jx_samples, axis=1, ddof=1)

    sigma2 = (2 * true_var**2) / (Nshots - 1)
    sigma = np.sqrt(sigma2) * np.ones(n_times)

    jx2_exp = Jx2_exp / (N / 4)
    sigma = sigma / (N / 4) / squeeze

    # --- Memoized model prediction for normalized ⟨J_z⟩ ---
    @lru_cache(maxsize=64)
    def get_model_jx2(delta):
        Ncode = int(ncode)
        tmin_code = min(t_exp) * N * gm * delta / fsup
        tmax_code = max(t_exp) * N * gm * delta / fsup
        t, jx2, _, _ = solvex(
            Ncode,
            gp_ratio=gratio,
            gd_ratio=Ncode,
            p_init=p_init,
            ti=tmin_code,
            tf=tmax_code,
            ntimes=n_times
        )
        return t, jx2 / (Ncode / 4)

    delta_list = np.geomspace(d_init, d_fin, ndelta)
    chi2_min = np.inf
    delta_best = None
    delta_crit = None

    chi2l = []

    for delta in delta_list:
        _, jx2_pred = get_model_jx2(delta)
        chi2 = np.sum(((jx2_exp - jx2_pred) / sigma) ** 2)
        chi2l.append(chi2)

        if chi2 < chi2_min:
            chi2_min = chi2
            delta_best = delta

        if delta_crit is None and chi2 - chi2_min > chi2_crit:
            delta_crit = delta
            break

    if delta_crit is None:
       print("No delta found within scan range for J^2_x")

    return delta_crit, chi2l, chi2-chi2_min, w0



