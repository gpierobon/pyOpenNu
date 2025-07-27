import numpy as np
from scipy.integrate import dblquad
from mpmath import mp, quad, sqrt, exp, inf, fabs, findroot

A = 1
Z = 1
ga = np.array([-A/2 + Z, -A/2 + Z, -A/2 + Z])

A = 129
Z = 54
ga_Xe = np.array([-A/2 + Z, -A/2 + Z, -A/2 + Z])

# PMNS Matrix angles and CP phase
theta12 = np.deg2rad(33.82)
theta13 = np.deg2rad(8.62)
theta23 = np.deg2rad(45.0)
delta   = np.deg2rad(197)

# PMNS matrix
U = np.array([
    [np.cos(theta12)*np.cos(theta13), np.sin(theta12)*np.cos(theta13), np.sin(theta13)*np.exp(-1j*delta)],
    [-np.sin(theta12)*np.cos(theta23) - np.cos(theta12)*np.sin(theta23)*np.sin(theta13)*np.exp(1j*delta),
     np.cos(theta12)*np.cos(theta23) - np.sin(theta12)*np.sin(theta23)*np.sin(theta13)*np.exp(1j*delta),
     np.sin(theta23)*np.cos(theta13)],
    [np.sin(theta12)*np.sin(theta23) - np.cos(theta12)*np.cos(theta23)*np.sin(theta13)*np.exp(1j*delta),
     -np.cos(theta12)*np.sin(theta23) - np.sin(theta12)*np.cos(theta23)*np.sin(theta13)*np.exp(1j*delta),
     np.cos(theta23)*np.cos(theta13)]
])

gav = np.zeros((3, 3), dtype=complex)
for i in range(3):
    for j in range(3):
        gav[i, j] = sum(ga[k] * np.conj(U[k, i]) * U[k, j] for k in range(3))

gav_Xe = np.zeros((3, 3), dtype=complex)
for i in range(3):
    for j in range(3):
        gav[i, j] = sum(ga[k] * np.conj(U[k, i]) * U[k, j] for k in range(3))



def compute_gammas(w_val, m_vals, gav):
    eV_2_Hz = 1/6.58e-16
    mp.dps = 70  # High precision

    T = mp.mpf("0.00016809")  # Temperature 1.95*8.62e-5
    Gf = mp.mpf("1.17e-23")   # Fermi constant (example)
    w = mp.mpf(w_val)
    m = [mp.mpf(mval) for mval in m_vals]

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

    prefactor = (4 * Gf**2) / (2 * mp.pi)**3
    gammap = prefactor * sum(sum(row) for row in ina)
    gammam = prefactor * sum(sum(row) for row in inam)

    return float(gammap), float(gammam), float(gammap*eV_2_Hz), float(gammam*eV_2_Hz)


def compute_ratio(m, w):
    mnu = mp.mpf(m) 
    w0 = mp.mpf(w)

    dm21 = mp.mpf("7.4e-5")
    dm31 = mp.mpf("2.5e-3")

    def mass_sum_difference(m1_guess, target_sum):

        m1 = mp.mpf(m1_guess)
        m2 = sqrt(m1**2 + dm21)
        m3 = sqrt(m1**2 + dm31)
        return m1 + m2 + m3 - target_sum

    def get_masses(m_sum):
        # Solve m1 from total mass sum
        m1 = findroot(lambda m1: mass_sum_difference(m1, m_sum), 0.001)
        m2 = sqrt(m1**2 + dm21)
        m3 = sqrt(m1**2 + dm31)
        return [m1, m2, m3]

    mm = get_masses(mnu)
    gp_eV, gm_eV, gp, gm = compute_gammas(w0, [mm[0], mm[1], mm[2]], gav)
    return gp/gm, gm


def oldGammas(m, w0):
    u  = 0.25     # CKM coefficient
    de = 3        # 3 Dirac, 6 Majorana

    int_lim = 10

    eV_2_Hz = 1/6.58e-16           # convert eV to Hz
    K_2_eV = 8.62e-5               # convert Kelvin to eV

    Tnu  = 1.95*K_2_eV    # eV
    GF   = 1.17e-23       # eV^-2

    norm = de * eV_2_Hz**2 * (4 * np.pi * GF**2 * m * u) / (np.pi**4 * 8)

    def integrand_m(k, w):
        return 2 * k**2 * np.sqrt(k**2 - 2 * w * m) / (np.exp(k / Tnu) + 1)

    def integrand_p(k, w):
        return 2 * k**2 * np.sqrt(k**2 + 2 * w * m) / (np.exp(k / Tnu) + 1)

    gp, _ = dblquad(lambda x, k: norm*integrand_p(k, w0), np.sqrt((m + w0)**2 - m**2), int_lim, -1, 1)
    gm, _ = dblquad(lambda x, k: norm*integrand_m(k, w0), 0, int_lim, -1, 1)

    return gm*eV_2_Hz, gp*eV_2_Hz


