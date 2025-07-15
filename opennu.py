import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


def solve2nd(Na, state='G', gp_ratio=0.95, gd_ratio=1, ti=1e-4, tf=1000, p_init=1.0):
    gm = 1
    td = 1/(Na*gm)
    gm = 1*td        # gamma-
    gp = gp_ratio*td # gamma+
    gd = gd_ratio*td # gamma_phi

    t_span = (ti, tf)
    t_eval = np.geomspace(*t_span, 200)

    # Initial conditions
    # For product styate (coherent spin in equatorial plane)
    if state == 'P':
        jj0 = 0.5 * Na * (0.5 * Na + 1)
        jz0 = 0
        jjz0 = Na / 4

    # Defaulting to Ground state
    else:
        jj0 = 0.5*Na + 0.25*Na**2*p_init**2 + 0.25*Na*(1-p_init**2)
        jz0 = -0.5 * Na * p_init
        jjz0 = Na**2 / 4 * p_init**2 + Na/4 * (1-p_init**2)

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
    return t, jz, jz2, jx2, jj
