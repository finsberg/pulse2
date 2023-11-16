from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable


def aortic_valve(p_lv, p_aortic, R_aortic):
    # If LV pressure is higher than the aortic pressure
    # there is a l
    if p_lv > p_aortic:
        return (p_lv - p_aortic) / R_aortic
    else:
        return 0


def mitral_valve(p_arterial, p_lv, R_mitral_valve):
    # Only flow from LA to LV if arterial pressure is
    # larger than ventricular pressure
    if p_arterial > p_lv:
        return (p_arterial - p_lv) / R_mitral_valve
    else:
        return 0


def elastance(t, T, T_sys, Tir):
    heart_cycle = np.floor(t / T)

    t = t - heart_cycle * T
    ret = np.zeros_like(t)

    # inds0 = (T_sys + Tir < t) & (t < T)
    # ret[inds0] = 0

    inds1 = (T_sys < t) & (t < T_sys + Tir)
    ret[inds1] = 0.5 * (1 + np.cos(np.pi * ((t[inds1] - T_sys) / Tir)))

    inds2 = (0 < t) & (t < T_sys)
    ret[inds2] = 0.5 * (1 - np.cos(np.pi * (t[inds2] / T_sys)))
    return ret


def time_varying_elastance(t, v_lv, Vd, E_max, E_min, T, T_sys, Tir):
    E = elastance(t=t, T=T, T_sys=T_sys, Tir=Tir)

    return E * (E_max - E_min) * (v_lv - Vd) + E_min


def rhs(
    t,
    y,
    lv_pressure_func: Callable[[float, float], float],
    R_mitral_valve,
    R_aortic,
    R_p,
    Ca,
    Cv,
):
    v_lv, p_aortic, p_arterial = y
    p_lv = lv_pressure_func(t, v_lv)
    # Flow from LA to LV
    q_in = mitral_valve(
        p_arterial=p_arterial,
        p_lv=p_lv,
        R_mitral_valve=R_mitral_valve,
    )
    q_out = aortic_valve(p_lv=p_lv, p_aortic=p_aortic, R_aortic=R_aortic)
    q_p = (p_aortic - p_arterial) / R_p

    return np.array(
        [
            q_in - q_out,
            (q_out - q_p) / Ca,
            (q_p - q_in) / Cv,
        ]
    )


def main():
    Vd = 1.123
    E_max = 1.5
    E_min = 0.037

    heart_rate = 70
    # Number of seconds for 1 beat
    heart_cycle = 60 / heart_rate
    # Systolic time period
    T_sys = 0.3 * np.sqrt(heart_cycle)
    # heart_cycleime for isovolumic relaxation
    Tir = 0.5 * T_sys

    lv_pressure_func = partial(
        time_varying_elastance,
        Vd=Vd,
        E_max=E_max,
        E_min=E_min,
        T=heart_cycle,
        T_sys=T_sys,
        Tir=Tir,
    )

    # Number of cycles
    N = 10
    # End time in seconds
    T = N * heart_cycle

    y0 = [
        135,  # Initial LV volume
        96,  # Initial aortic pressure
        6,  # Initial arterial pressure
    ]

    # Resistance in mitral valve
    R_mitral_valve = 0.08782
    # Resistance in aortic valve
    R_aortic = 0.1
    # Resistance in ?
    R_p = 1.3
    Ca = 1.601
    Cv = 1.894

    t = np.arange(0, T, 0.01)

    res = solve_ivp(
        fun=rhs,
        t_span=[0, T],
        y0=y0,
        t_eval=t,
        args=(
            lv_pressure_func,
            R_mitral_valve,
            R_aortic,
            R_p,
            Ca,
            Cv,
        ),
    )

    v_lv, p_aortic, p_arterial = res.y
    t = res.t
    p_lv = lv_pressure_func(t, v_lv)

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    ax[0].plot(v_lv, p_lv)
    ax[1].plot(t, p_lv, label="p_lv")
    ax[1].plot(t, p_aortic, label="p_aortic")
    ax[1].plot(t, p_arterial, label="p_arterial")
    ax[1].legend()
    fig.savefig("pv_loop_0D.png")


if __name__ == "__main__":
    main()
