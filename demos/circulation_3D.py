from functools import lru_cache
import math
import pprint
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable

import logging
import dolfin
from pathlib import Path
from pulse2.itertarget import itertarget
import pulse2
from pulse2.geometry import LVGeometry
import cardiac_geometries


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


def default_parameters() -> dict[str, float]:
    r"""Default parameters for the activation model

    Returns
    -------
    Dict[str, float]
        Default parameters

    Notes
    -----
    The default parameters are

    .. math::
        t_{\mathrm{sys}} &= 0.16 \\
        t_{\mathrm{dias}} &= 0.484 \\
        \gamma &= 0.005 \\
        a_{\mathrm{max}} &= 5.0 \\
        a_{\mathrm{min}} &= -30.0 \\
        \sigma_0 &= 150e3 \\
    """
    return dict(
        t_sys=0.16,
        t_dias=0.484,
        gamma=0.005,
        a_max=5.0,
        a_min=-30.0,
        sigma_0=150e3,
    )


def activation_function(
    t_span: tuple[float, float],
    t_eval: np.ndarray | None = None,
    parameters: dict[str, float] | None = None,
) -> np.ndarray:
    r"""Active stress model from the Bestel model [3]_.

    Parameters
    ----------
    t_span : Tuple[float, float]
        A tuple representing start and end of time
    parameters : Dict[str, float]
        Parameters used in the model, see :func:`default_parameters`
    t_eval : Optional[np.ndarray], optional
        Time points to evaluate the solution, by default None.
        If not provided, the default points from `scipy.integrate.solve_ivp`
        will be used

    Returns
    -------
    np.ndarray
        An array of activation points

    Notes
    -----
    The active stress is taken from Bestel et al. [3]_, characterized through
    a time-dependent stress function \tau solution to the evolution equation

    .. math::
        \dot{\tau}(t) = -|a(t)|\tau(t) + \sigma_0|a(t)|_+

    being a(\cdot) the activation function and \sigma_0 contractility,
    where each remaining term is described below:

    .. math::
        |a(t)|_+ =& \mathrm{max}\{a(t), 0\} \\
        a(t) :=& \alpha_{\mathrm{max}} \cdot f(t)
        + \alpha_{\mathrm{min}} \cdot (1 - f(t)) \\
        f(t) =& S^+(t - t_{\mathrm{sys}}) \cdot S^-(t - t_{\mathrm{dias}}) \\
        S^{\pm}(\Delta t) =& \frac{1}{2}(1 \pm \mathrm{tanh}(\frac{\Delta t}{\gamma}))

    .. [3] J. Bestel, F. Clement, and M. Sorine. "A Biomechanical Model of Muscle Contraction.
        In: Medical Image Computing and Computer-Assisted Intervention - MICCAI 2001. Springer
        Berlin Heidelberg, 2001, pp. 1159{1161.

    """
    params = default_parameters()
    if parameters is not None:
        params.update(parameters)

    print(f"Solving active stress model with parameters: {pprint.pformat(params)}")

    f = (
        lambda t: 0.25
        * (1 + math.tanh((t - params["t_sys"]) / params["gamma"]))
        * (1 - math.tanh((t - params["t_dias"]) / params["gamma"]))
    )
    a = lambda t: params["a_max"] * f(t) + params["a_min"] * (1 - f(t))

    def rhs(t, tau):
        return -abs(a(t)) * tau + params["sigma_0"] * max(a(t), 0)

    res = solve_ivp(
        rhs,
        t_span,
        [0.0],
        t_eval=t_eval,
        method="Radau",
    )
    return res.y.squeeze()


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


def kPa2mmHg(p: float) -> float:
    return p * 7.50061683


def main():
    heart_rate = 70
    # Number of seconds for 1 beat
    heart_cycle = 60 / heart_rate
    # Systolic time period

    # Number of cycles
    N = 10
    # End time in seconds
    T = N * heart_cycle

    # Resistance in mitral valve
    R_mitral_valve = 0.08782
    # Resistance in aortic valve
    R_aortic = 0.1
    # Resistance in ?
    R_p = 1.3
    Ca = 1.601
    Cv = 1.894

    t_eval = np.arange(0, T, 0.01)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("pulse2")
    logger.setLevel(logging.INFO)

    # Read geometry from file. If the file is not present we regenerate it.
    geofolder = Path("lv")
    if not geofolder.is_dir():
        cardiac_geometries.create_lv_ellipsoid(
            geofolder,
            create_fibers=True,
        )

    geo = cardiac_geometries.geometry.Geometry.from_folder(geofolder)
    geo = LVGeometry(
        mesh=geo.mesh,
        markers=geo.markers,
        ffun=geo.ffun,
        cfun=geo.cfun,
        f0=geo.f0,
        s0=geo.s0,
        n0=geo.n0,
    )
    geo.mesh.coordinates()[:] *= 2 / 5

    material_params = pulse2.HolzapfelOgden.transversely_isotropic_parameters()
    material = pulse2.HolzapfelOgden(f0=geo.f0, s0=geo.s0, parameters=material_params)

    Ta = dolfin.Constant(0.0)
    active_model = pulse2.ActiveStress(geo.f0, activation=Ta)
    comp_model = pulse2.Incompressible()

    model = pulse2.CardiacModel(
        material=material,
        active=active_model,
        compressibility=comp_model,
    )
    problem = pulse2.LVProblem(
        model=model, geometry=geo, parameters={"bc_type": "fix_base"}
    )
    problem.control_mode = "volume"

    y0 = [
        geo.inner_volume(),  # Initial LV volume
        96,  # Initial aortic pressure
        6,  # Initial arterial pressure
    ]

    params = default_parameters()
    params["t_sys"] = 0.14
    params["t_dias"] = 0.28
    params["sigma_0"] = 300e3

    def act(t):
        return (
            float(activation_function((t - 0.1, t), t_eval=[t], parameters=params))
            / 1000.0
        )

    volumes = []
    pressures = []
    gammas = []

    @lru_cache
    def save(time_step):
        logger.info(f"Save time step {time_step}")
        with dolfin.XDMFFile("circulation.xdmf") as ofile:
            ofile.write_checkpoint(
                problem.displacement,
                "u",
                time_step,
                dolfin.XDMFFile.Encoding.HDF5,
                True,
            )
        volumes.append(problem.Vendo)
        pressures.append(problem.pendo)
        gammas.append(problem.get_control_parameter("gamma"))
        fig, ax = plt.subplots()
        ax.plot(volumes, pressures)
        fig.savefig("pv_loop_circulation_3D.png")

    @lru_cache
    def lv_pressure_func(t, v_lv):
        a = act(t)
        # First get the correct gamma
        itertarget(
            problem,
            target_end=a,
            target_parameter="gamma",
            control_step=0.01,
            control_parameter="gamma",
            control_mode="volume",
            data_collector=None,
        )
        # Then change pressure to get the correct volume
        itertarget(
            problem,
            target_end=v_lv,
            target_parameter="volume",
            control_step=0.01,
            control_parameter="pressure",
            control_mode="pressure",
            data_collector=None,
        )
        save(t)
        return kPa2mmHg(problem.pendo)

    solve_ivp(
        fun=rhs,
        t_span=[0, T],
        y0=y0,
        t_eval=t_eval,
        args=(
            lv_pressure_func,
            R_mitral_valve,
            R_aortic,
            R_p,
            Ca,
            Cv,
        ),
    )

    # v_lv, p_aortic, p_arterial = res.y
    # t = res.t
    # p_lv = lv_pressure_func(t, v_lv)


if __name__ == "__main__":
    main()
