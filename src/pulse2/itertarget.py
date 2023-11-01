from __future__ import annotations
import operator as op
import numpy as np
import dolfin
from scipy.interpolate import splev, splrep

import logging
from .solver import Solver
from .problem import LVProblem, ControlMode
from .exceptions import InvalidControl


__all__ = ["itertarget"]

logger = logging.getLogger(__name__)


def numpyfy(
    x: float | list[float] | tuple[float] | np.ndarray, write: bool = True
) -> np.ndarray:
    if isinstance(x, (tuple, list)):
        y = np.array(x)
    elif np.isscalar(x):
        y = np.array([x])
    else:
        assert isinstance(x, np.ndarray)
        y = x

    y.setflags(write=write)

    return y


def target_close(
    target_value: np.ndarray,
    control_step: np.ndarray,
    target_end: np.ndarray,
) -> np.ndarray:
    new_control_step = np.zeros_like(control_step)

    for i, (tv, cs, te) in enumerate(zip(target_value, control_step, target_end)):
        # Next target value
        next_tv = tv + cs

        if abs(next_tv - te) > abs(tv - te):
            # We are stepping in the wrong direction
            cs = -cs
            next_tv = tv + cs

        if np.sign(te - next_tv) != np.sign(te - tv):
            # We have crossed the end value
            new_control_step[i] = te - tv
        else:
            new_control_step[i] = cs

    return new_control_step


def prediction_step(
    control_values: list[np.ndarray],
    prev_states: list[dolfin.Function],
    control_value: np.ndarray,
    problem: LVProblem,
) -> None:
    c0, c1 = control_values[-2:]
    s0, s1 = prev_states
    delta = ((control_value - c0) / (c1 - c0)).mean()

    problem.state.vector().zero()
    problem.state.vector().axpy(1.0 - delta, s0.vector())
    problem.state.vector().axpy(delta, s1.vector())


def reset_problem_state(problem: LVProblem, state_old: dolfin.Function):
    problem.state.vector().zero()
    problem.state.vector().axpy(1.0, state_old.vector())


def stepping_in_wrong_direction(
    first_step: bool,
    iterating: bool,
    target_value: np.ndarray,
    target_end: np.ndarray,
    target_value_old: np.ndarray,
):
    return (
        first_step
        and not iterating
        and (abs(target_value - target_end) > abs(target_value_old - target_end)).all()
    )


def interpolate_control(
    control_values: list[np.ndarray],
    control_value: np.ndarray,
    iterating: bool,
    target_values: list[np.ndarray],
    target_end: np.ndarray,
    control_step: np.ndarray,
):
    c0, c1 = control_values[-2:]
    t0, t1 = target_values[-2:]
    delta = (t1 - t0) / (c1 - c0)
    control_opt = 1 / delta * (target_end - t0) + c0

    if iterating or (abs(control_opt - c0) < abs(control_step)).all():
        # If we have enough values for a spline
        if len(target_values) >= 4:
            # Sort and create a spline representation of the
            # values and use that for interpolation/extrapolation
            def take_first(x):
                return tuple(map(op.itemgetter(0), x))

            inds = np.argsort(take_first(target_values))
            tck = splrep(
                np.array(take_first(target_values))[inds],
                np.array(take_first(control_values))[inds],
                k=1,
                s=0,
            )

            new_control = float(splev(target_end[0], tck))  # type: ignore
            control_step = new_control - control_value

        # If not we do a linear interpolation/extrapolation
        else:
            control_step = control_opt - c0
            control_value = c0

        iterating = True
    return control_value, control_step, iterating


def check_target_reached(target_value, target_end, tol):
    return (abs(target_value - target_end) <= abs(tol * target_end)).any()


def itertarget(
    problem: LVProblem,
    target_end: float | list[float] | tuple[float] | np.ndarray,
    data_collector=None,
    target_parameter: str = "pressure",
    control_parameter: str = "pressure",
    control_step: float | list[float] | tuple[float] | np.ndarray | None = None,
    control_mode: ControlMode | str = ControlMode.pressure,
    tol: float = 1e-6,
    adapt_step: float = True,
    max_adapt_iter: int = 7,
) -> None:
    """Continuation of the control parameter by simple homotopy, i.e.
    the control is increased by control_step (and adapted) until the
    target is reached. The control mode selects which problem must
    be solved: pressure or volume driven.

    If the target is different from the control, then we solve the
    nonlinear equation

    .. math::

        g(control) = target_{end}

    where g is implicitly defined as the target value for a given
    control value.

    Parameters
    ----------
    problem : LVProblem
        The problem containing the geometry and material laws
    target_end : float | list[float] | tuple[float] | np.ndarray
        The target value we are iterating towards
    data_collector : _type_, optional
        An object to collect data during iterations
    target_parameter : str, optional
        The name of the target parameter, by default "pressure"
    control_parameter : str, optional
        The name of the control parameter, by default "pressure"
    control_step : float | list[float] | tuple[float] | np.ndarray | None, optional
        The size of the step the control value will be changed each iteration, by default None
    control_mode : ControlMode | str, optional
        Either 'pressure' or 'volume'. If 'pressure' we set the cavity
        'pressure' and solve for the volume. If 'volume' we set the
        cavity volume and solve for the pressure, by default ControlMode.pressure
    tol : float, optional
        Relative tolerance for reaching the target value., by default 1e-6
    adapt_step : float, optional
        If True the control step can be expanded if the previous
        iteration converged in less than 6 newton iterations, by default True
    max_adapt_iter : int, optional
        _description_, by default 7

    """

    target_is_control = target_parameter == control_parameter
    if isinstance(control_mode, str):
        control_mode = ControlMode[control_mode]

    # Sanity checks
    if (
        control_parameter in ["pressure", "volume"]
        and control_parameter != control_mode.value
    ):
        raise InvalidControl(
            msg="Control mode and control parameter have to be the same.",
            target_parameter=target_parameter,
            control_mode=control_mode.value,
            control_parameter=control_parameter,
        )

    if not target_is_control:
        if target_parameter not in ["volume", "pressure"]:
            msg = (
                "Expected target_parameter to be one of volume "
                "or pressure if different from control parameter."
            )
            raise InvalidControl(
                msg=msg,
                target_parameter=target_parameter,
                control_mode=control_mode.value,
                control_parameter=control_parameter,
            )
        if target_parameter == control_mode.value:
            raise InvalidControl(
                msg="Target parameter cannot be the same as the control mode.",
                target_parameter=target_parameter,
                control_mode=control_mode.value,
                control_parameter=control_parameter,
            )

    # Initialize the nonlinear solver
    solver = Solver(use_snes=True)

    # Selects the control mode
    problem.control_mode = control_mode

    # Check if the state is a solution
    solver.solve(problem)

    if data_collector:
        data_collector.save()
        data_collector.plot()
        data_collector.print_status()

    target_end = numpyfy(target_end, write=False)
    # Get present value of both target and control parameters
    target_value = numpyfy(problem.get_control_parameter(target_parameter), write=False)
    control_value = numpyfy(
        problem.get_control_parameter(control_parameter), write=False
    )
    if control_step is None:
        control_step = numpyfy(target_end - target_value)
    else:
        control_step = numpyfy(control_step)

    logger.info(f"ITER TARGET {target_parameter} {target_end}")

    # Some flags
    target_reached = False

    # Used during prediction
    target_values = [np.copy(target_value)]
    control_values = [np.copy(control_value)]
    prev_states = [problem.state.copy(True)]

    # The main loop
    iterating = False
    while not target_reached:
        first_step = len(target_values) < 2

        # Old values
        target_value_old = target_values[-1]
        control_value_old = control_values[-1]
        state_old = prev_states[-1]

        # If target is control we simply increase control until we reach target
        if target_is_control:
            # If we are close to the target
            control_step = target_close(
                target_value,
                control_step,
                target_end,
            )

        # Else we need to iterate until target is passed and then find
        # correct control by interpolation.
        else:
            # Check if we cross the target
            # Interpolate!
            if not first_step:
                control_value, control_step, iterating = interpolate_control(
                    control_values=control_values,
                    control_value=control_value,
                    iterating=iterating,
                    target_values=target_values,
                    target_end=target_end,
                    control_step=control_step,
                )

        # New control value
        new_control_value = control_value + control_step
        control_value = new_control_value

        # Prediction step
        # ---------------
        if not first_step:
            prediction_step(
                control_values=control_values,
                prev_states=prev_states,
                control_value=control_value,
                problem=problem,
            )

        # Correction step
        # ---------------
        logger.info(
            "\nTRYING NEW CONTROL VALUE: {}={}{}\ntarget: {}={}, "
            'control parameter: "{}", control_mode: "{}"'.format(
                control_parameter,
                control_value,
                " (iterating)" if iterating else "",
                target_parameter,
                target_end,
                control_parameter,
                control_mode,
            )
        )
        problem.set_control_parameters(**{control_parameter: control_value})

        try:
            nliter, nlconv = solver.solve(problem)
            if not nlconv:
                raise RuntimeError("Solver did not converge...")
        except RuntimeError:
            logger.info("\nNOT CONVERGING")

            # Reset solution
            reset_problem_state(problem=problem, state_old=state_old)
            control_value = control_value_old

            # Reduce step
            control_step *= 0.5
            logger.info(f"REDUCING control_step = {control_step}")
            continue

        # Get target value
        target_value = numpyfy(problem.get_control_parameter(target_parameter))

        # If we are going in the wrong direction
        if stepping_in_wrong_direction(
            first_step=first_step,
            iterating=iterating,
            target_value=target_value,
            target_end=target_end,
            target_value_old=target_value_old,
        ):
            logger.info("STEPPING IN WRONG DIRECTION")

            # Reset solution
            reset_problem_state(problem=problem, state_old=state_old)
            control_value = control_value_old

            # Reduce step
            # May happen that the control step is too large
            control_step *= -0.8

            continue

        # Adapt control_step
        if not iterating and nliter < max_adapt_iter and adapt_step:
            control_step *= 2.0
            logger.info(f"INCREASING control_step = {control_step}")

        # Check if target has been reached

        if check_target_reached(target_value, target_end, tol):
            target_str = f"TARGET REACHED {target_parameter} = {target_value}"
            if target_parameter != control_parameter:
                target_str += f" WITH {control_parameter} = {control_value}"
            logger.info(target_str)
            target_reached = True

        else:
            # Save state
            target_values.append(np.copy(target_value))
            control_values.append(np.copy(control_value))
            if first_step:
                prev_states.append(problem.state.copy(True))
            else:
                # Switch place of the state vectors
                prev_states = [prev_states[-1], prev_states[0]]

                # Inplace update of last state values
                prev_states[-1].vector().zero()
                prev_states[-1].vector().axpy(1.0, problem.state.vector())

            logger.info(f"SUCCESFULL STEP: pendo={problem.pendo} Vendo={problem.Vendo}")

        # output
        if data_collector:
            if not iterating or target_reached:
                data_collector.save()
            data_collector.plot()
            data_collector.print_status()

    if data_collector:
        data_collector.flush()
