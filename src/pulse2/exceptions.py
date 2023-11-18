import operator
from dataclasses import dataclass

import dolfin
import numpy as np
from mpi4py import MPI


def check_value_greater_than(
    f: float | dolfin.Function | dolfin.Constant | np.ndarray,
    bound: float,
    inclusive: bool = False,
) -> bool:
    """Check that the value of f is greater than the given bound

    Parameters
    ----------
    f : float | dolfin.Function | dolfin.Constant | np.ndarray
        The variable to be checked
    bound : float
        The lower bound
    inclusive: bool
        Whether to include the bound in the check or not, by default False

    Returns
    -------
    bool
        True if value is greater than the lower bound,
        otherwise false
    """
    op = operator.ge if inclusive else operator.gt
    if np.isscalar(f):
        return op(f, bound)
    elif isinstance(f, dolfin.Constant):
        return op(f.values().max(), bound)
    elif isinstance(f, dolfin.Function):
        return op(
            f.function_space()
            .mesh()
            .mpi_comm()
            .allreduce(f.vector().get_local().max(), op=MPI.MAX),
            bound,
        )

    raise PulseException(  # pragma: no cover
        f"Invalid type for f: {type(f)}. Expected 'float', "
        "'dolfin.Constant', 'numpy array' or 'dolfin.Function'",
    )


def check_value_lower_than(
    f: float | dolfin.Function | dolfin.Constant,
    bound: float,
    inclusive: bool = False,
) -> bool:
    """Check that the value of f is lower than the given bound

    Parameters
    ----------
    f : float | dolfin.Function | dolfin.Constant | np.ndarray
        The variable to be checked
    bound : float
        The upper bound
    inclusive: bool
        Whether to include the bound in the check or not, by default False

    Returns
    -------
    bool
        True if value is greater than the lower bound,
        otherwise false
    """
    op = operator.le if inclusive else operator.lt
    if np.isscalar(f):
        return op(f, bound)
    elif isinstance(f, dolfin.Constant):
        return op(f.values().min(), bound)
    elif isinstance(f, dolfin.Function):
        return op(
            f.function_space()
            .mesh()
            .mpi_comm()
            .allreduce(f.vector().get_local().min(), op=MPI.MIN),
            bound,
        )

    raise PulseException(  # pragma: no cover
        f"Invalid type for f: {type(f)}. Expected 'float', "
        "'dolfin.Constant', 'numpy array' or 'dolfin.Function'",
    )


def check_value_between(
    f: float | dolfin.Function | dolfin.Constant,
    lower_bound: float,
    upper_bound: float,
    inclusive: bool = False,
) -> bool:
    """Check if value of `f` is between lower and upper bound

    Parameters
    ----------
    f : float | dolfin.Function | dolfin.Constant
        The variable to check
    lower_bound : float
        The lower bound
    upper_bound : float
        The upper bound
    inclusive: bool
        Whether to include the bound in the check or not, by default False

    Returns
    -------
    bool
        Return True if the value is between lower_bound and upper_bound,
        otherwise return False
    """
    return check_value_greater_than(
        f,
        lower_bound,
        inclusive=inclusive,
    ) and check_value_lower_than(f, upper_bound, inclusive=inclusive)


class PulseException(Exception):
    pass


@dataclass
class InvalidRangeError(ValueError, PulseException):
    name: str
    expected_range: tuple[float, float]

    def __str__(self) -> str:
        return (
            f"Invalid range for variable {self.name}. "
            f"Expected variable to be in the range: {self.expected_range}"
        )


@dataclass
class MissingModelAttribute(AttributeError, PulseException):
    attr: str
    model: str

    def __str__(self) -> str:
        return f"Missing required attributed {self.attr!r} for model {self.model!r}"


@dataclass
class InvalidControl(ValueError, PulseException):
    target_parameter: str
    control_parameter: str
    control_mode: str
    msg: str

    def __str__(self) -> str:
        return (
            f"{self.msg}\n"
            f"target_parameter = {self.target_parameter}\n"
            f"control_parameter = {self.control_parameter}\n"
            f"control_mode = {self.control_mode}\n"
        )


@dataclass
class InvalidMarker(KeyError, PulseException):
    marker: str
    valid_markers: tuple[str, ...]

    def __str__(self) -> str:
        return (
            f"Invalid marker {self.marker}. Possible options are "
            f"{self.valid_markers}"
        )


@dataclass
class SolverDidNotConverge(RuntimeError, PulseException):
    target_end: np.ndarray
    target_value: np.ndarray
    control_parameter: str
    control_mode: str
    control_value: np.ndarray
    control_step: np.ndarray
    n_failures: int

    def __str__(self) -> str:
        return (
            f"Solver failed after trying {self.n_failures} times. "
            f"Target value: {self.target_value}. "
            f"Target end: {self.target_end}. "
            f"Control mode: {self.control_mode}. "
            f"Control parameter: {self.control_parameter}. "
            f"Control step: {self.control_step}. "
        )


@dataclass
class ControlOutOfBounds(ValueError, PulseException):
    control_parameter: str
    control_value: np.ndarray
    min_control: np.ndarray | None
    max_control: np.ndarray | None

    def __str__(self) -> str:
        return (
            f"Value of control parameter {self.control_parameter} is "
            f"out of bounds with value {self.control_value} and "
            f"bounds: ({self.min_control}, {self.max_control})"
        )
