import typing

import dolfin


class NeumannBC(typing.NamedTuple):
    traction: float | dolfin.Constant | dolfin.Function
    marker: int


class RobinBC(typing.NamedTuple):
    value: float | dolfin.Constant | dolfin.Function
    marker: int


class BoundaryConditions(typing.NamedTuple):
    neumann: typing.Sequence[NeumannBC] = ()
    dirichlet: typing.Sequence[
        typing.Callable[
            [dolfin.FunctionSpace],
            typing.Sequence[dolfin.DirichletBC],
        ]
    ] = ()
    robin: typing.Sequence[RobinBC] = ()
    body_force: typing.Sequence[float | dolfin.Constant | dolfin.Function] = ()
