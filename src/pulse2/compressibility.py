import abc
from dataclasses import dataclass
from dataclasses import field
from typing_extensions import override

import dolfin
import ufl_legacy as ufl


from . import exceptions


class Compressibility(abc.ABC):
    @abc.abstractmethod
    def strain_energy(self, J: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        pass

    def register(self, *args, **kwargs) -> None:
        pass


@dataclass(slots=True)
class Incompressible(Compressibility):
    p: dolfin.Function = field(default=None, init=False)

    @override
    def register(self, p: dolfin.Function) -> None:
        self.p = p

    def strain_energy(self, J: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        if self.p is None:
            raise exceptions.MissingModelAttribute(attr="p", model=type(self).__name__)
        return self.p * (J - 1.0)


@dataclass(slots=True)
class Compressible(Compressibility):
    kappa: float | dolfin.Function | dolfin.Constant = 1e3

    def strain_energy(self, J: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        return self.kappa * (J * ufl.ln(J) - J + 1)
