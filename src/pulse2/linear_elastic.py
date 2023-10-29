from dataclasses import dataclass

import dolfin
import ufl_legacy as ufl


from . import exceptions
from . import kinematics
from .material_model import Material


@dataclass(frozen=True, slots=True)
class LinearElastic(Material):
    """Linear elastic material

    Parameters
    ----------
    E: float | dolfin.Function | dolfin.Constant
        Youngs module
    nu: float | dolfin.Function | dolfin.Constant
        Poisson's ratio
    """

    E: float | dolfin.Function | dolfin.Constant
    nu: float | dolfin.Function | dolfin.Constant

    def __post_init__(self):
        # The poisson ratio has to be between -1.0 and 0.5
        if not exceptions.check_value_between(self.nu, -1.0, 0.5):
            raise exceptions.InvalidRangeError(name="mu", expected_range=(-1.0, 0.5))

    @property
    def parameters(self) -> dict[str, float | dolfin.Constant | dolfin.Function]:
        return {"E": self.E, "nu": self.nu}

    def sigma(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        r"""Cauchy stress for linear elastic material

        .. math::
            \sigma = \frac{E}{1 + \nu} \left( \varepsilon +
            \frac{\nu}{1 + \nu} \mathrm{tr}(\varepsilon) \mathbf{I} \right)


        Parameters
        ----------
        F : ufl.core.expr.Expr
            The deformation gradient

        Returns
        -------
        ufl.core.expr.Expr
            _description_
        """
        e = kinematics.EngineeringStrain(F)
        I = kinematics.SecondOrderIdentity(F)
        return (self.E / (1 + self.nu)) * (
            e + (self.nu / (1 - 2 * self.nu)) * ufl.tr(e) * I
        )
