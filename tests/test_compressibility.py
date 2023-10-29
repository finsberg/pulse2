import math

import dolfin
import pytest
import ufl_legacy as ufl

import pulse2


def test_Incompressible(u, P1) -> None:
    p = dolfin.Function(P1)
    p.vector()[:] = 3.14
    u.interpolate(dolfin.Expression(("x[0]", "x[1]", "x[2]"), degree=1))
    F = pulse2.kinematics.DeformationGradient(u)
    J = pulse2.kinematics.Jacobian(F)
    comp = pulse2.compressibility.Incompressible()
    comp.register(p)
    psi = comp.strain_energy(J)
    value = dolfin.assemble(psi * ufl.dx)
    assert math.isclose(value, 3.14 * (8 - 1))


def test_Incompressible_with_missing_pressure_raises_MissingModelAttribute(u) -> None:
    u.interpolate(dolfin.Expression(("x[0]", "x[1]", "x[2]"), degree=1))
    F = pulse2.kinematics.DeformationGradient(u)
    J = pulse2.kinematics.Jacobian(F)
    comp = pulse2.compressibility.Incompressible()
    with pytest.raises(pulse2.exceptions.MissingModelAttribute):
        comp.strain_energy(J)


def test_Compressible(u) -> None:
    u.interpolate(dolfin.Expression(("x[0]", "x[1]", "x[2]"), degree=1))
    F = pulse2.kinematics.DeformationGradient(u)
    J = pulse2.kinematics.Jacobian(F)
    kappa = 1234
    comp = pulse2.compressibility.Compressible(kappa=kappa)
    psi = comp.strain_energy(J)
    value = dolfin.assemble(psi * ufl.dx)
    assert math.isclose(value, kappa * (8 * math.log(8) - 8 + 1))
