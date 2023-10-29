import math

import dolfin
import pulse2
import pytest
import ufl_legacy as ufl

import utils


@pytest.mark.parametrize("obj_str", ("float", "Constant", "Function"))
def test_linear_elastic_model(obj_str, mesh, P1, u) -> None:
    E = 2.0
    _nu = 0.2
    nu = utils.float2object(f=_nu, obj_str=obj_str, V=P1)
    model = pulse2.LinearElastic(E=E, nu=nu)

    u.interpolate(dolfin.Expression(("x[0]", "x[1]", "x[2]"), degree=1))
    F = pulse2.kinematics.DeformationGradient(u)
    # F = 2I, e = I, tr(e) = 3
    # sigma = (E / (1 + nu)) * (e + (nu / (1 - 2 * nu)) * tr(e) * I
    # sigma = (E / (1 + nu)) * (1 + (nu / (1 - 2 * nu)) * 3) * I
    sigma = model.sigma(F)
    I = ufl.Identity(3)
    zero = sigma - (E / (1 + _nu)) * (1 + (_nu / (1 - 2 * _nu)) * 3) * I
    assert utils.matrix_is_zero(zero)


@pytest.mark.parametrize("obj_str", ("float", "Constant", "Function"))
def test_linear_elastic_model_with_invalid_range(obj_str, P1) -> None:
    E = 2.0
    _nu = 0.5
    nu = utils.float2object(f=_nu, obj_str=obj_str, V=P1)

    with pytest.raises(pulse2.exceptions.InvalidRangeError):
        pulse2.LinearElastic(E=E, nu=nu)


@pytest.mark.parametrize(
    "params_func, expected_value",
    (
        (pulse2.HolzapfelOgden.orthotropic_parameters, 1.2352937267),
        (pulse2.HolzapfelOgden.partly_orthotropic_parameters, 1.435870273157),
        (
            pulse2.HolzapfelOgden.transversely_isotropic_parameters,
            53.6468124607508,
        ),
    ),
)
def test_holzapfel_ogden(params_func, expected_value, mesh, u) -> None:
    params = params_func()
    f0 = dolfin.Constant((1.0, 0.0, 0.0))
    s0 = dolfin.Constant((0.0, 1.0, 0.0))
    model = pulse2.HolzapfelOgden(f0=f0, s0=s0, **params)

    u.interpolate(
        dolfin.Expression(("x[0] / 10.0", "x[1] / 10.0", "x[2] / 10.0"), degree=1)
    )
    F = pulse2.kinematics.DeformationGradient(u)
    # F = I + 0.1 I, C = 1.21 I
    psi = model.strain_energy(F)
    value = dolfin.assemble(psi * ufl.dx)
    assert math.isclose(value, expected_value)


def test_holzapfel_ogden_invalid_range():
    with pytest.raises(pulse2.exceptions.InvalidRangeError):
        pulse2.HolzapfelOgden(a=-1.0)


@pytest.mark.parametrize(
    "params, attr",
    (
        ({"a_f": 1}, "f0"),
        ({"a_s": 1}, "s0"),
        ({"a_fs": 1}, "f0 and/or s0"),
    ),
)
def test_holzapfel_ogden_raises_MissingModelAttribute(params, attr):
    with pytest.raises(pulse2.exceptions.MissingModelAttribute) as e:
        pulse2.HolzapfelOgden(**params)
    assert e.value == pulse2.exceptions.MissingModelAttribute(
        attr=attr,
        model="HolzapfelOgden",
    )


def test_holzapfel_ogden_neohookean(u):
    model = pulse2.HolzapfelOgden(a=1.0)
    u.interpolate(
        dolfin.Expression(("x[0] / 10.0", "x[1] / 10.0", "x[2] / 10.0"), degree=1)
    )
    F = pulse2.kinematics.DeformationGradient(u)
    psi = model.strain_energy(F)
    value = dolfin.assemble(psi * ufl.dx)
    # F = I + 0.1 I, C = 1.21 I, I1= 3*1.21
    # psi = (a / 2) * (I1 - 3) = 0.5 (3 * 1.21 - 3) = 0.315
    assert math.isclose(value, 0.315)


def test_holzapfel_ogden_pure_fiber(u, mesh):
    f0 = dolfin.Constant((1.0, 0.0, 0.0))
    model = pulse2.HolzapfelOgden(a_f=1.0, f0=f0)
    u.interpolate(
        dolfin.Expression(("x[0] / 10.0", "x[1] / 10.0", "x[2] / 10.0"), degree=1)
    )
    F = pulse2.kinematics.DeformationGradient(u)
    psi = model.strain_energy(F)
    value = dolfin.assemble(psi * ufl.dx)
    # F = I + 0.1 I, C = 1.21 I, I4f = 1.21
    # psi = (a_f / 2) * (I4 - 1)**2 = 0.5 * 0.21**2
    assert math.isclose(value, 0.5 * 0.21**2)


def test_holzapfel_ogden_pure_fiber_sheets(u, mesh):
    f0 = dolfin.Constant((1.0, 0.0, 0.0))
    model = pulse2.HolzapfelOgden(a_fs=1.0, f0=f0, s0=f0)
    u.interpolate(
        dolfin.Expression(("x[0] / 10.0", "x[1] / 10.0", "x[2] / 10.0"), degree=1)
    )
    F = pulse2.kinematics.DeformationGradient(u)
    psi = model.strain_energy(F)
    value = dolfin.assemble(psi * ufl.dx)
    # F = I + 0.1 I, = 1.1 -> I8fs = 1.21
    # psi = (a_f / 2) * I8fs**2 = 0.5 * 1.21**2
    assert math.isclose(value, 0.5 * 1.21**2)
