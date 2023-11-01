import pytest
import dolfin
import numpy as np
import pulse2


@pytest.fixture
def problem(lvgeo):
    geo = pulse2.LVGeometry(
        mesh=lvgeo.mesh,
        markers=lvgeo.markers,
        ffun=lvgeo.ffun,
        cfun=lvgeo.cfun,
        f0=lvgeo.f0,
        s0=lvgeo.s0,
        n0=lvgeo.n0,
    )

    material_params = pulse2.HolzapfelOgden.transversely_isotropic_parameters()
    material = pulse2.HolzapfelOgden(f0=geo.f0, s0=geo.s0, **material_params)

    Ta = dolfin.Constant(0.0)
    active_model = pulse2.ActiveStress(geo.f0, activation=Ta)
    comp_model = pulse2.Incompressible()

    model = pulse2.CardiacModel(
        material=material,
        active=active_model,
        compressibility=comp_model,
    )
    return pulse2.LVProblem(model=model, geometry=geo)


@pytest.mark.parametrize(
    "control_parameter, control_mode",
    (
        ("pressure", pulse2.ControlMode.volume),
        ("volume", pulse2.ControlMode.pressure),
    ),
)
def test_itertarget_invalid_targets_raises_InvalidTargets(
    control_parameter, control_mode
):
    with pytest.raises(pulse2.exceptions.InvalidControl) as e:
        pulse2.itertarget.itertarget(
            problem=None,
            target_end=1.0,
            target_parameter="pressure",
            control_step=0.02,
            control_parameter=control_parameter,
            control_mode=control_mode,
        )
    assert e.value.msg == "Control mode and control parameter have to be the same."


@pytest.mark.parametrize(
    (
        "target_parameter, control_parameter, control_mode, "
        "target_end, change_v, change_p"
    ),
    (
        ("pressure", "pressure", pulse2.ControlMode.pressure, 0.1, True, True),
        ("gamma", "gamma", pulse2.ControlMode.pressure, 0.1, True, False),
        ("pressure", "gamma", pulse2.ControlMode.volume, 0.1, False, True),
        ("gamma", "gamma", pulse2.ControlMode.volume, 0.1, False, True),
    ),
)
def test_itertarget_modes(
    target_parameter,
    control_parameter,
    control_mode,
    target_end,
    change_v,
    change_p,
    problem: pulse2.LVProblem,
):
    pendo_old = problem.pendo
    Vendo_old = problem.Vendo
    target_old = problem.get_control_parameter(target_parameter)
    pulse2.itertarget.itertarget(
        problem=problem,
        target_end=target_end,
        target_parameter=target_parameter,
        control_parameter=control_parameter,
        control_mode=control_mode,
    )
    pendo_new = problem.pendo
    Vendo_new = problem.Vendo
    target_new = problem.get_control_parameter(target_parameter)

    assert np.isclose(target_old, 0.0)
    assert np.isclose(target_new, target_end)

    if change_p:
        assert not np.isclose(pendo_old, pendo_new)
    else:
        assert np.isclose(pendo_old, pendo_new)

    if change_v:
        assert not np.isclose(Vendo_old, Vendo_new)
    else:
        assert np.isclose(Vendo_old, Vendo_new)


@pytest.mark.parametrize(
    "target_value, target_end, tol, expected",
    (
        (1.0, 1.0, 1e-12, True),
        (1.0, 1.1, 1e-12, False),
        (1.0, 1.1, 0.2, True),
        (-1.0, -1.0, 1e-12, True),
        (-1.0, -1.1, 1e-12, False),
        (-1.0, -1.1, 0.2, True),
    ),
)
def test_check_target_reached(target_value, target_end, tol, expected):
    assert (
        pulse2.itertarget.check_target_reached(
            np.array([target_value]), np.array([target_end]), tol
        )
        == expected
    )


@pytest.mark.parametrize(
    "first_step, iterating, target_value, target_end, target_value_old, expected",
    (
        (True, False, 0.9, 1.0, 0.8, False),
        (True, False, 0.8, 1.0, 0.9, True),
        (False, False, 0.8, 1.0, 0.9, False),
        (False, True, 0.8, 1.0, 0.9, False),
    ),
)
def test_stepping_in_wrong_direction(
    first_step, iterating, target_value, target_end, target_value_old, expected
):
    assert (
        pulse2.itertarget.stepping_in_wrong_direction(
            first_step,
            iterating,
            np.array([target_value]),
            np.array([target_end]),
            np.array([target_value_old]),
        )
        == expected
    )


@pytest.mark.parametrize(
    "target_value, control_step, target_end, expected",
    (
        (0.8, 0.1, 1.0, 0.1),
        (0.95, 0.1, 1.0, 0.05),
        (1.2, -0.1, 1.0, -0.1),
        (1.05, -0.1, 1.0, -0.05),
        (1.2, 0.1, 1.0, -0.1),
        (1.05, 0.1, 1.0, -0.05),
    ),
)
def test_target_close_1D(target_value, control_step, target_end, expected):
    assert np.allclose(
        pulse2.itertarget.target_close(
            np.array([target_value]),
            np.array([control_step]),
            np.array([target_end]),
        ),
        expected,
    )


@pytest.mark.parametrize(
    "target_value1, control_step1, target_end1, expected1",
    (
        (0.8, 0.1, 1.0, 0.1),
        (0.95, 0.1, 1.0, 0.05),
        (1.2, -0.1, 1.0, -0.1),
        (1.05, -0.1, 1.0, -0.05),
        (1.2, 0.1, 1.0, -0.1),
        (1.05, 0.1, 1.0, -0.05),
    ),
)
@pytest.mark.parametrize(
    "target_value2, control_step2, target_end2, expected2",
    (
        (0.8, 0.1, 1.0, 0.1),
        (0.95, 0.1, 1.0, 0.05),
        (1.2, -0.1, 1.0, -0.1),
        (1.05, -0.1, 1.0, -0.05),
        (1.2, 0.1, 1.0, -0.1),
        (1.05, 0.1, 1.0, -0.05),
    ),
)
def test_target_close_2D(
    target_value1,
    control_step1,
    target_end1,
    expected1,
    target_value2,
    control_step2,
    target_end2,
    expected2,
):
    assert np.allclose(
        pulse2.itertarget.target_close(
            np.array([target_value1, target_value2]),
            np.array([control_step1, control_step2]),
            np.array([target_end1, target_end2]),
        ),
        np.array([expected1, expected2]),
    )
