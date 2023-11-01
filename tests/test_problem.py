import pytest
import dolfin
import pulse2
import numpy as np


@pytest.mark.parametrize("bc_type", pulse2.problem.BCType)
def test_LVProblem(lvgeo, bc_type):
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

    # traction = dolfin.Constant(0.0)
    # neumann = pulse2.NeumannBC(traction=traction, marker=2)

    # robin_value = dolfin.Constant(0.0)
    # robin = pulse2.RobinBC(value=robin_value, marker=3)

    # body_force = dolfin.Constant((0.0, 0.0, 0.0))
    # breakpoint()
    # bcs = pulse2.BoundaryConditions(
    #     dirichlet=(dirichlet_bc,),
    #     neumann=(neumann,),
    #     robin=(robin,),
    #     body_force=(body_force,),
    # )

    problem = pulse2.LVProblem(
        model=model, geometry=geo, parameters={"bc_type": bc_type}
    )
    solver = pulse2.Solver()
    result = solver.solve(problem)
    assert result[-1]

    u0, p0, *extra = problem.state.split(deepcopy=True)

    # With the HolzapfelOgden model the hydrostatic pressure
    # should equal the negative of the material parameter a
    assert np.allclose(p0.vector().get_local(), -material_params["a"])
    # And with no external forces, there should be no displacement
    assert np.allclose(u0.vector().get_local(), 0.0)

    # Simple inflation
    pendo = 0.1
    problem.pendo = pendo
    result = solver.solve(problem)
    assert result[-1]

    u1, p1, *extra1 = problem.state.split(deepcopy=True)
    # And both values should now be changed
    assert not np.allclose(u1.vector().get_local(), 0.0)
    assert not np.allclose(p1.vector().get_local(), -material_params["a"])

    # Now change the control mode to volume so that we keep
    # the volume constant
    # First we compute the volume
    vol1 = geo.inner_volume(u1)
    problem.control_mode = pulse2.ControlMode.volume
    # And solve without any changes
    solver.reset()
    result = solver.solve(problem)
    assert result[-1]

    u2, p2, *extra2 = problem.state.split(deepcopy=True)
    vol2 = geo.inner_volume(u1)
    # In this case everything should be the same
    assert np.allclose(u1.vector().get_local(), u2.vector().get_local())
    assert np.allclose(p1.vector().get_local(), p2.vector().get_local())
    # And the pressure should be the same
    assert np.allclose(extra2[-1].vector().get_local(), pendo)
    # And the volume should be the same
    assert np.isclose(vol1, vol2)

    # Now let us increase the active tension and solve for the
    # pressure with fixed volume

    problem.set_control_parameter("gamma", 0.1)
    result = solver.solve(problem)
    assert result[-1]
    u3, p3, *extra3 = problem.state.split(deepcopy=True)
    vol3 = geo.inner_volume(u1)
    # The volume should be the same
    assert np.isclose(vol2, vol3)
    # But the pressure should be increased
    assert (extra3[-1].vector().get_local() > pendo).all()

    # u and p should also be different
    assert not np.allclose(u1.vector().get_local(), u3.vector().get_local())
    assert not np.allclose(p1.vector().get_local(), p3.vector().get_local())
