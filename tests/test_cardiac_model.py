import math

import dolfin
import pulse2
import ufl_legacy as ufl


def test_CardiacModel(mesh, u):
    material_params = pulse2.HolzapfelOgden.transversely_isotropic_parameters()
    f0 = dolfin.Constant((1.0, 0.0, 0.0))
    s0 = dolfin.Constant((0.0, 1.0, 0.0))
    material = pulse2.HolzapfelOgden(f0=f0, s0=s0, **material_params)

    active_model = pulse2.ActiveStress(f0)
    comp_model = pulse2.Compressible()

    model = pulse2.CardiacModel(
        material=material,
        active=active_model,
        compressibility=comp_model,
    )
    u.interpolate(
        dolfin.Expression(("x[0] / 10.0", "x[1] / 10.0", "x[2] / 10.0"), degree=1)
    )
    F = pulse2.kinematics.DeformationGradient(u)
    psi = model.strain_energy(F)
    value = dolfin.assemble(psi * ufl.dx)
    assert math.isclose(value, 103.22036041941614)
