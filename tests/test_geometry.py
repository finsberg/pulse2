from pathlib import Path
import dolfin
import numpy as np
import pulse2


def test_save_load_geometry(tmp_path: Path):
    mesh = dolfin.UnitCubeMesh(3, 3, 3)
    cfun = dolfin.MeshFunction("size_t", mesh, 3)
    cfun.set_all(43)
    ffun = dolfin.MeshFunction("size_t", mesh, 2)
    ffun.set_all(42)
    rfun = dolfin.MeshFunction("size_t", mesh, 1)
    rfun.set_all(41)
    vfun = dolfin.MeshFunction("size_t", mesh, 0)
    vfun.set_all(40)
    V = dolfin.VectorFunctionSpace(mesh, "CG", 1)
    W = dolfin.VectorFunctionSpace(mesh, "DG", 0)
    f0 = dolfin.Function(V)
    f0.vector()[:] = 1
    s0 = dolfin.Function(V)
    s0.vector()[:] = 2
    n0 = dolfin.Function(W)
    n0.vector()[:] = 3
    markers = {"ENDO": (1, 2), "EPI": (2, 2), "BASE": (3, 2), "EPIRING": (4, 1)}
    long_axis = pulse2.geometry.LongAxis.y

    h5group = "group"
    pulse2.geometry.save_geometry(
        tmp_path.with_suffix(".h5"),
        mesh=mesh,
        markers=markers,
        f0=f0,
        s0=s0,
        n0=n0,
        long_axis=long_axis,
        cfun=cfun,
        ffun=ffun,
        rfun=rfun,
        vfun=vfun,
        h5group=h5group,
    )

    geo = pulse2.LVGeometry.from_file(tmp_path.with_suffix(".h5"), h5group=h5group)

    assert np.allclose(geo.f0.vector().get_local(), f0.vector().get_local())
    assert np.allclose(geo.s0.vector().get_local(), s0.vector().get_local())
    assert np.allclose(geo.n0.vector().get_local(), n0.vector().get_local())

    assert np.allclose(geo.cfun.array(), cfun.array())
    assert np.allclose(geo.ffun.array(), ffun.array())
    assert np.allclose(geo.rfun.array(), rfun.array())
    assert np.allclose(geo.vfun.array(), vfun.array())

    assert np.allclose(geo.long_axis, 1)
    assert geo.markers == markers
