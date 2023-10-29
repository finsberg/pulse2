import dolfin
import pytest
import cardiac_geometries
import ufl_legacy as ufl


@pytest.fixture(scope="session")
def mesh():
    return dolfin.UnitCubeMesh(3, 3, 3)


@pytest.fixture(scope="session")
def P1(mesh):
    return dolfin.FunctionSpace(mesh, ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1))


@pytest.fixture(scope="session")
def P2(mesh):
    return dolfin.FunctionSpace(mesh, ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2))


@pytest.fixture
def u(P2):
    return dolfin.Function(P2)


@pytest.fixture(scope="session")
def lvgeo(tmpdir_factory):
    return cardiac_geometries.create_lv_ellipsoid(
        tmpdir_factory.mktemp("lvgeo"), create_fibers=True
    )
