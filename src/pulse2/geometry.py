from __future__ import annotations
from dataclasses import dataclass, field
import logging
from enum import Enum
from pathlib import Path
import dolfin
import json
import ufl_legacy as ufl


logger = logging.getLogger(__name__)


class LongAxis(Enum):
    x = 0
    y = 1
    z = 2


def save_geometry(
    h5name: Path | str,
    mesh: dolfin.Mesh,
    markers: dict[str, tuple[int, int]] | None = None,
    f0: dolfin.Function | None = None,
    s0: dolfin.Function | None = None,
    n0: dolfin.Function | None = None,
    long_axis: LongAxis | int = LongAxis.x,
    cfun: dolfin.MeshFunction | None = None,
    ffun: dolfin.MeshFunction | None = None,
    rfun: dolfin.MeshFunction | None = None,
    vfun: dolfin.MeshFunction | None = None,
    h5group="",
    overwrite_file: bool = False,
):
    logger.info(f"Save geometry to {h5name}:{h5group}")
    h5name = Path(h5name)
    file_mode = "a" if h5name.is_file() and not overwrite_file else "w"

    data = {
        "mesh": mesh,
        "cfun": cfun,
        "ffun": ffun,
        "rfun": rfun,
        "vfun": vfun,
        "f0": f0,
        "s0": s0,
        "n0": n0,
    }

    if markers is None:
        markers = {}

    with dolfin.HDF5File(mesh.mpi_comm(), h5name.as_posix(), file_mode) as h5file:
        # Mesh functions
        for name, value in data.items():
            if value is None:
                continue
            h5file.write(value, f"{h5group}/{name}")

        h5file.attributes(h5group)["markers"] = json.dumps(markers)
        h5file.attributes(h5group)["long_axis"] = _get_longaxis(long_axis)


def _get_longaxis(long_axis: str | int | LongAxis) -> int:
    if isinstance(long_axis, str):
        return LongAxis[long_axis].value
    elif isinstance(long_axis, LongAxis):
        return long_axis.value
    else:
        assert isinstance(long_axis, int)
        return long_axis


@dataclass(slots=True)
class Geometry:
    mesh: dolfin.Mesh
    markers: dict[str, tuple[int, int]] = field(default_factory=dict)
    f0: dolfin.Function | None = None
    s0: dolfin.Function | None = None
    n0: dolfin.Function | None = None
    long_axis: int = 0
    cfun: dolfin.MeshFunction | None = None
    ffun: dolfin.MeshFunction | None = None
    rfun: dolfin.MeshFunction | None = None
    vfun: dolfin.MeshFunction | None = None

    def __post_init__(self):
        self._endoring_offset = self.compute_endoring_offset()

    @classmethod
    def from_file(
        cls, h5name, h5group="", comm=None, use_partition_from_file: bool = True
    ):
        comm = comm if comm is not None else dolfin.MPI.comm_world

        h5name = Path(h5name)
        if not h5name.is_file():
            raise FileNotFoundError(f"File {h5name} does not exist")

        mesh = dolfin.Mesh(comm)
        fields: dict[str, dolfin.Function] = {}
        mshfuncs: dict[str, dolfin.MeshFunction] = {}

        # load the microstructure
        with dolfin.HDF5File(mesh.mpi_comm(), h5name.as_posix(), "r") as h5file:
            if h5file.has_dataset(f"{h5group}/mesh"):
                h5file.read(mesh, f"{h5group}/mesh", use_partition_from_file)
            else:
                raise RuntimeError(f"No mesh found in file {h5name}")

            attrs = h5file.attributes(h5group).to_dict()
            markers = {
                k: (int(v[0]), int(v[1]))
                for k, v in json.loads(attrs.get("markers", "{}")).items()
                if len(v) == 2
            }
            long_axis = attrs.get("long_axis", 0)
            d = mesh.geometry().dim()

            for fname, dim in [
                ("cfun", d),
                ("ffun", d - 1),
                ("rfun", d - 2),
                ("vfun", d - 3),
            ]:
                fgroup = f"{h5group}/{fname}"
                if not h5file.has_dataset(fgroup) or d < 0:
                    continue

                logger.info(f"{cls}: loading '{fname}' from file")

                f = dolfin.MeshFunction("size_t", mesh, dim)
                h5file.read(f, fgroup)
                mshfuncs[fname] = f

            for f in ["f0", "s0", "n0"]:
                fgroup = f"{h5group}/{f}"
                if not h5file.has_dataset(fgroup):
                    continue
                logger.info(f"{cls}: loading '{f}' from file")

                fe = eval(h5file.attributes(fgroup)["signature"], ufl.__dict__)
                V = dolfin.FunctionSpace(mesh, fe)
                fun = dolfin.Function(V, name=f)
                h5file.read(fun, fgroup)
                fields[f] = fun

            return cls(
                mesh=mesh, markers=markers, long_axis=long_axis, **fields, **mshfuncs
            )

    def save(self, h5name, h5group=""):
        # open the file
        save_geometry(
            mesh=self.mesh,
            f0=self.f0,
            s0=self.s0,
            n0=self.n0,
            long_axis=self.long_axis,
            cfun=self.cfun,
            ffun=self.ffun,
            rfun=self.rfun,
            vfun=self.vfun,
            h5name=h5name,
            h5group=h5group,
            markers=self.markers,
            overwrite_file=True,
        )

    def compute_endoring_offset(self):
        return 0.0
        # ids = np.where(self.rfun.array() == self.ENDORING)[0]
        # self.mesh.init(1, 0)
        # pts = np.unique(np.hstack(map(self.mesh.topology()(1, 0), ids)))
        # coords = self.mesh.coordinates()[pts, self._long_axis]
        # quota_range = np.ptp(coords)
        # quota_range = MPI.max(self.mesh.mpi_comm(), quota_range)

        # if quota_range < DOLFIN_EPS:
        #     return coords[0]
        # else:
        #     return None

    @property
    def dim(self) -> int:
        return self.mesh.geometry().dim()

    @property
    def X(self) -> ufl.SpatialCoordinate:
        return ufl.SpatialCoordinate(self.mesh)

    @property
    def N(self) -> ufl.FacetNormal:
        return ufl.FacetNormal(self.mesh)

    @property
    def ds(self):
        return ufl.ds(domain=self.mesh, subdomain_data=self.ffun)

    def deformation_gradient(self, u=None):
        return ufl.Identity(self.dim) + ufl.grad(u)

    def inner_volume_form(self, u=None):
        # In general the base is not flat nor at quota = 0, so we need
        # a correction at least for the second case
        if self._endoring_offset is None:
            raise ValueError("The endoring at the base is not flat!")

        xshift = [0.0, 0.0, 0.0]

        xshift[self.long_axis] = self._endoring_offset
        xshift = dolfin.Constant(tuple(xshift))

        u = u or dolfin.Constant((0.0, 0.0, 0.0))

        x = self.X + u - xshift
        F = ufl.grad(x)
        n = ufl.cofac(F) * self.N

        return -1 / float(self.dim) * ufl.inner(x, n)

    def surface_area_form(self, u=None):
        # domain and boundaries

        u = u or dolfin.Constant((0.0, 0.0, 0.0))

        x = self.X + u
        F = ufl.grad(x)
        n = ufl.cofac(F) * self.N

        return ufl.sqrt(ufl.inner(n, n))

    def inner_volume(self, u=None, form_compiler_parameters=None, surf="ENDO"):
        """
        Compute the inner volume of the cavity for a given displacement u
        """

        # Create integration measure providing a mesh function for the
        # endocardial domain
        # FIXME: Include logic for bi ventricular endocardial volum
        ds_endo = self.ds(self._get_surf(surf))

        Vendo_form = self.inner_volume_form(u) * ds_endo

        ffc_params = form_compiler_parameters or {}
        V = dolfin.assemble(Vendo_form, form_compiler_parameters=ffc_params)
        return V

    def _get_surf(self, surf: int | str) -> int:
        assert isinstance(surf, (int, str))

        if isinstance(surf, str):
            assert surf in self.markers.keys()
            surf = self.markers[surf][0]
        return surf

    def surface_area(self, surf: int | str, u=None, form_compiler_parameters=None):
        """
        Compute the surface area of a given exterior facet domain.
        """

        ds_endo = self.ds(self._get_surf(surf))

        area_form = self.surface_area_form(u) * ds_endo
        ffc_params = form_compiler_parameters or {"quadrature_degree": 4}
        A = dolfin.assemble(area_form, form_compiler_parameters=ffc_params)

        return A


class LVGeometry(Geometry):
    ...


class BiVGeometry(Geometry):
    ...
