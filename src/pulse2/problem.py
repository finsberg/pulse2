import typing
import logging
from enum import Enum
from dataclasses import dataclass
from dataclasses import field

import dolfin
import ufl_legacy as ufl

from .cardiac_model import CardiacModel
from .geometry import Geometry
from .exceptions import InvalidMarker


logger = logging.getLogger(__name__)


class ControlMode(Enum):
    pressure = "pressure"
    volume = "volume"


class BCType(Enum):
    fix_base = "fix_base"
    fix_base_ver = "fix_base_ver"
    free = "free"


@dataclass(slots=True)
class LVProblem(dolfin.NonlinearProblem):
    model: CardiacModel
    geometry: Geometry
    parameters: dict[str, typing.Any] = field(default_factory=dict)

    _virtual_work: ufl.Form = field(init=False, repr=False)
    _control_mode: ControlMode = field(
        init=False, repr=False, default=ControlMode.pressure
    )
    _prev_residual: float = field(init=False, default=1.0, repr=False)
    _recompute_jacobian: bool = field(init=False, repr=False, default=True)
    _first_iteration: bool = field(init=False, repr=False, default=True)
    _Mspace: dolfin.FunctionSpace = field(init=False, repr=False)
    state: dolfin.Function = field(init=False, repr=False)
    _G: ufl.Form = field(init=False, repr=False)
    _dG: ufl.Form = field(init=False, repr=False)
    _bcs: list[dolfin.DirichletBC] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        dolfin.NonlinearProblem.__init__(self)
        parameters = type(self).default_parameters()
        parameters.update(self.parameters)
        if isinstance(parameters["bc_type"], str):
            parameters["bc_type"] = BCType[parameters["bc_type"]]
        self.parameters = parameters
        self._init_space()
        self._init_forms()

    @staticmethod
    def default_parameters():
        p = {
            "bc_type": BCType.fix_base_ver,
            "recompute_jacobian": {"always": False, "residual_ratio": 0.5},
            "always_recompute_jacobian": False,
            "form_compiler_parameters": {"quadrature_degree": 4},
        }
        return p

    @property
    def dG(self):
        return self._dG

    @property
    def G(self):
        return self._G

    def F(self, b, x):
        logger.debug("Assemble F")
        ffc_params = self.parameters["form_compiler_parameters"]
        dolfin.assemble(self.G, tensor=b, form_compiler_parameters=ffc_params)

        for bc in self._bcs:
            bc.apply(b)

        # Compute residual
        residual = b.norm("l2")
        residual_ratio = residual / self._prev_residual
        self._recompute_jacobian = (
            residual_ratio > self.parameters["recompute_jacobian"]["residual_ratio"]
        )
        if not self._first_iteration:
            logger.debug(
                f"\nresidual: {residual:e} "
                + f"\nprevious residual: {self._prev_residual:e} "
                + f"\nratio: {residual_ratio:e}"
            )
        self._prev_residual = residual

    def J(self, A, x):
        ffc_params = self.parameters["form_compiler_parameters"]
        if (
            self.parameters["recompute_jacobian"]["always"]
            or self._first_iteration
            or self._recompute_jacobian
        ):
            logger.debug("Assemble J")
            dolfin.assemble(self.dG, tensor=A, form_compiler_parameters=ffc_params)
            for bc in self._bcs:
                bc.apply(A)
            self._first_iteration = False

    def _init_space(self):
        """Initialize MixedFunctionSpace"""
        mesh = self.geometry.mesh

        cell = mesh.ufl_cell()
        vlist = [ufl.VectorElement("CG", cell, 2, 3)]
        vlist += [ufl.FiniteElement("CG", cell, 1)]

        vlist += {
            BCType.fix_base: [],
            BCType.fix_base_ver: [ufl.VectorElement("Real", cell, 0, 3)],
            BCType.free: [ufl.VectorElement("Real", cell, 0, 6)],
        }[self.parameters["bc_type"]]

        if self._control_mode == ControlMode.volume:
            vlist += [ufl.FiniteElement("Real", cell, 0)]

        self._Mspace = dolfin.FunctionSpace(mesh, ufl.MixedElement(vlist))
        self.state = dolfin.Function(self._Mspace)

    def _init_forms(self):
        """Initialize Nonlinear and tangent problem forms"""
        param = self.parameters
        geo = self.geometry

        # unknowns
        state = self.state
        u = ufl.split(state)[0]
        p = ufl.split(state)[1]

        if self._control_mode == ControlMode.volume:
            pendo = ufl.split(state)[-1]
            param["V_endo"] = dolfin.Constant(geo.inner_volume())
            Vendo = param["V_endo"]
        else:
            param["p_endo"] = dolfin.Constant(0.0)
            pendo = param["p_endo"]
            Vendo = None

        # Deformation gradient tensor
        # ---------------------------
        Jgeo = 1.0
        F = ufl.Identity(3) + ufl.grad(u)

        # Internal energy
        # ---------------
        L = self.model.strain_energy(F, p) * Jgeo * dolfin.dx

        # Inner pressure
        # --------------
        L += self._inner_volume_constraint(u, pendo, Vendo, geo.markers["ENDO"][0])

        # bcs
        # ---
        bcs, L = self._handle_bcs(L, u, param["bc_type"])

        # tangent problem
        # ---------------
        # test and trial *required* because of a bug in DOLFIN
        # when domain is coordinate-specific (domain obtained from
        # the function is different from the real domain).
        self._G = ufl.derivative(L, self.state, ufl.TestFunction(self._Mspace))
        self._dG = ufl.derivative(self._G, self.state, ufl.TrialFunction(self._Mspace))
        self._bcs = bcs

    def _handle_bcs(
        self, L, u, bc_type: BCType
    ) -> tuple[list[dolfin.DirichletBC], ufl.Form]:
        if bc_type == BCType.fix_base_ver:
            if "BASE" not in self.geometry.markers:
                raise InvalidMarker(
                    marker="BASE", valid_markers=tuple(self.geometry.markers.keys())
                )
            c = ufl.split(self.state)[2]

            # No translations in plane of the base
            if self.geometry.long_axis == 0:
                ct = ufl.as_vector([0.0, c[0], c[1]])
            elif self.geometry.long_axis == 1:
                ct = ufl.as_vector([c[0], 0.0, c[1]])
            else:
                ct = ufl.as_vector([c[0], c[1], 0.0])

            # No rotations around the long axis
            cr = [0.0, 0.0, 0.0]
            cr[self.geometry.long_axis] = c[2]
            cr = ufl.as_vector(cr)
            L += self._rigid_motion_constraint(u, ct, cr)

            # No vertical displacement at the base
            Vsp = self._Mspace.sub(0).sub(self.geometry.long_axis)
            bcs = [
                dolfin.DirichletBC(
                    Vsp,
                    dolfin.Constant(0.0),
                    self.geometry.ffun,
                    self.geometry.markers["BASE"][0],
                )
            ]

        elif bc_type == BCType.fix_base:
            if "BASE" not in self.geometry.markers:
                raise InvalidMarker(
                    marker="BASE", valid_markers=tuple(self.geometry.markers.keys())
                )
            # No displacement at the base
            Vsp = self._Mspace.sub(0)
            bcs = [
                dolfin.DirichletBC(
                    Vsp,
                    dolfin.Constant((0.0, 0.0, 0.0)),
                    self.geometry.ffun,
                    self.geometry.markers["BASE"][0],
                )
            ]

        elif bc_type == BCType.free:
            X = self.geometry.X
            r = ufl.split(self.state)[2]
            RM = [
                dolfin.Constant((1, 0, 0)),
                dolfin.Constant((0, 1, 0)),
                dolfin.Constant((0, 0, 1)),
                dolfin.cross(X, dolfin.Constant((1, 0, 0))),
                dolfin.cross(X, dolfin.Constant((0, 1, 0))),
                dolfin.cross(X, dolfin.Constant((0, 0, 1))),
            ]
            bcs = []
            L += sum(dolfin.dot(u, zi) * r[i] * dolfin.dx for i, zi in enumerate(RM))

        else:
            raise NotImplementedError

        return bcs, L

    def _inner_volume_constraint(self, u, pendo, V, sigma):
        """
        Compute the form
            (V(u) - V, pendo) * ds(sigma)
        where V(u) is the volume computed from u and
            u = displacement
            V = volume enclosed by sigma
            pendo = Lagrange multiplier
        sigma is the boundary of the volume.
        """

        geo = self.geometry

        # ufl doesn't support any measure for duality
        # between two Real spaces, so we have to divide
        # by the total measure of the domain
        ds_sigma = geo.ds(sigma)
        area = dolfin.assemble(dolfin.Constant(1.0) * ds_sigma)

        V_u = geo.inner_volume_form(u)
        L = -pendo * V_u * ds_sigma

        if V is not None:
            L += dolfin.Constant(1.0 / area) * pendo * V * ds_sigma

        return L

    def _surface_area_constraint(self, u, p, A, sigma):
        geo = self.geometry

        ds_sigma = geo.ds(sigma)
        refarea = dolfin.assemble(dolfin.Constant(1.0) * ds_sigma)

        A_u = geo.surface_area_form(u)
        L = -p * A_u * ds_sigma

        if A is not None:
            L += dolfin.Constant(1.0 / refarea) * p * A * ds_sigma

        return L

    def _rigid_motion_constraint(self, u, ct, cr):
        """
        Compute the form
            (u, ct) * dx + (cross(u, X), cr) * dx
        where
            u  = displacement
            ct = Lagrange multiplier for translations
            ct = Lagrange multiplier for rotations
        """

        Lt = ufl.inner(ct, u) * ufl.dx
        X = self.geometry.X
        dim = self.geometry.dim

        if dim == 2:
            # rotations around z
            Lr = ufl.inner(cr, X[0] * u[1] - X[1] * u[0]) * ufl.dx

        elif dim == 3:
            # rotations around x, y, z
            Lr = ufl.inner(cr, ufl.cross(X, u)) * ufl.dx
        else:
            raise RuntimeError(f"Invalid dimension {dim}")

        return Lt + Lr

    def get_real_space_value(self, num):
        real_space_value = dolfin.Function(
            dolfin.FunctionSpace(self.geometry.mesh, "R", 0)
        )
        dolfin.assign(real_space_value, self.state.sub(num))
        return float(real_space_value)

    def set_real_space_value(self, num, value):
        assert isinstance(value, float), "expected a float for the value argument"
        real_space_value = dolfin.Function(
            dolfin.FunctionSpace(self.geometry.mesh, "R", 0)
        )
        real_space_value.interpolate(dolfin.Constant(value))
        dolfin.assign(self.state.sub(num), real_space_value)

    @property
    def control_mode(self) -> ControlMode:
        """Return the control model 'pressure' or 'volume'"""
        return self._control_mode

    @control_mode.setter
    def control_mode(self, mode: ControlMode | str) -> None:
        """Set the control model 'pressure' or 'volume'"""
        if isinstance(mode, str):
            mode = ControlMode[mode]
        self._change_mode_and_reinit(mode)

    def _change_mode_and_reinit(self, control_mode: ControlMode) -> None:
        assert control_mode in ControlMode
        if self.control_mode == control_mode:
            return

        # Save the current state
        state_old = self.state.copy(True)
        pendo_old = self.pendo
        Vendo_old = self.Vendo

        # Reinit problem
        self._control_mode = control_mode
        self._init_space()
        self._init_forms()

        # Assign old values
        dolfin.assign(self.state.sub(0), state_old.sub(0))
        dolfin.assign(self.state.sub(1), state_old.sub(1))

        if self.parameters["bc_type"] not in [BCType.fix_base]:
            dolfin.assign(self.state.sub(2), state_old.sub(2))

        self.pendo = pendo_old
        if self._control_mode == ControlMode.volume:
            self.Vendo = Vendo_old

    @property
    def displacement(self):
        return dolfin.Function(self.state, 0, name="displacement")

    @property
    def epiarea(self):
        """Return the value of the epicardial surface"""
        geo = self.geometry
        state = self.state
        u = ufl.split(state)[0]
        return geo.surface_area(geo.markers["EPI"][0], u)

    @property
    def pendo(self):
        """Return the value of the endo pressure"""
        if self._control_mode == ControlMode.volume:
            pnum = self._Mspace.num_sub_spaces() - 1
            return self.get_real_space_value(pnum)
        else:
            return float(self.parameters["p_endo"])

    @pendo.setter
    def pendo(self, p):
        """Set the value of the endo pressure"""
        if self._control_mode == ControlMode.volume:
            pnum = self._Mspace.num_sub_spaces() - 1
            self.set_real_space_value(pnum, p)
        else:
            self.parameters["p_endo"].assign(p)

    @property
    def Vendo(self):
        param = self.parameters
        if self._control_mode == ControlMode.pressure:
            geo = self.geometry
            state = self.state
            u = ufl.split(state)[0]
            return geo.inner_volume(u)
        else:
            return float(param["V_endo"])

    @Vendo.setter
    def Vendo(self, V):
        """Set the value of a the endo volume"""
        if self._control_mode == ControlMode.pressure:
            raise RuntimeError("Cannot assign Vendo!")
        else:
            self.parameters["V_endo"].assign(V)

    def set_control_parameter(self, name, value):
        """Set the value of a given control parameters"""
        # Volume or pressure parameters
        if name in ["pressure", "volume"]:
            if name != self.control_mode.value:
                logger.error(
                    "Problem is in {} control mode. "
                    "Cannot use {} as control.".format(self._control_mode, name)
                )

            if name == "pressure":
                self.pendo = value
            else:
                self.Vendo = value

        # Material parameters
        elif name in self.model.material.parameters:
            self.model.material.parameters[name].assign(value)

        # Activation parameters
        elif name in self.model.active.parameters:
            self.model.active.parameters[name].assign(value)

        else:
            logger.error(f"{name} is not a parameter.")

    def set_control_parameters(self, **params):
        """Set the value of any given control parameters"""
        for param, value in params.items():
            self.set_control_parameter(param, value)

    def get_control_parameter_list(self):
        """Returns a list of valid control parameter"""
        # FIXME should be independent from the specific mat
        p = ["pressure", "volume"]
        p.extend([k for k in self.model.material.parameters.keys() if k[1] == "_"])
        p.extend(self.model.active.parameters.keys())
        return p

    def get_control_parameter(self, param):
        """Returns the value of the given control parameter"""
        if param == "pressure":
            return self.pendo

        elif param == "volume":
            return self.Vendo

        elif param in self.model.material.parameters:
            return float(self.model.material.parameters[param])

        # Activation parameters
        elif param in self.model.active.parameters:
            return float(self.model.active.parameters[param])

        else:
            logger.error(f"{param} is not a parameter.")
