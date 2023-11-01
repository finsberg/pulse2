# # Simple PV - loop
# TBW

# Local imports
import logging
import dolfin

# from pulse2.problem import Problem
from pulse2.itertarget import itertarget
import pulse2
from pulse2.geometry import LVGeometry
from cardiac_geometries.geometry import Geometry

# from pulse2.material import HolzapfelOgden, Guccione

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# We only want standard output on rank 0. We therefore set the log level to
# ERROR on all other ranks

comm = dolfin.MPI.comm_world


# Read geometry from file. If the file is not present we regenerate it.

geo = Geometry.from_folder("lv")
geo = LVGeometry(
    mesh=geo.mesh,
    markers=geo.markers,
    ffun=geo.ffun,
    cfun=geo.cfun,
    f0=geo.f0,
    s0=geo.s0,
    n0=geo.n0,
)
geo.mesh.coordinates()[:] /= 3

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
problem = pulse2.LVProblem(
    model=model, geometry=geo, parameters={"bc_type": "fix_base"}
)

# Numerical constants setting the boundaries for the pV loop

p_atrium = 0.2
p_ED = 1.2
p_aortic = 17.0
ESV = 70.0
max_adapt_iter = 12


# Save displacements for later processing
volumes = []
pressures = []
gammas = []

# ofile.write(Function(problem.displacement,name = "u"),0.0)
# ofile.write(problem.displacement, 0.0)
with dolfin.XDMFFile("simple_pvloop.xdmf") as ofile:
    ofile.write_checkpoint(
        problem.displacement,
        "u",
        0.0,
        dolfin.XDMFFile.Encoding.HDF5,
        True,
    )
logger.info(f"VOLUME = {problem.Vendo}")
volumes.append(problem.Vendo)
pressures.append(problem.pendo)
gammas.append(problem.get_control_parameter("gamma"))


itertarget(
    problem,
    target_end=p_atrium,
    target_parameter="pressure",
    control_step=0.02,
    control_parameter="pressure",
    control_mode="pressure",
    data_collector=None,
    max_adapt_iter=max_adapt_iter,
)

# ofile.write(problem.displacement, 1.0)
with dolfin.XDMFFile("simple_pvloop.xdmf") as ofile:
    ofile.write_checkpoint(
        problem.displacement,
        "u",
        1.0,
        dolfin.XDMFFile.Encoding.HDF5,
        True,
    )
logger.info(f"VOLUME = {problem.Vendo}")
volumes.append(problem.Vendo)
pressures.append(problem.pendo)
gammas.append(problem.get_control_parameter("gamma"))

itertarget(
    problem,
    target_end=p_ED,
    target_parameter="pressure",
    control_step=0.02,
    control_parameter="pressure",
    control_mode="pressure",
    data_collector=None,
    max_adapt_iter=max_adapt_iter,
)

# ofile.write(problem.displacement, 1.0)
with dolfin.XDMFFile("simple_pvloop.xdmf") as ofile:
    ofile.write_checkpoint(
        problem.displacement,
        "u",
        2.0,
        dolfin.XDMFFile.Encoding.HDF5,
        True,
    )
logger.info(f"VOLUME = {problem.Vendo}")
volumes.append(problem.Vendo)
pressures.append(problem.pendo)
gammas.append(problem.get_control_parameter("gamma"))

# Isovolumic contraction. Adjusting gamma to reach aortic pressure

itertarget(
    problem,
    target_end=p_aortic,
    target_parameter="pressure",
    control_step=0.01,
    control_parameter="gamma",
    control_mode="volume",
    data_collector=None,
    max_adapt_iter=max_adapt_iter,
)

# ofile.write(Function(problem.displacement, name="u"), 2.0)
with dolfin.XDMFFile("simple_pvloop.xdmf") as ofile:
    ofile.write_checkpoint(
        problem.displacement,
        "u",
        3.0,
        dolfin.XDMFFile.Encoding.HDF5,
        True,
    )
logger.info(f"VOLUME = {problem.Vendo}")
volumes.append(problem.Vendo)
pressures.append(problem.pendo)
gammas.append(problem.get_control_parameter("gamma"))

# Isotonic contraction. Adjusting gamma to reach gamma max

itertarget(
    problem,
    target_end=ESV,
    target_parameter="volume",
    control_step=0.01,
    control_parameter="gamma",
    control_mode="pressure",
    data_collector=None,
    max_adapt_iter=max_adapt_iter,
)

# ofile.write(Function(problem.displacement, name="u"), 3.0)
with dolfin.XDMFFile("simple_pvloop.xdmf") as ofile:
    ofile.write_checkpoint(
        problem.displacement,
        "u",
        4.0,
        dolfin.XDMFFile.Encoding.HDF5,
        True,
    )
logger.info(f"VOLUME = {problem.Vendo}")
volumes.append(problem.Vendo)
pressures.append(problem.pendo)
gammas.append(problem.get_control_parameter("gamma"))

# Isovolumic relaxation. Adjusting gamma negatively to reach atrial pressure.

itertarget(
    problem,
    target_end=p_atrium,
    target_parameter="pressure",
    control_step=-0.01,
    control_parameter="gamma",
    control_mode="volume",
    data_collector=None,
    max_adapt_iter=max_adapt_iter,
)


# ofile.write(Function(problem.displacement, name="u"), 4.0)
with dolfin.XDMFFile("simple_pvloop.xdmf") as ofile:
    ofile.write_checkpoint(
        problem.displacement,
        "u",
        5.0,
        dolfin.XDMFFile.Encoding.HDF5,
        True,
    )
logger.info(f"VOLUME = {problem.Vendo}")
volumes.append(problem.Vendo)
pressures.append(problem.pendo)
gammas.append(problem.get_control_parameter("gamma"))

# Isotonic relaxation. Adjusting gamma negatively to reach zero active force.

itertarget(
    problem,
    target_end=0.0,
    target_parameter="gamma",
    control_step=-0.01,
    control_parameter="gamma",
    control_mode="pressure",
    data_collector=None,
    max_adapt_iter=max_adapt_iter,
)

# ofile.write(Function(problem.displacement, name="u"), 5.0)
with dolfin.XDMFFile("simple_pvloop.xdmf") as ofile:
    ofile.write_checkpoint(
        problem.displacement,
        "u",
        6.0,
        dolfin.XDMFFile.Encoding.HDF5,
        True,
    )
logger.info(f"VOLUME = {problem.Vendo}")
volumes.append(problem.Vendo)
pressures.append(problem.pendo)
gammas.append(problem.get_control_parameter("gamma"))

import matplotlib.pyplot as plt

fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(volumes)
ax[1].plot(pressures)
ax[2].plot(gammas)
fig.savefig("pressure_volume_gamma.png")

fig, ax = plt.subplots()
ax.plot(volumes, pressures)
fig.savefig("pv_loop.png")

print(f"{volumes = }")
print(f"{pressures = }")
print(f"{gammas = }")
