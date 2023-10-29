import dolfin
import numpy as np
import ufl_legacy as ufl


def matrix_is_zero(A: ufl.core.expr.Expr) -> bool:
    n = ufl.domain.find_geometric_dimension(A)
    for i in range(n):
        for j in range(n):
            value = dolfin.assemble(A[i, j] * ufl.dx)
            print(i, j, value)
            is_zero = np.isclose(value, 0)
            if not is_zero:
                return False
    return True


def IsochoricDeformationGradient(u) -> ufl.core.expr.Expr:
    from pulse2 import kinematics

    return kinematics.IsochoricDeformationGradient(kinematics.DeformationGradient(u))


def float2object(
    f: float,
    obj_str: str,
    V: dolfin.FunctionSpace,
):
    if obj_str == "float":
        return f
    if obj_str == "Constant":
        return dolfin.Constant(f)
    if obj_str == "Function":
        v = dolfin.Function(V)
        v.vector()[:] = f
        return v
    raise ValueError(f"Invalid object string {obj_str!r}")
