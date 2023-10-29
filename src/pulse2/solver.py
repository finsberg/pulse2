import dolfin

__all__ = ["Solver"]


class Solver:
    __slots__ = ("solver",)

    def __init__(self, use_snes=True, max_iterations=50):
        self.reset(use_snes, max_iterations=max_iterations)

    def solve(self, problem):
        problem._first_iteration = True
        problem._prev_residual = 1.0
        problem._recompute_jacobian = True
        result = self.solver.solve(problem, problem.state.vector())
        return result

    def reset(self, use_snes=True, max_iterations=50):
        dolfin.PETScOptions.set("pc_factor_mat_solver_type", "mumps")
        # dolfin.PETScOptions.set("pc_factor_mat_solver_package", "superlu_dist")
        dolfin.PETScOptions.set("ksp_type", "preonly")
        dolfin.PETScOptions.set("mat_mumps_icntl_7", 6)
        if use_snes:
            solver = dolfin.PETScSNESSolver()
            solver.parameters["report"] = False
            dolfin.PETScOptions.set("snes_monitor")
        else:
            solver = dolfin.NewtonSolver()

        solver.parameters["linear_solver"] = "mumps"
        # solver.parameters["linear_solver"] = "gmres"
        # solver.parameters["preconditioner"] = "petsc_amg"
        solver.parameters["maximum_iterations"] = max_iterations
        # solver.parameters["lu_solver"]["symmetric"] = True
        self.solver = solver

    # def reset(self, use_snes=True, max_iterations=14):
    #     dolfin.PETScOptions.set("pc_type", "gamg")
    #     # dolfin.PETScOptions.set("pc_factor_mat_solver_package", "superlu_dist")
    #     # dolfin.PETScOptions.set("ksp_type", "preonly")
    #     # dolfin.PETScOptions.set("mat_mumps_icntl_7", 6)

    #     dolfin.PETScOptions.set("pc_gamg_agg_nsmooths", 1)
    #     dolfin.PETScOptions.set("pc_gamg_square_graph", 2)
    #     dolfin.PETScOptions.set("pc_gamg_coarse_eq_limit", 2000)
    #     dolfin.PETScOptions.set("pc_gamg_esteig_ksp_type", "cg")
    #     dolfin.PETScOptions.set("pc_gamg_esteig_ksp_max_it", 20)
    #     dolfin.PETScOptions.set("pc_gamg_threshold", 0.01)
    #     dolfin.PETScOptions.set("mg_levels_ksp_type", "chebyshev")
    #     dolfin.PETScOptions.set("mg_levels_esteig_ksp_type", "cg")
    #     dolfin.PETScOptions.set("mg_levels_esteig_ksp_max_it", 20)
    #     dolfin.PETScOptions.set("mg_levels_ksp_max_it", 5)
    #     dolfin.PETScOptions.set("mg_levels_pc_type", "jacobi")
    #     if use_snes:
    #         solver = dolfin.PETScSNESSolver()
    #         solver.parameters["report"] = False
    #         dolfin.PETScOptions.set("snes_monitor")
    #     else:
    #         solver = dolfin.NewtonSolver()

    #     # solver.parameters["linear_solver"] = "lu"
    #     solver.parameters["linear_solver"] = "gmres"
    #     # solver.parameters["preconditioner"] = "petsc_amg"
    #     # solver.parameters["maximum_iterations"] = max_iterations
    #     # solver.parameters["lu_solver"]["symmetric"] = True
    #     self.solver = solver
