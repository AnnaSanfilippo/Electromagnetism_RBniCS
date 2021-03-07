# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# 3D extension of the thermal toy problem, present in "Reduced Basis Methods for Partial   Differential Equations: An Introduction" by Alfio Quarteroni, Andrea Manzoni and Federico Negri. In this case the sampling is the exponential one.

from dolfin import *
from rbnics import *

class ThermalBlock(EllipticCoerciveCompliantProblem):
    
    # Default initialization of members
    def __init__(self, V, **kwargs):
        # Call the standard initialization
        EllipticCoerciveCompliantProblem.__init__(self, V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        self.u = TrialFunction(V)
        self.v = TestFunction(V)
        self.dx = Measure("dx")(subdomain_data=self.subdomains)
        self.ds = Measure("ds")(subdomain_data=self.boundaries)
    
#    # Return the alpha_lower bound.
#    def get_stability_factor_lower_bound(self):
#        return min(self.compute_theta("a"))
    
    # Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        mu = self.mu
        if term == "a":
            theta_a0 = 10**(-mu[0])
            theta_a1 = 1.
            return (theta_a0, theta_a1)
        elif term == "f":
            theta_f0 = mu[1]
            return (theta_f0,)
        else:
            raise ValueError("Invalid term for compute_theta().")
    
    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        v = self.v
        dx = self.dx
        if term == "a":
            u = self.u
            a0 = inner(grad(u), grad(v))*dx(1)
            a1 = inner(grad(u), grad(v))*dx(2)
            return (a0, a1)
        elif term == "f":
            ds = self.ds
            f0 = v*ds(1)
            return (f0,)
        elif term == "dirichlet_bc":
            bc0 = [DirichletBC(self.V, Constant(0.0), self.boundaries, 3)]
            return (bc0,)
        elif term == "inner_product":
            u = self.u
            x0 = inner(grad(u), grad(v))*dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")
        
# 1. Read the mesh for this problem
#    The files are in .xdmf extension

mesh = Mesh()
with XDMFFile("data/thermal_block.xdmf") as infile:
    infile.read(mesh)

subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
with XDMFFile("data/thermal_block_subdomains.xdmf") as infile:
    infile.read(subdomains)

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
with XDMFFile("data/thermal_block_boundaries.xdmf") as infile:
    infile.read(boundaries)

# 2. Create Finite Element space (Lagrange P1)
V = FunctionSpace(mesh, "Lagrange", 1)

# 3. Allocate an object of the ThermalBlock class
thermal_block_problem = ThermalBlock(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(-2.0, 2.0), (-1.0, 1.0)]
thermal_block_problem.set_mu_range(mu_range)

# 4. Prepare reduction with a POD-Galerkin method
pod_galerkin_method = PODGalerkin(thermal_block_problem)
pod_galerkin_method.set_Nmax(9)
#pod_galerkin_method.set_tolerance(2e-4)

# 5. Perform the offline phase
pod_galerkin_method.initialize_training_set(50)
reduced_thermal_block_problem = pod_galerkin_method.offline()

# 6. Perform an online solve
online_mu = (-0.78, -1.0)
reduced_thermal_block_problem.set_mu(online_mu)
reduced_thermal_block_problem.solve()
reduced_thermal_block_problem.export_solution(filename="online_solution")

thermal_block_problem.solve()
thermal_block_problem.export_solution(filename="truth_solution")

reduced_thermal_block_problem.export_error(filename="error")

# 7. Perform an error analysis
pod_galerkin_method.initialize_testing_set(50)
pod_galerkin_method.error_analysis()

# 8. Perform a speedup analysis
pod_galerkin_method.initialize_testing_set(50)
pod_galerkin_method.speedup_analysis()
