# Copyright (C) 2015-2019 by the RBniCS authors
#
# This file is part of RBniCS.
#
# RBniCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS. If not, see <http://www.gnu.org/licenses/>.
#

from dolfin import *
from rbnics import *

class Electromagnetism_Solution(EllipticCoerciveCompliantProblem):

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
        x, y, z = SpatialCoordinate(mesh)
        self.f = as_vector(((z**2-1)*(2*x*y-x**2+1)-(y**2-1)*(x**2-1-2*x*z)+0.5*(x**2-1)*(y**2-1)*(z**2-1),
                (x**2-1)*(2*y*z-y**2+1)-(z**2-1)*(y**2-1-2*x*y)+0.5*(x**2-1)*(y**2-1)*(z**2-1),
                (y**2-1)*(2*x*z+1-z**2)-(x**2-1)*(z**2-1-2*y*z)+0.5*(x**2-1)*(y**2-1)*(z**2-1)))

#    # Return the alpha_lower bound.
#    def get_stability_factor_lower_bound(self):
#        return min(self.compute_theta("a"))

    # Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        mu = self.mu
        if term == "a":
            theta_a0 = 10**(-mu[0]) # Per sampling esponenziale
            theta_a1 = 1.
            theta_a2 = 1.
            return (theta_a0, theta_a1, theta_a2)
        elif term == "f":
            theta_f0 = 1
            return (theta_f0,)
        else:
            raise ValueError("Invalid term for compute_theta().")

    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        v = self.v
        dx = self.dx
        if term == "a":
            u = self.u
            a0 = inner(curl(u), curl(v))*dx(1)
            a1 = inner(curl(u), curl(v))*dx(2)
            a2 = inner(u, v)*dx
            return (a0, a1, a2)
        elif term == "f":
            f = self.f
            f0 = inner(f, v)*dx
            return (f0,)
        elif term == "dirichlet_bc":
            bc0 = [DirichletBC(self.V, Constant((0.0,0.0,0.0)), self.boundaries, 1),
                   DirichletBC(self.V, Constant((0.0,0.0,0.0)), self.boundaries, 2),
                   DirichletBC(self.V, Constant((0.0,0.0,0.0)), self.boundaries, 3)]
            return (bc0,)
        elif term == "inner_product":
            u = self.u
            x0 = inner(curl(u), curl(v))*dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")

# 1. Read the mesh for this problem
#    The files are in .xdmf extension

mesh = Mesh()
with XDMFFile("data/lcar=0.1/thermal_block.xdmf") as infile:
    infile.read(mesh)

subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
with XDMFFile("data/lcar=0.1/thermal_block_subdomains.xdmf") as infile:
    infile.read(subdomains)

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
with XDMFFile("data/lcar=0.1/thermal_block_boundaries.xdmf") as infile:
    infile.read(boundaries)

# 2. Create Finite Element space (Lagrange P1)
V = FunctionSpace(mesh, "N1curl", 1)

# 3. Allocate an object of the ThermalBlock class
electromagnetism_problem = Electromagnetism_Solution(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(0., 4.)] #[(0.0001, 1.0)]
electromagnetism_problem.set_mu_range(mu_range)

# 4. Prepare reduction with a POD-Galerkin method
pod_galerkin_method = PODGalerkin(electromagnetism_problem)
pod_galerkin_method.set_Nmax(15)
#pod_galerkin_method.set_tolerance(1e-2)

# 5. Perform the offline phase
pod_galerkin_method.initialize_training_set(50)
reduced_electromagnetism_problem = pod_galerkin_method.offline()

# 6. Perform an online solve
online_mu = (2.,)
reduced_electromagnetism_problem.set_mu(online_mu)
reduced_electromagnetism_problem.solve()
reduced_electromagnetism_problem.export_solution(filename="online_solution")

electromagnetism_problem.solve()
electromagnetism_problem.export_solution(filename="truth_solution")

reduced_electromagnetism_problem.export_error(filename="error")


# 7. Perform an error analysis
pod_galerkin_method.initialize_testing_set(50)
pod_galerkin_method.error_analysis()

# 8. Perform a speedup analysis
pod_galerkin_method.initialize_testing_set(50)
pod_galerkin_method.speedup_analysis()
