# We try to solve the elliptic problem curl(curl(u)) + k u = f in [-1,1]^3 (we have two subdomains: a sphere centered in the origin of radius 0.5 and [-1,1]^3) with BCs u x n = 0 on all the boundaries, where k = k_0 in [10^(-3),10^3] in the sphere and k = k_1 = 1 in the rest of the domain.
# We also know the exact solution, i.e. u_e=((x^2-1)(y^2-1)(z^2-1),(x^2-1)(y^2-1)(z^2-1),(x^2-1)(y^2-1)(z^2-1)), so we can compute f and as a consequence the absolute and relative Hcurl and the L2 errors.

from fenics import *

# Create mesh and define function space

# 1. Read the mesh for this problem
#    The files are in .xdmf extension
#    We have to change the directory as we want the dimension of the mesh: data/lcar=*
mesh = Mesh()
with XDMFFile("data/lcar=0.1/electro_block.xdmf") as infile:
    infile.read(mesh)

#    The sphere label is 1, the box without the sphere label is 2
subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
with XDMFFile("data/lcar=0.1/electro_block_subdomains.xdmf") as infile:
    infile.read(subdomains)

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
with XDMFFile("data/lcar=0.1/electro_block_boundaries.xdmf") as infile:
    infile.read(boundaries)

# 2. Define the Edge Elements function space
V = FunctionSpace(mesh, "N1curl", 1)

# Find the dimension of the function space
print(V.dim())

# Compute maximum cell size in mesh, measured greatest distance between any two vertices of a cell 
hmax = mesh.hmax()
print('hmax = ',hmax)

# Compute minimum cell size in mesh, measured greatest distance between any two vertices of a cell.
hmin = mesh.hmin()
print('hmin = ',hmin)

# Compute average cell size in mesh, measured greatest distance between any two vertices of a cell.
total_h = []
for cell in cells(mesh):
    total_h.append(cell.h())
print('hmean = ',sum(total_h)/len(total_h))

# Print the maximum and the minimum cell sizes from the list
print('hmax_m = ',max(total_h))
print('hmin_m = ',min(total_h))

# Define boundary conditions (we have to ways to implement them)
# 1. Dirichlet boundary

#def DirichletBoundary(x, on_boundary):
#    return on_boundary

# 2. Boundary conditions

bc1 = DirichletBC(V, Constant((0,0,0)), boundaries, 1)
bc2 = DirichletBC(V, Constant((0,0,0)), boundaries, 2)
bc3 = DirichletBC(V, Constant((0,0,0)), boundaries, 3)

bc = [bc1, bc2, bc3]

# Define the different subdomains integration zones
dx = Measure("dx", domain=mesh, subdomain_data=subdomains)

# Define the exact solution
x, y, z = SpatialCoordinate(mesh)

u_e = as_vector((0.5*(x**2-1)*(y**2-1)*(z**2-1),0.5*(x**2-1)*(y**2-1)*(z**2-1),0.5*(x**2-1)*(y**2-1)*(z**2-1)))

# Define input data
k0 = Constant(1000) # It can be changed from 0.001 to 1000
k1 = Constant(1.0)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f0 = as_vector(((z**2-1)*(2*x*y-x**2+1)-(y**2-1)*(x**2-1-2*x*z)+0.5*k0*(x**2-1)*(y**2-1)*(z**2-1),
                (x**2-1)*(2*y*z-y**2+1)-(z**2-1)*(y**2-1-2*x*y)+0.5*k0*(x**2-1)*(y**2-1)*(z**2-1),
                (y**2-1)*(2*x*z+1-z**2)-(x**2-1)*(z**2-1-2*y*z)+0.5*k0*(x**2-1)*(y**2-1)*(z**2-1)))
f1 = as_vector(((z**2-1)*(2*x*y-x**2+1)-(y**2-1)*(x**2-1-2*x*z)+0.5*k1*(x**2-1)*(y**2-1)*(z**2-1),
                (x**2-1)*(2*y*z-y**2+1)-(z**2-1)*(y**2-1-2*x*y)+0.5*k1*(x**2-1)*(y**2-1)*(z**2-1),
                (y**2-1)*(2*x*z+1-z**2)-(x**2-1)*(z**2-1-2*y*z)+0.5*k1*(x**2-1)*(y**2-1)*(z**2-1)))
a = (inner(curl(u), curl(v)) + k0*inner(u, v))*dx(1) + (inner(curl(u), curl(v)) + k1*inner(u, v))*dx(2)
L = inner(f0, v)*dx(1) + inner(f1, v)*dx(2)

# Compute the solution
u = Function(V)
solve(a == L, u, bc, solver_parameters={'linear_solver':'mumps'})

# Save the solution
sol = XDMFFile("Solution_Perm.xdmf")
u.rename("Sol_N1curl","Sol_N1curl")
sol.write(u)

# Save the exact solution
Pu_e= project(u_e, V)
sol_e = XDMFFile("Exact_Solution_Perm.xdmf")
Pu_e.rename("Sol_E_N1curl","Sol_E_N1curl")
sol_e.write(Pu_e)

## Norms
# 1. Compute error in L2 norm
error = inner((u_e - u),(u_e - u))*dx
errL2 = sqrt(abs(assemble(error)))

# 2. Compute the absolute error in H(curl) norm
error = inner(curl(u_e - u),curl(u_e - u))*dx
errHC = sqrt(errL2**2 + abs(assemble(error)))

# 3. Compute the relative error in H(curl) norm
err     = abs(assemble(inner(u_e,u_e)*dx))
error   = inner(curl(u_e),curl(u_e))*dx
error1  = sqrt(err + abs(assemble(error)))
errHC_r = errHC/error1

# 4. Print errors
print('error_L2 =', errL2)
print('error_H(curl) =', errHC)
print('relative error_H(curl) =', errHC_r)
