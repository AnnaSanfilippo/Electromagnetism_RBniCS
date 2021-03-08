# We try to solve the elliptic problem curl(k curl(u)) + u = f in [-1,1]^3 (we have two subdomains: a sphere centered in the origin of radius 0.5 and [-1,1]^3) with BCs u x n = 0 on all the boundaries, where k = k_0 in [10^(-3),1] (initially we used [10^(-2),10^2], but the values grater than 1 are unuseful) in the sphere and k = k_1 = 1 in the rest of the domain. We use the same f of Permeability.py.
# We do not know the exact solution, so we want to compute an approximate solution with a very fine mesh and use it as exact solution in order to have the L2 errors. In this script we do not compute the errors, but only the solutions.

from fenics import *
from mpi4py import MPI
import os

# Define the parameters and the meshes
meshes = [0.05,0.025,0.0125] #meshes = [0.40,0.20,0.10]
k0s    = [1,0.1,0.01,0.001]  #k0s = [1000,100,10]

for meshSize in meshes:
    for k0val in k0s:
        #assert MPI.COMM_WORLD.Get_size() > 1
        mpiRank = MPI.COMM_WORLD.Get_rank()
        
        if mpiRank==0:
            print("#####################################")
            print("Solving: lcar=%.4f   -  k0=%d " % (meshSize,k0val))

        # 1. Read the mesh for this problem
        #    The files are in .xdmf extension
        #    We have to change the directory for the dimension of the mesh: data/lcar=*
        mesh = Mesh()
        with XDMFFile("data/lcar=%.4f/electro_block.xdmf" % meshSize) as infile:
            infile.read(mesh)
        
        #    The sphere label is 1, the box without the sphere label is 2
        subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
        with XDMFFile("data/lcar=%.4f/electro_block_subdomains.xdmf" % meshSize) as infile:
            infile.read(subdomains)
        
        boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        with XDMFFile("data/lcar=%.4f/electro_block_boundaries.xdmf" % meshSize) as infile:
            infile.read(boundaries)
        
        # 2. Define the Edge Elements function space
        V = FunctionSpace(mesh, "N1curl", 1)
        
        # Find the dimension of the function space
        print(V.dim())
        
        # Compute maximum cell size in mesh, measured greatest distance between any two vertices of a cell 
        hmax = mesh.hmax()
        if mpiRank==0:
            print('hmax = ',hmax)
        
        # Compute minimum cell size in mesh, measured greatest distance between any two vertices of a cell.
        hmin = mesh.hmin()
        if mpiRank==0:
            print('hmin = ',hmin)
        
        # Compute average cell size in mesh, measured greatest distance between any two vertices of a cell.
        total_h = []
        for cell in cells(mesh):
            total_h.append(cell.h())
        if mpiRank==0:
            print('hmean = ',sum(total_h)/len(total_h))
        
        # Print the maximum and the minimum cell sizes from the list
        if mpiRank==0:
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
        
        # Define input data
        x, y, z = SpatialCoordinate(mesh)
        
        k0 = Constant(k0val)
        k1 = Constant(1.0)
        
        def k(x,y,z):
            if(x**2+y**2+z**2<=0.25):
                return(k0)
            else:
                return(k1)
        
        # Define variational problem
        u = TrialFunction(V)
        v = TestFunction(V)
        f = as_vector(((z**2-1)*(2*x*y-x**2+1)-(y**2-1)*(x**2-1-2*x*z)+0.5*(x**2-1)*(y**2-1)*(z**2-1),
                        (x**2-1)*(2*y*z-y**2+1)-(z**2-1)*(y**2-1-2*x*y)+0.5*(x**2-1)*(y**2-1)*(z**2-1),
                        (y**2-1)*(2*x*z+1-z**2)-(x**2-1)*(z**2-1-2*y*z)+0.5*(x**2-1)*(y**2-1)*(z**2-1)))
        a = (inner(k(0.1,0.1,0.1)*curl(u), curl(v)) + inner(u, v))*dx(1) + (k(0.9,0.9,0.9)*inner(curl(u), curl(v)) + inner(u, v))*dx(2)
        L = inner(f, v)*dx(1) + inner(f, v)*dx(2)
        
        # Compute the solution
        u = Function(V)
        solve(a == L, u, bc, solver_parameters={'linear_solver': 'gmres',
                         'preconditioner': 'amg'})

        # Save the solution
        path = ("k0=%d" % k0val)
        path += ('/lcar=%.4f' % meshSize) 
        if mpiRank==0:
            print("Solution written to ",path)
            if not os.path.exists(path):
                os.makedirs(path)
        sol = XDMFFile(path+"/Solution_Res.xdmf")
        u.rename("Sol_N1curl","Sol_N1curl")
        sol.write(u)
        if mpiRank==0:
            print("########## DONE  #################")

