# In this script we want to compute the L2 errors between the approximate solutions and the # "exact" solution, i.e. the solution obtained with the finest mesh possible.

from fenics import *
from mpi4py import MPI
import os

parameters['allow_extrapolation'] = True

# 1. Read the finest mesh possible for this problem
#    The files are in .xdmf extension
#    We have to change the directory as we want the dimension of the mesh: data/lcar=0.0125
mesh_e = Mesh()
with XDMFFile("data/lcar=0.0125/electro_block.xdmf") as infile:
    infile.read(mesh_e)

# 2. Define the Edge Elements function space for the "exact" solution
V_e = FunctionSpace(mesh_e, "N1curl", 1)

# Define the parameters and the meshes
meshes = [0.40,0.20,0.10,0.05,0.025]
k0s    = [1,0.1,0.01,0.001]

for meshSize in meshes:
    for k0val in k0s:
        mpiRank = MPI.COMM_WORLD.Get_rank()
        
        if mpiRank==0:
            print("#####################################")
            print("Errors for lcar=%.4f vs lcar=0.0125  -  k0=%.3f " % (meshSize,k0val))


        # 1. Read the mesh of the approximate solution for this problem
        #    The files are in .xdmf extension
        #    We have to change the directory for the dimension of the mesh: data/lcar=*
        mesh = Mesh()
        with XDMFFile("data/lcar=%.4f/electro_block.xdmf" % meshSize) as infile:
            infile.read(mesh)

        # 2. Define the Edge Elements function space for the "exact" solution
        V   = FunctionSpace(mesh, "N1curl", 1)


        # Read the solutions u and u_e in .xml extention
        file_e   = File("k0=%.3f/lcar=0.0125/Solution_Res.xml" % k0val) 
        file_a   = File("k0=%.3f/lcar=%.4f/Solution_Res.xml" % (k0val,meshSize))

        u_e = Function(V_e)
        u   = Function(V)

        file_e >> u_e
        file_a >> u

        # Interpolate u on the u_e function space
        u = interpolate(u, V_e)

        # Norms
        # Compute error in L2 norm
        error = inner((u_e - u),(u_e - u))*dx
        errL2 = abs(assemble(error))          

        # 4. Print errors
        print('error_L2 =', errL2)

        if mpiRank==0:
            print("########## DONE  #################")

