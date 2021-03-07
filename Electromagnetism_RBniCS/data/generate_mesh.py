# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
import dolfin
import pygmsh
import meshio

# Define 3D geometry
lcar = 0.0125
geom = pygmsh.built_in.Geometry()
sphere = geom.add_ball([0, 0, 0], 0.5, lcar=lcar)
box = geom.add_box(-1, 1, -1, 1, -1, 1, lcar=lcar, holes=[sphere.surface_loop])
geom.add_physical(box.surface_loop.surfaces[0], label=1)
geom.add_physical([box.surface_loop.surfaces[i] for i in range(1, 5)], label=2)
geom.add_physical(box.surface_loop.surfaces[5], label=3)  
geom.add_physical(sphere.volume, label=11)  
geom.add_physical(box.volume, label=12)    
pygmsh_mesh = pygmsh.generate_mesh(geom)

# Save mesh in pygmsh format
meshio.write("electro_block.xdmf", meshio.Mesh(
    points=pygmsh_mesh.points,
    cells={"tetra": pygmsh_mesh.cells_dict["tetra"]}))

# Read back in mesh in dolfin format
mesh = dolfin.Mesh()
with dolfin.XDMFFile("electro_block.xdmf") as infile:
    infile.read(mesh)

# Save subdomains in pygmsh format
meshio.write("electro_block_subdomains_tmp.xdmf", meshio.Mesh(
    points=pygmsh_mesh.points,
    cells={"tetra": pygmsh_mesh.cells_dict["tetra"]},
    cell_data={"subdomains": [pygmsh_mesh.cell_data_dict["gmsh:physical"]["tetra"] - 10]}
))

# Read back in subdomains in dolfin format
subdomains_mvc = dolfin.MeshValueCollection("size_t", mesh, mesh.topology().dim())
with dolfin.XDMFFile("electro_block_subdomains_tmp.xdmf") as infile:
    infile.read(subdomains_mvc, "subdomains")
subdomains = dolfin.cpp.mesh.MeshFunctionSizet(mesh, subdomains_mvc)
with dolfin.XDMFFile("electro_block_subdomains.xdmf") as outfile:
    outfile.write(subdomains)
os.remove("electro_block_subdomains_tmp.xdmf")
os.remove("electro_block_subdomains_tmp.h5")

# Save boundaries in pygmsh format
meshio.write("electro_block_boundaries_tmp.xdmf", meshio.Mesh(
    points=pygmsh_mesh.points,
    cells={"triangle": pygmsh_mesh.cells_dict["triangle"]},
    cell_data={"boundaries": [pygmsh_mesh.cell_data_dict["gmsh:physical"]["triangle"]]}
))

# Read back in boundaries in dolfin format
boundaries_mvc = dolfin.MeshValueCollection("size_t", mesh, mesh.topology().dim() - 1)
with dolfin.XDMFFile("electro_block_boundaries_tmp.xdmf") as infile:
    infile.read(boundaries_mvc, "boundaries")
boundaries_mvc_dict = boundaries_mvc.values()
for c in dolfin.cells(mesh):
    for f, _ in enumerate(dolfin.facets(c)):
        if (c.index(), f) not in boundaries_mvc_dict:
            boundaries_mvc.set_value(c.index(), f, 0)
boundaries = dolfin.cpp.mesh.MeshFunctionSizet(mesh, boundaries_mvc)
with dolfin.XDMFFile("electro_block_boundaries.xdmf") as outfile:
    outfile.write(boundaries)
os.remove("electro_block_boundaries_tmp.xdmf")
os.remove("electro_block_boundaries_tmp.h5")
