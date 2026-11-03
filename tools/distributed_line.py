"""A line's S, a field and a gradient across ranks: ``distributed_line.py direct``.

The script of the 'Run a distributed solve' recipe. It reads ``line.msh`` from the
folder it runs in and writes one ``.s2p`` and one ``.vtu`` there.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI
from scipy.constants import epsilon_0, mu_0

import gfem

GHz = 1e9
rank = MPI.COMM_WORLD.rank
linear_solver = sys.argv[1]  # "direct", or "bddc_gmres", which needs no PETSc

mesh = gfem.Mesh.from_gmsh("line.msh", length_unit="mm")  # rank 0 reads it once
z_line = float(np.sqrt(mu_0 / epsilon_0) * 1e-3 / 6e-3)  # 1 mm between 6 mm plates
line = gfem.Model(
    mesh,
    materials={"feed": gfem.Material(), "design": gfem.Material()},
    boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
    ports=[
        gfem.LumpedPort("p1", "port1", "+z", impedance=z_line),
        gfem.LumpedPort("p2", "port2", "+z", impedance=z_line),
    ],
)
options = gfem.SolverOptions(linear_solver=linear_solver)

problem = gfem.Driven((5 * GHz, 8 * GHz), keep_fields=[5 * GHz])
result = gfem.solve(line, problem, options=options)
result.e_field(5 * GHz, "p1").to_vtu(f"e-{linear_solver}.vtu")  # every rank calls this
if rank == 0:  # S is the same on every rank, so one rank writes it
    result.s.to_touchstone(f"line-{linear_solver}.s2p")


def transmission(eps_r):
    """|S21|^2 at 5 GHz, for a permittivity per element of the design volume."""
    model = line.with_material("design", gfem.MaterialField(eps_r=eps_r))
    s = gfem.solve(model, gfem.Driven(5 * GHz), options=options).s
    return jnp.abs(s["p2", "p1"][0]) ** 2


eps_r = jnp.full(len(mesh.elements("design")), 4.0)
power, gradient = jax.value_and_grad(transmission)(eps_r)
if rank == 0:
    print(f"ranks {MPI.COMM_WORLD.size}, {linear_solver}")
    print(f"S21 at 5 GHz, air: {complex(result.s['p2', 'p1'][0]):.12e}")
    print(f"|S21|^2, eps_r = 4: {float(power):.12f}")
    print(f"gradient norm: {float(jnp.linalg.norm(gradient)):.12e}")
