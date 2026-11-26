# Solver progress and logs

See what gfem does while it solves: which solver each frequency got, whether
a factorization was reused, how long each phase took, and how far a long
sweep has gone. gfem writes all of it as standard `logging` records under
`gfem` and prints nothing by default. One
call turns on the {term}`log display`: a line per record on stderr and a
{term}`progress row` for each long loop. The display changes no result. It
colours with rich, which the `log` extra installs; without rich the lines are
plain text:

{{ install_log }}

## Turn it on

Solve the line of {doc}`../get-started` at three frequencies with the display on:

```python
import urllib.request

import gfem

url = "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/data/line.msh"
urllib.request.urlretrieve(url, "line.msh")
mesh = gfem.Mesh.from_gmsh("line.msh", length_unit="mm")

z_line = 376.73 / 6  # ohm: the 6 mm wide, 1 mm high parallel-plate line in air
plates = [gfem.PEC("bottom"), gfem.PEC("top")]
ports = [
    gfem.LumpedPort("p1", "port1", "+z", impedance=z_line),
    gfem.LumpedPort("p2", "port2", "+z", impedance=z_line),
]
ceramic = {"feed": gfem.Material(), "design": gfem.Material(eps_r=10.0)}
model = gfem.Model(mesh, materials=ceramic, boundaries=plates, ports=ports)

gfem.log.enable()
result = gfem.solve(model, gfem.Driven(frequency=(2e9, 5e9, 8e9), excite="p1"))
```

```text
14:02:29.512 INFO  solver linear_solver='auto' chose 'direct' for this driven problem: 24,074 unknowns, est. 1.1 GiB factorization <= 8.0 GiB budget (half of 16 GiB)
14:02:29.512 INFO  solver Driven: tree-cotree gauge on
14:02:29.936 INFO  solver Tree-cotree gauge basis: 14130 -> 14130 free unknowns, 2 conductors (1 floating), template 390852 entries (ungauged 362628), basis 0.024 s, template 0.312 s
14:02:30.744 INFO  solver driven forward solve at 2 GHz: direct factorized (first), iterations []
14:02:31.082 INFO  solver driven forward solve at 5 GHz: direct refactorized (first), iterations []
14:02:31.220 INFO  solver driven forward solve at 8 GHz: direct refactorized (other frequency), iterations []
```

- The lines go to stderr, never into what your script prints.
- `gfem.log.enable("DEBUG")` shows more, `"WARNING"` only warnings and errors.
  A second `enable` replaces the first settings.
- A command-line job takes the same switch:
  `python -m gfem run job.json --out result --log info`.

## Read a log line

A line holds the time to the millisecond, the level (`DEBUG`, `INFO`, `WARN`,
`ERROR`, `CRIT`), the logger's short name (`solver`, `timing`, `cache`, `mpi`,
`mesh`, `mesh_refinement`), under MPI the rank tag `rN/M`
({ref}`watch-under-mpi`), and the message.

A `solver` line of a driven solve names the frequency, the direction
(`forward`, or `backward` for the {term}`adjoint solve` of a gradient), the
linear solver and what it did, with the reason in brackets:

- `factorized`: a new factorization or preconditioner.
- `refactorized`: the matrix changed (another frequency or design), so the
  solver factorized again.
- `reused`: an earlier factorization or preconditioner served again, as in a
  design loop ({doc}`s-parameter-gradient`).

`iterations` lists the Krylov iterations per excited port; it is empty for a
direct solve.

## How long each phase took

`timing=True` adds one `timing` line per phase of each solve, whatever level
you asked for:

```python
gfem.log.enable(timing=True)
again = gfem.solve(model, gfem.Driven(frequency=5e9, excite="p1"))
```

```text
14:02:32.295 INFO  solver linear_solver='auto' chose 'direct' for this driven problem: 24,074 unknowns, est. 1.1 GiB factorization <= 8.0 GiB budget (half of 16 GiB)
14:02:32.297 INFO  solver Driven: tree-cotree gauge on
14:02:32.462 INFO  timing driven discretize: 0.159 s
14:02:33.370 INFO  timing driven assemble at 5 GHz: 0.871 s
14:02:33.832 INFO  timing driven setup at 5 GHz: 0.469 s
14:02:33.852 INFO  timing driven solve at 5 GHz: 0.0196 s
14:02:33.853 INFO  solver driven forward solve at 5 GHz: direct factorized (first), iterations []
```

To add the seconds up per phase, attach a handler of your own, as in
{doc}`solve-a-device` ("Time each phase").

## Progress rows

A long loop draws one {term}`progress row` below the log lines: its name, a
bar, the units done out of the total (`k/?` when unknown), the phase
(`assemble`, `setup`, `solve`), the latest values (frequency in GHz,
iterations, error estimate) and the elapsed time. The rows are:

| Row | One unit is | Total |
| --- | --- | --- |
| `uniform sweep` | one frequency of a {py:obj}`gfem.Driven` sweep | the frequencies |
| `uniform sweep (adjoint)` | one frequency of the adjoint solves of a gradient | the frequencies |
| `adaptive sweep` | one {term}`sample frequency` | unknown |
| `port mode sweep` | the frequencies settled so far | the output frequencies |
| `mode search` | a seed solve or a mode refined | unknown |
| `mesh refinement` | one refinement cycle | unknown |

- A phase outside these loops gets a spinner row of its own.
- A loop inside another gets its own row under its parent's.
- With {term}`frequency workers <frequency worker>`, each frequency a worker
  finishes advances the one row of your process.
- A row disappears when its loop ends, also on an error; the log lines stay.

Rows are drawn where they can be redrawn: in a terminal, and in Jupyter with
`ipywidgets` installed. In a pipe, a file or a batch job's output you get the
log lines and no rows, and none on a rank other than 0.

(watch-under-mpi)=
## Under MPI

On several ranks the display shows rank 0's lines, tagged with the rank and
the number of ranks, `r0/4`. Warnings and errors show from every rank. To read
every rank's lines:

```python
gfem.log.enable(rank0_only=False)
```

`enable` works before anything imports mpi4py. The launch itself is in
{doc}`distributed-solve`.

## With your own handlers

A handler of your own on a `gfem` logger reads the records and their `extra`
attributes (`frequency`, `solver`, `action`, `phase`, `seconds`, ...) with the
display on or off, so you may skip `enable`. While the display is on, the
`gfem` logger stops propagating, so a root handler does not print a record
twice. A stream handler on the same terminal tears through the rows; turn the
rows off:

```python
gfem.log.enable(progress=False)
```

`rich=False` gives plain text even where rich is installed. `NO_COLOR` turns
colour off, `FORCE_COLOR` on.

## Turn it off

```python
gfem.log.disable()
```

`disable` removes the display and restores the level and propagation the
`gfem` logger had before the first `enable`, overwriting a level you set while
it was on.

## Limits

- With frequency workers, the workers' log lines arrive together when the
  sweep ends; their row advances as they go.
- Rows are never written to a file. A batch job's output file holds the log
  lines only.
- The thermal transient's time loop has no row.
- In Jupyter without rich, the plain-text lines are not coloured: a kernel's
  stderr is not a terminal.

## Next

- {py:obj}`gfem.log.enable`, {py:obj}`gfem.log.disable`: every argument.
- {doc}`solve-a-device`: the phase records and their fields.
- {doc}`s-parameter-gradient`: what a design loop reuses between designs.
- {doc}`distributed-solve`: running on several ranks.
- {doc}`save-and-export`: the command line `python -m gfem run`.
