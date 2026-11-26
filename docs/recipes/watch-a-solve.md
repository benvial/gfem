# Solver progress and logs

See what gfem does while it solves: which solver each frequency got, whether
a factorization was reused, how long each phase took, and how far a long
sweep has gone. gfem reports all of it as standard `logging` records on
loggers under `gfem` and prints nothing by default. One call turns on the
{term}`log display`: one line per record on stderr, and a {term}`progress row`
for each long loop while it runs.

The display is optional and changes no result: S, fields and gradients are
the same with it on or off. It colours its lines with rich, which the `log`
extra installs:

{{ install_log }}

Without rich it prints the same lines in plain text, with ANSI colours on a
terminal.

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

- The lines go to stderr, so they never mix with what your script prints on
  stdout.
- `gfem.log.enable("DEBUG")` shows more, `gfem.log.enable("WARNING")` only
  warnings and errors. The level is a name in any case, or an int such as
  `logging.DEBUG`.
- Calling `enable` again replaces the earlier settings; it never adds a
  second display.
- A job run from the command line takes the same switch:
  `python -m gfem run job.json --out result --log info`.

## Read a log line

Each line holds, in order:

1. the wall-clock time the record was made, to the millisecond;
2. the level: `DEBUG`, `INFO`, `WARN`, `ERROR` or `CRIT`;
3. the logger's short name: `solver` for `"gfem.solver"`, `timing` for
   `"gfem.timing"`, `cache` for `"gfem.solve_cache"`, `mpi`, `mesh`,
   `mesh_refinement`;
4. under MPI, the rank tag `rN/M` ({ref}`watch-under-mpi`);
5. the record's message, as gfem wrote it.

A `solver` line of a driven solve names the frequency, the direction
(`forward`, or `backward` for the {term}`adjoint solve` of a gradient), the
linear solver and what it did, with the reason in brackets:

- `factorized`: a new factorization or preconditioner, here for the first
  frequency;
- `refactorized`: the matrix changed (another frequency, another design), so
  the solver factorized again;
- `reused`: the factorization or preconditioner of an earlier solve served
  again, as in a design loop ({doc}`s-parameter-gradient`).

`iterations` lists the Krylov iterations per excited port; it is empty for a
direct solve. On a colour terminal the solver and its action stand out:
`factorized` and `refactorized` in magenta, `reused` in green.

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

`gfem.log.enable("WARNING", timing=True)` shows the phases and the warnings
only. To add the seconds up per phase rather than read them line by line,
attach a handler of your own, as in {doc}`solve-a-device` ("Time each phase").

## Progress rows

While a long loop runs, the display draws one {term}`progress row` for it
below the log lines: its name, a bar, the units done out of the total
(`k/?` when the total is not known in advance), the phase it is in
(`assemble`, `setup`, `solve`), the latest values of the loop (the frequency
in GHz, the iterations, the error estimate) and the elapsed time. The rows are:

| Row | One unit is | Total |
| --- | --- | --- |
| `uniform sweep` | one frequency of a {py:obj}`gfem.Driven` sweep | the frequencies |
| `uniform sweep (adjoint)` | one frequency of the adjoint solves of a gradient | the frequencies |
| `adaptive sweep` | one {term}`sample frequency` | unknown |
| `port mode sweep` | the frequencies settled so far | the output frequencies |
| `mode search` | a seed solve or a mode refined | unknown |
| `mesh refinement` | one refinement cycle | unknown |

- A phase that runs outside these loops (assembling a single solve, say) gets
  a spinner row of its own with its elapsed time.
- A loop inside another gets its own row under its parent's. A fixed-angle
  sweep's one-frequency solves advance its one `uniform sweep` row.
- With {term}`frequency workers <frequency worker>`, each frequency a worker
  finishes advances the one row of your process.
- A row disappears when its loop ends, also when it stops on an error: only
  the log lines stay.

Rows are drawn only where they can be erased and redrawn: in a terminal, and
in Jupyter when `ipywidgets` is installed (there each finished loop leaves an
empty output line). In a pipe, a file or a batch job's output file you get
the log lines and no rows; nothing is printed in their place. On any process
but rank 0 of an MPI job there are no rows either.

(watch-under-mpi)=
## Under MPI

Launched on several ranks, the display shows rank 0's lines only, each
tagged with the rank and the number of ranks, `r0/4`. Warnings and errors
show from every rank, since a rank-local warning is often the one that
matters. To read every rank's lines, tagged `r0/4` to `r3/4`:

```python
gfem.log.enable(rank0_only=False)
```

gfem reads the rank from the launcher's environment, so `enable` works before
anything imports mpi4py. The {doc}`distributed solve recipe <distributed-solve>` covers the launch itself.

## With your own handlers

The records and their `extra` attributes (`frequency`, `solver`, `action`,
`phase`, `seconds`, ...) are the same with the display on or off: a handler of
your own on a `gfem` logger reads them either way, and you may skip `enable`
altogether. While the display is on, the `gfem` logger stops propagating, so
a root handler (`logging.basicConfig`) does not print gfem's records a second
time.

If you keep a stream handler of your own that writes to the same terminal,
its lines tear through the rows; turn the rows off:

```python
gfem.log.enable(progress=False)
```

`rich=False` gives the plain-text lines even where rich is installed, and
`rich=True` raises `ImportError` where it is not. Colour follows the usual
switches: `NO_COLOR` turns it off, `FORCE_COLOR` on.

## Turn it off

```python
gfem.log.disable()
```

`disable` removes the display and gives the `gfem` logger back the level and
propagation it had before the first `enable`, so a level you set on `gfem`
while the display was on is overwritten. With the display off, `disable`
does nothing.

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
