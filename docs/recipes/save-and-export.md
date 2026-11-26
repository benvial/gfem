# Result files and exports

Take a result out of Python: hand S to a circuit simulator as a Touchstone
file, label it for analysis with xarray, hand it to scikit-rf, open a field in ParaView, and write
the whole run as a {term}`job` file that solves again on another machine and
leaves a {term}`result bundle` to read back. The device is the line of
{doc}`solve-a-device`; every value is in SI units and every phase follows
$\exp(+i\omega t)$ ({doc}`../conventions`).

| To | Write | Read it with |
| --- | --- | --- |
| a circuit simulator | `s.to_touchstone("line.s2p")` | any Touchstone reader |
| an analysis in Python | `s.to_xarray()` | xarray |
| scikit-rf | `s.to_skrf(name="line")` | scikit-rf |
| a SAX circuit | `s.to_sdict(circuit="rf")` | SAX ({doc}`sax-circuit`) |
| ParaView or PyVista | `field.to_vtu("line_e.vtu")` | ParaView, meshio, PyVista |
| a run on another machine | `gfem.Job(...).to_json("job.json")` | `python -m gfem run`, then `gfem.load_result` |

## A result to save

Solve the line over a band and keep the electric field at 5 GHz.

```python
import json
import urllib.request
from pathlib import Path

import numpy as np

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

problem = gfem.Driven(frequency=np.linspace(2e9, 8e9, 7), keep_fields=5e9)
result = gfem.solve(model, problem)
s = result.s
```

`keep_fields=5e9` keeps the field at that frequency for every excited port;
without it a driven result holds S only. `keep_fields=True` keeps every
frequency, at the memory cost of one field per frequency and port.

## S to a Touchstone file

```python
s.to_touchstone("line.s2p")
s.to_touchstone("line50.s2p", z0=50.0)  # S referred to 50 ohm first
for name in ("line.s2p", "line50.s2p"):
    print(next(line for line in open(name) if line.startswith("#")).strip())

rows = np.loadtxt("line.s2p", comments=("!", "#"))  # Hz, then Re and Im of S
s_file = (rows[:, 1::2] + 1j * rows[:, 2::2]).reshape(-1, 2, 2).transpose(0, 2, 1)
print(abs(s_file - s.values).max() < 1e-15)  # the file holds S to the last digit
```

```text
# Hz S RI R 62.7883333333
# Hz S RI R 50
True
```

- The file is Touchstone 1.1: frequency in Hz, S as real and imaginary parts,
  rows in increasing frequency. The port names and the conventions are in its
  `!` comment lines.
- Without `z0` the file is labelled with the ports' own reference impedance,
  `s.z_ref`: 62.79 Ω here, the resistance of each lumped port. gfem never
  labels S with an impedance it was not computed for, so nothing is silently
  referred to 50 Ω.
- `z0=50.0` writes `s.renormalize(50.0)`: the same network, described by
  power waves referred to 50 Ω. Use it to compare with a measurement made at
  50 Ω, and whenever the ports' reference impedances differ.
- The name must end in `.s<n>p` for n ports, and every port must have been
  excited (`Driven(excite=None)`, the default).
- In Python, the same network is also `s.y()` (siemens) and `s.z()` (ohm), one
  matrix per frequency in port order, and `s.db("p2", "p1")` is |S21| in dB.

## S to xarray

`s.to_xarray()` labels S with its frequencies and port names, for selecting by
value and plotting. It needs the `xarray` extra (`pip install "gfem[xarray]"`).

```python
ds = s.to_xarray()
print(dict(ds.sizes))
print(ds.frequency.attrs["units"], ds.z_ref.attrs["units"])
print(abs(ds.s.sel(frequency=5e9, port="p2", excited="p1").item()))  # |S21| at 5 GHz

# netCDF has no complex numbers: store the real and imaginary parts.
parts = ds.assign(s_real=ds.s.real, s_imag=ds.s.imag).drop_vars("s")
parts.to_netcdf("line_s.nc")
```

```text
{'frequency': 7, 'port': 2, 'excited': 2}
Hz ohm
0.9013322305443373
```

- `s` has dimensions `frequency`, `port` (the wave leaving) and `excited` (the
  port fed), in the order of `s.values`; `z_ref` has `frequency` and `port`.
- {py:obj}`gfem.EigenmodeResult.to_xarray`,
  {py:obj}`gfem.ElectrostaticResult.to_xarray` and
  {py:obj}`gfem.PortModeResult.to_xarray` label the other problem types the
  same way.

## S to scikit-rf

`s.to_skrf()` gives a [scikit-rf](https://scikit-rf.org) network of the same
S on the ports' own reference impedances, for scikit-rf's cascades, plots and
conversions. It needs the `skrf` extra (`pip install "gfem[skrf]"`).

```python
nw = s.to_skrf(name="line")
print(nw.s_def, round(float(nw.z0[0, 0].real), 4))
nw.renormalize(50.0)  # scikit-rf's own renormalization, in place
print(abs(nw.s - np.asarray(s.renormalize(50.0).values)).max() < 1e-12)
```

```text
pseudo 62.7883
True
```

- The network holds pseudo-waves on a reference per port and frequency
  (`s_def="pseudo"`), so S and the references pass unchanged and its `z`, `y`
  and `renormalize` are exact. Frequencies are in increasing order.
- scikit-rf's ABCD matrix (`nw.a`) assumes power waves: renormalize to a real
  reference first.
- It is not differentiable: it copies S to NumPy. For a block of a SAX circuit
  use `s.to_sdict(circuit="rf")` ({doc}`sax-circuit`).

## Fields to ParaView or PyVista

A kept field is read by frequency and excited port. It writes to a VTU file
for ParaView, and gives values at points in metres.

```python
e = result.e_field(5e9, "p1")  # the field at 5 GHz, port 1 fed with 1 W
e.to_vtu("line_e.vtu")

grid = e.to_meshio()  # what the file holds
print(grid.points.min(axis=0), grid.points.max(axis=0))  # m
print(list(grid.point_data))
```

```text
[0. 0. 0.] [0.04  0.006 0.001]
['E_real', 'E_imag']
```

- Coordinates are in metres and values in SI units (V/m here), whatever unit
  the mesh file was in.
- A complex field is written as two arrays, its real and imaginary parts.
  The phasor is the peak value: at time $t$ the field is
  $\operatorname{Re}(\mathbf{E} e^{i\omega t})$ = `E_real` $\cos\omega t$ − `E_imag` $\sin\omega t$.
- Each tetrahedron of order $p$ is cut into $p^3$ smaller ones for the file, so
  the file has more cells than the mesh: 8 times more here.
- `e.to_pyvista()` returns the same grid as a PyVista mesh, for a plot in a
  notebook (the `viz` extra, `pip install "gfem[viz]"`).

Read values where you need them with `at`. Beyond the ceramic only the wave
sent on to port 2 travels, so the field there is the transmitted voltage over
the plate gap, $|S_{21}| \sqrt{2 P Z} / h$ with $P$ = 1 W:

```python
point = [35e-3, 3e-3, 0.5e-3]  # m: in the air between the ceramic and port 2
print(abs(e.at(point)[2]))  # |Ez|, V/m
print(abs(s["p2", "p1"][3]) * np.sqrt(2 * 1.0 * z_line) / 1e-3)  # by hand, V/m
print(np.linalg.norm(abs(e.b().at(point))) * 299792458.0)  # c |B|, V/m
```

```text
10100.420969107126
10100.418684690692
10101.755545950038
```

- `at` takes one point `(3,)` or many `(n, 3)`, and returns the components
  along x, y and z.
- `e.b()` is the magnetic flux density, in T, from the curl of E. It is one
  {term}`element order` lower than E, so c|B| agrees with |E| of this plane
  wave to about 1e-4 rather than to every digit.
- For the power through a surface, use {py:obj}`gfem.power_flow` rather than
  sampled fields ({doc}`measurements`).

## A run as a file

A {term}`job` is the model, the problem and the solver options in one
object. Its file is what you copy to a cluster or a cloud machine.

```python
job = gfem.Job(model, problem)
job.to_json("job.json")
print(json.loads(Path("job.json").read_text())["job"]["model"]["mesh"])
```

```text
{'$type': 'Mesh', 'path': 'line.msh', 'sha256': 'a08f23ededbdecfa1ef42d913a13ed5bddb4f0e96e7c87a7052b99b3294cef97', 'length_unit': 'mm'}
```

The job file does not hold the mesh: it refers to the mesh file by its path
and its sha256. The path is relative to the job file's directory when the
mesh lies inside it, as here, so copy the directory with `job.json` and
`line.msh` together. A mesh outside it is stored by its absolute path. A
mesh with no file (a refined one) is stored whole in `job.npz`, next to
`job.json`, which also holds every array of the job (a material field, moved
mesh nodes) and is written only when the job has arrays. Reading the job
reads the mesh again, and refuses a mesh file whose sha256 changed.

Solve the job from the command line, on this machine or another:

```bash
python -m gfem run job.json --out result
```

and read the result back in Python:

```python
saved = gfem.load_result("result")
print(sorted(p.name for p in Path("result").iterdir()))
print(abs(saved.s.values - s.values).max())  # the same solve, the same S
print(abs(saved.e_field(5e9, "p1").at(point)[2]))  # the kept field came back too

run = json.loads(Path("result/result.json").read_text())["run"]
print(sorted(run))
```

```text
['result.json', 'result.npz']
0.0
10100.420969107126
['device', 'device_override', 'options', 'ranks', 'wall_time_s']
```

- `result/` is a {term}`result bundle`: `result.json` holds the labels and
  `result.npz` the arrays. {py:obj}`gfem.load_result` takes the directory or
  its `result.json`, and returns the same type the solve returned.
- A kept field in the bundle refers to the mesh file the same way, by path
  and sha256: here by its absolute path, since the mesh is outside `result/`.
  Keep the mesh where it was, or read a bundle without fields.
- The `run` entry records how it ran: the device, the options, the number of
  ranks and the wall time in seconds.
- From Python, `gfem.Job.from_json("job.json").run(out="result")` does the
  same as the command line.
- `--log info` (or `debug`, `warning`, ...) shows gfem's log lines on stderr
  while the job runs, as `gfem.log.enable` does ({doc}`watch-a-solve`);
  without it the command prints nothing more.
- `--device cpu` (or `cuda`, `cuda:1`) overrides the job's device, so a job
  written for a GPU runs on a CPU machine, and the other way round
  ({doc}`gpu-direct-solver`). The bundle records the device used.
- The command exits with 0 on success, 1 when the job cannot be read or
  solved (the message is on stderr), and 2 for a command-line error such as a
  job file that does not exist.
- Under a launcher (`srun --mpi=pmix python -m gfem run ...`) the job solves across ranks
  and rank 0 writes one bundle, which reads on a single process
  ({doc}`distributed-solve`).

## Solve once, read it back

Re-running a notebook solves every cell again. Name a directory with
`cache=` and gfem keeps each finished result there, filed under the model,
the problem and the solver options; the same call again reads the result
back instead of solving. Nothing is kept unless a call names a directory.

```python
import logging
import sys

log = logging.getLogger("gfem.solve_cache")  # silent unless you turn it on
log.addHandler(logging.StreamHandler(sys.stdout))
log.setLevel(logging.INFO)

result = gfem.solve(model, problem, cache="solves")  # solves and stores
again = gfem.solve(model, problem, cache="solves")  # reads back, no solve
print(abs(again.s.values - result.s.values).max())
```

```text
miss 4c0b6f1e9d27: solved and stored (solves/4c0b6f1e9d27a8530f62c1d9b4e7a03f5c8d21e6b9f0a47c3d5e8b1f2a6c904e)
hit 4c0b6f1e9d27 (solves/4c0b6f1e9d27a8530f62c1d9b4e7a03f5c8d21e6b9f0a47c3d5e8b1f2a6c904e)
0.0
```

`gfem.log.enable()` shows these records too, as log lines on stderr beside
the rest of gfem's ({doc}`watch-a-solve`).

A job takes the same directory, from Python as `Job.run(out=..., cache=...)`
or from the command line:

```bash
python -m gfem run job.json --out cached --cache solves
```

```python
run = json.loads(Path("cached/result.json").read_text())["run"]
print(run["cache"]["status"], run["cache"]["key"])
```

```text
hit 4c0b6f1e9d27a8530f62c1d9b4e7a03f5c8d21e6b9f0a47c3d5e8b1f2a6c904e
```

- The entry is found by its solve key, a hash of the mesh (its nodes and
  elements, not its file), the materials, the problem, the solver options
  other than `threads`, `frequency_workers`, `retain_memory` and `device`,
  gfem's version and code, and the versions of NGSolve and, when the
  options select them, PETSc or SLEPc. Change any of them and the call
  solves and adds an entry. A new gfem version, or an edit to gfem's code,
  never reads an older one's entries.
- A result read back has the same type and the same values as the solve
  that stored it, with its fields on your `model.mesh`, and the solve's
  warnings are shown again at your line. A wrong argument still raises,
  cache or not.
- The stored result may have been solved on another machine, device,
  thread count or number of ranks. It then agrees with a solve here to the
  precision gfem documents, not to every digit. A `device="cuda"` call that
  reads its result back needs no GPU; one that solves does.
- The logger above prints one `hit` or `miss` line per call, and a line
  when old entries are deleted. A job's bundle is written either way, and
  its `run` entry gains a `cache` item: the status, the key, and on a hit
  `solved`, the device, ranks and wall time of the run that stored the
  entry ({py:obj}`gfem.Job.run`). The bundle's own `wall_time_s` is this
  call's.
- Some calls solve and store nothing, even with `cache=`: a solve under
  `jax.grad`, `jax.jit` or `jax.vmap` (gradients), a solve that raises (a
  sweep that stopped), and one with an input the cache cannot file, such
  as a subclass of a gfem class or a {py:obj}`gfem.IncidentField`, whose
  function is code, not data ({py:obj}`gfem.SolveCacheWarning` names it;
  {py:obj}`gfem.PlaneWave` and {py:obj}`gfem.GaussianBeam` are filed). {py:obj}`gfem.refine_mesh` never uses a cache.
- The directory is created when it does not exist. A directory that exists
  and is not a solve cache is refused before solving.
- Once the entries pass 10 GiB in total, the least recently used ones are
  deleted. To change that limit, edit `max_bytes` (in bytes) in
  `solves/solve-cache.json`. A result larger than the limit is not stored.
- To solve again, drop `cache=` for that call or delete its entry,
  `solves/<key>/`. To empty the cache, delete the directory.
  {py:obj}`gfem.release_memory` frees memory and leaves the directory alone.
- On a cluster, put the directory on a shared filesystem, never on a
  node's local disk. Under a launcher, rank 0 reads and writes it for
  every rank. A solve that keeps fields on more than one rank is not
  cached: write those with `Job.run(out=)`, or solve with
  `comm=MPI.COMM_SELF` ({doc}`distributed-solve`).

## When a sweep stops

An iterative linear solve that does not converge at one frequency of a
uniform sweep raises {py:obj}`gfem.LinearSolverConvergenceError`, after
solving the frequencies before it. Run as a job, what was solved is not
lost: it is written as a {term}`partial bundle`, `result/partial.json` and
`result/partial.npz`, never as `result.json`. The command line exits with 1
and prints a second line, `gfem: note: ...`, that says where it is.

- Read it by name: `gfem.load_result("result/partial.json")` returns a
  {py:obj}`gfem.DrivenResult` over the solved frequencies only.
  `gfem.load_result("result")` refuses it, and its message points to the file.
- Its `run` entry also holds `unsolved`, the frequencies with no result, in
  Hz, and `error`, the solver's message. Solve those again with
  `gfem.SolverOptions(linear_solver="direct")`.
- A `result.json` from an earlier run in the same directory is left alone,
  and the note says so. A later run that completes removes the partial
  bundle.

## Check your results

- Read back what you wrote: `gfem.load_result` gives the S of the solve, and
  `np.loadtxt` the S of a Touchstone file, as above.
- To trust a result read from a solve cache, solve once without `cache=`
  and compare its S with the one read back.
- Open the `.s2p` file in your circuit simulator and compare one value with
  `s["p2", "p1"]` at the same frequency, with the reference impedance on the
  file's `#` line.
- Open `line_e.vtu` in ParaView and probe a point with a value you know, as
  the hand value above.

## Limits

- A Touchstone 1.1 file has one real reference impedance for every port.
  When the ports' `s.z_ref` differ, or one is complex, pass `z0=`.
  A lumped port without a resistance cannot be written: its row of S is NaN.
  When no port has a reference impedance (every port a
  {py:obj}`gfem.WavePort` without a `voltage_path`, as in a photonic
  network), `z0=` is required and is a nominal label only: S is written
  unchanged, normalized to unit modal power, and a comment line in the file
  says so. A network that mixes such ports with ports that have a reference
  impedance cannot be written.
- A result holds the fields `keep_fields` asked for, and only those; a field
  you did not keep needs a new solve. A solve cache does not change that:
  it reads back the fields the stored solve kept.
- The solve cache files a result under the inputs of the solve, as a job
  file records them. An input it cannot record, such as a subclass of a
  gfem class, is never guessed at: that call solves without the cache
  ({py:obj}`gfem.SolveCacheWarning`).
- Plane waves, Gaussian beams and point dipoles are written with the
  problem and the result. A {py:obj}`gfem.IncidentField` is not: its
  function is code, not data, and a file without it would describe another
  problem, so saving a result or a job that holds one raises `TypeError`
  naming the source and its function. Save what you read from the result
  instead (S, cross sections, fields to VTU), or describe the field as a
  `PlaneWave` or `GaussianBeam` when it is one ({doc}`incident-fields`).
- Across ranks, write S and the job from rank 0, and call a field's
  `to_vtu` on every rank ({doc}`distributed-solve`, "Write files").
- A job or result file records the gfem version that wrote it. gfem reads
  files of an older format and refuses a newer one, and refuses by name any
  entry it does not know.

## Next

- {doc}`distributed-solve`: run a job across the cores of a cluster.
- {doc}`gpu-direct-solver`: run the same job on a GPU.
- {doc}`measurements`: values at points, the power through a surface, losses.
- {doc}`rational-model`: S as a SPICE subcircuit or a Verilog-A model.
- {py:obj}`gfem.Job`, {py:obj}`gfem.NetworkParameters` and
  {py:obj}`gfem.Field` in the API reference.
