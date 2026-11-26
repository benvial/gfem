# Result files and exports

Take a result out of Python: hand S to a circuit simulator as a Touchstone
file, label it for analysis with xarray, hand it to scikit-rf, open a field in
ParaView, and write the whole run as a {term}`job` file that solves again on
another machine and leaves a {term}`result bundle` to read back. The device is
the line of {doc}`solve-a-device`.

| To | Write |
| --- | --- |
| a circuit simulator | `s.to_touchstone("line.s2p")` |
| xarray, scikit-rf | `s.to_xarray()`, `s.to_skrf(name="line")` |
| ParaView or PyVista | `field.to_vtu("line_e.vtu")` |
| another machine | `gfem.Job(...).to_json("job.json")`, then `python -m gfem run` |

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
without it a result holds S only. `keep_fields=True` keeps every frequency.

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

- The file is Touchstone 1.1, frequency in Hz, S as real and imaginary parts;
  the name ends in `.s<n>p`, and every port must have been excited.
- Without `z0` it is labelled with the ports' own reference, `s.z_ref`: 62.79 Ω
  here, never silently 50 Ω. `z0=50.0` writes `s.renormalize(50.0)`, the same
  network referred to 50 Ω.

## S to xarray

`s.to_xarray()` labels S with its frequencies and port names (the `xarray`
extra, `pip install "gfem[xarray]"`).

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

- `s` has dimensions `frequency`, `port` (the wave leaving) and `excited`.

## S to scikit-rf

`s.to_skrf()` gives a [scikit-rf](https://scikit-rf.org) network of the same S
on the ports' own references (the `skrf` extra, `pip install "gfem[skrf]"`).

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

- The network holds pseudo-waves (`s_def="pseudo"`), so S and the references
  pass unchanged; its ABCD matrix (`nw.a`) assumes power waves, so renormalize
  to a real reference first. It is not differentiable.

## Fields to ParaView or PyVista

A kept field is read by frequency and excited port, and written to a VTU file.

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

- Coordinates are in metres, values in SI units (V/m). A complex field is
  written as real and imaginary parts of the peak phasor: at time $t$ it is
  `E_real` $\cos\omega t$ − `E_imag` $\sin\omega t$.
- Each order-$p$ tetrahedron is cut into $p^3$ cells: 8 times more here.

Read values with `at`. Beyond the ceramic the field is $|S_{21}| \sqrt{2 P Z} / h$
with $P$ = 1 W:

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

- `at` takes one point `(3,)` or many `(n, 3)`. `e.b()` is the magnetic flux
  density in T, one {term}`element order` lower than E, so c|B| agrees with |E|
  here to about 1e-4. For power through a surface use
  {py:obj}`gfem.power_flow`.

## A run as a file

A {term}`job` is the model, problem and solver options in one file to copy to
a cluster.

```python
job = gfem.Job(model, problem)
job.to_json("job.json")
print(json.loads(Path("job.json").read_text())["job"]["model"]["mesh"])
```

```text
{'$type': 'Mesh', 'path': 'line.msh', 'sha256': 'a08f23ededbdecfa1ef42d913a13ed5bddb4f0e96e7c87a7052b99b3294cef97', 'length_unit': 'mm'}
```

The job refers to the mesh file by path (relative when the mesh lies in its
directory: copy the two together) and sha256; a changed mesh file is refused.
A mesh with no file, and the job's arrays, go in `job.npz`.

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
  `result.npz` the arrays; {py:obj}`gfem.load_result` takes the directory and
  returns the type the solve returned. A kept field refers to the mesh by path
  and sha256 (here its absolute path, since the mesh is outside `result/`).
- `gfem.Job.from_json("job.json").run(out="result")` does the same from Python.
  `--device cpu` or `cuda` overrides the job's device
  ({doc}`gpu-direct-solver`). The command exits with 0 on success, 1 when the
  job cannot be read or solved, 2 for a command-line error.

## Solve once, read it back

Name a directory with `cache=` and gfem keeps each finished result there; the
same call again reads it back instead of solving. Nothing is kept unless a call
names a directory.

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

`gfem.log.enable()` shows a `hit` or `miss` line per call. A job takes the same
directory, as `Job.run(out=..., cache=...)` or on the command line:

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

- The entry is found by a hash of the mesh, materials, problem, solver options
  (not `threads`, `frequency_workers`, `retain_memory` or `device`) and the
  versions of gfem and its solvers: change any and the call solves again. A
  result read back has the same type and values, with its fields on your
  `model.mesh` ({py:obj}`gfem.Job.run`).
- A solve under `jax.grad`, `jax.jit` or `jax.vmap`, one that raises, or one
  with an input the cache cannot file (a {py:obj}`gfem.IncidentField`) stores
  nothing ({py:obj}`gfem.SolveCacheWarning`).
- Past 10 GiB the least recently used entries go (`max_bytes` in
  `solves/solve-cache.json`); to solve again, drop `cache=`. On a cluster use a shared filesystem
  ({doc}`distributed-solve`).

## When a sweep stops

An iterative solve that does not converge at one frequency of a uniform sweep
raises {py:obj}`gfem.LinearSolverConvergenceError`. Run as a job, what was
solved is written as a {term}`partial bundle`, `result/partial.json`, never
`result.json`; the command exits with 1 and a `gfem: note: ...` line says
where.

- `gfem.load_result("result/partial.json")` returns a {py:obj}`gfem.DrivenResult`
  over the solved frequencies; `gfem.load_result("result")` refuses it. Its
  `run` entry holds `unsolved` (Hz): solve those with
  `gfem.SolverOptions(linear_solver="direct")`.

## Check your results

- `gfem.load_result` and `np.loadtxt` give the solve's S back, as above.
- Compare one value of the `.s2p` file in your circuit simulator with
  `s["p2", "p1"]`, at the reference on its `#` line.

## Limits

- A Touchstone 1.1 file has one real reference per port: when `s.z_ref` differ
  or one is complex, pass `z0=`. With no port reference (wave ports without a
  `voltage_path`), `z0=` is a nominal label; a mix cannot be written.
- A {py:obj}`gfem.IncidentField` is not saved (its function is code) and
  raises `TypeError`; use a `PlaneWave` or `GaussianBeam`
  ({doc}`incident-fields`).
- A result holds only the fields `keep_fields` asked for.

## Next

- {doc}`distributed-solve`, {doc}`gpu-direct-solver`, {doc}`measurements`,
  {doc}`rational-model`.
- {py:obj}`gfem.Job`, {py:obj}`gfem.NetworkParameters` and
  {py:obj}`gfem.Field` in the API reference.
