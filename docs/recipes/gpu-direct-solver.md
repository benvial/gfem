# Run the direct solver on an NVIDIA GPU

Run the `direct` linear solver's factorization and solves on an NVIDIA GPU,
with NVIDIA cuDSS. Only that step moves to the GPU. Assembly, the
`bddc_gmres` solver, the eigenvalue loop and the gradients stay on the CPU,
and the results are the same CPU arrays as from a CPU solve. The GPU solver
is experimental: whether it beats the CPU depends on the card's
double-precision speed, so measure before you rely on it.

## Install

{{ install_cuda }}

The extras exist on Linux x86_64 and Windows x86_64. On macOS they install
nothing, and a GPU request says that CUDA is not supported there.

## Solve on the GPU

```python
import gfem

device = "cuda" if gfem.cuda_available() else "cpu"
options = gfem.SolverOptions(linear_solver="direct", device=device)
result = gfem.solve(model, problem, options=options)
```

- `device="cuda"` is the first GPU, `"cuda:1"` the second.
- A GPU needs `linear_solver="direct"`; `SolverOptions` refuses any other
  solver with a GPU.
- There is no automatic choice: you pick the device, and a saved
  {term}`job` keeps it.
- {py:obj}`gfem.cuda_available` starts a CUDA context, so it is `False` on a machine
  whose GPU is missing or in an error state, even when the CUDA libraries
  import.
- Wave port modes are 2D problems and always run on the CPU. Inside a driven
  sweep with wave ports, only the 3D solve uses the GPU. A standalone
  `gfem.PortMode` solve with a `device` other than `"cpu"`, or any
  `linear_solver` but the default, raises `ValueError`: leave both at their
  defaults.

When the GPU cannot be used, the solve raises `gfem.DeviceUnavailableError`.
Its message gives the reason and the fix: install the extra, the CUDA
driver's error, or the number of devices. gfem never falls back to the CPU on its own. To rerun
a GPU job on a CPU machine without editing it:

```bash
python -m gfem run job.json --out results --device cpu
```

## Leave GPU memory for the solver when JAX also uses the GPU

gfem does not need JAX on the GPU. If you installed a CUDA build of JAX
anyway, JAX preallocates 75% of the GPU memory at its first operation, which
can leave too little for the factorization. gfem then warns once with
`gfem.DeviceMemoryWarning`. Set, before Python starts:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

or cap JAX with `XLA_PYTHON_CLIENT_MEM_FRACTION` (for example `.25`).

When the factorization would not fit in the free GPU memory, gfem keeps the
factors in CPU memory (cuDSS's hybrid memory mode): slower, but it runs.

## Check your results

- Solve once with `device="cpu"` and once with `device="cuda"`: S, or the
  eigenfrequencies, should agree to the solver tolerance.

How gfem itself is checked: {doc}`../verification`.

## Next

{py:obj}`gfem.SolverOptions`, {py:obj}`gfem.cuda_available`,
{py:obj}`gfem.DeviceUnavailableError`, {doc}`../capabilities`.

