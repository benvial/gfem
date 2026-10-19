# Run the direct solver on an NVIDIA GPU

gfem can factorize and solve with NVIDIA cuDSS instead of NGSolve's sparse
Cholesky. Only the direct solve moves to the GPU: assembly, BDDC, the
eigenvalue loop and the gradients stay on the CPU, and every array you get
back is a host array. The GPU solver is experimental: whether it beats the CPU
depends on the card's double-precision speed, so measure before you rely on
it.

## Install

```bash
pip install "gfem[cuda12]"  # CUDA 12 drivers
pip install "gfem[cuda13]"  # CUDA 13 drivers (check `nvidia-smi`)
```

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
- There is no automatic choice: a job records the device it ran on.
- `gfem.cuda_available()` starts a CUDA context, so it is `False` on a machine
  whose GPU is missing or in an error state, even when the CUDA libraries
  import.

When the GPU cannot be used, the solve raises `gfem.DeviceUnavailableError`
with the reason and the fix (install the extra, the CUDA driver's error, or
the number of devices). gfem never falls back to the CPU on its own. To rerun
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
factors in host memory (cuDSS's hybrid memory mode): slower, but it runs.
