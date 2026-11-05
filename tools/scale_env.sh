#!/bin/bash
# On the login node: the environment of one frozen commit, for tools/scale_job.sbatch.
#
#   scale_env.sh SCALE_DIR BASE_VENV WHEEL COMMIT PETSC_WHEELHOUSE [WHEEL_DIR ...]
#
# SCALE_DIR       is made: env/, COMMIT, manifest.txt, records/, logs/steps/, steps/, the scripts.
# BASE_VENV       a venv with gfem's dependencies; hard-linked, so nothing in it is copied.
# WHEEL           gfem's wheel, built from COMMIT.
# COMMIT          the full SHA, read by every record as SCALE_COMMIT.
# PETSC_WHEELHOUSE what tools/petsc_wheelhouse.sh built against this machine's MPI.
# WHEEL_DIR       directories of further wheels, mpi4py among them: a compute cluster has no index.
set -eu
DIR=${1:?scale_env.sh SCALE_DIR BASE_VENV WHEEL COMMIT PETSC_WHEELHOUSE [WHEEL_DIR ...]}
BASE=${2:?base venv} WHEEL=${3:?wheel} COMMIT=${4:?commit} PETSC=${5:?petsc wheelhouse}
shift 5
HERE=$(cd "$(dirname "$0")" && pwd)
E=$DIR/env
mkdir -p "$DIR/logs/steps" "$DIR/records" "$DIR/steps"
echo "$COMMIT" > "$DIR/COMMIT"
rm -rf "$E"
cp -al "$BASE" "$E"
LINKS=(--find-links "$PETSC")
for w in "$@"; do LINKS+=(--find-links "$w"); done
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1
"$E/bin/python" -m pip uninstall -y -q gfem
"$E/bin/python" -m pip install -q --no-index "${LINKS[@]}" "$WHEEL"
"$E/bin/python" -m pip install -q --no-index "${LINKS[@]}" "$PETSC"/*.whl
"$E/bin/python" -m pip install -q --no-index "${LINKS[@]}" mpi4py==4.1.2
cp "$HERE"/scale_*.py "$HERE/scale_job.sbatch" "$HERE/split_mesh.py" "$DIR/"
{
  echo "commit $COMMIT"
  sha256sum "$WHEEL"
  sha256sum "$PETSC"/*.whl
  "$E/bin/python" -m pip freeze
} > "$DIR/manifest.txt"
grep -iE "^(gfem|ngsolve|petsc|slepc|mpi4py|jax==|numpy)" "$DIR/manifest.txt"
