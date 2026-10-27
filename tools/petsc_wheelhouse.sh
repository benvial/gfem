#!/usr/bin/env bash
# Build the PETSc stack that gfem's PETSc and SLEPc solvers use: PETSc and
# SLEPc with complex scalars and without MPI, with their Python bindings
# petsc4py and slepc4py, as a wheelhouse (a folder of four wheels) to
# install into the environment that runs gfem.
#
# Usage:
#   bash petsc_wheelhouse.sh OUTPUT_DIR   build the four wheels into OUTPUT_DIR
#                                         (a new or empty folder)
#   bash petsc_wheelhouse.sh --check      check the prerequisites only
# then, in the environment that runs gfem:
#   pip install --no-index --find-links OUTPUT_DIR petsc petsc4py slepc slepc4py
#
# Prerequisites: Linux (macOS is not tested), uv, curl, make, cmake, and the
# C, C++ and Fortran compilers gcc, g++ and gfortran. The script downloads
# the four source distributions from PyPI, and PETSc's configure downloads
# MUMPS, METIS and OpenBLAS; all of them are compiled from source. It is a
# long build, done once per Python version.
#
# Target Python: the bindings are compiled for one CPython version (3.12 or
# newer), so build with the Python of the environment that runs gfem. The
# script uses the Python that `uv python find` gives: the active virtual
# environment's, or the .venv of the project in the current folder, if any.
# Set PYTHON to choose another, as a path or a version, for example
# PYTHON=/path/to/venv/bin/python or PYTHON=3.12.
#
# MAKE_NP sets the number of parallel compile jobs (default: PETSc's own
# choice from the number of cores).
set -euo pipefail

# The release gfem's PETSc tests run on: gfem needs this minor or newer.
PETSC_VERSION=3.26.0
SLEPC_VERSION=3.26.0
# sha256 of each source distribution on PyPI
SHA256_petsc=f5230023e6e22ee607802a13c82bc25f3d81e71654ad386a5b9bdff17fed93df
SHA256_petsc4py=0a34bd083e24bee1a6759c4639a45d10c0117855eb9f0466834450245f75d7e5
SHA256_slepc=a2f4cc2af76d55c078c30ad8bc66b44736dac921a7912266eb44136fc1b6029d
SHA256_slepc4py=07bf970c861736b9b977dfecd4ad46415d6c4a6f26b0ed128ecffae59613d6ff

CONFIGURE_OPTIONS=(
  --with-scalar-type=complex  # gfem's operators are complex
  --with-mpi=0                # one process: MUMPS on its serial libseq, no ParMETIS
  --download-mumps            # the sparse direct solver (CeCILL-C)
  --download-metis            # its ordering (Apache-2.0)
  --download-openblas         # BLAS and LAPACK (BSD-3-Clause)
  --download-openblas-use-pthreads  # threaded: OPENBLAS_NUM_THREADS sets the threads
  --with-x=0                  # no X11 graphics
)

usage() {
  echo "usage: bash petsc_wheelhouse.sh OUTPUT_DIR | --check" >&2
  exit 2
}

fail() {
  echo "petsc_wheelhouse.sh: $*" >&2
  exit 1
}

# Check every prerequisite, report all that are missing, and set PY to the
# target Python.
check_prerequisites() {
  local missing=() tool
  for tool in uv curl make cmake gcc g++ gfortran; do
    command -v "$tool" > /dev/null || missing+=("$tool")
  done
  [ "$(uname -s)" = Linux ] || echo "warning: only Linux is tested, not $(uname -s)" >&2
  ((${#missing[@]} == 0)) || fail "missing prerequisites: ${missing[*]} (install them and run again)"
  PY=$(UV_NO_CONFIG=1 uv python find ${PYTHON:+"$PYTHON"}) ||
    fail "no Python found for PYTHON='${PYTHON:-}' (set PYTHON to the Python of the environment that runs gfem)"
  "$PY" -c 'import sys; sys.exit(sys.implementation.name != "cpython" or sys.version_info < (3, 12))' ||
    fail "$PY is not CPython 3.12 or newer, which gfem needs"
  echo "compilers: $(gcc -dumpfullversion -dumpversion) (gcc), $(gfortran -dumpfullversion -dumpversion) (gfortran)"
  echo "target Python: $PY ($("$PY" -c 'import sys; print(sys.version.split()[0])'))"
}

# Download one source distribution from PyPI and check its sha256.
fetch() {
  local name=$1 version=$2 sha256=$3
  local sdist=$WORK/$name-$version.tar.gz
  curl -fsSL --retry 3 -o "$sdist" \
    "https://files.pythonhosted.org/packages/source/${name:0:1}/$name/$name-$version.tar.gz"
  "$PY" - "$sdist" "$sha256" << 'EOF'
import hashlib, sys
path, expected = sys.argv[1:]
with open(path, "rb") as f:
    digest = hashlib.sha256(f.read()).hexdigest()
sys.exit(f"{path}: sha256 {digest}, expected {expected}" if digest != expected else 0)
EOF
}

# Build one wheel into the wheelhouse. The wheelhouse itself provides the
# petsc and slepc a binding builds against; no cache, so a wheel built
# with other configure options is never reused.
build() {
  local name=$1 version=$2 start=$SECONDS
  echo "=== building $name $version"
  uv build --no-cache --wheel --python "$PY" --out-dir "$OUT" --find-links "$OUT" \
    --build-constraints "$WORK/constraints.txt" "$WORK/$name-$version.tar.gz"
  echo "=== $name built in $((SECONDS - start)) s"
}

# The petsc wheel folds the static MUMPS, PORD and libseq libraries into
# libpetsc.so and then deletes them, but its petscvariables, petsc.pc and
# petscmachineinfo.h still list them, so SLEPc fails to link against it
# ("Unable to link with PETSc"). Remove those flags from the wheel's text
# files (never from a binary) and update their hashes in its RECORD.
patch_petsc_wheel() {
  "$PY" - "$OUT"/petsc-"$PETSC_VERSION"-*.whl << 'EOF'
import base64, csv, hashlib, io, os, re, sys, zipfile

wheel = sys.argv[1]
flags = re.compile(rb" ?-l(?:smumps|dmumps|cmumps|zmumps|mumps_common|pord|mpiseq)\b")
with zipfile.ZipFile(wheel) as old:
    (record,) = [n for n in old.namelist() if n.endswith(".dist-info/RECORD")]
    changed = {}
    for name in old.namelist():
        data = old.read(name)
        if name != record and b"\0" not in data and flags.search(data):
            changed[name] = flags.sub(b"", data)
            print(f"patched {name}")
    rows = list(csv.reader(io.StringIO(old.read(record).decode())))
    for row in rows:
        if row[0] in changed:
            digest = base64.urlsafe_b64encode(hashlib.sha256(changed[row[0]]).digest())
            row[1:] = ["sha256=" + digest.rstrip(b"=").decode(), str(len(changed[row[0]]))]
    text = io.StringIO()
    csv.writer(text, lineterminator="\n").writerows(rows)
    changed[record] = text.getvalue().encode()
    patched = wheel + ".patched"
    with zipfile.ZipFile(patched, "w") as new:
        for info in old.infolist():  # same names, order, permissions, compression
            data = changed[info.filename] if info.filename in changed else old.read(info)
            new.writestr(info, data)
os.replace(patched, wheel)
EOF
}

case ${1:-} in
--check)
  (($# == 1)) || usage
  check_prerequisites
  echo "all prerequisites found"
  exit 0
  ;;
"" | -*) usage ;;
esac
(($# == 1)) || usage
check_prerequisites

mkdir -p "$1"
OUT=$(cd "$1" && pwd)
[ -z "$(ls -A "$OUT")" ] || fail "$OUT is not empty: give a new or empty folder"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# A PETSC_DIR / PETSC_ARCH of another PETSc would leak into the build.
unset PETSC_DIR PETSC_ARCH SLEPC_DIR
export UV_NO_CONFIG=1
export PETSC_CONFIGURE_OPTIONS="${CONFIGURE_OPTIONS[*]}${MAKE_NP:+ --with-make-np=$MAKE_NP}"
printf '%s\n' "petsc==$PETSC_VERSION" "petsc4py==$PETSC_VERSION" \
  "slepc==$SLEPC_VERSION" "slepc4py==$SLEPC_VERSION" > "$WORK/constraints.txt"

fetch petsc "$PETSC_VERSION" "$SHA256_petsc"
fetch petsc4py "$PETSC_VERSION" "$SHA256_petsc4py"
fetch slepc "$SLEPC_VERSION" "$SHA256_slepc"
fetch slepc4py "$SLEPC_VERSION" "$SHA256_slepc4py"

build petsc "$PETSC_VERSION"
patch_petsc_wheel
build petsc4py "$PETSC_VERSION"
build slepc "$SLEPC_VERSION"
build slepc4py "$SLEPC_VERSION"

echo "wheelhouse: $OUT"
ls "$OUT"
echo "install: pip install --no-index --find-links $OUT petsc petsc4py slepc slepc4py"
