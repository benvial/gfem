"""Skip the GPU tier when no CUDA device is usable (see `gfem.cuda_available`)."""

import pytest

import gfem


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip every tests/cuda test on a machine without a usable GPU."""
    if not any(item.get_closest_marker("cuda") for item in items):
        return
    if gfem.cuda_available():
        return
    skip = pytest.mark.skip(
        reason="no usable CUDA device (gfem.cuda_available() is False): needs an "
        "NVIDIA GPU and the cuda12 or cuda13 extra"
    )
    for item in items:
        if item.get_closest_marker("cuda"):
            item.add_marker(skip)
