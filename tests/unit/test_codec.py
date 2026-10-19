"""The generic JSON + .npz codec behind job.json / result.json."""

import dataclasses
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gfem import _codec


@dataclasses.dataclass(frozen=True)
class Toy:
    values: jax.Array
    table: dict
    label: str = "x"
    extra: float = 1.0


_codec.register(Toy, tag="_TestToy")
CONTEXT = _codec.Context(Path())


def _round_trip(value):
    enc = _codec.Encoder(CONTEXT)
    doc = json.loads(json.dumps(enc.encode(value), allow_nan=False))
    return doc, enc.arrays, _codec.Decoder(CONTEXT, enc.arrays).decode(doc)


def test_a_registered_dataclass_round_trips_with_arrays_by_path():
    toy = Toy(
        values=jnp.arange(3.0) + 1j,
        table={"a": np.ones(2), "$odd": (math.nan, -math.inf, 2 + 3j, None, True)},
        label="y",
    )
    doc, arrays, back = _round_trip(toy)
    assert set(arrays) == {".values", ".table['a']"}
    assert doc["$type"] == "_TestToy"
    assert isinstance(back.values, jax.Array)
    assert back.values.dtype == jnp.complex128
    np.testing.assert_array_equal(back.values, toy.values)
    assert isinstance(back.table["a"], np.ndarray)
    nan, inf, z, none, true = back.table["$odd"]
    assert math.isnan(nan)
    assert inf == -math.inf
    assert (z, none, true) == (2 + 3j, None, True)
    assert back.label == "y"


def test_arguments_missing_from_an_older_file_take_their_defaults():
    doc, arrays, _ = _round_trip(Toy(values=jnp.ones(1), table={}, extra=5.0))
    del doc["extra"]
    assert _codec.Decoder(CONTEXT, arrays).decode(doc).extra == 1.0


def test_unregistered_values_are_refused():
    with pytest.raises(TypeError, match="object is not a serializable"):
        _codec.Encoder(CONTEXT).encode(object())


def test_tags_are_unique():
    with pytest.raises(ValueError, match="already taken"):
        _codec.register(dataclasses.make_dataclass("Other", ["a"]), tag="_TestToy")
