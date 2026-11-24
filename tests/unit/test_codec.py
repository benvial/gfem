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


def test_a_reader_of_its_own_declares_the_keys_it_takes():
    cls = dataclasses.make_dataclass("_TestOwnReader", ["a"])
    with pytest.raises(TypeError, match="names"):
        _codec.register(cls, from_args=lambda args, _: cls(args["a"]))
    assert cls not in _codec._BY_TYPE


def test_an_unknown_key_is_refused_without_a_file_name():
    cls = dataclasses.make_dataclass("_TestKeyed", ["a", ("b", int, 0)])
    _codec.register(cls)
    decoder = _codec.Decoder(CONTEXT, {})
    assert decoder.decode({"$type": "_TestKeyed", "a": 1}) == cls(1)
    with pytest.raises(ValueError, match=r"^_TestKeyed has no 'c' in this gfem"):
        decoder.decode({"$type": "_TestKeyed", "a": 1, "c": 2})


def test_lumped_values_round_trip_as_plain_numbers():
    import gfem.job  # noqa: F401 - registers the gfem classes
    from gfem import Impedance, LumpedPort

    port = LumpedPort("p1", "s", "+z", resistance=50.0, inductance=2e-9)
    sheet = Impedance("a", resistance=1e3, capacitance=1e-15)
    doc, arrays, back = _round_trip((port, sheet))
    assert arrays == {}
    assert doc[0]["inductance"] == 2e-9
    assert back == (port, sheet)


def test_an_eigenmode_result_keeps_its_wave_port_s_coupling_q_in_model_order():
    import gfem.job  # registers the gfem classes

    result = gfem.EigenmodeResult(
        omega=jnp.array([1e10 + 1e6j, 2e10 + 3e6j]),
        error=jnp.array([1e-12, 2e-12]),
        q_port={"wave": jnp.array([4e3, math.inf]), "lumped": jnp.array([5e3, 6e3])},
    )
    _, _, back = _round_trip(result)
    assert list(back.q_port) == ["wave", "lumped"]
    for name, values in result.q_port.items():
        np.testing.assert_array_equal(back.q_port[name], values)


def test_an_eigenmode_result_keeps_its_requested_power_share_and_loads_without_it():
    import gfem.job  # registers the gfem classes

    result = gfem.EigenmodeResult(
        omega=jnp.array([1e10 + 1e6j, 2e10 + 3e6j]),
        error=jnp.array([1e-12, 2e-12]),
        requested_power_share={"b": jnp.array([0.97, math.nan]), "a": jnp.ones(2)},
    )
    doc, arrays, back = _round_trip(result)
    assert list(back.requested_power_share) == ["b", "a"]
    for name, values in result.requested_power_share.items():
        np.testing.assert_array_equal(back.requested_power_share[name], values)
    # a file written before the field existed reads through its default
    del doc["requested_power_share"]
    old = _codec.Decoder(CONTEXT, arrays).decode(doc)
    assert old.requested_power_share == {}
