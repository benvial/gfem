"""Pytree registration for gfem's frozen physics values."""

from __future__ import annotations

import dataclasses

import jax


def register_fields[T](cls: type[T], children: tuple[str, ...]) -> type[T]:
    """Register a frozen dataclass as a pytree.

    The ``children`` fields are subtrees (array leaves, ``None`` for absent
    values, nested gfem pytrees), keyed by attribute name; every other field
    is static and must be hashable. Unflattening bypasses ``__init__``, so
    validation runs once, when the user builds the object, never on the
    placeholder leaves JAX passes around.
    """
    names = tuple(f.name for f in dataclasses.fields(cls))  # ty: ignore[invalid-argument-type]
    statics = tuple(n for n in names if n not in children)

    def flatten_with_keys(obj):
        kids = tuple((jax.tree_util.GetAttrKey(n), getattr(obj, n)) for n in children)
        return kids, tuple(getattr(obj, n) for n in statics)

    def flatten(obj):
        kids = tuple(getattr(obj, n) for n in children)
        return kids, tuple(getattr(obj, n) for n in statics)

    def unflatten(aux, kids):
        obj = object.__new__(cls)
        for n, v in zip(children, kids, strict=True):
            object.__setattr__(obj, n, v)
        for n, v in zip(statics, aux, strict=True):
            object.__setattr__(obj, n, v)
        return obj

    jax.tree_util.register_pytree_with_keys(cls, flatten_with_keys, unflatten, flatten)
    return cls
