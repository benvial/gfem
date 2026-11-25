"""Frequencies in messages and reprs, in the unit that fits them."""

__all__ = ["complex_hertz", "hertz", "hertz_band"]

_UNITS = ((1e12, "THz"), (1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz"))


def _unit(f: float) -> tuple[float, str]:
    """The scale and unit that put ``|f|`` in [1, 1000), Hz below 1 kHz."""
    for scale, unit in _UNITS:
        if abs(f) >= scale:
            return scale, unit
    return 1.0, "Hz"


def hertz(f: float, digits: int = 6) -> str:
    """``f`` (Hz) as ``"50 GHz"``, with ``digits`` significant digits."""
    scale, unit = _unit(f)
    return f"{f / scale:.{digits}g} {unit}"


def complex_hertz(f: complex, digits: int = 6) -> str:
    """A complex frequency ``f`` (Hz) as ``"7.3 GHz + i 200 MHz"``."""
    sign = "-" if f.imag < 0 else "+"
    return f"{hertz(f.real, digits)} {sign} i {hertz(abs(f.imag), digits)}"


def hertz_band(lo: float, hi: float, digits: int = 6) -> str:
    """A band ``lo``-``hi`` (Hz) as ``"55-65 GHz"``, in the unit of ``hi``.

    A band of one frequency (``lo == hi``) is that frequency, `hertz`.
    """
    if lo == hi:
        return hertz(hi, digits)
    scale, unit = _unit(hi)
    return f"{lo / scale:.{digits}g}-{hi / scale:.{digits}g} {unit}"
