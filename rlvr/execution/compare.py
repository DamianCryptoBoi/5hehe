"""Structural, float-tolerant equality for comparing actual vs. expected.

Used inside the sandbox runner (so it must be self-contained / importable with
no project deps) and re-exported here for tests. Values arriving from the
sandbox are JSON round-tripped, so we treat list/tuple as equivalent sequences
and compare numbers with an absolute+relative tolerance of 1e-6.
"""

from __future__ import annotations

import math
from typing import Any

FLOAT_TOLERANCE = 1e-6


def values_equal(a: Any, b: Any, tol: float = FLOAT_TOLERANCE) -> bool:
    """Structural equality with float tolerance.

    - bool is compared strictly (True != 1) to avoid masking type bugs.
    - int/float compared with math.isclose(abs_tol=tol, rel_tol=tol); NaN==NaN.
    - lists/tuples compared element-wise (order-sensitive, length must match);
      list and tuple of equal contents are considered equal (JSON erases tuples).
    - dicts compared by key set then value-by-value.
    - sets compared as unordered collections.
    - everything else falls back to ``==``.
    """
    # Strict bool handling: don't let True == 1 sneak through. If either side is a
    # bool, both must be bools and equal.
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b

    # Integers must match EXACTLY: a float tolerance must never accept a wrong
    # integer answer (e.g. 1_000_000 vs 1_000_001 are ~1e-6 apart relatively).
    if isinstance(a, int) and isinstance(b, int):
        return a == b

    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        af, bf = float(a), float(b)
        if math.isnan(af) and math.isnan(bf):
            return True
        if math.isinf(af) or math.isinf(bf):
            return af == bf
        return math.isclose(af, bf, rel_tol=tol, abs_tol=tol)

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(values_equal(x, y, tol) for x, y in zip(a, b))

    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(values_equal(a[k], b[k], tol) for k in a)

    if isinstance(a, set) and isinstance(b, set):
        if len(a) != len(b):
            return False
        # order-insensitive match with tolerance
        remaining = list(b)
        for x in a:
            for i, y in enumerate(remaining):
                if values_equal(x, y, tol):
                    remaining.pop(i)
                    break
            else:
                return False
        return True

    return a == b
