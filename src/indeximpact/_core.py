"""Coordinate-only alignment impact report, not a data validator."""

from functools import reduce
from math import prod

import numpy as np
import pandas as pd
import xarray as xr
from xarray.indexes import PandasIndex


class Unsupported(ValueError):
    pass


def _index(obj, dim):
    ix = obj.xindexes.get(dim)
    if type(ix) is not PandasIndex or ix.dim != dim:
        raise Unsupported(
            "only same-name one-dimensional built-in PandasIndex supported"
        )
    p = ix.index
    if type(p) not in (pd.Index, pd.RangeIndex, pd.DatetimeIndex, pd.TimedeltaIndex):
        raise Unsupported("custom, multi, categorical and interval indexes unsupported")
    plain_strings = p.dtype.kind == "O" and all(type(value) is str for value in p)
    if getattr(p, "tz", None) is not None or (
        p.dtype.kind not in "biufMmTU"
        and not isinstance(p.dtype, pd.StringDtype)
        and not plain_strings
    ):
        raise Unsupported(
            "only plain finite numeric, boolean, string or timezone-naive time indexes supported"
        )
    if not p.is_unique or p.hasnans:
        raise Unsupported("coordinate indexes must be unique and nonmissing")
    if p.dtype.kind in "f" and not np.isfinite(p.to_numpy()).all():
        raise Unsupported("coordinate indexes must be finite")
    return p


def inspect(objects, *, join="inner", sample_limit=10):
    """Report supported index geometry without computing or changing data.

    ``join`` follows native xr.align semantics within the documented scope.
    The default inner join only describes potential changes; it never aligns
    inputs. Prefer native join="exact" when a mismatch should stop your work.
    Non-string/empty DataArray names use the literal placeholder <dataarray>.
    Raises Unsupported for unsupported shared indexes or invalid arguments.
    """
    if (
        not isinstance(objects, dict)
        or len(objects) < 2
        or any(not isinstance(k, str) or not k for k in objects)
    ):
        raise Unsupported("two or more explicitly named inputs required")
    if join not in ("inner", "outer", "left", "right", "exact"):
        raise Unsupported("unsupported join; override is deliberately not modeled")
    if type(sample_limit) is not int or not 0 <= sample_limit <= 100:
        raise Unsupported("sample limit must be integer from0to100")
    if any(not isinstance(o, (xr.DataArray, xr.Dataset)) for o in objects.values()):
        raise Unsupported("xarray DataArray or Dataset inputs required")
    if any(
        any(not isinstance(d, str) or not d for d in o.dims) for o in objects.values()
    ):
        raise Unsupported("nonempty string dimension names required")
    for o in objects.values():
        if any(
            type(ix) is not PandasIndex or name != ix.dim
            for name, ix in o.xindexes.items()
        ):
            raise Unsupported(
                "only one same-name built-in PandasIndex per dimension is supported"
            )
    dimensions = {}
    target_sizes = {k: dict(o.sizes) for k, o in objects.items()}
    retained = {k: dict(o.sizes) for k, o in objects.items()}
    findings = []
    for dim in sorted(set().union(*(o.dims for o in objects.values()))):
        members = {k: o for k, o in objects.items() if dim in o.dims}
        if len(members) < 2:
            dimensions[dim] = {
                "scope": "unshared, unchanged by alignment",
                "sources": {k: int(o.sizes[dim]) for k, o in members.items()},
            }
            continue
        indexes = {k: _index(o, dim) for k, o in members.items()}
        first = next(iter(indexes.values()))
        if any(p.dtype != first.dtype for p in indexes.values()):
            raise Unsupported("mixed coordinate dtypes are not modeled")
        equal = all(first.equals(p) for p in indexes.values())
        if join == "exact" and not equal:
            findings.append({"code": "EXACT_INDEX_MISMATCH", "dimension": dim})
            dimensions[dim] = {
                "scope": "exact join refuses alignment",
                "sources": {k: len(p) for k, p in indexes.items()},
            }
            continue
        if equal or join in ("left", "exact"):
            target = first
        elif join == "right":
            target = list(indexes.values())[-1]
        else:
            target = reduce(
                lambda a, b: a.union(b) if join == "outer" else a.intersection(b),
                indexes.values(),
            )
        parts = {}
        for k, p in indexes.items():
            lost = np.flatnonzero(~p.isin(target))
            added = np.flatnonzero(~target.isin(p))
            common = p[p.isin(target)]
            target_common = target[target.isin(p)]
            reordered = not common.equals(target_common)
            parts[k] = {
                "size": len(p),
                "retained": len(common),
                "dropped": len(lost),
                "inserted": len(added),
                "reordered": reordered,
                "dropped_positions": lost[:sample_limit].tolist(),
                "inserted_target_positions": added[:sample_limit].tolist(),
                "positions_truncated": len(lost) > sample_limit
                or len(added) > sample_limit,
            }
            target_sizes[k][dim] = len(target)
            retained[k][dim] = len(common)
            for cond, code in (
                (len(lost), "LABELS_DROPPED"),
                (len(added), "LABELS_INSERTED"),
                (reordered, "LABELS_REORDERED"),
            ):
                if cond:
                    findings.append({"code": code, "input": k, "dimension": dim})
        dimensions[dim] = {
            "scope": "shared supported index",
            "target_size": len(target),
            "sources": parts,
        }
    refused = join == "exact" and bool(findings)
    variables = {}
    for k, o in objects.items():
        vs = (
            {
                o.name
                if isinstance(o.name, str) and o.name
                else "<dataarray>": o.variable
            }
            if isinstance(o, xr.DataArray)
            else o.data_vars
        )
        if any(not isinstance(n, str) for n in vs):
            raise Unsupported("string data variable names required")
        variables[k] = {}
        for name, v in vs.items():
            before = prod(v.sizes.values())
            keep = prod(retained[k][d] for d in v.dims)
            after = prod(target_sizes[k][d] for d in v.dims)
            variables[k][name] = {
                "source_cells": before,
                "aligned_cells": None if refused else after,
                "retained_source_cells": None if refused else keep,
                "dropped_source_cells": None if refused else before - keep,
                "new_fill_positions": None if refused else after - keep,
            }
    return {
        "status": "REFUSED" if refused else "CHANGED" if findings else "UNCHANGED",
        "join": join,
        "dimensions": dimensions,
        "variables": variables,
        "findings": findings,
        "limits": [
            "Index alignment only, not arithmetic/broadcast/merge result prediction.",
            "Cell counts describe geometric index positions, not valid or nonmissing measurements.",
            "Non-index coordinates, variable overlap, attributes, units, calendars and data values are not checked.",
            "Positions are sampled, labels are not included; dimension, variable and input names remain visible.",
            "Built-in native join=exact is the simpler guard when no diagnostic report is needed.",
        ],
    }
