from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from indeximpact import inspect


@pytest.mark.parametrize("join", ["inner", "outer", "left", "right", "exact"])
def test_two_membership_calls_per_source_on_shared_axis(join):
    a = xr.DataArray([1, 2, 3], dims="x", coords={"x": [3, 1, 2]})
    b = a if join == "exact" else a.sel(x=[1, 2])
    original = pd.Index.isin
    calls = []

    def count(index, values, level=None):
        calls.append((len(index), len(values)))
        return original(index, values, level=level)

    with patch.object(pd.Index, "isin", count):
        report = inspect({"a": a, "b": b}, join=join)
    assert len(calls) == 4
    aligned = xr.align(a, b, join=join)
    for name, actual in zip(("a", "b"), aligned):
        assert report["variables"][name]["<dataarray>"]["aligned_cells"] == actual.size


@pytest.mark.parametrize("sample_limit", [0, 1, 100])
@pytest.mark.parametrize("join", ["inner", "outer", "left", "right"])
def test_masks_keep_position_order_counts_and_truncation(join, sample_limit):
    a = xr.DataArray(np.ones(4), dims="x", coords={"x": [8, 2, 6, 4]})
    b = xr.DataArray(np.ones(3), dims="x", coords={"x": [4, 2, 9]})
    report = inspect({"a": a, "b": b}, join=join, sample_limit=sample_limit)
    aa, bb = xr.align(a, b, join=join)
    for key, before, after in (("a", a, aa), ("b", b, bb)):
        source = list(before.x.values)
        target = list(after.x.values)
        dropped = [i for i, value in enumerate(source) if value not in target]
        inserted = [i for i, value in enumerate(target) if value not in source]
        common = [value for value in source if value in target]
        target_common = [value for value in target if value in source]
        part = report["dimensions"]["x"]["sources"][key]
        assert part["dropped_positions"] == dropped[:sample_limit]
        assert part["inserted_target_positions"] == inserted[:sample_limit]
        assert part["dropped"] == len(dropped)
        assert part["inserted"] == len(inserted)
        assert part["retained"] == len(common)
        assert part["reordered"] == (common != target_common)
        assert part["positions_truncated"] == (
            len(dropped) > sample_limit or len(inserted) > sample_limit
        )


def test_refused_exact_and_unshared_dimensions_do_not_scan_membership():
    a = xr.DataArray([1, 2], dims="x", coords={"x": [1, 2]})
    b = xr.DataArray([1, 2], dims="x", coords={"x": [2, 3]})
    with patch.object(pd.Index, "isin", side_effect=AssertionError("unexpected scan")):
        assert inspect({"a": a, "b": b}, join="exact")["status"] == "REFUSED"
        result = inspect({"a": a, "b": b.rename(x="y")})
        assert result["status"] == "UNCHANGED"
