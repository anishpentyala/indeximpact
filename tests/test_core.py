import numpy as np
import pandas as pd
import pytest
import xarray as xr

from indeximpact import Unsupported, inspect


def arr(labels, dims=("x",)):
    return xr.DataArray(np.arange(len(labels)), dims=dims, coords={dims[0]: labels})


@pytest.mark.parametrize("join", ["inner", "outer", "left", "right"])
@pytest.mark.parametrize(
    "left,right",
    [
        ([1, 2, 3], [2, 3, 4]),
        ([3, 2, 1], [1, 2, 3]),
        ([1, 2], []),
        ([], []),
        ([1, 2], [7, 8]),
    ],
)
def test_oracle_shapes_and_fill_geometry(join, left, right):
    a, b = arr(np.array(left, dtype="int64")), arr(np.array(right, dtype="int64"))
    report = inspect({"a": a, "b": b}, join=join)
    aligned = xr.align(a, b, join=join)
    for key, original, actual in zip(("a", "b"), (a, b), aligned):
        v = report["variables"][key]["<dataarray>"]
        assert v["aligned_cells"] == actual.size
        common = original.x.to_index().intersection(actual.x.to_index())
        assert v["retained_source_cells"] == len(common)
        assert v["dropped_source_cells"] == original.size - len(common)
        assert v["new_fill_positions"] == actual.size - len(common)


@pytest.mark.parametrize("join", ["inner", "outer", "left", "right"])
def test_multidimension_counts_do_not_doublecount_corners(join):
    a = xr.DataArray(
        np.ones((3, 4)), dims=["x", "y"], coords={"x": [1, 2, 3], "y": [2, 4, 6, 8]}
    )
    b = xr.DataArray(
        np.ones((2, 3)), dims=["x", "y"], coords={"x": [2, 4], "y": [1, 4, 8]}
    )
    r = inspect({"a": a, "b": b}, join=join)
    aa, bb = xr.align(a, b, join=join)
    for key, source, target in [("a", a, aa), ("b", b, bb)]:
        v = r["variables"][key]["<dataarray>"]
        keep = np.prod(
            [
                len(source[d].to_index().intersection(target[d].to_index()))
                for d in source.dims
            ]
        )
        assert v["aligned_cells"] == target.size and v["retained_source_cells"] == keep


def test_equality_of_aligned_coordinates_can_hide_dropped_labels():
    a = arr([1.0, 2.0, 3.0])
    b = arr([1.0, 2.000000000000001, 3.0])
    assert bool((a.x == b.x).all())
    r = inspect({"a": a, "b": b})
    assert r["dimensions"]["x"]["target_size"] == 2
    assert r["dimensions"]["x"]["sources"]["a"]["dropped_positions"] == [1]
    assert r["status"] == "CHANGED"
    with pytest.raises(ValueError):
        xr.align(a, b, join="exact")
    assert inspect({"a": a, "b": b}, join="exact")["status"] == "REFUSED"


def test_lazy_data_never_computed_and_inputs_unchanged():
    import dask.array as da
    from dask import delayed

    @delayed
    def boom():
        raise AssertionError("data computation forbidden")

    a = xr.DataArray(
        da.from_delayed(boom(), shape=(3,), dtype=float),
        dims="x",
        coords={"x": [1, 2, 3]},
    )
    b = a.isel(x=slice(1, None))
    graph_before = a.data.__dask_graph__()
    r = inspect({"a": a, "b": b}, join="outer")
    assert r["variables"]["b"][b.name]["new_fill_positions"] == 1
    assert (
        a.sizes["x"] == 3
        and b.sizes["x"] == 2
        and a.data.__dask_graph__() is graph_before
    )


def test_dataset_variable_specific_cells_and_no_arithmetic_claim():
    a = xr.Dataset(
        {"xy": (("x", "y"), np.ones((3, 2))), "xonly": ("x", [1, 2, 3]), "scalar": 42},
        coords={"x": [1, 2, 3], "y": [5, 6]},
    )
    b = a.sel(x=[2, 3])
    r = inspect({"a": a, "b": b})
    assert r["variables"]["a"]["xy"]["dropped_source_cells"] == 2
    assert r["variables"]["a"]["xonly"]["dropped_source_cells"] == 1
    assert r["variables"]["a"]["scalar"]["dropped_source_cells"] == 0


@pytest.mark.parametrize(
    "labels", [[1, 1, 2], [1.0, float("nan"), 2.0], [1.0, float("inf"), 2.0]]
)
def test_ambiguous_coordinate_inputs_fail_closed(labels):
    with pytest.raises(Unsupported):
        inspect({"a": arr(labels), "b": arr(labels)})


def test_unindexed_shared_dimension_is_not_assumed_safe():
    a = xr.DataArray([1, 2], dims="x")
    with pytest.raises(Unsupported):
        inspect({"a": a, "b": a})


def test_mixed_numeric_bool_dtype_not_alias():
    with pytest.raises(Unsupported):
        inspect({"a": arr([0, 1]), "b": arr([False, True])})


@pytest.mark.parametrize(
    "labels",
    [
        pd.date_range("2000", periods=3),
        pd.timedelta_range("1day", periods=3),
        ["a", "b", "c"],
    ],
)
def test_supported_time_and_string_indexes(labels):
    a = arr(labels)
    b = a.isel(x=[0, 2])
    assert inspect({"a": a, "b": b})["dimensions"]["x"]["sources"]["a"]["dropped"] == 1


def test_multiindex_explicitly_unsupported():
    a = xr.DataArray(
        [1, 2], dims="x", coords={"x": pd.MultiIndex.from_tuples([("a", 1), ("b", 2)])}
    )
    with pytest.raises(Unsupported):
        inspect({"a": a, "b": a})


def test_samples_bounded_counts_not_truncated():
    a = arr(np.arange(200))
    b = a.isel(x=[0])
    r = inspect({"a": a, "b": b}, sample_limit=2)
    s = r["dimensions"]["x"]["sources"]["a"]
    assert (
        s["dropped"] == 199
        and s["dropped_positions"] == [1, 2]
        and s["positions_truncated"]
    )


def test_reordered_not_reported_as_dropped():
    a = arr([3, 2, 1])
    b = arr([1, 2, 3])
    r = inspect({"a": a, "b": b}, join="left")
    assert (
        r["dimensions"]["x"]["sources"]["b"]["reordered"]
        and not r["dimensions"]["x"]["sources"]["b"]["dropped"]
    )


def test_outer_empty_indexes_preserve_unsorted_nonempty_native_order():
    a = arr(np.array([7, 1, 0], dtype="int64"))
    empty = arr(np.array([], dtype="int64"))
    r = inspect({"a": a, "b": empty, "c": empty}, join="outer")
    native = xr.align(a, empty, empty, join="outer")
    assert native[0].x.values.tolist() == [7, 1, 0]
    assert not r["dimensions"]["x"]["sources"]["a"]["reordered"]
    assert r["dimensions"]["x"]["sources"]["b"]["inserted_target_positions"] == [
        0,
        1,
        2,
    ]


def test_multiple_indexes_on_one_dimension_refused():
    a = arr([1, 2]).assign_coords(u=("x", [3, 4])).set_xindex("u")
    with pytest.raises(Unsupported):
        inspect({"a": a, "b": arr([1, 2])})


@pytest.mark.parametrize("join", ["inner", "outer", "left", "right", "exact"])
def test_dimension_membership_not_global_first_last(join):
    # x appears only in inputs 2 and 3; y is unshared and must not broadcast.
    a = xr.DataArray([10, 20], dims="y", coords={"y": [8, 9]})
    b = arr([2, 1, 3])
    c = arr([1, 2, 4])
    if join == "exact":
        assert inspect({"a": a, "b": b, "c": c}, join=join)["status"] == "REFUSED"
        with pytest.raises(ValueError):
            xr.align(a, b, c, join=join)
    else:
        r = inspect({"a": a, "b": b, "c": c}, join=join)
        aligned = xr.align(a, b, c, join=join)
        for key, v in zip(("a", "b", "c"), aligned):
            assert r["variables"][key]["<dataarray>"]["aligned_cells"] == v.size
        assert r["variables"]["a"]["<dataarray>"]["aligned_cells"] == 2


def test_exact_equal_has_full_scalar_and_cell_counts():
    a = arr([2, 1])
    r = inspect({"a": a, "b": a.copy()}, join="exact")
    assert r["status"] == "UNCHANGED" and not r["findings"]
    assert r["variables"]["a"]["<dataarray>"]["retained_source_cells"] == 2


def test_exact_refusal_never_publishes_partial_geometry():
    a = xr.Dataset(
        {"xv": ("x", [1, 2]), "yv": ("y", [3, 4]), "s": 42},
        coords={"x": [1, 2], "y": [1, 2]},
    )
    b = a.assign_coords(x=[3, 4])
    r = inspect({"a": a, "b": b}, join="exact")
    for values in r["variables"].values():
        for variable in values.values():
            assert variable["aligned_cells"] is None
            assert variable["retained_source_cells"] is None
            assert variable["new_fill_positions"] is None


def test_plain_object_strings_supported_but_mixed_object_refused():
    a = arr(np.array(["a", "b", "c"], dtype=object))
    b = a.isel(x=[0, 2])
    assert inspect({"a": a, "b": b})["dimensions"]["x"]["sources"]["a"]["dropped"] == 1
    mixed = arr(np.array(["a", 1], dtype=object))
    with pytest.raises(Unsupported):
        inspect({"a": mixed, "b": mixed})


def test_report_json_serializes_without_numpy_coercion():
    import json

    a = arr([1, 2, 3])
    b = arr([2, 4])
    r = inspect({"a": a, "b": b}, join="outer")
    assert json.loads(json.dumps(r, allow_nan=False)) == r


def test_synthetic_workflow_output_matches_checked_in_artifact():
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).parent.parent
    output = subprocess.check_output(
        [sys.executable, str(root / "examples/compare_joins.py")],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        text=True,
    )
    assert json.loads(output) == json.loads(
        (root / "examples/compare_joins.expected.json").read_text()
    )
