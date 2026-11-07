import random

import numpy as np
import xarray as xr

from indeximpact import inspect

random.seed(3473)
checks = 0
for trial in range(1000):
    axes = [
        [random.sample(range(8), random.randrange(9)) for d in range(2)]
        for n in range(3)
    ]
    objects = {
        str(i): xr.DataArray(
            np.zeros(tuple(map(len, ax))),
            dims=["x", "y"],
            coords=dict(zip(["x", "y"], ax)),
        )
        for i, ax in enumerate(axes)
    }
    # preserve integer dtype even for empty coordinates
    objects = {
        k: o.assign_coords({d: np.array(o[d], dtype="int64") for d in o.dims})
        for k, o in objects.items()
    }
    for join in ["inner", "outer", "left", "right"]:
        r = inspect(objects, join=join)
        aligned = xr.align(*objects.values(), join=join)
        for key, source, actual in zip(objects, objects.values(), aligned):
            v = r["variables"][key]["<dataarray>"]
            keep = 1
            for d in source.dims:
                src = source[d].to_index()
                tgt = actual[d].to_index()
                part = r["dimensions"][d]["sources"][key]
                assert (
                    part["dropped_positions"]
                    == np.flatnonzero(~src.isin(tgt))[:10].tolist()
                ), (trial, join, key, d, "drop")
                assert (
                    part["inserted_target_positions"]
                    == np.flatnonzero(~tgt.isin(src))[:10].tolist()
                ), (trial, join, key, d, "insert")
                assert part["reordered"] == (
                    not src[src.isin(tgt)].equals(tgt[tgt.isin(src)])
                ), (trial, join, key, d, "reorder", src, tgt, part, axes)
                keep *= sum(src.isin(tgt))
            assert (
                v["aligned_cells"] == actual.size and v["retained_source_cells"] == keep
            ), (trial, join, key, v, actual.shape)
            checks += 1
print(f"{checks} native-alignment input/join comparisons passed")
