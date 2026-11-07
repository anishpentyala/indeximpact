"""Synthetic notebook-sized example. These are not a user's measurements."""

import json

import numpy as np
import xarray as xr

from indeximpact import inspect

reference = xr.Dataset(
    {
        "temperature": (("day", "station"), np.ones((3, 3))),
        "quality": ("day", [0, 0, 0]),
        "calibration": 42,
    },
    coords={"day": [1, 2, 3], "station": ["a", "b", "c"]},
)
candidate = xr.Dataset(
    {"temperature": (("day", "station"), np.ones((3, 2)))},
    coords={"day": [2, 3, 4], "station": ["b", "d"]},
)
inputs = {"reference": reference, "candidate": candidate}
summary = {}
for join in ("exact", "inner", "outer", "left", "right"):
    report = inspect(inputs, join=join)
    summary[join] = {
        "status": report["status"],
        "reference_temperature": report["variables"]["reference"]["temperature"],
        "candidate_temperature": report["variables"]["candidate"]["temperature"],
    }
print(json.dumps(summary, indent=2, allow_nan=False))
