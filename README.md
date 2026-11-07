# indeximpact

Coordinate-only reports for explicit xarray alignment choices. Pre-release package, not a data validator.

Use xarray's native guard when your inputs should match:

```python
aligned = xr.align(reference, candidate, join="exact")
# Or guard arithmetic within its scope:
with xr.set_options(arithmetic_join="exact"):
    difference = reference - candidate
```

This package is for the next question: **if the mismatch is intentional, what index positions would each possible join remove, add or reorder?** It does not choose a scientifically correct join.

## Install

Requires Python3.10 or newer. From a source checkout:

```bash
python -m pip install .
# Optional tests and lazy-data validation:
python -m pip install ".[test]"
```

No command-line interface or data-file loader is included. The API receives existing xarray objects.

## Synthetic workflow

`examples/compare_joins.py` defines two small datasets, with no network or files. Reference temperature has3 days x3 stations, candidate has3 days x2 stations; they overlap on2 days and1 station. It compares explicit joins without aligning the data arrays:

| Join | Reference temperature | Candidate temperature |
| --- | --- | --- |
| exact | refuses; no partial cell prediction | refuses |
| inner |9 ->2 positions;7 dropped |6 ->2 positions;4 dropped |
| outer |9 ->16 positions;7 newly filled |6 ->16 positions;10 newly filled |
| left |9 ->9 positions, unchanged |6 ->9 positions;4 dropped,7 newly filled |
| right |9 ->6 positions;7 dropped,4 newly filled |6 ->6 positions, unchanged |

Dropping one row and two columns removes7 of9 reference positions, not9. The report computes retained Cartesian positions rather than summing row and column losses. Dataset variables that use only some dimensions are counted separately; a scalar stays one position. An alignment report is **not** a Dataset subtraction result: missing variable names, arithmetic broadcasting and non-index-coordinate merge behavior are outside its scope.

```python
from indeximpact import inspect
report = inspect({"reference": reference, "candidate": candidate}, join="inner")
report["dimensions"]["station"]["sources"]["reference"]
# dropped_positions are zero-based SOURCE positions.
# inserted_target_positions are zero-based TARGET positions.
report["variables"]["reference"]["temperature"]["dropped_source_cells"]
# 7
```

Default position samples contain at most10 positions; `sample_limit` accepts0 through100. Full counts are never truncated. Labels are omitted, but input, dimension and variable names remain visible. Output is JSON serializable without special encoders. Saving or sharing it is your decision. Non-string or empty DataArray names use the documented placeholder `<dataarray>`; it is not an attributed variable identity.

## Strict scope

- Explicit inner/outer/left/right/exact joins on at least2 named inputs. Dictionary order decides left/right, separately for each shared dimension.
- DataArray/Dataset inputs; string names; one same-name built-in PandasIndex per dimension. Supported shared-axis labels are unique, nonmissing, finite plain numeric/bool/string or timezone-naive datetime/timedelta, with matching dtypes across shared axes.
- Reject shared-axis mixed object labels, custom indexes, MultiIndex, multiple indexes per dimension, category/interval/tz indexes, duplicate/nonfinite/missing labels, mixed dtypes, shared unindexed dimensions and repeated dimension names within a data variable (even on unshared axes). It does not fall back to positional matching.
- Unshared dimensions are unchanged by alignment, not broadcast. Non-index coordinates, attributes, units, calendars and values are not checked. No override, exclude, explicit target indexes or tolerance matching.
- Reads index metadata into memory; never computes or changes data arrays. Large coordinate indexes can still cost memory and time. No constant-memory claim.
- Cell counts are geometric positions, not valid measurements. New fill positions may receive NaN or a caller's fill value during actual alignment. Existing NaNs are not counted.
- `UNCHANGED` means only no change in modeled supported index geometry. It never means scientifically valid, equivalent variables or safe arithmetic. `REFUSED` exact joins have no predicted geometry, even for unaffected variables.

## Validation performed

Run `python -m pytest -q` and `python tests/native_oracle.py` in an environment with xarray, NumPy, pandas, Dask and pytest. Run the example with `python examples/compare_joins.py`; its output is checked into the source tree.

72 fixed tests and12,000 deterministic input/join comparisons passed on each of:

- Python3.10.12, xarray2024.10.0, pandas2.2.3, NumPy2.1.3, Dask2024.10.0.
- Python3.13, xarray2026.9.0, pandas3.1.1, NumPy2.5.4, Dask2026.8.0.

Repeated variable dimensions are deliberately rejected: xarray permits their construction with a warning, but a dimension-size mapping collapses repeated names and cannot support this report's cell accounting. The causal control produced21 failures and51 passes before this refusal; after the fix all72 tests pass. The21 failures were the20 DataArray/Dataset, shared/unshared and join combinations plus the lazy-data refusal case. A distinct-axis negative control preserves native counts.

This is two tested environments, not proof of every version between them. Random comparisons cover integer axes; dedicated cases cover strings/times. A delayed Dask data bomb proves the tested report path does not compute data arrays. A causal regression reproduces the original private implementation's incorrect sorting against empty indexes and passes after matching native union/intersection semantics.

## Existing tools and evidence

This is not a claim that xarray is broken or lacks a guard. Its native align/exact options already prevent mismatches. Climate-indices documents dropped-timestep warnings; canVOD has domain-specific audit; mxalign is a meteorological alignment and verification engine. Prefer an existing workflow when it fits. Historical user reports motivate explaining coordinate loss but do not establish demand for this package. See `docs/feasibility.md` for source links and limits. No outside user, dataset or adoption is implied.

## Development disclosure

This project was developed with AI assistance and tested against native xarray behavior. No affiliation with or endorsement by xarray or the linked projects is implied.
