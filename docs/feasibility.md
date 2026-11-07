# Xarray alignment diagnostic feasibility
November 7, 2026

A narrow index-alignment impact report is technically feasible and has a plausible debugging use. It should not be sold as a missing safety guard: native exact joins already stop mismatched alignment, and domain libraries already report dropped data. Private work now covers geometric loss/fill/reorder diagnostics without computing data arrays, with strict support boundaries. Recommendation: continue only if a documented notebook workflow makes this materially easier than native exact plus a few index comparisons. No public repo or demonstrated adoption yet.

## Options

| Option | Best fit | Limitation |
| --- | --- | --- |
| Native `xr.align(..., join="exact")`, `arithmetic_join="exact"` | Stop accidental mismatch | Refusal is the desired result, not a detailed impact forecast |
| Existing domain warnings/audit | Climate indices or canVOD workflows | Domain-specific, use these when already in that stack |
| Private impact report | Intentionally choosing inner/outer/left/right joins; need reproducible counts | Extra dependency and narrow supported index model; not a scientific validity check |

## Evidence

Issue3473 is a concrete floating-coordinate drift example, including the trap that comparing coordinate arrays can itself align them. Issue7045 discusses predictability and float indexes; it is historical design discussion, not proof of a current bug or a blanket complaint about current xarray. Issue1354 distinguishes index alignment from conflicting-coordinate merge behavior. The prototype only addresses the former.

The native align documentation defines union/intersection and first/last indexes per dimension. Its `exclude`, explicit indexes and override options are outside this prototype. NCAR teaching material demonstrates alignment directly and is a useful baseline for how simple explanations can already be.

Alternatives matter. climate_indices documents automatic inner alignment with InputAlignmentWarning for dropped timesteps. canVOD audit documents domain alignment checks. mxalign describes a meteorological alignment and verification engine. These are reasons to avoid claiming novelty or universal missing validation.

## Technical findings

The private prototype accepts named xarray DataArray/Dataset inputs and explicit join. It reports per-input dimension positions dropped/inserted/reordered, and per-variable Cartesian geometry counts. Products of retained per-axis sizes avoid counting removed corners twice. It never inspects array values, interpolates, coerces coordinate dtype, or corrects inputs. Native exact remains the safer recommendation for workflows that expect identical coordinates.

50 fixed regression cases pass on two environments (Python3.10/xarray2024.10/pandas2.2 and Python3.13/xarray2026.9/pandas3.1). A deterministic random harness checked 1,000 three-input two-dimensional cases across four joins, totaling12,000 input/join comparisons with native xr.align. This caught an actual private prototype defect: pandas Index.join sorted an unsorted index against empty indexes where native xarray preserves order. The implementation now uses the union/intersection operations used by native PandasIndex. The original implementation fails the added ordering regression; the corrected implementation passes. Random testing is supporting evidence, not exhaustive verification.

The probe now imports documented xarray.indexes.PandasIndex rather than a private module. Only a single built-in same-name index per dimension is accepted. Extra/custom/multi/category/interval/tz/mixed-object indexes are refused, shared unindexed dimensions refused, mixed dtypes refused, duplicate/missing/infinite labels refused. Independent review reproduced both original pinned environments, 50 fixed tests and 12,000 native comparisons in each, plus a separate multi-dtype and refusal battery. Packaging review verified clean artifacts and installed tests on an older and an intermediate stack. These bounded checks do not establish support for every version or scientific validity.

## Boundaries and unknowns

This is not an arithmetic, broadcast, concat, merge, units, calendar or missing-data validator. Per-variable cell counts are geometric positions, not nonmissing measurements. Already lost coordinates cannot be reconstructed. Index coordinates are materialized; lazy array data are not computed. Counts can be large Python integers. Name privacy and potentially large index memory/time need explicit documentation. No known user dataset or outside demand has been established, and no performance benchmark or version compatibility claim is justified yet.

## Sources

All fetched November7,2026. Three source types: official reference/API docs, issue/community records, and educational/project documentation. Historical issue dates are not current feature-release dates.

- xarray native API, current reference: https://docs.xarray.dev/en/stable/generated/xarray.align.html
- xarray exact arithmetic option, current reference: https://docs.xarray.dev/en/latest/generated/xarray.set%5Foptions.html
- xarray documented index class: https://docs.xarray.dev/en/stable/generated/xarray.indexes.PandasIndex.html
- Actual float mismatch report, community primary record: https://github.com/pydata/xarray/issues/3473
- Alignment design discussion, historical community record: https://github.com/pydata/xarray/issues/7045
- Concat/merge distinction, historical community record: https://github.com/pydata/xarray/issues/1354
- NCAR alignment tutorial, education: https://ncar-hackathons.github.io/scientific-computing/xarray/06_alignment.html
- Climate indices own migration guide, existing dropped-data warnings: https://climate-indices.readthedocs.io/en/latest/xarray_migration.html
- canVOD own API guide, existing domain audit: https://nfb2021.github.io/canvodpy/api/canvod-audit/
- mxalign author-supplied package description, project claims rather than independent validation: https://pypi.org/project/mxalign/
