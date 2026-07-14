# IPUMS NHIS codebook review

Review date: 2026-07-14

## Exact input binding

- Extract: `data/raw/nhis/nhis_00001.csv.gz`
- Extract SHA-256: `f6fbee8a4a033e2661f74dc7fc6d2fee0deda0648ca433d54d9a1348505ff36c`
- DDI: `data/raw/nhis/nhis_00001.xml`
- DDI SHA-256: `03201d7b52ef0fc32166789d0a21223ddfd46089f582d4a3024223400298ae72`
- Rows: 2,061,980
- Variables: 38 in CSV and DDI, with identical names and order
- Samples: every year from 1997 through 2018

## Mapping decisions verified against DDI

- NHIS yes/no condition variables use `1 = No`, `2 = Yes`; `0/7/8/9`
  remain distinct missing reasons.
- `MORTSTAT` uses `1 = assumed deceased`, `2 = assumed alive`, `9 = NIU`.
- `MORTELIG` uses `1 = eligible`, `2 = under age 18`, `3 = ineligible`,
  `9 = NIU`.
- `MORTUCODLD` uses `01` for diseases of heart, `05` for cerebrovascular
  diseases, and `96 = NIU`. Canonical codes are zero-padded to three digits.
- `MORTDODY = 9999` and `MORTDODQ = 9` are NIU.
- Converted CSV BMI has two implied decimals already applied; `99.99` is the
  converted unknown code and `99.8` is the released top-code.
- `RACEA` and `HISPETH` are audit-only. The bridge to NHANES RIDRETH1 uses five
  categories: Mexican American, other Hispanic, non-Hispanic White,
  non-Hispanic Black, and other/multiracial. Hispanic ethnicity takes
  precedence over race.

## Observed-code coverage

A 4,367-row union sample containing every observed value of all mapped source
variables passed strict harmonization. This included every observed discrete
code, all BMI values, all years, and all mortality codes. No unknown observed
source code was silently converted or treated as a negative response.

## Status

`configs/feature_mapping.yaml` is marked `verified_against_user_extract` and is
bound to the exact SHA-256 digests above. Any change to the extract, DDI, or
mapping invalidates the binding and must stop cohort construction.
