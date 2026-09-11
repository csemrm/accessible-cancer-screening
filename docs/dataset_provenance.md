# SIPaKMeD dataset provenance and status

## Original source is not scriptable

The dataset authors' own page (`cs.uoi.gr/~marina/sipakmed.html`) links
directly to `.7z` archives on the same host, but the server returns
`403 Forbidden` at the Apache level for every tested request (varied
user-agent, with/without referer) — this is a server-side access
restriction, not a client-side fix. Verified 2026-09-10.

## Sources used instead (Kaggle mirrors)

| Kaggle ref | Content | Size | Role |
|---|---|---|---|
| `srijanshovit/sipakmed-cropped-5-class` | Pre-cropped single-cell images, resized to a fixed 66x66px | 10MB | Used only for pipeline smoke-testing — **too low-resolution for the explainability work (RQ3)**, which needs nucleus-level detail. Validated at 4,049 total images across 5 classes (793/813/831/825/787), exactly matching Plissiti et al. 2018's reported count — confirms this mirror is a faithful, non-augmented rendering of the full dataset. |
| `marinaeplissiti/sipakmed` | Raw: 966 cluster `.bmp` images + per-cell `_cytNN.dat`/`_nucNN.dat` polygon boundary files (x,y coordinate lists, one point per line) | 6.4GB | **Primary source.** Uploaded by the dataset's own author (name matches Plissiti et al.) — used to derive native-resolution single-cell crops via `crop_sipakmed.py`, rather than inheriting the fixed-resolution mirror's resize decision. |

## Status as of this session

- Raw source (`marinaeplissiti/sipakmed`) download in progress in the
  background (~6.4GB at ~0.6–1.2MB/s from Kaggle, multi-hour ETA).
- `.dat` polygon format confirmed by spot-checking one cyt/nuc file pair
  before committing to the full download (see git history).
- `crop_sipakmed.py` written against this confirmed format; not yet run
  against the full raw set (blocked on the download above).
- Once crops are produced, cross-check total count and per-class counts
  against the 66x66 mirror's 793/813/831/825/787 breakdown as an integrity
  check before treating the native-resolution crops as canonical.

## License

Kaggle lists `marinaeplissiti/sipakmed`'s license as "other" and
`srijanshovit/sipakmed-cropped-5-class`'s as "unknown" — neither is a
license this project can rely on without direct confirmation. Pull the
exact terms from the dataset authors' original page/paper (Plissiti et al.,
ICIP 2018) before any publication or redistribution of derived data.
