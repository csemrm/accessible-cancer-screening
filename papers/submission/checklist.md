# JMI (SPIE) submission checklist — "Trustworthiness is not one property"

Generated 2026-10-11 from the current state of `papers/paper_jmi.tex` / `paper_jmi_supp.tex` at commit `9e9d083`.

## Files to upload

| File | Purpose | Status |
|---|---|---|
| `papers/paper_jmi.pdf` | Main manuscript (23 pages) | Ready |
| `papers/paper_jmi_supp.pdf` | Supplement (12 pages, Tables S1–S14, Figures S1–S7) | Ready |
| `papers/submission/cover_letter.pdf` | Cover letter to the Editor-in-Chief | Ready |
| Separate title/author-info form, if the portal has one | Portal-specific; fill directly on SPIE's submission site | Not applicable here — JMI is not double-blind, author block is already in the manuscript |

## Manuscript facts for the portal's metadata fields

- **Title:** Trustworthiness is not one property: a multidimensional evaluation of cervical cytology AI
- **Authors:** Md. Mostafizur Rahman¹* (corresponding), Khairul Anam²
  1. Dept. of Electrical and Computer Engineering, San Francisco Bay University, Fremont, CA, US
  2. Dept. of Computer Science and Engineering, Pacific State University, US
- **Corresponding author email:** csemrm@gmail.com
- **ORCIDs:** Rahman 0009-0007-2022-8677; Anam 0009-0005-4052-8023
- **Keywords (6, within SPIE's 3–6 range):** cervical cytology, vision transformers, cross-dataset generalization, explainability, selective prediction, model efficiency
- **Abstract:** 249 words (structured: Purpose/Approach/Results/Conclusions), under JMI's 250-word limit
- **Article type:** Research article
- **Funding:** none (stated in Acknowledgments)
- **Conflicts of interest:** none declared
- **AI-tool use disclosure:** present (Methods + Acknowledgments) — generative AI assisted drafting/language only, authors verified all content
- **Data availability:** all 4 datasets are public with citations/DOIs; training/eval code and per-model result files "available from the corresponding author upon reasonable request" (repo kept private — author's deliberate choice, already revisited by 2 rounds of peer feedback and kept as-is)
- **References:** 60, numbered, superscript style

## Internal verification already done this session (not required again)

- Both PDFs rebuilt twice (pdflatex x2 passes) with zero `undefined reference` / `! ` LaTeX errors.
- Abstract word count checked programmatically at 249/250.
- Every externally-pasted peer-review claim (5 rounds) was independently checked against the underlying code/data before any text was changed — see commit messages `bd12bbe` through `9e9d083` for the specific verification done per item.
- Figure 1 (explainability heatmaps) regenerated at 300 dpi (was 150), larger labels, same checkpoints/predictions as before (accuracy reproduced exactly: DeiT-Tiny 0.9580536912751678, matches prior report of 0.958).

## Known, deliberate open items (not blockers, already discussed with the author)

- Public code/data repository: declined in favor of "available upon reasonable request" — raised by reviewers twice, author decision stands.
- A second explainability perturbation baseline (beyond deletion/insertion AUC): flagged as a possible follow-up experiment, not done — would need a new Kaggle run, not a text fix.
- Patient-level external validation: out of scope, noted as a limitation in the manuscript itself.

## Suggested reviewers

Not populated — fill in on the portal if JMI's form asks for suggested/excluded reviewers; nothing in this repo establishes a reviewer list.

## Before you click submit

1. Re-read `papers/paper_jmi.pdf` title page once more for typos in author info (last manual check was this session, commit `9e9d083`).
2. Confirm the corresponding-author email (`csemrm@gmail.com`) is one you check regularly — SPIE will send review correspondence there.
3. Check SPIE's current file-size limits on their submission portal against `paper_jmi.pdf`'s ~5.3 MB (grew from ~1.3 MB after the Figure 1 resolution bump); if the portal balks, the 300 dpi PNG can be stepped down to ~200 dpi without falling back to the old 150 dpi quality.
4. `docs/submission_tracker.md` in this repo still describes the old, abandoned 3-separate-papers conference track (TCCE/TEHI) — it is stale relative to this JMI submission and should probably be updated or removed once this submission goes in, so a future reader isn't misled.
