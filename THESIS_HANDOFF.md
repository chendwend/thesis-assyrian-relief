# Thesis project handoff

Current continuation: [FINAL_SUBMISSION_HANDOFF.md](FINAL_SUBMISSION_HANDOFF.md), updated 2026-09-06. Read that file first for the submission plan, completed pushes, and confirmed bilingual metadata.

Historical handoff below, last updated: 2026-09-01

## Critical 2026-09-01 update (supersedes stale metrics below)

The primary statistical unit is now the dependency group, not the image or the
189 curation-record identifiers. The frozen 664-image corpus contains 171
dependency groups: 80 train, 52 validation, and 39 test. By ruler, the group
counts are Ashurbanipal 20/15/13, Ashurnasirpal II 35/22/16, and Sargon II
25/15/10 for train/validation/test.

Across stretch-preprocessing seeds 24, 42, and 77, primary dependency-group
classification is:

| Seed | Correct | Accuracy | Macro-F1 |
|---:|---:|---:|---:|
| 24 | 32/39 | 0.821 | 0.824 |
| 42 | 32/39 | 0.821 | 0.814 |
| 77 | 33/39 | 0.846 | 0.841 |
| Mean +/- sample SD | -- | 0.829 +/- 0.015 | 0.826 +/- 0.014 |

Group nearest-centroid accuracy and macro-F1 are 0.838 +/- 0.015 and
0.835 +/- 0.019. Group retrieval Recall@1 is 0.872 +/- 0.051; Recall@3 and
Recall@5 are both 0.897 +/- 0.044. Image and 41-record results are secondary.

The aspect-preserving resize-and-pad sensitivity is complete. Its group accuracy
is 0.821 +/- 0.044, macro-F1 is 0.825 +/- 0.042, and Recall@1 is
0.803 +/- 0.039. It does not consistently improve classification and lowers
retrieval. The tracked report is
reports/aspect_ratio_ablation_2026-09-01.md.

All six chapters and the Abstract were aligned with the group-level analysis.
The Introduction, Literature Review, and Methodology were substantively
rewritten in response to the user's notes. Frahm is cited, together with
Manovich, Bracker, and Zauner. Project-history narration and the Conclusion's
submission to-do list were removed.

The corrected seed-42 confusion matrix and UMAP are dependency-group figures.
The latter is fitted on 80 training groups and projects validation and test
groups into the fixed space.

Latest PDF:

- path: /home/kostya/projects/thesis-overleaf/output/pdf/main.pdf
- pages: 41
- compiled: 2026-09-01
- SHA-256: d13bba9001950ddb03d95cef4746cad2a4c1503f5d9f1b7e422bb612afef2d42
- final log: no unresolved citations/references and no overfull/underfull boxes

Code verification: 20 tests passed; ruff passed on every changed code path.

Immediate next priorities are human scholarly review, not another model run:

1. verify page-specific support for the archaeological claims, especially the
   broad Frahm/Manovich/Bracker framing;
2. confirm registered title, department wording, required Hebrew frontmatter,
   signatures, and university formatting;
3. proofread the 41-page PDF for prose, captions, bibliography presentation,
   and page flow;
4. only then consider a partial-scene audit or room/programme feasibility table.

Source holdout is not currently a sound headline robustness experiment. The
available British Museum-only test subset has 19 records and only three Sargon
examples; source and monarch are too strongly confounded for a nominal
leave-one-source-out score to answer the thesis question.

This file is the starting point for a new Codex session. Read it before making
changes. The user's first priority is to finish and submit the MA thesis; the
articles and public dataset follow later.

## 1. Author, programme, and deadline

- Author: Konstantin Margulyan
- Degree: MA, Social Sciences and Humanities
- Track: Department of Israel Heritage, thesis track
- University: Ariel University
- Submission deadline: 7 September 2026
- Current working title: *Deep Visual Analysis of Monarch-Associated Patterns
  in Neo-Assyrian Reliefs*
- The registered title and official institutional wording still need final
  confirmation.

The planned outputs are three conceptually separate tracks, not Git branches:

1. the MA thesis, which is submission-critical;
2. a later thesis-derived article, provisionally aimed at PNAS and potentially
   extended beyond the submitted thesis; and
3. a database-construction article plus a Zenodo dataset release.

The structured database should be described in the thesis as provenance and
curation infrastructure, but it should not displace the visual-analysis
research question.

## 2. Repository map and Git warning

| Path | Purpose | Actual Git root |
|---|---|---|
| `/home/kostya/projects/thesis-assyrian-relief` | Modelling, evaluation, audits, plots | same directory |
| `/home/kostya/projects/thesis-overleaf` | LaTeX thesis and compiled PDF | same directory |
| `/home/kostya/projects/assyrian-relief-dataset` | Canonical metadata, PostgreSQL schema, ingestion and curation | `/home/kostya/projects` |
| `/home/kostya/projects/dataset_v2` | Locally reviewed modelling images | `/home/kostya/projects` |
| `/home/kostya/projects/manually_downloaded` | Extra manually acquired files | `/home/kostya/projects` |
| `/home/kostya/projects/THESIS_CONTEXT.md` | Earlier generated context | `/home/kostya/projects` |

Important: `assyrian-relief-dataset` and `dataset_v2` are not independent Git
repositories. Running `git status` in either displays the entire parent project
tree. Do not stage, commit, move, or delete unrelated sibling projects.

All relevant worktrees are dirty and contain user work plus previous Codex work.
Do not reset, discard, or overwrite existing changes. No commits were created in
the previous session.

## 3. Current thesis artifact

The latest compiled thesis is:

`/home/kostya/projects/thesis-overleaf/output/pdf/main.pdf`

- Pages: 42
- SHA-256: `009ded68ad1558dc28f8353f433a1c977e20b7ba66980eba878722ffd832daa1`
- Compiled: 2026-08-26
- Final LaTeX log had no unresolved citations, unresolved references,
  overfull boxes, or underfull boxes.
- Pages containing the new shortcut table, UMAP, occlusion figures,
  Discussion, and Conclusion were rendered and visually inspected.

The thesis currently includes substantive drafts of the Introduction,
Literature Review, Methodology, Results, Discussion, and Conclusion. It is not
submission-ready: literature coverage, robustness work, wording, and final
proofreading remain.

## 4. Data history and authoritative interpretation

### Legacy workbook

`main.xlsx` was the first curation arrangement. Each row could represent one or
more images. `style_analysis_use` was assigned after manual inspection according
to the best image in the record, so it is a record-level screening field, not a
per-image label.

### Local image corpus

`dataset_v2` was created later by inspecting the images within approved records
and downloading the good images individually. In the newer database these files
are represented at image level. The current local directory contains 777 images
treated as `core_train` eligibility. Experiment membership is separate from
eligibility.

The corrected three-ruler experiment uses 664 of those 777 files. The remaining
113 belong to Sennacherib or Tiglath-Pileser III and are excluded from the
three-class experiment, not relabelled as an `OTHER` class.

### Manual-download folder

The manual folder contains 10 files. Two MFA 35.731 images are byte-identical to
`BOS 35.731-1.jpg` and `BOS 35.731-2.jpg` in `dataset_v2`; those copies occur in
training. The other eight manual files are recorded as `inference_only` and were
not used in training, validation, or testing.

### Canonical database

The canonical export currently contains 560 museum-object records. The database
work includes a PostgreSQL schema for museum objects, raw source records, relief
groups, images, subjects, bibliography, curation links, and versioned
experiments. Current counts used in the thesis are 524 relief groups and mappings
for 777 selected local images. The canonical export has 2,004 image references
and 3,068 object-subject assignments.

The public database, rights audit, access layer, dataset paper, and Zenodo release
are later work. Do not expand them into a second thesis contribution before the
thesis is secure.

## 5. Important curation decisions already incorporated

- BM registration numbers beginning with `1847,0702...` must not be collapsed
  to a single `BM 1847` relief. The old Excel-derived rule caused that error.
- Duplicate AO 19904 workbook information was consolidated.
- AO 19864 and AO 19880 remain separate museum objects but are linked/grouped
  conservatively because one photograph is shared and Albenda suggests the
  fragment may belong to AO 19864.
- BM 124564, BM 124565, and BM 124566 are separate transported fragments of one
  court scene and are constrained to one experimental component.
- BM 124584 and BM 124585 follow the same linked-fragment principle.
- MET 31.143.4 and MET 32.143.6 are treated as linked parts of a divided panel,
  without erasing their separate accession identities.
- MET 32.143.3 is present in `dataset_v2`.
- BM 124928 was replaced with the correct images.
- Generic curator wording such as “in set with” is not automatically treated as
  proof of a physical join. Supported joins, continuous scenes, transport
  divisions, and explicitly reviewed possible same-frieze relations constrain
  the split; broad museum-set relations remain contextual metadata.

Use three identifiers conceptually:

- `object_id`: museum accession/registration object;
- `relief_group_id`: physical join, continuous scene, or conservatively linked
  archaeological group; and
- `image_id`: photograph, scan, crop, or detail.

Do not merge museum objects merely to simplify analysis. Link them through the
group and image mapping tables.

## 6. Frozen three-ruler experiment

Manifest:

`/home/kostya/projects/thesis-assyrian-relief/data/splits/image_level_dataset_v2_grouped.csv`

SHA-256:

`4942ec4984fa5a5b261cd9b2c7cc141e37a17ae84d82cf0e41becd83d518690e`

Composition:

| Authority | Train records (images) | Validation records (images) | Test records (images) | Total records (images) |
|---|---:|---:|---:|---:|
| Ashurbanipal | 25 (194) | 16 (55) | 13 (28) | 54 (277) |
| Ashurnasirpal II | 39 (166) | 22 (48) | 17 (24) | 78 (238) |
| Sargon II | 31 (104) | 15 (30) | 11 (15) | 57 (149) |
| Total | 95 (464) | 53 (133) | 41 (67) | 189 (664) |

The 189 legacy `Relief_ID` values were assigned through 171 inseparable connected
components. Curated archaeological links and exact identical checksums stay in
one partition. The component/split construction used seed 24. Model
initialisation was repeated with seeds 24, 42, and 77 on the same frozen split.

Configs:

- `configs/style_dinov2_grouped_seed24.yaml`
- `configs/style_dinov2_grouped_seed42.yaml`
- `configs/style_dinov2_grouped_seed77.yaml`

Checkpoints and evaluations are under `outputs/grouped_seed24`,
`outputs/grouped_seed42`, and `outputs/grouped_seed77`. The `outputs/` directory
is Git-ignored because it contains large checkpoints.

Model:

- frozen DINOv2 ViT-S/14 backbone;
- 384-dimensional backbone representation;
- trainable projection to 256 dimensions plus a three-class linear classifier;
- class-weighted cross-entropy;
- AdamW, learning rate `1e-3`, weight decay `1e-4`;
- batch size 16, maximum 50 epochs;
- validation macro-F1 checkpoint selection and early stopping.

## 7. Verified quantitative results

Across model seeds 24, 42, and 77 on one fixed split:

| Evaluation | Accuracy | Macro-F1 |
|---|---:|---:|
| Image level, n=67 | `0.856 +/- 0.017` | `0.856 +/- 0.016` |
| Record level, mean logits, n=41 | `0.829 +/- 0.024` | `0.827 +/- 0.023` |
| Nearest class centroid, n=41 | `0.837 +/- 0.014` | `0.833 +/- 0.013` |

Retrieval:

- Recall@1: `0.870 +/- 0.037`
- Recall@3: `0.886 +/- 0.037`
- Recall@5: `0.886 +/- 0.037`

Selected epochs were 4, 7, and 5 for seeds 24, 42, and 77. The representative
seed-42 record-level result is 34/41 correct, accuracy 0.829 and macro-F1 0.825.

Four record errors were shared by every initialisation:

- AO 19902: Ashurbanipal -> Ashurnasirpal II
- BM 124867: Ashurbanipal -> Ashurnasirpal II
- BM 124931: Ashurbanipal -> Ashurnasirpal II
- BM 118829: Sargon II -> Ashurnasirpal II

Seven other records changed correctness or predicted class between seeds.

### Five-ruler exploratory result

The older five-ruler validation run is retained only as evidence for the corpus
scope decision, not as a controlled ablation:

- validation image accuracy 0.722, macro-F1 0.595, n=115;
- saved record accuracy 0.686, macro-F1 0.630, n=35;
- majority-vote record accuracy 0.714, macro-F1 0.647;
- retrieval Recall@1 0.686 and Recall@3/5 0.800;
- only four Sennacherib and three Tiglath-Pileser III validation records;
- no audited five-class test result.

The thesis states that three rulers were chosen because the assembled supervised
corpus did not support a stable balanced five-class baseline. It does not claim
that the excluded programmes lack meaningful style.

## 8. Leakage and shortcut audits completed

Implementation:

`scripts/audit_dataset_shortcuts.py`

Tracked summary:

`reports/dataset_shortcut_audit_2026-08-26.md`

Machine outputs:

`outputs/dataset_audit/`

Duplicate result:

- zero exact SHA-256 crossings;
- zero DCT pHash pairs across splits within Hamming distance 6;
- zero credible automated perceptual matches;
- one dHash-only review candidate: BM 124542-2 (train) versus BM 124583-2
  (validation), pHash distance 28 and resized correlation -0.114; visual review
  showed unrelated scenes, so this is a hash collision.

The audit does not rule out partial-scene overlap or substantially different
views of the same archaeological panel.

Source association:

- source/ruler normalized mutual information: 0.350;
- 100 of 104 AO images are Sargon II;
- 273 of 277 Ashurbanipal images use a BM prefix;
- 19 of 21 MET images are Ashurnasirpal II.

Shortcut baselines, trained only on training images and evaluated on 67 test
images:

| Predictor | Accuracy | Macro-F1 |
|---|---:|---:|
| Source-prefix majority | 0.672 | 0.664 |
| Geometry and file size | 0.522 | 0.521 |
| Border statistics | 0.433 | 0.413 |
| Combined simple acquisition features | 0.552 | 0.551 |

Seed-42 image accuracy is 0.866, so the full model is stronger than these simple
baselines. However, the benchmark contains substantial non-stylistic label
signal. Do not describe the result as “pure style classification.”

## 9. Corrected UMAP and occlusion work

The corrected relief-level UMAP uses the seed-42 grouped checkpoint. UMAP was
fitted only on training-record embeddings, then validation and test records were
transformed into the fixed projection.

Files:

- `outputs/grouped_seed42/umap/relief_umap_fit_train_all_splits.csv`
- `outputs/grouped_seed42/umap/relief_umap_fit_train_all_splits.html`
- `outputs/grouped_seed42/umap/relief_umap_fit_train_all_splits.png`

`scripts/umap_style.py` now supports `--png-out` and a two-panel static plot:
ruler colouring on the left, museum/source-prefix colouring on the right.

Interpretation: the labels are strongly separated, but the embedding was trained
to predict those labels and source is correlated with them. The plot is a model
diagnostic, not independent proof of three historical schools.

Corrected grouped-model occlusion outputs were generated for:

- AO 19894, stable correct Sargon II, seed-42 probability 0.915; strongest
  Sargon-II-over-Ashurnasirpal-II support around the eye/upper face;
- MET 32.143.7, stable correct Ashurnasirpal II;
- AO 19909, stable correct Ashurbanipal despite AO usually indicating Sargon II,
  probability 0.885; its margin is sensitive near the fragment boundary and
  pale background;
- BM 124931, stable Ashurbanipal -> Ashurnasirpal II error, probability 0.729;
  sensitivity is broad and includes upper scene, tree, frame/background, and
  lower register.

The thesis includes figures for AO 19894 and BM 124931 and discusses AO 19909.
The evidence supports the cautious conclusion that the model uses some local
object features but also framing/acquisition shortcuts.

## 10. Thesis files changed in the latest block

In `/home/kostya/projects/thesis-overleaf`:

- `chapters/(3)Methodology.tex`
  - added the perceptual duplicate procedure and result;
  - added source/acquisition shortcut audit methodology.
- `chapters/(4)Results.tex`
  - added shortcut-baseline table;
  - added corrected UMAP and interpretation;
  - added AO 19894 and BM 124931 occlusion case study;
  - updated validity status.
- `chapters/(5)Discussion.tex`
  - interpreted measured shortcut risk;
  - rejected a “carving morphology alone” interpretation;
  - updated data/software validity.
- `chapters/(6)Conclusion.tex`
  - added the measured limitation;
  - replaced completed items in the remaining-work list.
- `figures/umap_grouped_seed42.png`
- `figures/occlusion_AO19894_margin.png`
- `figures/occlusion_BM124931_margin.png`

Earlier in the same extended workstream, all other principal chapters,
frontmatter, bibliography, preamble, and audit notes were also edited and remain
uncommitted.

## 11. Zotero and bibliography state

Zotero desktop exposed its local API at `http://localhost:23119/api/`. The main
archaeological collection is named `Neo-Assyrian`. The user also created and
imported the requested additional collection/materials for methodology; verify
the exact current collection name in Zotero rather than assuming it.

Better BibTeX is not installed. It is useful later for stable citekeys and
repeatable export, but the thesis currently compiles without it. Do not install
software without a clear need and user authorization.

The Overleaf bibliography is:

`/home/kostya/projects/thesis-overleaf/references.bib`

Archaeological sources added or confirmed during the work include Barnett,
Albenda, *Sculptures from the North Palace of Ashurbanipal at Nineveh*, relevant
chapters from *Moving on from Ebla* and *Making Pictures of War*, Albenda's 2008
hunt study, and material on the Balawat Gates. The literature review still needs
systematic archaeological and deep-learning/methodology coverage.

Efficient Zotero workflow for the new session:

1. inventory titles, item types, authors, dates, abstracts, tags, and attachments;
2. map sources to specific thesis claims/sections;
3. search within PDFs for targeted terms and inspect relevant chapters/pages;
4. read complete articles or relevant book chapters when needed;
5. do not ingest whole books indiscriminately into context;
6. search the web for missing scholarship only when the local bibliography is
   insufficient, preferring primary/official sources for technical methods;
7. record page-specific support before adding a historical claim.

If the Zotero API is unavailable, the prior session successfully inspected the
local Zotero SQLite database read-only. Never write directly to `zotero.sqlite`.

## 12. What remains, in priority order

### Submission-critical

1. **Source-balanced robustness feasibility.** First measure whether each ruler
   has enough cross-source coverage for leave-one-source-out or source-balanced
   evaluation. Do not train a nominal holdout whose held-out source is almost a
   proxy for one class. Produce a feasibility table before choosing the design.
2. **Background/inscription/preprocessing ablations.** Compare at least one
   controlled alternative such as foreground masking, inscription masking, or
   aspect-preserving resize-and-pad. Keep the frozen split and declared seeds.
3. **Partial-scene duplicate audit.** Current hashes cover whole-image similarity;
   partial overlaps and very different views remain possible.
4. **Room/programme sensitivity.** Audit whether a palace room or decorative
   programme can be held out while preserving all three classes. If not feasible,
   document the limitation rather than forcing a weak experiment.
5. **Literature review.** Complete both archaeological foundations and deep
   learning/evaluation/shortcut/explainability methodology with verified,
   page-specific sources.
6. **Full-thesis consistency review.** Reconcile terminology (`record`, `object`,
   `relief group`, `image`, `component`), figure/table numbering, claims, counts,
   abstract, title, institutional wording, and bibliography.
7. **Proofreading and submission formatting.** Confirm university requirements,
   registered title, department wording, date format, signatures/approval pages,
   and any Hebrew frontmatter requirements.

### Later, not thesis-critical unless explicitly reprioritized

- public database UI/API and cloud deployment;
- image/source rights audit for public release;
- Zenodo deposit and DOI;
- database-construction article;
- expanded PNAS-oriented thesis article;
- additional rulers or out-of-distribution objects such as the White Obelisk and
  Balawat bronze plates.

## 13. Recommended exact next action

Start with a read-only source-holdout feasibility audit, not another expensive
training run. Create a table by ruler, source prefix, relief group/component, and
split. Determine which of these designs is statistically possible:

1. leave-one-source-out;
2. source-balanced test subset;
3. BM-only within-source evaluation for rulers with BM coverage;
4. paired reweighting or matching by source;
5. if none are adequate, a documented limitation plus foreground/background
   ablation.

Only after choosing a defensible design should new model runs begin. Preserve the
current 664-image frozen baseline and store any new manifest/config as a new,
hashed experiment version.

Suggested opening instruction to the next session:

> Read `THESIS_HANDOFF.md` completely. Preserve all dirty worktrees. Begin with a
> read-only source-holdout feasibility audit on the frozen 664-image manifest.
> Explain the feasible experimental options in simple concrete language before
> launching new training. The MA thesis is the priority; database publication and
> articles are later tracks.

## 14. Commands and environment notes

### Python and GPU

Use the project virtual environment:

```bash
cd /home/kostya/projects/thesis-assyrian-relief
.venv/bin/python ...
```

PyTorch previously detected one CUDA device: NVIDIA GeForce RTX 3060 Laptop GPU.
DINOv2 is cached under `/home/kostya/.cache/torch/hub`.

Useful verification commands:

```bash
.venv/bin/ruff check scripts/audit_dataset_shortcuts.py scripts/umap_style.py
.venv/bin/python scripts/audit_dataset_shortcuts.py
.venv/bin/python scripts/summarize_grouped_runs.py
```

### UMAP regeneration

```bash
.venv/bin/python scripts/umap_style.py \
  --config configs/style_dinov2_grouped_seed42.yaml \
  --umap-fit-split train \
  --umap-plot-splits train val test \
  --html-out outputs/grouped_seed42/umap/relief_umap_fit_train_all_splits.html \
  --csv-out outputs/grouped_seed42/umap/relief_umap_fit_train_all_splits.csv \
  --png-out outputs/grouped_seed42/umap/relief_umap_fit_train_all_splits.png
```

### PDF compilation

The thesis compiled successfully with Windows MiKTeX, not WSL LaTeX. The sequence
was:

1. `pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf main.tex`
2. `biber --input-directory output/pdf --output-directory output/pdf main`
3. `pdflatex` twice more.

Executables were under:

`C:\Users\chend\AppData\Local\Programs\MiKTeX\miktex\bin\x64\`

When editing or inspecting the PDF in a new session, use the available PDF skill
and follow its operation-marker and render/visual-QA requirements. Poppler's
`pdftoppm.exe` and `pdfinfo.exe` are available in the same MiKTeX directory.

### Tooling caveat

The Windows filesystem sandbox helper intermittently returned
`helper_unknown_error`. WSL commands with correctly scoped approval worked. Use
`apply_patch` for edits and preserve unrelated changes. Do not work around
permission failures with broad or destructive shell operations.

## 15. Communication preference

The user found abstract project-management terminology overwhelming. Use simple,
concrete explanations and define necessary technical terms. Prefer short numbered
next actions and explain what each action will change. Do not ask the user to do
work that can be completed locally. Ask only for choices, credentials, specialist
judgment, or information that cannot be recovered from the repositories.

The current normal `high` reasoning setting is adequate for implementation and
routine analysis. Extra-high is most useful for the final whole-thesis argument,
consistency, and submission review.
