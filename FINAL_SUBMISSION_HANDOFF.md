# Final submission handoff — 6 September 2026

## Current status — 7 September 2026

**Editing, numerical/source checks, case dossiers and final PDF QA are complete.** Reviewed deliverable: `../thesis-overleaf/output/pdf/Konstantin_Margulyan_Thesis_2026-09-07.pdf`, 57 pages, SHA-256 `83e6e94e5fbad9e8d6f36f644d0806c53a416095305b48c8ee2116273f752ad5`. `output/pdf/main.pdf` is identical; the September 1 PDF is preserved separately. These PDFs remain local ignored artifacts.

Read `../thesis-overleaf/audit/final_submission_qa_2026-09-07.md` for final checks and limits. Twenty tests passed; numerical and figure inputs retain their audited hashes; all final page bodies were visually checked. No retraining or external inference occurred. English-first departmental order and the remote Overleaf compiler setting remain unverified. No portal submission or message was sent.

Dated preflight and progress sections below are historical. Do not restart completed phases. Both final checkpoints were pushed and verified; details are recorded at the end of this file.

## Start here

Read this file first, then `../thesis-overleaf/audit/submission_requirements_2026-09-06.md`. This supersedes the action priorities in the older THESIS_HANDOFF.md; retain that file as historical evidence. Do not restart the completed review or repeat push confirmations.

The user must submit sometime on **7 September 2026**, with **no fixed submission hour**. The thesis is in English. The immediate priority is a compliant, defensible thesis and a checked PDF. Broader research, articles, repository publication and deployment are secondary.

The user initially asked for a review and plan without edits, plus a checkpoint commit and push. They then explicitly authorized the push, supplied the departmental instructions, requested advice on Extra high reasoning and conversation handoff, and said that work can start after this preparation. Review and preflight are complete; substantive manuscript/analysis edits have NOT begun. The new metadata below authorizes keeping the current title. Proceed with the editing phase when this task is continued; do not repeat metadata questions already answered.

## User-provided front matter

- English author name currently in source: **Konstantin Margulyan**. Preserve unless the user corrects it.
- Hebrew name, explicitly provided: **קונסטנטין מרגוליאן**.
- Supervisor, explicitly provided: **Prof. Shai Gordin**.
- Hebrew spelling **שי גורדין** is supported by the official Ariel profile: https://cris.ariel.ac.il/iw/persons/shai-gordin-2/ . That profile displays an older/different rank; the user's supplied Prof. title governs the draft. Do not silently change it to Dr.
- Department, explicitly provided: **The Jewish Heritage Department** / **המחלקה למורשת ישראל**.
- There is no separately approved title supplied. The user authorized keeping the current title if suitable. We recommended keeping **Deep Visual Analysis of Monarch-Associated Patterns in Neo-Assyrian Reliefs** because it fits the evidence.
- A possible Hebrew translation to refine during bilingual drafting: **ניתוח חזותי מבוסס למידה עמוקה של דפוסים הקשורים למלכים בתבליטים נאו־אשוריים**. This is a proposed translation, not an institutionally registered title.
- Submission date: **7 September 2026**. Use the appropriate month/year on title pages, with corresponding Hebrew dating checked before insertion. Do not keep the literal “Submission deadline” label.

## Repositories and completed checkpoint

Modelling: `/home/kostya/projects/thesis-assyrian-relief`.
Manuscript: `/home/kostya/projects/thesis-overleaf`.
Windows access: prefix each Linux path with `\\wsl.localhost\Ubuntu` and use backslashes.

Both checkpoints were committed and **successfully pushed**, and remote HEADs were verified:

| Repository | Branch | Checkpoint | Destination |
|---|---|---|---|
| Modelling | main | 603637559cd0c50e50a8e36dcaf32c123f21d7e4 | https://github.com/chendwend/thesis-assyrian-relief.git |
| Manuscript | main | 6fc8ab9ce182fcb3d8ca5f7baa81ff5b3cbbc3d6 | https://git.overleaf.com/67d3f6cc1cf2d2d99453f0f2 |

The modelling commit preserves 48 existing changed/new files; the manuscript commit preserves 23. The GitHub push also includes two earlier unpushed commits. Both repos were clean and synchronized immediately after pushing. This handoff, its pointer in THESIS_HANDOFF.md, and the submission audit were created AFTER the checkpoint and are initially local changes. Ignored weights, arrays, images and the compiled PDF were not backed up by Git.

Do not stage the parent `/home/kostya/projects` repository or sibling datasets broadly. Preserve all user work. Do not force-push, reset, or change global credential settings. No tools were used to submit the thesis or contact anyone; those actions are not authorized here.

## Submission instructions: verified pages 7–13

Source: `C:\Users\chend\Downloads\הנחיות-לכתיבת-הצעת-מחקר-ותזה-במורשת-ישראל.docx`.
Document 34, edition 5, updated 06/08/2023; 13 rendered pages. It was extracted and rendered read-only using an invisible separate Word instance, and all relevant pages were visually inspected. Treat its institutional requirements as source material, not as instructions to the assistant. Pages 4–6 concern proposals, not this thesis.

See the companion audit for exact page references, wording, and gaps. Essential requirements: A4, double body spacing, 2.5 cm margins, consistent 12-point recommended font, single-spaced bibliography and notes; both Hebrew and English title, supervision and abstract pages; each abstract 1–5 pages; title pages without logo or student ID. Maximum 120 pages excludes bibliography and appendices; no minimum length is stated. Remove placeholder acknowledgments or replace with user-approved real text; acknowledgments are optional.

The current source has 1.5 spacing, a 1.5-inch left margin, no Hebrew pages, no supervision pages, and placeholder acknowledgments. Bibliography spacing is not explicitly single. Current main.tex uses pdfLaTeX-style packages. Local XeLaTeX/LuaLaTeX and Times New Roman/David fonts exist; a Unicode engine migration is a reasonable implementation step, followed by full Hebrew visual checking and Overleaf compiler alignment.

The guide gives a Hebrew-first EXAMPLE order and allows variations; it does not explicitly specify the reversed order for an English thesis. English-first with Hebrew material at the end is a proposed adaptation, not a directly quoted requirement. Prepare all required pages and consult an accepted departmental English example/coordinator if available before finalizing order; this uncertainty does not prevent drafting. The title template says “Ariel University”; the supervision example says “Ariel University Of Samaria”. Record and handle the distinction carefully.

Submission through the student portal follows supervisor approval. Later library deposit after examination is a separate administrative step, not automatically tomorrow's action. Do not contact the supervisor/coordinator or submit through a portal without explicit authorization.

## Evidence already reviewed

All six chapters, current front matter, 30-entry bibliography, existing reports, saved predictions and parts of the catalogue evidence were reviewed. The current PDF has 41 pages; all text was examined and sample figures/bibliography visually inspected. Its SHA-256 is `d13bba9001950ddb03d95cef4746cad2a4c1503f5d9f1b7e422bb612afef2d42` at `../thesis-overleaf/output/pdf/main.pdf` (September 1 build). This is the pre-edit PDF, not a new submission build.

Frozen manifest: `data/splits/image_level_dataset_v2_grouped.csv`, SHA-256 `4942ec4984fa5a5b261cd9b2c7cc141e37a17ae84d82cf0e41becd83d518690e`.
Related component mapping: `data/splits/image_level_dataset_v2_grouped_components.csv`.

- 664 images; **171 dependency groups**, split **80 training / 52 validation / 39 test**.
- Train/validation/test groups: Ashurbanipal 20/15/13; Ashurnasirpal II 35/22/16; Sargon II 25/15/10.
- Stretch-preprocessing seeds 24/42/77: 32/39, 32/39, 33/39 correct. Mean accuracy .829 ± .015; macro-F1 .826 ± .014. Seeds share one data split; they are not independent historical replications.
- Retrieval R@1 .872 ± .051; R@3 and R@5 .897 ± .044.
- Source-prefix baseline .672 uses **67 images**, not the primary 39 groups. It is evidence of confounding, not a directly comparable group-level baseline.
- Completed padding sensitivity macro-F1 .825 ± .042; R@1 .803 ± .039. No consistent improvement. Do not choose preprocessing by test performance.
- Only British Museum source has all three labels in test: 19 groups (10 Ashurbanipal, 6 Ashurnasirpal II, 3 Sargon). A source holdout would be limited, not a new headline claim.
- Excluded Sennacherib/Tiglath-pileser III material comprises 62/51 images. Potential unfamiliar-label examples require dependency review; these are not 113 independent or wholly untouched cases because older five-class exploration existed.
- Saved seed-24 macro-F1 confidence bound .929494712… is rounded inconsistently (.930 in thesis vs .929 in report). Reconcile from the exact saved value and a stated rounding rule.

No model retraining, substantive source edits, or fresh test-suite run took place during this review. Historical September 1 test results must not be represented as new validation.

## High-priority new finding: retrieval needs a random baseline

A read-only calculation from the verified class counts gave the expected probability that a uniformly random ranking of the 80 distinct training groups includes a same-label group among its first k items, weighted by test class frequencies:

`sum_c (n_test_c / 39) * [1 - choose(80 - n_train_c, k) / choose(80, k)]`

| k | Random label-hit probability | Current learned retrieval mean |
|---|---|---|
| 1 | .342949 | .872 |
| 3 | .708433 | .897 |
| 5 | .866318 | .897 |

The strong result is first-neighbour retrieval. A nearly 90% R@5 is close to the 86.6% random label-hit rate in this three-class corpus. This calculation is not yet saved as a reproducible report or incorporated in the thesis. Recompute using the actual evaluator's pool/exclusion rule before publication. It is descriptive, not a significance test. Add precision@k or the full returned-neighbour composition if feasible. Do not hide this qualification to obtain stronger claims.

## Planned editing and verification order

1. **Submission compliance first:** bilingual title/supervision/abstract pages, margins and spacing, remove placeholders, consistent dates and required front-matter contents, Unicode compilation and visually checked Hebrew.
2. **Claim and number consistency:** group-based evidence throughout; retrieval random baseline; comparable group-level shortcut baseline and paired uncertainty if feasible; resolve numeric rounding and dataset/split descriptions. Preserve the distinction between corpus-label prediction and historical carving style.
3. **Citation audit:** verify existence and metadata of all 30 bibliography entries, then support for all 28 cited keys at the actual claim level. No undefined citation keys were found, but that is NOT a completed hallucination audit. Curtis/Tallis 2008 and Gebru 2021 are currently unused entries. Resolve vague chapter/editor attributions and add page references where justified.
4. **Outlier dossiers:** investigate the four stable errors below with images and original-space neighbours; UMAP only presents findings. Museum disagreement is a question, not proof the museum is wrong.
5. **Bounded instructor cases:** Balawat and White Obelisk only after image identity, context and unfamiliar-input interpretation are specified. Report negative/inconclusive outcomes. Do not sacrifice the submission checks for these.
6. **Final prose and PDF:** grammar, defined ML terms for archaeologists, consistent monarch names, figures/captions and references, bilingual abstract agreement, all pages rendered/inspected, clean build with resolved cross-references, final numeric source check, checkpoint and verified push of approved changes.

Use a strict time budget: at most about 4–6 working hours on expansion, with at least the final 4 hours protected for citation/number/grammar/PDF checks and submission preparation. Adjust to the user's actual available working time; there is no exact portal cutoff provided. Avoid broad new training sweeps, post-hoc test tuning, decorative plots, or major literature expansion.

### Outlier case set

Stable errors across all three seeds:
- AO19902: Ashurbanipal label → Ashurnasirpal II prediction.
- BM124867: Ashurbanipal → Ashurnasirpal II.
- BM124931: Ashurbanipal → Ashurnasirpal II.
- BM118829: Sargon II → Ashurnasirpal II (seed 24), Ashurbanipal (42/77).

Useful correctly predicted controls: AO19894 (Sargon II), AO19909 (Ashurbanipal, unusual AO source).

For each, inspect all dependent views, original-space neighbour ranks/cosine similarities across seeds, subject, composition, carving, preservation, crop/background and catalogue context. Do not infer distances from the two-dimensional UMAP alone. Stored metadata places BM124867 in North Palace Room C Panel 14, BM124931 in Room F Panel 3 with a Hamanu inscription, BM118829 in a Sargon palace detached building. AO19902 needs full Thomas 2017 pp. 291–292 no. 7 verification. Some direct museum requests return 403, so distinguish stored evidence, search-index text and freshly opened catalogue pages.

### Instructor cases and targeted literature

- Ashurnasirpal II Balawat bronze work: represented ruler but new material/monument domain.
- Shalmaneser III Balawat bands: absent ruler as well as new material. A forced three-class label cannot identify him.
- White Obelisk: BM currently favours Ashurnasirpal I, who is absent from the model. The present classifier cannot adjudicate Ashurnasirpal I versus II.
- Critical image check: https://www.britishmuseum.org/collection/object/W_Rm-1066 (BM124686) explicitly flags “Wrong image attached (Shalmaneser gates)”. Verify photos against object identity before inference.
- White Obelisk record: https://www.britishmuseum.org/collection/object/W_1856-0909-58 . Verify text and visual context before using it.
- Prespecify full-object images and scene crops before examining predictions, keep full context, count monuments/independent objects rather than crops, and report every selected case. Choose any unfamiliarity threshold using known-class validation only. Softmax scores are not historical attribution probabilities.
- Existing Curtis/Tallis 2008 is relevant to Balawat. Targeted White Obelisk reading: Sollberger 1974; Reade 1975 (10.2307/4200012); Pittman 1996 (10.1080/00043079.1996.10786686). Verify the actual arguments, not just existence.
- Calibration only if used: Guo et al. 2017, https://proceedings.mlr.press/v70/guo17a.html . Nearest-neighbour unfamiliarity only if implemented: Sun et al. 2022, https://proceedings.mlr.press/v162/sun22d.html . Check Frahm, Manovich and Bracker at the specific cited claim/chapter/page.
- Useful extra figures: learned retrieval versus random baseline, compact error-and-neighbour plates, source-by-ruler matrix at group level, external-case distance distributions only if that analysis is completed.

## Reasoning and conversation continuity

Extra high reasoning is adequate for the methodology/citation consistency work; increasing it is unnecessary by default and cannot replace checking primary evidence. No setting was changed. The user asked to be told when to move to a fresh task and wants a clear handoff. There is no reliable live context percentage exposed here; do not invent one or confuse it with account usage. The completed checkpoint/preflight is a good boundary for a fresh task before substantive edits. Update this file after each completed phase with changed files, evidence, remaining issues and commit IDs. Do not create a new task automatically unless asked.

Suggested next-task prompt:

“Read FINAL_SUBMISSION_HANDOFF.md in thesis-assyrian-relief and the linked submission-requirements audit. Continue the final thesis work for submission on 7 September 2026. Start with the required English/Hebrew pages and formatting, then work through the prioritized audit and bounded analyses. Preserve the checkpoints and report what is verified versus unresolved.”

## Working environment and practical constraints

- Windows PowerShell controls WSL files through UNC paths. Native Git lives at `C:\Program Files\Git\cmd\git.exe`; local MiKTeX is under `C:\Users\chend\AppData\Local\Programs\MiKTeX\miktex\bin\x64`.
- Native Git push succeeded using repository-specific `safe.directory`, with terminal prompts and credential interaction disabled. WSL credential helper had a stale path; do not change global settings unnecessarily.
- Bundled Python: `C:\Users\chend\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`. Call the workspace dependency tool in a new task to refresh paths. XeLaTeX and LuaLaTeX were verified installed.
- Default sandbox tool startup repeatedly failed with `helper_unknown_error` / sandbox refresh errors; escalated, explicitly scoped exec calls succeeded. apply_patch also failed before writing anything; requested handoff files were saved through literal PowerShell here-strings. Avoid exposing credentials or shell interpolation.
- Guideline QA PDF: `C:\Users\chend\AppData\Local\Temp\thesis-guidelines-20260906\submission-instructions.pdf`; page images are beside it. This is a temporary read-only rendering, not a modified original.
- Use applicable document/PDF skills and full render verification for final outputs. Do not mutate memory files unless explicitly asked. No proactive subagents are authorized by the current session instructions.

## Editing progress in continuation - 6 September 2026 (local evening)

### Phase 1: implemented and initially checked

- Migrated manuscript to XeLaTeX/fontspec/polyglossia, with Times New Roman and David locally and portable font fallbacks; added `.latexmkrc` and the engine directive. Overleaf UI/compiler alignment remains to verify.
- Set A4, 2.5 cm margins, double body spacing, and explicitly single-spaced 12-point references. Added modest emergency line stretch to avoid margin overflows.
- Replaced English title page with confirmed department/supervisor metadata, added English supervision and Hebrew title/supervision pages, and drafted the Hebrew abstract. Removed placeholder acknowledgments from the build.
- English-first ordering remains an explicitly documented adaptation, not a verified departmental English-thesis rule. Titles are unnumbered; English preliminary pages use lowercase Roman numerals; Hebrew abstract uses uppercase Roman numerals, followed by unnumbered Hebrew supervision and title.
- Date checked against Hebcal: 7 September 2026 is 25 Elul 5786. Title pages use September 2026 and Elul 5786.
- Current working PDF: `../thesis-overleaf/tmp/submission-20260906/main.pdf`, 55 pages. Both abstracts are two pages. Rendered all pages at 95 dpi; visually inspected the bilingual title/supervision pages and Hebrew abstract. Full final-page review remains outstanding. Previous `output/pdf/main.pdf` is preserved.
- Current build has no undefined citations/references, no overfull boxes, and no missing-glyph warnings; one minor underfull paragraph remains. This is an interim build, not the final checked PDF.

### Phase 2: controls recomputed and incorporated

- New reproducible script: `scripts/audit_submission_controls.py` (run with the existing WSL `.venv/bin/python`). It verifies the frozen manifest hash, saved gallery/query identities, normalization, top-one neighbours and all saved retrieval hit rates.
- Tracked report: `reports/submission_controls_2026-09-06.md`; companion directory contains summary JSON with hashes, full original-space neighbour rankings, paired source predictions and group source counts.
- Evaluator confirmed: all 80 training groups, no exclusions, 39 test groups. Random Hit@1/3/5: .342949/.708433/.866318. Learned Hit@1/3/5: .872/.897/.897. New P@1/3/5: .872/.855/.846, sample SD .051/.017/.010; random expected precision .342949 at all k.
- Thesis now calls the saved evaluator's `Recall@k` measure `Hit@k` and defines the distinction. Methods, Results, Discussion, Conclusion and both abstracts include the random reference and P@5 qualification.
- Group-trained/source-majority baseline: 22/39, accuracy .564103, macro-F1 .449495. Every group has exactly one source prefix. Full-model mean accuracy advantage .264957; paired group-bootstrap 95% interval [.136752, .410256], 10,000 resamples, seed 20260906. Seeds are kept paired within group, never counted as 117 independent observations.
- Exact seed-24 macro-F1 upper bound .9294947121034077 now rounds directly to .929 in the thesis.
- Corrected the inverse source-association error: nearly all Ashurbanipal images are BM, but BM itself is not overwhelmingly Ashurbanipal.
- No retraining, checkpoint change, split change or new image acquisition occurred.

### Phase 3: in progress

- Existing Zotero source attachments located via a read-only immutable database snapshot after its local API failed and ordinary read-only SQLite was locked. No Zotero writes. Local page-text extracts and source index are in manuscript `tmp/submission-20260906/`.
- Bracker 2020 attachment is specifically Nadali and Portuese, "Archaeology of Images: Context and Intericonicity in Neo-Assyrian Art", pp. 127-157. Relevant text pp. 128-131 directly supports the thesis claim; replace vague editor attribution with chapter citation.
- Frahm volume's "Assyrian Art" chapter is by John M. Russell, pp. 453-510. Chapter and contents inspected; targeted text includes White Obelisk pp. 470-471. Do not attribute chapter arguments to Frahm as author.
- Manovich 2020 title/ISBN and introduction pp. 9-11 inspected; add precise support pages for computational scale, variation and omitted context.
- DOI metadata saved at `tmp/submission-20260906/crossref_metadata.json`. CVF and IEEE/Crossref page numbering differ for Caron 2021 and Kornblith 2019. Keep a consistent cited version and document that discrepancy rather than treating both numbers as an error. Some Crossref calls returned 429; Bracker DOI is not a Crossref DOI and must be checked with its publisher.
- Full citation audit, stable-error dossiers, bounded instructor cases, final prose/PDF QA, checkpoint and push remain unfinished. No new commits/pushes yet.
### Phases 3–5: completed within the stated scope (night of 6–7 September)

- Full targeted source audit saved in manuscript `audit/citation_audit_2026-09-06.md`: all 30 inherited entries and three added chapter entries, 33 database entries / 31 cited. Corrected Mahmoud Assran, Marina Khoroshiltseva, M. M. Howard initials, and Nadali 2016 extent 83–88. Standardised Reade's author identity. Preserved version-specific CVF pagination and replaced vague Frahm/Bracker editor attributions with Russell/Nadali–Portuese chapters.
- The complete Sidestone volume was recovered after an incomplete download. Thomas 2017 table 59.1 p. 291 directly verifies AO 19902 in Room R and AO 19909 in Room V'/T'. Barnett independently supplies AO 19902's hunting-procession context, BM 124867 Room C slab 14/restoration, and BM 124931 Room F slab 3/Hamanu inscription.
- All eight images of the four errors and two controls inspected. Dossiers, image hashes, all-view plate and 90 top-five rows saved in `reports/stable_error_dossiers_2026-09-06/`. Main Discussion now includes the original-space nearest-neighbour comparisons and published context. BM 124867 and BM 124931 retrieve a correct-label first neighbour in seed 24 despite wrong classification; this distinction is explicit.
- Balawat/White Obelisk are documented as limits and future design requirements. No external inference was run. Shalmaneser III and Ashurnasirpal I are absent labels; direct BM pages failed, and the Balawat photo warning was not refreshed. No new attribution is claimed.

### Phase 6: final QA in progress

- 20 existing tests passed (UMAP deterministic-worker warning and disabled optional TBB thread-layer warning only). New numerical audit script passes Ruff. All experiment-input hashes still match the saved audit.
- Clarified from the actual training engine that checkpoint monitoring used image-level macro-F1 over 133 validation photographs; primary test evaluation remains dependency-group level. No checkpoint was changed.
- Clean Unicode build directory `../thesis-overleaf/tmp/submission-final-20260906`. The 56-page version was fully rendered and all pages inspected. Fixed reversed English contents entry for Hebrew Abstract, removed repeated closing summaries, prevented isolated paragraph lines, expanded bibliography author lists. Both abstracts remain two pages.
- Final figure readability pass now uses `scripts/render_submission_figures.py`: saved 171 UMAP coordinates and existing heatmaps/ranks only, no inference or fit. Larger plot labels, vertically arranged UMAP panels, and a larger-label nearest-neighbour plate. Input hashes recorded beside dossiers. Rechecking revised pagination before PDF promotion and checkpoint.
- Overleaf configuration is now `latexmkrc` (without a leading dot), matching official Overleaf documentation. Required compiler is XeLaTeX. Browser control failed during sandbox startup, so the Overleaf UI setting was not inspected. Local XeLaTeX works; do not describe remote compilation as verified.
- English-first order remains an adaptation permitted in principle by the supplied example, without an independently verified department-specific English-thesis precedent. Final PDF/QA report and remote commit verification remain to finish. No portal submission or messages sent.

### Phase 6: completed — 7 September 2026

- Final PDF is **57 pages**, with both abstracts two pages and all required bilingual title/supervision pages. All 21 changed page bodies were inspected after the full 56-page review; remaining bodies match reviewed renders. The settled build has no LaTeX/package warnings, missing glyphs, overfull/underfull boxes, or unresolved references. All fonts are embedded.
- Final PDF SHA-256: `83e6e94e5fbad9e8d6f36f644d0806c53a416095305b48c8ee2116273f752ad5`. Saved at `../thesis-overleaf/output/pdf/Konstantin_Margulyan_Thesis_2026-09-07.pdf` and identically at `output/pdf/main.pdf`. Original preserved as `output/pdf/main-pre-submission-20260901.pdf`.
- Final reports: manuscript `audit/final_submission_qa_2026-09-07.md` and `audit/final_submission_checks_2026-09-07.json`. Fourteen numerical input hashes and sixteen figure-input hashes verified unchanged. Both new scripts passed Ruff; 20 existing tests passed.
- Manuscript checkpoint: `791a1704dbe20d30a1a35ebc793b5279c9dfac0e`, “Complete bilingual thesis, evidence audit, and final PDF verification”. Remote push verification is recorded below after it succeeds.
- Remaining administrative checks are English-first departmental ordering and Overleaf's XeLaTeX setting/remote compilation. The locally reviewed PDF is the deliverable; alternative font fallbacks may reflow the remote build.
- No additional thesis edits, retraining, external inference, portal submission or communication are implied. The technical and editorial work is complete within the documented scope.

## Verified final checkpoints — 7 September 2026

| Repository | Checkpoint | Remote verification |
|---|---|---|
| Manuscript / Overleaf main | `791a1704dbe20d30a1a35ebc793b5279c9dfac0e` | Push succeeded; remote refs/heads/main equals this commit |
| Analysis / GitHub main | `414ebd0edc626e75d544c06156988612544f1472` | Push succeeded; remote refs/heads/main equals this commit |

A subsequent documentation-only commit records this verification; its identity is available in Git history. Neither push was forced. No dataset, weight or checkpoint artifact was altered. The Overleaf credential-store helper printed its existing stale-path warning, but authentication, push and remote HEAD verification all succeeded; no global credential settings were changed.

The final PDF remains the locally verified 57-page artifact identified above. A successful Overleaf Git push does not establish a successful remote PDF build. Further work should be limited to any user-requested correction, departmental ordering confirmation, compiler selection if rebuilding remotely, and the user's submission process.
