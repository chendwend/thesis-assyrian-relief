# Stable-error dossiers — 6 September 2026

## Evidence and selection

The four groups were selected because their group predictions were wrong in all three frozen runs (seeds 24, 42, 77). AO 19894 and AO 19909 were selected in advance as correctly predicted controls. These are descriptive, post-evaluation case studies; they are not a new test set or a causal explanation experiment. No model, split, image, or label was changed.

`image_inventory.json` lists the eight reviewed photographs and SHA-256 hashes. `all_views.jpg` displays every one. `nearest_neighbours.jpg` compares the four errors with their first training neighbour in each seed; its first-view displays represent group-averaged embeddings, not individual-image distances. `top_five.csv` contains all five neighbours for all six cases and three seeds (90 rows). The full 80-neighbour ranking and the reproducible evaluator check are in `../submission_controls_2026-09-06/` and `scripts/audit_submission_controls.py`.

Cosine similarities come from each seed's learned projection. Their scales must not be compared as calibrated probabilities or used to invent an unfamiliarity threshold. Neighbours were selected in the original embedding space, not from UMAP coordinates.

## Findings

- **AO 19902, Ashurbanipal → Ashurnasirpal II in all runs.** One photograph shows a horseman fragment with a pale museum wall, oblique viewpoint and missing continuation. First neighbours: BM 124586 (0.954, seed 24), BM 124575 (0.367, 42), BM 124586 (0.868, 77), all Ashurnasirpal II. These depict different figure subjects; broad proximity does not establish a shared scene type or carving school. The stored Louvre record supplies Nineveh but no room. Crucially, Thomas 2017 table 59.1 p. 291 no. 7 directly supplies North Palace Room R, and Barnett 1976 pp. 48–49/pls. XXXIX–XL identifies slab 9 in a hunting procession. This evidence was inspected and is stronger than treating the accession as an unexplained Louvre exception. The catalogue's generic war-scene description and Barnett's hunting context should remain distinct, not silently harmonised.
- **BM 124867, Ashurbanipal → Ashurnasirpal II in all runs.** Two views: close royal-head detail and a wide chariot/lion scene, with different scale and illumination. Barnett p. 37/pl. VIII identifies Room C slab 14 and local restorations. First neighbours: BM 118918 (Ashurbanipal, 0.930), BM 124575 (Ashurnasirpal II, 0.899), BM 124537 (Ashurnasirpal II, 0.917). Seed 24 thus retrieves a correct-label neighbour despite a wrong class prediction. View aggregation can conceal differences in the photographed information; no per-view causal attribution is claimed.
- **BM 124931, Ashurbanipal → Ashurnasirpal II in all runs.** One photograph emphasises the dense city-assault register, with a dark surround and incomplete wider composition. Barnett p. 40/pl. XVII directly identifies Room F slab 3 and the Hamanu inscription. First neighbours: BM 118918 (Ashurbanipal, 0.942), BM 124575 (Ashurnasirpal II, 0.909), BM 124537 (Ashurnasirpal II, 0.906). Its shared neighbour sequence with BM 124867, despite different subjects, cautions against interpreting the neighbourhood as a literal iconographic match.
- **BM 118829, Sargon II → Ashurnasirpal II (24), Ashurbanipal (42/77).** Two photographs show different extents/illumination of figures among trees. The stored BM record places the relief in a detached building of Sargon II's palace and describes bird hunting; this record was read locally, not refreshed live. First neighbours: BM 124537 (Ashurnasirpal II, 0.935), BM 118917 (Ashurbanipal, 0.542), BM 118918 (Ashurbanipal, 0.586). The failure is stable but the alternative attribution is not. No competing historical attribution follows.
- **AO 19894, Sargon II, correct throughout.** A head fragment on a pale wall; stored catalogue gives a tribute bearer from Sargon's palace, façade/panel 21. First neighbours are Sargon II in every seed: AO 19870 (24), AO 22197 (42/77). Existing occlusion findings localise part of the decision around the eye/upper face; this is a model-sensitive region, not a proved diagnostic trait.
- **AO 19909, Ashurbanipal, correct throughout.** A broken two-register chariot scene on a pale wall. Thomas table 59.1 p. 291 no. 14 gives Room V'/T'. First neighbours BM 124801 (24), BM 136774 (42), BM 124938 (77), all Ashurbanipal. This is a useful counterexample to a deterministic AO→Sargon shortcut, but it does not remove corpus-wide source confounding.

## Catalogue provenance

Stored records: `../assyrian-relief-dataset/data/processed/canonical_objects.jsonl`. Original URLs:
- AO 19902: https://collections.louvre.fr/ark:/53355/cl010123112
- AO 19909: https://collections.louvre.fr/ark:/53355/cl010123119
- AO 19894: https://collections.louvre.fr/ark:/53355/cl010122717
- BM 124867: https://www.britishmuseum.org/collection/object/W_1856-0909-16_3
- BM 124931: https://www.britishmuseum.org/collection/object/W_1856-0909-17-18_2
- BM 118829: https://www.britishmuseum.org/collection/object/W_1851-0902-34

Published context was checked in the user's Barnett PDF and the publisher's complete Thomas volume. See the manuscript's `audit/citation_audit_2026-09-06.md`. No new image was downloaded for these plates. Existing photographs retain their Louvre/British Museum provenance.

## Bounded instructor cases

No Balawat or White Obelisk inference was run. Shalmaneser III and Ashurnasirpal I are absent from the three-class model. Balawat also changes medium and monument, and the flagged BM photograph identity was not refreshed successfully. The thesis now explains these limitations and defines the requirements for a future comparison. This does not constitute a negative performance result, because no such test was performed.

## Final presentation pass — 7 September 2026

`../../scripts/render_submission_figures.py` regenerates final thesis figures from saved inputs. It increases neighbour-label size and records hashes in `figure_input_hashes.json`. The first stored view remains the plate representative; all dependent views remain in `all_views.jpg` and `image_inventory.json`. No neighbour, cosine score or source photo changed.

The script arranges existing 171 UMAP coordinates vertically and renders the two existing occlusion maps with readable labels. It does not fit UMAP or run inference. Occlusion limits retain the 98th percentile of absolute map values, zero-centred normalisation and 0.4 overlay alpha. All 16 figure-input hashes were reverified, and final figures were visually checked.
