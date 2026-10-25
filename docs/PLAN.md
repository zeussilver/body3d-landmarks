# body3d-landmarks — plan

Portfolio project covering two problems a 3D body-scanning product has to solve:

1. **Landmark detection on 3D body scans**, with a measured improvement in
   generalisation across body types.
2. **3D visualisation of body-shape change over time** from mesh/STL data.

**All data is synthetic** (parametric body model + simulated turntable scanner).
Every table, figure and viewer page says so. Nothing here claims
real-scanner accuracy.

Status: started 2026-10-18. **Done** 2026-10-19: phases 1–4 complete (see the progress log).
Published at github.com/zeussilver/body3d-landmarks with a live demo on GitHub Pages.

---

## 1. Body-model licence decision

| Option | Licence | Registration | Verdict |
|---|---|---|---|
| **Anny** (NAVER LABS Europe, `pip install anny`) | Code Apache-2.0; MakeHuman/MPFB2 assets CC0-1.0; SOMA topology Apache-2.0 | none | **Chosen** |
| SMPL / SMPL-X (MPI) | non-commercial research only; commercial via Meshcapade; no redistribution | user must register | rejected |
| STAR (MPI) | non-commercial scientific research only | via MPI site | rejected |
| MakeHuman directly | software AGPL, exports CC0 | none | not needed: Anny wraps the same CC0 assets as a PyTorch model |

Rules that follow from the choice:
- Pin `anny==0.6.1`. Use only the default `anny` topology. Never request the
  `smplx` / `smpl` topologies or install the `anny[smpl]` extra (those assets are
  non-commercial).
- Never commit body-model files or the Anny cache (`~/.cache/anny`, about 741 MB).
  Generated meshes and datasets stay out of the repo; scripts regenerate them.
- Cite the Anny paper (arXiv 2511.03589) in the README.

Sources: [Anny repo](https://github.com/naver/anny),
[Anny paper](https://arxiv.org/abs/2511.03589),
[SMPL-X licence](https://smpl-x.is.tue.mpg.de/modellicense.html),
[STAR repo](https://github.com/ahmedosman/STAR),
[MakeHuman licence FAQ](https://static.makehumancommunity.org/oldsite/faq/what_changed_regarding_the_license_in_2020.html).

### Feasibility spike (2026-10-18, Anny 0.6.1, CPU, M4 Pro)

- Default topology: 13,718 vertices, 27,420 triangles; z-up, metres; rest pose is an A-pose.
- Shape controls: 6 phenotype parameters (`gender, age, muscle, weight, height,
  proportions`) plus 254 local controls (`measure-waist-circ-incr`,
  `measure-hips-circ-incr`, `stomach-pregnant-incr`, `l/r-upperleg-fat-incr`, ...).
- `gender <= 0.5` is the male end of the blend (Anny's own sampler uses the boys'
  tables there; mean height 1.76 m vs 1.65 m).
- Built-in WHO-calibrated sampler, adults aged 20–70: 400 bodies in 0.2 s.
  Height p1–p99 = 1.46–1.94 m. Mesh BMI p1/p10/p50/p90/p99 = 14.6 / 18.4 / 26.1 / 36.2 / 39.2.
- "Mesh BMI" = Anny `Anthropometry`: enclosed volume × 980 kg/m³ / height². It is a
  proxy, labelled as such everywhere.
- First model build parses the CC0 assets into a 741 MB cache (about 15 s). The
  default install downloaded nothing extra.

---

## 2. Design

### 2.1 Synthetic bodies (`bodies.py`)
- Sample adults with Anny's `SimpleShapeDistribution` (age 20–70), plus random
  torso/hip/stomach/thigh/bust local controls for shape diversity.
- Reporting strata: **mesh BMI band** (<18.5, 18.5–25, 25–30, 30–35, ≥35, WHO
  cut-offs), **sex** (gender drawn from [0, 0.2] or [0.8, 1]; Anny's gender is a
  continuous blend, binary only for reporting), **height band** (<1.60, 1.60–1.75, ≥1.75 m).
- Small pose jitter (shoulder abduction ±10°, stance ±5°) so armpit/crotch occlusion
  varies. Dropped if Anny's pose API makes it costly (see risks).
- Splits are subject-disjoint with fixed seeds. Test set: 2,000 bodies, 400 per BMI
  band, sex-balanced, so the tails have enough subjects.

### 2.2 Ground-truth landmarks (`landmarks.py`, `landmarks.json`)
16 landmarks in the spirit of ISO 8559-1, each a fixed template vertex index:

vertex (top of head), cervicale, acromion L/R, bust point L/R, waist front, waist
back, waist side L/R, buttock point, crotch, knee L/R (mid-patella), lateral malleolus L/R.

Indices are picked once on the mean adult template with geometric rules, checked
visually, then frozen. Because Anny keeps topology fixed, the same index is the same
anatomical point on every body. That is what makes cross-body-type error meaningful.

### 2.3 Simulated turntable scanner (`scanner.py`)
Turntable body scanners typically capture the subject during one rotation of about a
minute. Simplified model:
- 8 orthographic depth maps at yaw 0°, 45°, …, 315° (known turntable angles),
  256×160 px at 8.5 mm/px (2.18 m × 1.36 m, wide enough for the A-pose arm span).
  Subject facing ±15° and offset ±2 cm from the turntable centre.
- Rasteriser: edge-function test at each pixel centre inside a triangle's bounding
  box, barycentric depth (exact for orthographic projection), z-buffer with
  `np.minimum.at`. Deterministic, easy to mirror in C#.
- Noise: depth σ ≈ 1–3 mm, dropout at grazing angles, random blob holes, 1 mm
  quantisation. Self-occlusion (armpit, crotch, between thighs) comes from the geometry.
- Training renders on the fly in DataLoader workers (augmentation needs the mesh).
  The test set is rendered once with a fixed seed and kept.
- Simplifications stated in the README: orthographic not perspective, noise
  parameters invented, not calibrated against any real device.

### 2.4 Baseline (`baseline.py`)
B0: fuse the 8 depth maps into a point cloud, align the mean template by ICP
(rotation about z, translation, uniform scale), take the nearest scan point to each
template landmark. It does not adapt to body shape, so its error should grow with BMI.

### 2.5 Model (`model.py`, `train.py`)
- Shared small U-Net (about 2 M parameters) per view. Input: normalised depth plus a
  validity mask. Output: 16 heatmaps at 1/2 resolution, soft-argmax for sub-pixel
  2D positions, plus a per-view confidence.
- **Fusion by triangulation.** In an orthographic turntable view at yaw θ, a point
  (x, y, z) lands at u = x·cosθ + y·sinθ, w = z. So 3D comes from a closed-form
  confidence-weighted least-squares solve over the 8 views. It is differentiable,
  so the loss is heatmap MSE (visibility-weighted) plus 3D L1 in mm.
- PyTorch on MPS (`PYTORCH_ENABLE_MPS_FALLBACK=1`). Training sets are 8,000 bodies
  per condition (same size in every condition). Model selection uses the validation
  set only. Each final configuration touches the test set once.

### 2.6 Evaluation (`evaluate.py`)
Per landmark and overall: mean / median 3D error (mm), PCK@10 mm, PCK@20 mm. Each is
broken down by BMI band, sex and height band, with 95% subject-level bootstrap CIs and
n per band. Each run reports its **worst band** and the gap to the in-range band.
Outputs go to `results/*.json` and are rendered into README tables by a script.

### 2.7 Generalisation experiment (headline)

| Run | Training data | Purpose |
|---|---|---|
| E0 | full adult range | upper-bound reference |
| E1 | mesh BMI 18.5–27 only | the gap: narrow train, full test |
| E2 | E1 + **normal-offset inflation**: offset the mesh along vertex normals by a smooth, region-varying amount (−10…+60 mm), landmarks move with it; plus height/sex-balanced sampling | model-free fix that would also work on real labelled scans |
| E3 | E1 + **shape-space extrapolation**: same training subjects re-generated with weight/waist/hip/stomach controls pushed past the training range | model-based fix; optimistic, because test bodies come from the same generator (said in README) |

3 training seeds each. Headline = worst-band error and gap closure for E1 → E2/E3, as
mean ± std over seeds. If E1 shows no real gap, the README says so, and the headline
becomes model vs baseline across bands. No tuning on the test set to rescue a number.

### 2.8 Body-shape change over time (`change.py`, `viewer/`)
- Trajectories: 6 synthetic subjects × 4 scans (weeks 0, 8, 16, 24), weight and
  waist/hip/stomach/thigh controls easing down along an illustrative GLP-1-style curve
  (about −10% mesh mass by week 24). Each scan gets a new rigid placement and small
  pose jitter, as on a real turntable. Illustrative only, no clinical claim.
- Change map, two paths:
  - same topology: per-vertex signed displacement along the week-0 normals (mm);
  - arbitrary STL: own point-to-plane ICP (trimmed, weighted towards low-change
    regions such as head and shins) then signed nearest-point distance via KD-tree.
  The two paths are compared on the synthetic trajectories.
- Circumferences at waist, hip and thigh (5 cm below crotch) over time: own
  mesh-plane slicer, chaining edge crossings into loops. It keeps the torso loop
  (A-pose arms cut the hip plane) and reports both contour length and convex-hull
  length (what a tape measures).
- Own binary/ASCII STL reader/writer (`stl.py`, mirrored in C# in phase 3).
- Viewer: one static page, three.js (pinned version) + `STLLoader` + `OrbitControls`.
  Timeline slider over the 4 scans, diverging colour map with a legend in mm,
  circumference-vs-week chart, "SYNTHETIC DATA" banner. Served with
  `python -m http.server`. Screenshot and GIF go in the README.

### 2.9 Optional phase 3: C# service (do last, cut first)
ASP.NET Core minimal API on .NET 10 (current LTS):
- own STL parser, change map and circumferences;
- landmark inference: C# port of the rasteriser, ONNX Runtime for the exported
  per-view CNN, triangulation in C#;
- xUnit tests, including Python-vs-C# parity.

Needs the .NET SDK installed (not on this machine yet; ask first). Optional S3 in
ap-southeast-2 with `--profile aws-agent`: check `aws freetier get-account-plan-state`
first, ask before creating anything, ask again about clean-up.

### Layout
```
body3d-landmarks/
  README.md  pyproject.toml  docs/PLAN.md
  src/body3d/  bodies.py landmarks.py landmarks.json scanner.py baseline.py
               model.py train.py evaluate.py change.py stl.py
  viewer/      index.html viewer.js
  results/     *.json *.png   (small, committed later)
  tests/       pytest
  dotnet/      (phase 3 only)
```
Environment: `uv` venv, Python 3.12, deps pinned in `pyproject.toml`. No Makefile;
entry points are `python -m body3d.<module>`.

---

## 3. Phases and estimates

| Phase | Content | Days |
|---|---|---|
| P0 | Licence check, spike, this plan | 0.5 (done) |
| P1a | Body sampling + strata, landmark indices, scanner sim, tests | 1.5 |
| P1b | Baseline B0, evaluation harness with bootstrap CIs | 1.0 |
| P1c | Model + triangulation, training E0/E1 (overnight runs) | 1.5 |
| P1d | E2/E3 augmentations, 3-seed final runs, results tables | 1.0 |
| P2 | Trajectories, change maps, ICP, slicer, viewer, screenshot/GIF | 2.0 |
| P4 | README (< 150 lines), every number from `results/` | 0.5 |
| **Core total** | | **≈ 8** |
| P3 (optional) | C# API, ONNX Runtime, xUnit + parity, optional S3 | +2.5–3 |

Machine time: about 12 training runs (4 conditions × 3 seeds) at roughly 1 h each on
MPS, mostly overnight. If a run passes 1.5 h, shrink to 4,000 bodies per condition,
the same for all.

Lean fallback (≈ 4 days) if time runs out: E1 vs E2 only, no E3, 1 seed reported as
such, no pose jitter, screenshot without GIF, no P3.

Schedule: core phases first; P3 only if time allows.

---

## 4. Acceptance numbers

Targets are fixed before any results exist. The README reports actual values even
when a target is missed.

| ID | What | Target |
|---|---|---|
| A1 | Rasteriser vs trimesh ray casting (5 bodies × 8 views, clean) | median abs depth diff ≤ 1 mm; ≥ 99% of pixels hit by both within 3 mm |
| A1b | Render speed, 8 views, CPU | ≤ 40 ms per body |
| A2 | Landmark definitions | L/R pairs mirror within 10 mm on the template; every landmark *observable* (depth-visible within 1 cm, or within 1 px of the silhouette) in ≥ 2 of 8 clean views for ≥ 99% of test bodies. Revised after the first measurement, see log |
| A3 | E0 model, noisy scans, 2,000-body test set | mean error ≤ 15 mm; PCK@20 ≥ 80%; ≥ 30% lower mean error than B0 |
| A4 | Generalisation (headline) | E1 worst BMI band ≥ 1.3× its in-range error (premise check); E2 or E3 closes ≥ 50% of the E1→E0 worst-band gap; in-range error worsens ≤ 1 mm; 3 seeds, mean ± std + 95% CI |
| A5 | Change tools | slicer within 0.5% of analytic perimeter (cylinder, elliptic cylinder); ICP recovers a known rigid transform (≤ 15°, ≤ 50 mm) to ≤ 2 mm RMS; topology vs ICP/nearest-point displacement agree within 3 mm median |
| A6 | Viewer | 4-scan timeline, signed heatmap with mm legend, circumference chart, loads < 2 s locally; screenshot + GIF in README |
| A7 | Hygiene | pytest green in < 60 s (no training); README < 150 lines; every README number traceable to `results/*.json` |
| A8 (P3) | Python vs C# parity | circumference ≤ 0.1 mm; depth maps ≤ 1 mm on ≥ 99.9% of pixels; landmarks ≤ 1 mm |

**Done** = this plan approved, P1 + P2 built, real numbers in the README (E1 vs
E2/E3 per-band table, viewer screenshot/GIF) and tests green.

---

## 5. Risks

- **Synthetic-to-real gap.** Bodies and scanner are simulated; the noise model is a
  guess. The project shows method and engineering, not field accuracy.
- **Premise risk.** E1's gap may be small. Then report it honestly (A4 premise check).
- **Landmark semantics.** A fixed vertex is anatomically consistent in Anny by
  construction, but on very large bodies (bust point, buttock point) it may drift
  from where a human measurer would put it. Said in the README.
- **Anny API drift.** Pinned to 0.6.1. The pose API is untested (pose jitter is
  droppable).
- **MPS gaps.** Some ops may fall back to CPU; budget is in the overnight runs.
- **Arms in the hip plane.** The slicer must pick the torso loop; tested.

## 6. Out of scope / guard-rails

- No public release without an explicit decision.
- No AWS resources without asking first (phase 3 only, ap-southeast-2 only).
- No SMPL/STAR/SMPL-X files, ever; no third-party accounts created for the project.
- No real person's scan data.

---

## 7. Progress log

### 2026-10-18 — P1a done
Built `bodies.py`, `landmarks.py` (+ frozen `landmarks.json`), `scanner.py`, 15 pytest
tests (6.5 s). Numbers from `results/scanner_check.json`:

- **A1 met.** Rasteriser vs trimesh ray casting, 5 bodies × 8 views, 248,396 pixels:
  identical pixel coverage, max depth difference < 1e-10 mm.
- **A1b met.** 25.5 ms per body for 8 views on CPU (27.6 ms in the re-run, which shared the CPU with pytest).
- **A2 mirror part met** (0.0 mm, the template is exactly symmetric).
- **A2 observability: 15 of 16 landmarks met, crotch not met.** On the 2,000-body
  stratified test set every landmark except the crotch is observable in ≥ 2 views for
  100% of bodies. The crotch is short of 2 views for 23.6% of bodies: 2.5% / 5.0% /
  23.5% / 38.3% / 48.8% by mesh-BMI band (<18.5 … ≥35). (Re-measured after the BMI ≥ 15
  floor added in P1b.) Cause: the thighs touch and close the gap
  between the legs. This is a real, body-type-dependent occlusion, so the crotch stays
  in the set and its error is reported separately per band.

Design changes against the approved plan, and why:
- Image 256×128 → 256×160 at 8.5 mm/px: the A-pose arm span (up to about 1.25 m) did
  not fit.
- Point-splat → edge-function rasteriser: the splat version moved each sample to the
  pixel centre along its triangle plane. On convex surfaces that plane lies outside the
  body, so the z-buffer picked depths that were too close: only 85% of pixels within
  3 mm of ray casting (median error 13 mm at grazing angles). The edge-function version
  is exact and 4× faster (108 → 26 ms).
- A2 criterion: depth-visibility alone said the top of the head was visible in 0.99
  views on average. Horizontal cameras see upward/downward-facing points only as
  silhouette extremes, so "observable" now also counts the silhouette. Strict
  depth-visibility is still reported. Changed before any model results existed.
- Landmark rule for the natural waist: excluding leg-weighted vertices put the waist at
  56% of stature (hip level). Now: min convex-hull perimeter of non-arm vertices,
  57–70% of stature (lands at 64%).
- Bodies are centred on the pelvis root joint, not the vertex mean (dense hand and
  face vertices pulled the mean forward).

### 2026-10-18 — P1b done
Built `data.py` (fixed, cached test split), `evaluate.py` (per-landmark and per-band
metrics, 1,000-sample subject-level bootstrap CIs, worst band, gap to the 18.5–25 band),
`baseline.py` (B0). 24 pytest tests green (7.4 s).

- **Test set**: 2,000 bodies, 400 per mesh-BMI band, 1,000 per sex, one noisy scan each
  with a per-body seed. Cached as uint16 mm under `data/test/` (1.5 GB, git-ignored),
  36 s to rebuild on 10 worker processes.
- **Plausibility floor**: the local controls produced adults down to mesh BMI 11.4.
  All splits now reject mesh BMI < 15 (and ≥ 50). The test set was rebuilt before any
  model result existed.
- **B0** (`results/b0.json`, 62 s for 2,000 scans): mean 30.8 mm (95% CI 30.2–31.4),
  median 24.2 mm, PCK@10 13.3%, PCK@20 39.5%.

| run | overall | <18.5 | 18.5-25 | 25-30 | 30-35 | >=35 | worst / in-range |
|---|---|---|---|---|---|---|---|
| B0 | 30.8 | 20.3 | 22.0 | 26.5 | 36.5 | 48.9 | 2.23x (>=35) |
| B0 without snapping | 36.4 | 23.8 | 24.7 | 31.8 | 44.1 | 57.7 | 2.34x (>=35) |

  The rigid template fails where body shape departs from it: from the 18.5–25 band to
  ≥ 35, crotch error goes 19.1 → 99.1 mm and waist-front 18.7 → 103.7 mm. Male 33.6 mm
  vs female 28.0 mm (template gender is 0.5). Height bands are flat (30.4–31.3 mm),
  since uniform scale absorbs stature.
- Consequence for A3: the learned model must reach ≤ 21.6 mm mean (30% below B0).

### 2026-10-18 — P1c done (E0, E1, seed 0)
Built `model.py` (shared per-view U-Net, 1.96 M parameters, heatmaps + per-view confidence,
soft-argmax, closed-form weighted least-squares triangulation) and `train.py` (scans
rendered on the fly (placement and noise depend on worker and body index, so at most 8
variants per body; see the 2026-10-25 note); model selection on the
validation split of the same distribution; test scored once). 29 pytest tests green.

- Training: 10 epochs × 1,000 steps of 8 bodies, AdamW + one-cycle (max lr 2e-3), MPS.
  GPU-bound at about 36 bodies/s (loader delivers 234/s). E0 took 49 min, E1 68 min.
  Mixed precision gave no speed-up on MPS; channels-last was 4× slower. Both left off.
- Triangulation from ground-truth pixels reproduces the 3D points to 0.0002 mm.

| run | overall | <18.5 | 18.5-25 | 25-30 | 30-35 | >=35 | worst / in-range |
|---|---|---|---|---|---|---|---|
| B0 | 30.8 | 20.3 | 22.0 | 26.5 | 36.5 | 48.9 | 2.23x (>=35) |
| E0 (full range) | 1.30 | 1.57 | 1.39 | 1.22 | 1.18 | 1.16 | 1.13x (<18.5) |
| E1 (BMI 18.5–27) | 3.64 | 2.32 | 1.31 | 1.70 | 4.20 | 8.65 | 6.59x (>=35) |

(mean 3D error, mm, 2,000-body test set; 95% CIs are all within ±0.35 mm, see `results/`.)

- **A3 met**, by a wide margin: E0 1.30 mm, PCK@10 100%, 96% below B0. The targets were
  set before the difficulty of the synthetic task was known and are left unchanged.
- **A4 premise met**: E1's worst band (≥ 35) is 6.6× its in-range error. Where E1 fails at
  ≥ 35: crotch 14.1, waist front 14.0, buttock 13.6, waist side 11.0, knee 8.9, bust 7.6 mm
  (all 1.2–1.7 mm in range). E1 is also worse below its range (<18.5: 2.32 vs 1.31 mm).
  The gap P1d must close at ≥ 35 is 8.65 mm (E1) → 1.16 mm (E0).
- **Leakage check**: in the 18-D generator parameter space (6 phenotype + 9 local + 3
  pose), the median distance from a test body to its nearest training body is 1.33, the
  same as training-to-training (1.32). No duplicates.
- **Why the absolute numbers are so small**: labels are exact vertex positions; the
  generator has only 18 degrees of freedom, so the visible surface pins down occluded
  points (E0 crotch error at ≥ 35 is 1.28 mm although about half of those crotches are
  observable in fewer than 2 views); noise is mild and independent per view; the body
  is perfectly still during the scan. The relative gap is the meaningful result.
  Absolute mm are not comparable to real scanners, where manual landmarking alone
  varies by several mm.
- Open decision before P1d: add turntable realism (body sway between views) so the
  absolute numbers mean more. **Decided 2026-10-18: add it** (next entry). The no-sway
  numbers above are kept for reference only; their files moved to `runs/nosway/`.

### 2026-10-18 — subject sway added; P1d set up
Decided to add sway before P1d. Test and validation scans were regenerated and B0,
E0 and E1 re-run on them; all later numbers include sway.

- **Sway model** (`scanner.Sway`, `sway_views`): views are ~7.5 s apart on a ~60 s
  turntable, longer than a postural-sway cycle, so each view gets an independent rigid
  perturbation: horizontal drift σ 3 mm, lean about the floor σ 0.25° (≈ 7 mm at the
  head), twist σ 0.4°, plus arm abduction drift σ 1.5° per side, applied by linear blend
  skinning around a shoulder pivot (centroid of the 0.3–0.7 arm-weight ring, 20 mm
  from Anny's joint). Per-scan means are removed, so ground truth is the average
  position (mean of the 8 views is within 0.3 mm of it). Per-view vertex displacement:
  median 7 mm, max ≈ 34 mm at the hands. Values are guesses, not measurements.
- B0 on the sway test set: 31.4 mm (was 30.8): 20.9 / 22.5 / 27.0 / 37.1 / 49.4 by band.
- **E2** (`augment.Inflate`): offset along outward vertex normals, per-region ranges in
  mm (abdomen −10…60, hips −10…50, thighs −10…45, chest −10…35, upper arms −5…25,
  calves and neck −5…20, forearms −3…10; head, hands and feet 0). Regions are Anny
  skinning-weight groups; amplitudes share one body-wide level plus σ 0.25 regional
  spread; applied to 80% of samples. Mesh BMI of inflated narrow bodies: p1–p99 17.5–44.6.
- **E3** (`augment.extrapolate`, `data.train_meshes_extrapolated`): one variant per
  narrow training subject. 80% heavier (weight +0.3…0.8, adiposity controls +0…0.6),
  20% lighter; gender, age, muscle, proportions and pose kept. Anny's weight control
  shifted stature by up to 108 mm for some bodies, so the height control is corrected
  by 3 Newton steps (stature error now < 0.05 mm for all 8,000). Variant mesh BMI
  p1–p99 15.4–43.2. Training draws the variant for 50% of samples.
- **Deviation**: E2 drops the planned sex/height-balanced sampling. The narrow set is
  already close to balanced (46% male), and leaving it out keeps E2 a single
  intervention.
- Runs: E0/E1/E2/E3 × seeds 0–2, 10 epochs each, queued back to back (~9–10 h on MPS).
  `python -m body3d.report` aggregates them into `results/summary.json`
  (mean ± std over seeds; gap closure paired by seed).
- 37 pytest tests green.
- To do when the machine is idle: re-measure A1b. The last `scanner_check.json` run
  shared the CPU with training (145 ms). Done 2026-10-19: 26.2 ms on an idle machine.

### 2026-10-19 — E0/E1 with sway (seed 0)
| run | overall | <18.5 | 18.5-25 | 25-30 | 30-35 | >=35 | worst / in-range |
|---|---|---|---|---|---|---|---|
| B0 | 31.37 | 20.85 | 22.48 | 27.04 | 37.09 | 49.40 | 2.20x |
| E0 | 2.73 | 2.97 | 2.76 | 2.70 | 2.57 | 2.67 | 1.08x (<18.5) |
| E1 | 5.53 | 3.64 | 2.66 | 3.04 | 5.93 | 12.40 | 4.67x (>=35) |

Sway roughly doubles E0's error (1.30 → 2.73 mm) and keeps the gap: E1 is 4.7× worse at
≥ 35 than in range, so P1d must move ≥ 35 from 12.40 mm towards 2.67 mm. E1 is slightly
better than E0 in range (2.66 vs 2.76 mm): the specialised model gives up nothing there.
E2/E3 and seeds 1–2 are running.

### 2026-10-19 — P1d done (all runs with sway, 3 seeds)
All 12 runs finished cleanly (46–67 min each). From `results/summary.json`
(`python -m body3d.report`), mean ± std over seeds 0–2, mean 3D error in mm on the
2,000-body test set:

| model | seeds | overall | <18.5 | 18.5-25 | 25-30 | 30-35 | >=35 |
|---|---|---|---|---|---|---|---|
| B0 rigid template + nearest point | 1 | 31.37 | 20.85 | 22.48 | 27.04 | 37.09 | 49.40 |
| E0 trained on full range | 3 | 2.78 ± 0.05 | 3.02 ± 0.07 | 2.80 ± 0.06 | 2.73 ± 0.04 | 2.62 ± 0.05 | 2.72 ± 0.05 |
| E1 trained on BMI 18.5-27 | 3 | 5.35 ± 0.14 | 3.69 ± 0.15 | 2.64 ± 0.07 | 3.03 ± 0.08 | 5.74 ± 0.14 | 11.63 ± 0.54 |
| E2 = E1 + normal-offset inflation | 3 | 5.41 ± 0.11 | 4.13 ± 0.06 | 2.86 ± 0.05 | 3.42 ± 0.09 | 6.10 ± 0.20 | 10.54 ± 0.18 |
| E3 = E1 + shape-space extrapolation | 3 | 3.02 ± 0.11 | 3.41 ± 0.17 | 2.71 ± 0.07 | 2.81 ± 0.09 | 2.95 ± 0.10 | 3.20 ± 0.12 |

PCK@10 mm: B0 10.9%, E0 100%, E1 87.5%, E2 89.3%, E3 99.8%.

- **Headline (A4 met by E3)**: on the ≥ 35 band, E3 closes 95% ± 1% of the E1 → E0
  gap (per seed 95.0 / 95.6 / 93.1%), 11.63 → 3.20 mm, while the in-range band worsens
  by only 0.06 mm. Optimistic by construction: the extrapolated training bodies and the
  test bodies come from the same generator.
- **E2 does not meet A4**: 12% ± 6% closure (18.8 / 11.5 / 5.3%), in-range +0.22 mm,
  and the <18.5 band gets worse (3.69 → 4.13 mm). Per landmark at ≥ 35 it helps 9 of 16,
  where extra tissue really is roughly a normal offset (buttock 19.1 → 12.3, waist back
  15.5 → 10.1, waist sides ≈ 13.0 → 9.0, bust ≈ 9.4 → 5.9, knees ≈ 11.7 → 8.8 mm). It
  hurts 7, worst the waist front (21.2 → 29.7 mm) and acromion (≈ 9.7 → 12.8 mm), and
  does not help the crotch (23.4 → 24.6 mm). Reading: weight gain moves the abdomen
  front, shoulder top and thigh junction in ways a normal offset does not reproduce, so
  the network learns the wrong geometry there. A model-free fix would need a better
  deformation model than uniform normal offsets (follow-up idea, not planned).
- **Acceptance so far**: A1 met (rasteriser = ray casting); A1b met (26.2 ms, idle
  machine); A2 met for 15/16 landmarks, crotch not (thigh contact); A3 met (E0 2.78 mm,
  PCK@20 100%, 91% below B0); A4 met by E3, not by E2; A7 so far: 37 tests in ~20 s.

### 2026-10-19 — P2 done
Built `stl.py` (own binary/ASCII STL reader/writer, corner welding), `change.py`
(trajectories, Kabsch and point-to-plane ICP, two change-map paths, mesh-plane slicer,
girths at model-predicted landmark heights), the three.js viewer (`viewer/`) and
`viewer/capture.py` (headless-Chrome screenshot + GIF → `docs/img/`). 44 pytest tests,
9 s. Numbers from `results/change_check.json`:

- **Trajectories**: 6 subjects (3 male, 3 female, start mesh BMI 30.1–37.4, stature
  1.42–2.06 m), weeks 0/8/16/24, loss curve 1 − e^(−t/10 wk) normalised to week 24,
  calibrated per subject by bisection to −10.0% mesh mass at week 24 with stature kept.
  Arms ±3° and stance ±1° differ per visit; each visit gets a new turntable placement
  (±15°, ±2 cm). Raw STLs (mm) go to `data/trajectories/`.
- **A5 met.** Slicer within 0.5% of the analytic perimeter for circular and elliptic
  cylinders (test). ICP recovers a 14° + 2° rotation and 5 cm shift of the template to
  < 2 mm RMS (test). Path A (correspondence) vs path B (STL only, welded and re-ordered
  vertices, ICP): median |difference| per scan 0.08–1.81 mm over 18 scans (target ≤ 3 mm).
  ICP point-to-plane residual 2.8 mm mean, 5.1 mm max (it includes real shape change).
- **Girths**: heights from E0 (seed 0) landmarks predicted on a simulated scan of every
  visit. Against ground-truth landmark heights, tape girths differ by
  0.75 mm on average (5.4 mm max). Week-24 change (tape):
  waist -88 to -51 mm, hip -60 to -25 mm, thigh -38 to -34 mm.
- **A6 met.** Timeline slider, subject picker, signed change colours (±20 mm; arm weight
  ≥ 0.5 greyed out because pose dominates there), translucent week-0 shell, girth
  chart, mass/BMI panel, SYNTHETIC banner, phone layout checked. First render 275 ms
  locally (three.js already cached). Screenshot and 4-frame GIF in `docs/img/`.
- Notes: the colour scale was narrowed from ±40 to ±20 mm once real ranges were known
  (week-24 change spans about −24…+10 mm; the faint positive colour on feet and
  shoulders is posture difference between visits, said in the viewer).

### 2026-10-19 — P3 done (C# / .NET, no AWS)
.NET SDK 10.0.402 installed with Homebrew (`brew install dotnet`, no sudo; ASP.NET Core
runtime 10.0.13). `dotnet/` holds `Body3d.Core` (STL reader/writer with corner welding,
turntable rasteriser, lower-median input normalisation, slicer + convex-hull girths,
point-to-plane ICP with a k-d tree, ONNX Runtime 1.31 landmark inference with
soft-argmax and weighted least-squares triangulation), `Body3d.Api` (ASP.NET Core
minimal API) and `Body3d.Tests` (xUnit, 10 tests, all green). About 990 lines of C#.
`python -m body3d.export` writes the ONNX model (per-view CNN only) and the parity
fixtures; ONNX vs PyTorch outputs differ by at most 1.0e-5.

- **A8 met** (`results/parity_csharp.json`, 3 synthetic scans): depth maps 100% of
  pixels within 1 mm, identical coverage, max difference 6.0e-5 mm; landmarks max
  0.0032 mm (target 1 mm); girths max 6.3e-5 mm (target 0.1 mm).
- API: `GET /health`, `POST /scans/landmarks` and `/scans/girths` (STL body, mm by
  default), `POST /scans/change` (multipart `baseline` + `followup`). Local run on the
  M4 Pro: render 6–9 ms for 8 views, ONNX inference 150–170 ms on CPU; change for s2
  week 0 → 24: ICP 1.1 s, residual 3.43 mm, tape Δ waist −51.4 / hip −25.2 / thigh
  −34.6 mm (Python pipeline: −50.9 / −25.4 / −34.2 mm). Bad STL → 400.
- Nearest-neighbour search started as a uniform grid with expanding shells; the
  multi-start ICP made it slow (24 s per test), so it became a k-d tree (2 s).
- AWS (optional S3 storage): added the same day, see next entry.
- Fixtures (`dotnet/fixtures/*.stl`, `*.f32`, `viewnet.onnx`, about 16 MB) and build
  output are git-ignored; `python -m body3d.export` regenerates them.

### 2026-10-19 — P3 S3 storage (bucket plan and cost reviewed before creating anything)
- Account: new AWS experience, free plan; S3 is on the Free Tier service list (only S3
  Access Grants excluded).
  Prices checked against the AWS Price List API for ap-southeast-2 (published
  2026-10-15): Standard $0.025/GB-month, PUT/COPY/POST/LIST $0.0055 per 1,000, GET $0.0044
  per 10,000; data transfer out is free for the first 100 GB/month. Estimated cost of
  this step: well under US$0.01, covered by the free plan.
- Bucket `body3d-scans-apse2-<random suffix>` in ap-southeast-2: Block Public Access (all four), ACLs disabled
  (BucketOwnerEnforced), SSE-S3 default encryption, bucket policy denying non-TLS
  requests, versioning off (synthetic, regenerable), lifecycle: `scans/` objects expire
  after 30 days and incomplete multipart uploads are aborted after 1 day; tags
  `project=body3d-landmarks`, `data=synthetic`. All settings read back and verified.
  Not enabled, on purpose for synthetic data: server access logging, CloudTrail data
  events, SSE-KMS; for real scan (health) data all three should be on.
- API: `IScanStore` with `LocalScanStore` (default, used by tests) and `S3ScanStore`
  (SHA-256 upload checksums, `ExpectedBucketOwner` on every call). New endpoints
  `PUT /subjects/{id}/scans/{week}`, `GET /subjects/{id}/scans`,
  `GET /subjects/{id}/change?from=&to=`. Credentials come from `aws login` (the SDK needs
  the `AWSSDK.Signin` package for that). 12 xUnit tests green, without AWS.
- Integration (`results/s3_integration.json`): 24 synthetic STLs (32.9 MB) uploaded
  through the API, weeks listed correctly for all 6 subjects, sampled SHA-256 checksums
  match the local files, week 0 → 24 girth change computed from S3 matches the Python
  pipeline within 0.42 mm on average (2.1 mm max; the two sides take girth heights from
  differently rendered scans). Upload median 1.7 s per 1.37 MB file (home uplink),
  change from S3 median 2.0 s.
- Clean-up: the bucket is kept; the lifecycle rule expires the objects 30 days
  after upload, and an empty bucket costs nothing.

### 2026-10-19 — P4 done
- `README.md`: 149 lines. Results table straight from `results/summary.json`, viewer GIF,
  pipeline, change tracking, C#/S3, reproduce commands, limitations, credits. A script
  checked all 105 numbers in the prose: every one is in `results/*.json`, a design
  constant (views, seeds, visits), an identifier (ISO 8559, arXiv id), or one of two
  stated ratios (4.4× = 11.63 / 2.64; 91% = 1 − 2.78 / 31.37). All 19 local links resolve.
- Final state: pytest 44 passed, xUnit 12 passed. Committed tree is 77 files / 1.25 MB.
  Data, runs, the ONNX model and fixtures are regenerated by scripts.
- Acceptance summary: A1 ✔, A1b ✔, A2 15/16 (crotch ✘, explained), A3 ✔, A4 ✔ via E3
  (E2 ✘, explained), A5 ✔, A6 ✔, A7 ✔, A8 ✔.
- Published: code (MIT) on `main`, the viewer demo on the `gh-pages` branch (GitHub Pages),
  trained weights and the ONNX model as Release v0.1 assets with SHA-256 checksums.

### 2026-10-25 — wording fix: training perturbations
- A review found that README and this plan said each epoch gets fresh noise and sway. The
  dataset seeds its generator with (seed, `torch.initial_seed()`, index), and the DataLoader
  uses 8 persistent workers, so a body repeats the draw whenever it lands on the same worker:
  at most 8 perturbations per body over 10 epochs. Wording corrected in README, this plan and
  the `TrainScans` docstring. The code and all results are unchanged; they were produced by
  this behaviour. Adding the epoch to the seed would need every run retrained.
