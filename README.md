# Task–Manifold Geometry

*How neural networks preserve, transform, and discard the geometry of their input —
studied one controlled variable at a time.*

When a classifier learns, it reshapes the geometry of its input: some structure is
kept, some is torn apart, some is built. This is a three-experiment program studying
*which* geometry survives and *why*, using synthetic 2-D manifolds (spiral, sinewave,
circle) where the ground-truth structure is known exactly, and small MLPs where every
variable can be held fixed but one.

The through-line across the three experiments:

> **What geometry a network keeps is governed by the task, by capacity, and by depth —
> and the early layers are where it is decided.**

Each experiment isolates one of those factors. The third explains the first two
mechanistically.

---

## The trilogy

**[Experiment 1 — Task Structure Governs *Which* Geometry a Classifier Keeps](experiments/exp1_task/README.md)**
Holding the data fixed and varying only the labelling scheme (at matched class
counts), a classifier preserves local geometry when the task is *aligned* with the
manifold and tears it when the task cuts *across* it. Binary tasks collapse the
representation onto a single axis regardless of shape. The effect is graded and
shape-dependent — the sharper the cut across the manifold, the more geometry is lost.

**[Experiment 2 — Capacity, Not Its Arrangement, Governs *How Much* Geometry Survives](experiments/exp2_capacity/README.md)**
Holding the task fixed and matching total parameter budget, the *arrangement* of
capacity across layers barely matters — total capacity is what drives preservation.
The one exception: a narrow early layer under scarce capacity destabilises *global*
structure specifically, occasionally failing outright while local structure stays
robust.

**[Experiment 3 — The Early Layers Decide: *Where Through Depth* Geometry Is Reshaped](experiments/exp3_trajectory/README.md)**
Measuring geometry layer by layer rather than only at the output localises *where*
the reshaping happens. Extrinsic tasks tear geometry early, in the first transitions,
and the damage is largely done before the output. Under scarce capacity, the network
builds the manifold's intrinsic coordinate through depth — but a narrow early layer
makes that construction variable and sometimes incomplete. This experiment explains
the endpoint findings of the first two: the *what* and the *how much* are both
decided early.

---

## How the three connect

Experiment 1 asks *what* geometry survives and answers: the task decides. Experiment 2
asks *how much* survives and answers: capacity decides, not its arrangement — except
when an early bottleneck destabilises things. Experiment 3 opens up the layers to ask
*where* through depth these effects happen, and finds the early layers are decisive
for both: extrinsic tearing is front-loaded, and the scarce-capacity instability
originates at the narrow first layer. The trajectory tooling built for Experiment 3 —
per-layer probes with a geodesic reference that corrects for the ambient-distance
contamination on folded manifolds — is the instrument the later work builds on.

A recurring methodological thread runs through all three: fix everything but one
variable; separate mechanical effects (forced by architecture) from behavioural ones;
report local and global geometry separately, because they can trade off; verify
across seeds; and report negative results honestly.

---

## Repository structure

```
src/representation_geometry/   # shared library: worlds, model, probes, architecture, targets
  experiments/                 #   per-experiment orchestration + configs
experiments/                   # run scripts, per-experiment writeups (README.md), and outputs
  exp1_task/
  exp2_capacity/
  exp3_trajectory/
```

The `src/` package is the single source of truth for all shared code (the MLP, the
manifold generators, the geometry probes). Each experiment is a thin configuration
over that shared core.

## Reproducing

```
pip install -e .
python experiments/exp1_task/run.py
python experiments/exp2_capacity/run.py
python experiments/exp3_trajectory/run.py
```

Each run regenerates its manifolds from config and reproduces the figures and result
tables in the corresponding `experiments/<name>/outputs/` directory.

## Probes

The geometry probes (in `src/representation_geometry/probes.py`) are reusable beyond
these experiments — they measure the geometric change between any two representational
states:

- **k-NN preservation** (local neighbourhood structure)
- **geodesic k-NN preservation** (local structure against an intrinsic-coordinate
  reference, correcting ambient-distance contamination on folded manifolds)
- **distance correlation** (global pairwise-distance structure)
- **PCA spectrum** (effective dimensionality)
- **latent reconstruction R²** (recoverability of the manifold's intrinsic coordinate)
- **transition / delta variants** (change localised between adjacent layers)
