# Shaped From The Start: Where Through Depth a Classifier Reshapes Geometry

Measuring geometry progressively through layers rather than only at the output
can reveal at what point through the depth of a network geometric structure
is preserved or destroyed. In that regard, it appears that the early layers
are often decisive, though full trajectory matters.

When a task cuts across the manifold, the tearing of local geometry happens in
the first layers and is mostly complete before the output. The arc-aligned task,
by contrast, maintains structure throughout. And, when capacity is scarce, an
early bottleneck destabilises the construction of the manifold's intrinsic
coordinate. The network will still attempt to build it through depth, but the
early narrow layer makes the process variable and sometimes incomplete, 
distinctly failing through global collapse.


## The Question

Experiments 1 and 2 showed that tasks govern which geometry survives, and
capacity governs how much of it survives, respectively. However, both of 
these experiments measured the endpoint only (input vs logit), so neither
could say where throughout the depth of the network did these reshapings occur.

**The guiding question of the experiment:**

_*Where through depth are local and global gemoetry preserved or destroyed,
and how does depend on architecture and task alignment?*_


## Why Measure Per Layer, and Why Geodesic

An endpoint measurement collapses the whole network into a single before/after, but
"the extrinsic task tears local geometry" (Experiment 1) and "a narrow early layer
destabilises geometry" (Experiment 2) are both claims that a trajectory can
localise: early or late, gradual or abrupt, recoverable or not. This is measured
here.

This test also adds an essential dimension to these experiments by measuring 
local geometry preservation in proximity to a geodesic reference. That is, neighbours 
defined by proximity along the manifold's intrinsic coordinate, and not by ambient
distance.

## Setup

Two sub-studies, each a controlled trajectory comparison holding everything fixed
except one variable.

**A. The scarce-capacity trajectory (spiral, 32-class arc, back-loaded).** The
low-capacity back-loaded architecture from Experiment 2 - a (4, 8, 16) MLP,
~740 parameters, whose narrow first layer produced unstable geometry - run over 10
seeds, measured at every layer. This localises where through depth the
Experiment 2 instability appears.

**B. The aligned-versus-torn trajectory (sinewave, 32-class).** The manifold and
task where Experiment 1's extrinsic tearing was strongest. Two conditions, same
manifold, same architecture, same class count, differing only in the label
geometry: **intrinsic arc** (aligned, the task follows the manifold) versus
**observed y** (extrinsic, a horizontal cut across a curve that is multivalued in
y). This localises where the extrinsic tearing happens, against a baseline that
does not tear.

**Metrics**, computed at every layer relative to input: Euclidean and geodesic
k-NN preservation (local), distance correlation (global), arc reconstruction R²
(latent-coordinate recoverability), and PCA profile. Reported both as absolute
per-layer values and as layer-to-layer deltas, which locate where change
concentrates.


## Results

**1. The aligned task maintains geometry flat through depth, the extrinsic task tears it early.**

On the sinewave, the intrinsic-arc task holds local geometry almost unchanged at
every layer, geodesic k-NN loss stays near zero throughout. The extrinsic y-split
tears local geometry early, with the largest geodesic k-NN loss occurring at the first 
transitions (input → hidden_1 and hidden_1 → hidden_2), which then tails off. Global 
structure follows the same pattern, with the y-split's largest distance- correlation loss also 
landing at the first transition. The network satisfies the extrinsic task by scrambling 
geodesically-close points up front, not by a late reshuffle near the output.

![Layerwise geodesic local geometry loss, sinewave 32-class](outputs/sinewave_geodesic_loss.png)

*Layer-to-layer geodesic k-NN loss (higher = more local tearing at that transition).
The aligned task (blue) stays flat; the extrinsic y-split (orange) tears early and
tails off. Transitions in network order, input → hidden_1 first.*

**2. The arc-aligned task's intrinsic coordinate survives, the extrinsic task's does not.**

Arc reconstruction R² makes the contrast explicit. The intrinsic-arc task holds arc
recoverability flat, it never has to destroy the coordinate because the coordinate
is the task. The extrinsic y-split progressively loses arc recoverability
through the middle and later transitions. The representation is reorganised away
from the manifold's intrinsic structure to satisfy a task that cuts across it.

**3. Under scarce capacity, the network builds the intrinsic coordinate through depth,
variably and sometimes incompletely**

On the back-loaded spiral, local geometry is robust across all ten seeds and all
layers (geodesic k-NN barely moves). The action is in arc reconstruction R²: the
coordinate is nearly unreadable at the input (~0.15) and the network builds it
through depth, climbing toward high recoverability. But the narrow first layer makes
this construction variable; arc R² at the first hidden layer ranges from ~0.29
to ~0.87 across seeds. And while most seeds recover substantially by the last
hidden layer, a tail ends well short (logit arc R² ranges ~0.66 to ~0.99). The
instability of Experiment 2 is, mechanistically, instability in how reliably and
completely the early layers establish the intrinsic coordinate.

**4. Scarcity produces a distinct second failure mode: progressive global collapse.**

One of the ten seeds failed differently. Rather than under-building the arc
coordinate, it destroyed global structure monotonically through depth, with distance
correlation falling 1.00 → 0.88 → 0.81 → 0.72 → 0.58 from input to logits, while
its arc coordinate recovered normally. No other seed exhibited this kind of behaviour. 
So, a narrow early layer at low capacity does not have a single failure, it destabilises the 
trajectory in at least two distinguishable ways - slow or incomplete construction of the
intrinsic coordinate, and, more rarely, progressive collapse of global distance structure.


## Caveat: trajectory shape, not absolute per-layer value

Layer widths differ within an architecture, and a layer's width bounds how much
geometry it can express, so absolute per-layer values are not directly comparable
across layers of different width. The trustworthy signal is the shape of a
trajectory and the difference between conditions at the same layer of the same
architecture (which is why every comparison here holds the architecture fixed and
varies one thing (seed, or task)). The two sub-studies use different architectures
and are not cross-compared. The single progressive-collapse failure (Result 4) is
one seed in ten; a documented failure mode, not a quantified rate.

## What this does and does not show

**Does**. It localises, through depth, the endpoint findings of Experiments 1 and
2. Extrinsic tearing of geometry is front-loaded in the early layers, while an
aligned task maintains structure throughout. Under scarce capacity, the network
builds the intrinsic coordinate through depth with a variability and an occasional
incompleteness that originates early, plus a distinct rarer failure of progressive
global collapse. Local-geometry claims use the geodesic reference, which corrects
the ambient measure's contamination on folded manifolds. The scarce-capacity study
uses 10 seeds; the task-contrast study holds architecture fixed across conditions.

**Does not**. Absolute per-layer values are not compared across differing widths,
only trajectory shape and same-architecture cross-condition differences are
interpreted. The progressive-collapse failure is a single-seed vignette, not a
quantified frequency. Two manifolds, one small MLP depth, and 2-D data; the
depth-localisation may not transfer to deeper networks or higher-dimensional
manifolds. The Euclidean k-NN trajectory is reported only as the naive measure the
geodesic reference corrects.


## Next

The trajectory tooling built here, per-layer probes with a geodesic reference,
is the instrument for the questions that follow - whether the depth-location of
tearing is predictable from a measure of how the task's boundaries align with the
manifold's tangent, defined before training - and how the geometry a network builds
through depth relates to what it can generalise to. The latter opens the next line
of work.


### Reproducing

```bash
python experiments/exp3_trajectory/run.py
```

Regenerates the per-layer trajectories for both sub-studies and reproduces the
figures in `outputs/`, including the geodesic-referenced local-geometry
trajectories.

_*380D33*_
