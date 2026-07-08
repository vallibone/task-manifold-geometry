# Task Structure Governs Which Geometry a Classifier Keeps

Across three synthetic manifolds (circle, sinewave, spiral), the network
preserves local neighbourhood structure when the task is aligned with the
manifold (labels assigned as continuous sectors along its surface), and
preserves less of it when the task is split extrinsically, by ambient
coordinates (x, y, quadrant) that cut across the manifold. The severity of
collapse scales with task simplicity - high-class tasks retain multi-dimensional
structure, while binary tasks compress the representation onto a single axis
(≥ 99.8% of variance) regardless of shape or label-alignment with the manifold.


## The Question

It is well known that classification representations discard 
information, keeping only the geometry that separates its labels.
The usual way that this is studied is to increase the number of classes that the task
requires and monitor changes in the representation's dimensionality. However, class
counting confounds two separate aspects - how hard the task is, and how well the labels 
relate to the manifold that the data lies on.

**The guiding question of the experiment:**

_*Given a task with a particular relationship to the underlying manifold, what 
geometry does the network preserve, and what does it tear apart?*_


## Why the Naive Version is Confounded

The obvious experiment is to fix a dataset, train classifiers with 2, 8, and 32
classes, and measure how representation geometry changes with class count. The
problem is that the labelling scheme is not held constant across those runs.
If the 2-class labels split the data one way and the 32-class labels split it
another, then changing the class count also changes how the labels sit relative
to the manifold. Any geometry effect could be the class count, the labelling
relationship, or both. In other words, they are entangled.

To break the entanglement, this experiment fixes the data and varies the
labelling scheme deliberately, at matched class counts. Some schemes 
follow the manifold (intrinsic; along the arc) and others cut across it
(extrinsic, by x, y, or quadrant). Comparing aligned against extrinsic 
at the same class count isolates the effect of the label-manifold relationship
while holding task difficulty fixed.


## Setup

**Data**: Three 2-D manifolds - circle, sinewave, spiral - were generated from 
a parameterised generator, so each shape is reproducible from its configuration.

**Model**: A multilayered perceptron trained to classify the sampled points.

**Labelling Schemes**:

**Intrinsic Arc**: 
Labels assigned as contiguous bins along the manifold's arc 
length, aligned with the geometry of the manifold.

**Observed x/y**: 
Labels assigned by binning an ambient coordinate. Extrinsic 
boundaries that cut across the manifold.

**Quadrant**: 
Extrinsic cuts across the manifold, separating into four evenly 
split quarters.

**Vertical/Horizontal halfspace**:
Binary extrinsic split.

*Class counts of 2, 8, and 32 (4 for quadrant).*


## Metrics (All computed at the output (logit) layer, relative to input):

**k-NN Preservation (k = 10)**:
 A measure of each point's input neighbours that remain
neighbours in the representation. A measure of the model's ability to preserve local
geometry.

**Distance Correlation**: 
Correlation between input and representation pairwise 
distances. A measure of the model's ability to preserve global geometry.

**PCA Variance Profile**: 
Dimensionality reduction (effective dimensionality), 
measuring the share of variance on the first principal components (1 - 3) and the 
minimum number of components required to reach 99% variance.


## Results

**1. Binary tasks collapse the representation onto one axis, invariantly.**

Every binary condition (All three shapes, every splitting labelling scheme) places 
~ 99.8% of its variance on the first principal component (std < 0.002 across seeds, 
for the seeded schemes), and needs only one component to reach 99% of the total. 
Local geometry is correspondingly low (k-NN preservation between 0.11 - 0.42), 
showing that regardless of the shape of the manifold, or how the boundary is drawn, 
the network will squeeze the manifold onto a line.

**2. At matched class count, aligned tasks preserve more local geometry than extrinsic 
ones, robustly across seeds.**

At 32 classes, arc-aligned k-NN preservation is 0.90 ± 0.01 (circle), 0.96 ± 0.01 (spiral), 
and 0.81 ± 0.01 (sinewave). The same shapes under extrinsic x/y splits preserve less: 
circle 0.70 ± 0.02–0.04, spiral 0.81–0.83 ± 0.02–0.05, sinewave 0.26–0.51 ± 0.03. 
The aligned-over-extrinsic ordering held in every seed, at both 8 and 32 classes, 
with the gap exceeding the combined seed spread in all conditions - the spiral has the 
tightest margin and still shows no band overlap. Aligned tasks also retain global 
structure better (distance correlation ~0.96 aligned vs ~0.66 extrinsic at 32 classes
on the circle) and classify more accurately, consistent with the task fitting the data's 
shape.

For this comparison, the class count is held fixed and both sides have identical logit 
dimensionality, so the difference reflects the label-to-manifold alignment.

![Seed-verified metrics across all three manifolds](outputs/seed_summary_grid.png)

*k-NN preservation, distance correlation, and first-PC variance across circle, spiral, 
and sinewave, over 5 seeds (error bars = seed spread). Arc-aligned (blue) preserves more 
local geometry than extrinsic splits at matched class count; the sinewave y-split 
declining with class count shows the shape-dependence.*

**3. The tearing is graded and shape-dependent, not absolute.**

Extrinsic splits do not fully destroy local geometry, but preserve less of it than 
aligned splits by an amount that depends on the severity of the cuts across the boundaries 
on the specific manifold. The sinewave under a y-split is the most extreme case of this, 
where preservation falls to 0.26. This, of course, makes perfect sense, as a horizontal 
cut through a curve that is multivalued in y separates points that are close along the 
manifold. Circle and spiral, whose extrinsic splits are less destructive geometrically, 
retain more local geometry.


## Caveat: The class count axis is partly mechanical

Local geometry preservation rises with class count for every scheme, which is, 
partly, not behavioural. Because the metrics are measured at the logit layer, 
and logit layer's dimensionality equals the class count, a binary task only 
has a 1-2 dimensional output in which to preserve neighbourhoods, while a 32 
class task has 32. More classes mechanically provide more room to preserve local 
structure.


## Where This Sits

This phenomenon connects to several established lines of work:

The binary collapse result is consistent with the Intrinsic-dimension and 
Manifold-untangling literature. Ansuini et al. (2019, Intrinsic dimension of
data representations in deep neural networks) showed that representation
dimensionality first rises then falls through depth, reaching low values near the
output. Chung, Lee and Sompolinsky (2018, Classification and geometry of general
perceptual manifolds) give the theoretical account of what manifold structure a
classifier must keep to separate classes.

The local, region-dependent framing (geometry preserved where the task respects
adjacency and torn where it must cross) is closest to Li and Wang (2023, Toward
a computational theory of manifold untangling: from global embedding to local
flattening), which treats untangling as a local operation.

The controlled design (fix the data, vary the task) appears in two recent
papers. "Should We Always Train Models on Fine-Grained Classes?" (2025) uses
synthetic 2D data with tunable boundary structure; "Causes and Consequences of
Representational Similarity in Machine Learning Models" (2025) holds data points
fixed while varying the task definition. This experiment arrives at the same
methodology independently and applies it specifically to local-versus-global
geometry preservation under aligned versus extrinsic labels.

## What this does and does not show

**Does**. It demonstrates, under controlled conditions, that label–manifold
alignment governs how much local geometry a classifier's output layer preserves,
and that binary tasks collapse the representation onto a single axis regardless
of shape or split. The central aligned-versus-extrinsic comparison was verified
across 5 seeds; the ordering held in every seed. The toy setting is
deliberate - clean control over the label–manifold relationship is only possible
with synthetic manifolds and known ground truth.

**Does not**. The metrics are measured input-to-output, not layer by layer, so
this says nothing yet about where through depth the geometry changes. The
seed verification covers the core schemes (intrinsic arc, observed x, observed
y) at 2, 8, and 32 classes; the additional quadrant and halfspace conditions
are single runs. The architecture is a single MLP, and the manifolds are 2D.
The class-count trend is partly an artefact of logit dimensionality, as noted.

## Next

The natural extension is to measure geometry preservation *per layer* rather than
only at the output, turning the input-to-output comparison into a trajectory
through depth, and to verify the alignment effect across seeds and with a predictor
- a measure of how the label boundaries align with the manifold's
tangent.


### Reproducing 

```bash
python experiments/exp1_task/run.py
```

Regenerates the data from config and reproduces the headline figures in
`outputs/`.

_*380D33*_
