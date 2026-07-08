# Task-Manifold Geometry

This repository unifies three small experiments on learned representation geometry.

The trilogy asks how much of a manifold's geometry a classifier keeps, and where
that geometry is altered:

1. **Task structure**: aligned labels preserve geometry better than extrinsic labels.
2. **Capacity**: total parameter budget matters more than how capacity is distributed, though scarcity exposes allocation effects.
3. **Depth**: layerwise probes show when intrinsic coordinates are built, preserved, or collapsed.

Run each experiment from its folder:

```bash
pip install -e .
python experiments/exp1_task/run.py
python experiments/exp2_capacity/run.py
python experiments/exp3_trajectory/run.py
```

Existing generated outputs are committed under each experiment's `outputs/` folder.
