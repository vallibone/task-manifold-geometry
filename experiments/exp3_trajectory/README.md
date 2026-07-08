# Experiment 3: Trajectory Through Layers

Tracks local, global, geodesic, and intrinsic-coordinate geometry through network depth.

Key results:

1. The aligned task maintains geometry flat through depth; the extrinsic task tears it early.
2. In aligned tasks, intrinsic coordinates survive. In extrinsic tasks, they do not.
3. Under scarce capacity, the network builds intrinsic coordinates through depth, variably.
4. Scarcity produces a distinct second failure mode: progressive global collapse.

Run:

```bash
python experiments/exp3_trajectory/run.py
```
