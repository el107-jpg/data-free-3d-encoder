# Data-free latent space: connectivity and cavity check

This is the detail for section 2 ("Shape Validity training") of
`Data-free latent space.md`. It covers how to find disconnected pieces and
cavities during training without a voxel grid and with almost no extra network
evaluations.

Status: design only. Nothing here has been implemented or tested.

Sign convention: positive = inside the shape, negative = free space.

## Idea

Every training step already evaluates the field at many points for the Eikonal
and SDF-consistency losses. Those points, with their signs and values, are
enough to decide whether the shape is one piece. We build a graph over them
instead of sampling a grid.

Two facts make this work:

- The sign of a sample is exact by definition: the shape is wherever the
  network is positive. Only the distance value is a guess.
- If the field is close to a true distance, a sample with value `f` has a ball
  of radius `|f|` around it that lies entirely on its own side of the surface.

## Inputs (all reused)

For each latent code in the batch:

- the volume samples `x`, with `f(x)` and the gradient `∇f(x)`;
- the one-step surface guesses `x' = x − f(x)∇f(x)`, with `f(x')`. These are
  already evaluated for the consistency loss `|f(x')|`. They are not refined
  with Newton steps;
- the measured consistency error `e` (a running typical value of `|f(x')|`),
  used as the uncertainty everywhere below;
- the origin, which the anchor loss keeps inside the shape.

## Nodes

- **Inside points:** every sample or guess with `f > 0`.
- **Band points:** every sample or guess with `|f| ≤ δ`, of either sign, where
  `δ` is about the size of `e`.

Band points are needed for thin shapes (sheets, cables). There, inside points
are rare and have tiny balls, and a one-step guess can jump straight through
the shape so that no sign change records it. The small value is the only
evidence.

- **Confident outside points:** samples with `f < −δ`. These are not nodes of
  the solid graph. They are used to cut edges, and they are the nodes of the
  mirrored outside graph used for cavities.

## Edges

Candidate edges join nodes closer than a few times the local sample spacing.
Neighbours come from a hash grid.

Each candidate edge is put in one of three states by looking at the segment
between its endpoints and the samples near it.

| State | Test | Meaning |
|---|---|---|
| Certified solid | Balls of nearby inside points cover the whole segment | Definitely connected here |
| Certified cut | See cut rules below | Definitely crosses free space |
| Unknown | Neither | May hide a gap no wider than the uncovered length |

Ball radii are shrunk by `e` before any test, so an overestimating field
cannot certify falsely.

### Cut rules

An edge is cut if either holds:

1. **Outside ball.** The free ball of a confident outside point reaches the
   segment (its distance to the segment is less than its shrunk radius).
2. **Normal-aligned crossing.** A surface crossing lies on the segment and its
   normal points roughly along the edge. Crossings and normals come from the
   gradient at a near-zero sample, or from a sample and its guess that have
   opposite signs.

A nearby outside point alone does not cut an edge. Outside points flank every
thin neck, and their balls stop at the neck wall. A crossing whose normal is
perpendicular to the edge is a side wall, not a gap.

A cut removes one edge only. The two endpoints can still be connected another
way (for example around a U-shape).

## Two graphs, three actions

Union-find is run twice on the same edges.

- **Optimistic graph:** certified-solid plus unknown edges.
- **Pessimistic graph:** certified-solid edges only.

The main body is the component containing the origin.

| Finding | Action | Why it is safe |
|---|---|---|
| Fragment separate in the optimistic graph | Delete it, or bridge it if it is large and close | It is separate for real |
| Fragment separate only in the pessimistic graph | Bridge only | A false alarm just asks an already-solid neck to be solid |
| Enclosed free region in the optimistic outside graph | Fill | It is enclosed for real |

Delete versus bridge for a real fragment is decided by its size (largest
inside value among its points, an estimate of its inscribed radius) and its
gap to the main body (uncovered length of the shortest joining edge).
Thresholds to be set by experiment.

### Cavities

The same construction is run on the confident outside points, with inside
balls and normal-aligned crossings as the cuts. A component with no far-field
point (beyond the shape's bounding radius) is a cavity.

## Tolerance policy

Gaps thinner than the resolution are treated as connected. The resolution is
roughly `2δ` plus the local sample spacing. A gap the robot cannot enter does
not matter to the planner.

Merging in the graph does not close the gap in the field. If the single-piece
criterion in `scripts/check_bar.py` counts pieces by sign on a grid, such
fragments will still count as separate. In that case, also apply the bridging
loss across merged gaps. To be checked before the first run.

## Losses

The graph is built without gradients. It only chooses points. The losses are
ordinary evaluations of the network at those points.

- **Bridge:** for the joining edge of a fragment, take the existing samples in
  a tube around the segment (plus a few points on the segment if the tube is
  empty) and penalise `relu(−f)²`. This is a hinge with zero margin: it is
  zero wherever the field is already inside. A positive margin would steadily
  thicken sheets and cables, because many of their edges are uncertified.
- **Delete:** on a fragment's inside and band points, penalise `relu(f)²`.
- **Fill:** on a cavity's points, penalise `relu(−f)²`.

Points found in fragments and cavities are kept in the anchor-code cache as
hard negatives, so a feature that was found once stays visible on later steps.

## When to turn it on

- **Early training:** the zero set is noise and distance values are
  unreliable. Use only the non-empty term (origin positive, soft volume).
- **Once the consistency error is moderate:** sign-only version. Nothing is
  certified solid; an edge is cut only by an outside point lying almost on the
  segment or by a normal-aligned crossing.
- **Once the consistency error is small:** full version with balls.

## Implementation notes

- Hash grid for neighbour lookup; no k-d tree.
- Point-to-segment distance tests against nearby samples for each candidate
  edge.
- Union-find, or scatter-min label propagation on GPU. Shapes are batched by
  offsetting node indices.
- A few thousand nodes per shape, so this should be millisecond-scale. Not
  measured.

## Known limits

- **Specks.** A stray piece is seen only if a sample lands in or next to it.
  A spurious speck is exactly the case where the surrounding field still
  points to the main body, so reuse cannot beat the sampling odds. The
  hard-negative cache is the mitigation.
- **Thin gaps and thin necks.** Both are unresolved below the resolution. By
  the tolerance policy they count as connected.
- **Thin hollow shells.** A closed wall thinner than the resolution does not
  cut outside edges, so the cavity inside it is missed.
- **Value accuracy.** All ball tests assume the field does not overestimate
  distance by more than `e`.

## Check before using it in a loss

Run the graph method and plain grid connected-component labelling on a few
hundred codes from the current decoder, including sheets and cables. Compare
the number of pieces and cavities per code. If they disagree on more than a
few percent, tune the edge tests first.

## Background

- GINN (Berzins et al., ICML 2025, arXiv 2402.14009) enforces connectedness
  with persistent homology on a grid: a sorted union-find sweep finds the
  saddle where each extra piece would merge, and a squared loss pulls the
  field to the surface level there. It bridges pieces; it does not delete
  them.
- GINN's first version found critical points by gradient descent instead. The
  authors reported it took over 90% of training time and was unreliable on
  sine networks.
- This plan gets the same kind of information from points already sampled,
  and supervises a tube or a whole fragment, not a single saddle cell.
