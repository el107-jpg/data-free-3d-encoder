The current system for encoding 3d shapes and generating a good latent space is utilizing a massive library of 3d shapes and then including an attention based point cloud encoder which then outputs latents which are decoded by the decoder as a signed distance field. 

This idea is to bypass the encoder and library entirely and include just a decoder via random sampling within a ball. So the idea is that we will use purely losses that will encourage both diversity of 3d shapes and something like eikonal loss which makes sure the sdf is doing the right thing. 

The decoder can just remain a big mlp (possibly bigger given it's larger job now), but the training and loss is now very different.

The decoder outputs a sphere plus a bounded learned correction: `f(x, z) = (R − |x|) + A · tanh(net(x, z) / A)` with `R = 0.5` and `A = 0.35`, the correction initialised near zero. Every code therefore starts close to a sphere, always contains the ball of radius `R − A` (never empty, origin inside) and always lies within radius `R + A` (bounded), while a true distance field stays representable everywhere. (An earlier version faded the correction to zero beyond a bounding radius; that forced a sphere's far field onto every shape, which the Eikonal and consistency losses would then pull the shape back towards.)

1. Diversity training:

So one idea for diversity training would be to literally just sample a bunch of points n the object latent space and then basically create a loss term for how similar they are. Now the measure of similarity is going to be simply the difference in the sdf value between a bunch of sampled points along them and we will incentivize similarity and disimilarity based on a linear relationship with the distance between the points (this should be a a very steep loss where it's far less of a signal close to the lienar relationship and a lot of signal when it strays way off to not constrain it too much) Also, the difference of sdf value should be absolute not proportional to the sdf value itself since the proportional one will approach zero as we go to infinity regardless. The absolute difference is what matters here since.

For these sampled latent points, we also need to make sure they actually have a diversity of distance so we can sample points in some randomly selected epsilons around some point and then fit the entire combination of them which should offer lots of diversity signal without too much hassle.

To prevent trivial diversity such as just moving the object around, we will also anchor it in the middle by making the origin of the 3d space positive via a loss term. 

Now later on in training when everything is more settled in, we can add in more complex diversity training  by chanign the similarity metric to also include things like the shapes' surface to volume ratio, thickness, convexity, elongation, etc) This way, we can reward some more complexity over just the simple blobs it may overfitt toward. This may be part of a phase 2 we will have. This can be done quite easily using the internal points we already sample which means no additional sampling. 

2. Shape Validity training:

We will also be making sure that any embedding within the ball will have a valid 3d shape that follows the properties that we want. These are namely that there is a 3d shape available in there somewhere (like it's not an empty signed distance field where everything is below 0), that it is generally connected in one contiguous shape, that there are no cavities (empty regions within a filled region).

These are actually quite hard to create a differentiable loss term for, especially early on where there won't be many actual shapes anyways. So we might start out by just making sure that there are zeroes, so by random sampling and searching for positive points (i.e points that are inside the shape), we can directly search for them and add a small loss term which specifically penalizes having too little positive points. We can also add a few gradient steps along the network toward local maxima for this. 



3. sdf regularization training:
We can incentivize it actually working like a sdf by stepping along the gradient direction in the distance field's amount and penalize |f(x-f(x)df(x))|

Of course, we also want to include an eikonal loss given that sdfs should have a gradient norm of 1 in pretty much all cases. We can also penalize having a change in the gradient direction given small changes in location, though we should keep this small since it might not exist.

4. Equivariance training:

Given the cheap sampling of boundary and internal points we already have (refer to the plan below) we can literally just add a loss value to push the centroid of those points toward the origin along with a fixed covariance trace and getting rid of off diagonal covariance terms, which makes it so that our diversity remains non trivial shape things not easy variations. We can then add in scale translation and rotation later.

We will not pin axis order or mirror flips. They are a finite set of variations, so the diversity they offer saturates quickly, and forcing one canonical choice could break continuity of the latent space.

5. Spherical latent space 

One idea i have for this is to make it not a sphere but a shell so that the latent space lies on a spherical manifold. This might help given that it would get rid of any boundaries we have on our latent space while maintianing a bound on the total volume of our expert region. The boundaries on our sampling might cause issues with locality training near it and how symmetric locality is around the entire latent space. 

How it works:

- Every latent code is a unit vector in 32 numbers, so the codes live on the surface of a sphere (31 free dimensions).
- To sample a code, draw a 32-D Gaussian vector and divide it by its length. This is uniform over the sphere.
- The decoder normalises its input to unit length before using it. Any non-zero vector is then a valid code, including the output of a later encoder or an average of several codes.
- The latent distance used in the diversity loss is the chord length between the two unit codes, `sqrt(2 - 2cos)`, where `cos` is their cosine similarity. This replaces raw cosine similarity: `1 - cos` grows with the square of the distance for nearby codes, which does not fit a linear relationship. The shape distance is a straight-line distance between two fields, and straight-line distances can match chord lengths on a sphere exactly but cannot match arc lengths at large separations. For the small epsilon clusters, chord and angle are nearly identical.
- To interpolate between two codes, move along the sphere (slerp), not along the straight line. Exactly opposite codes have no unique midpoint; this is a rare edge case.
- Size and pose stay out of the latent (section 4).

Advantages:

- No boundary. Every code has neighbours in all directions, so the local diversity loss behaves the same everywhere. In a ball, codes near the edge have one-sided neighbourhoods.
- It matches what a ball would give anyway. About 97% of a 32-D ball's volume lies beyond radius 0.9, so the interior would almost never be sampled and would be poorly trained.
- Sampling is trivial and exactly uniform.
- Codes have a fixed, bounded scale, which suits the planner's shape tokens.

On periodicity: this is neutral in both directions. The sphere has no special affinity for periodic traits such as rotation (a loop of shapes fits equally well in a ball, and rotation is pinned out anyway). It also does not hurt non-periodic traits such as size: a trait only needs to be a single-valued function on the sphere, for example the component of the code along some direction. The one real consequence is that walking straight in any direction returns to the start, which is harmless.

Things to watch:

- All directions are forced equal. The sphere's symmetry plus a linear distance target asks for 31 equally strong modes of shape variation at every code, while real shape families have a few large modes and many small ones. The network may fill weak directions with high-frequency wobble. The steep-outside, flat-near-target loss shape helps; a per-axis weighting on the chord distance is the fallback. In early runs, check whether some latent directions produce only tiny or noisy shape changes.
- No centre. A ball has a natural average shape at the origin; a sphere has none. We do not expect to need one.

6. Loss balancing (adaptive multipliers)

Diversity is the objective; every other term (validity, Eikonal, SDF consistency, pose pinning, connectivity) is a constraint with a tolerance. Each constraint's weight rises automatically while its tolerance is violated and relaxes once it is met, so we do not hand-tune the weights against each other.

7. Dead-direction diagnostic (logged, not a loss)

At a few hundred probe codes, step in each of the 31 tangent directions and convert the field change at the surface points into how far the surface moves along its normal. Remove the part explained by translation, rotation and uniform scale, smooth the rest over the surface graph at the planner's resolution (real deformation survives, wobble cancels), and take the singular values. A direction is live if its motion over a typical cluster radius exceeds the SDF consistency error; we log live directions per code, the share of motion removed by smoothing (wobble), and how stable the live subspace is between neighbouring codes.

8. Keeping transformations continuous along the sphere

Topology can only change when two facing walls meet, so we never let walls closer than a minimum feature size `t` (at least a couple of sample spacings) persist. Every code starts as a sphere, so every shape stays one piece with no cavities and no handles, and thin spikes and sheets cannot vanish by thinning, which is what would make the distance field jump between nearby codes. Enforcement is by the graph (widen thin gaps within one piece, thicken thin walls; see the actions table below), plus a free barrier loss on existing samples: where `|f| < t/2`, penalise `relu(0.5 − |∇f|)²`. With this in place the field must be a true distance everywhere, and we count handles on anchor codes with a coarse voxel grid now and then as a check.

Training this whole thing is going to be quite complicated, so there are some computational speedups that will be necessary, so we will not be skimping on reusing caches or doing mutliple gradient steps to reuse already finished computation. See below for how we will efficiently find the losses for everything without needing to sample lots of additional points. It should be nice and parallelizeable. 


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

## Two graphs, five actions

Union-find is run twice on the same edges.

- **Optimistic graph:** certified-solid plus unknown edges.
- **Pessimistic graph:** certified-solid edges only.

The main body is the component containing the origin.

| Finding | Action | Why it is safe |
|---|---|---|
| Fragment separate in the optimistic graph | Delete it, or bridge it if it is large and close | It is separate for real |
| Fragment separate only in the pessimistic graph | Bridge only | A false alarm just asks an already-solid neck to be solid |
| Enclosed free region in the optimistic outside graph | Fill | It is enclosed for real |
| Gap narrower than `t` between two parts of the same piece | Widen: push the free samples in the gap to `−t/2` or below | Bridging here would create a handle |
| Wall thinner than `t` (found in the outside graph) | Thicken: push the inside samples in the wall to `t/2` or above | Stops punctures and parts vanishing by thinning |

Delete versus bridge for a real fragment is decided by its size (largest
inside value among its points, an estimate of its inscribed radius) and its
gap to the main body (uncovered length of the shortest joining edge).
Thresholds to be set by experiment.

### Cavities

The same construction is run on the confident outside points, with inside
balls and normal-aligned crossings as the cuts. A component with no far-field
point (beyond the shape's bounding radius) is a cavity.

## Tolerance policy

Gaps thinner than the resolution cannot be told apart from contact. The
resolution is roughly `2δ` plus the local sample spacing, and the minimum
feature size `t` (section 8) must be at least this.

A thin gap between two different pieces is treated as connected and gets the
bridging loss, so it really closes. A thin gap between two parts of the same
piece is never bridged, because that would create a handle; it is widened
instead.

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


Architectural notes on the decoder network itself:

We don't need anything other than a normal mlp, especially since we want the field to be relatively low frequency to avoid specks and also add a bias to smoothness which an sdf certainly has. We will use a modulated MLP with eight layers of width 512, softplus activations (or another smooth one), raw coordinates straight into the network (no fourier encoding), and a skip connection which feeds the coordinates back in halfway through the layers. 