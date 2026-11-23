---
name: feedback-learned-base-head
description: The data-free decoder's base shape must come from a learned head, never a fixed function of the code; confirm design choices before building
metadata:
  type: feedback
---

For the data-free decoder's code-set base (rounded box), the base must be produced by a learned head on the side network. A fixed, untrained code-to-base map is unacceptable ("a horrible way of doing this", "defeats the entire point"; 2026-11-15, B2 cancelled before it ran).

**Why:** the point of the latent space is that the network learns what each code means; hand-wiring code directions to shape parameters bypasses that.
**How to apply:** fix learned-head problems inside the head (parametrisation, costs, incentives), not by replacing it with a fixed map. When an option means something structurally different from what the user pictures (e.g. "fixed map" vs "head"), spell it out and confirm before building. Related: [[data-free-latent-work]].
