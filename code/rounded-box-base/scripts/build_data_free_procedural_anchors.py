"""Procedural anchor objects for the data-free latent, chosen by KIND of geometry.

Every shape is a skeleton (points on curves, surfaces or small lumps) grown by a radius: the solid is
everything within the radius of the skeleton, and its field is ``radius - distance to the skeleton``
(positive inside; an exact distance outside the solid). So the families are skeleton families:

  rods     a disc swept along a line, an L, a C, a U, an S, a fold-back, a spiral, a helix, junctions ...
  sheets   a plate, folded and curved plates, a bowl, a saddle, a twisted ribbon, crossing plates ...
  mixed    a plate or a ball with rods attached
  blobs    thick lumps

Each candidate is put in the trainer's pose convention (soft centroid at the origin, soft covariance
diagonal with the configured trace), must fit inside the outer radius, be one simple piece, and pass the
same thin-feature and thin-gap test as the Thingi anchors. Descriptors (scripts/data_free_shape_descriptors.py)
are measured from the field grid. Selection: one candidate per family first, then the rest, each time the
candidate whose descriptor is farthest from those already chosen.

usage: build_data_free_procedural_anchors.py --output-dir <new folder> [--count 64] [--sheet picture.png]
Writes anchors.pt (same layout as the Thingi anchor files), anchors.json and, optionally, a contact sheet.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import scipy.ndimage as ndi
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_free_shape_descriptors as SD  # noqa: E402

from arm_planner.data_free import diagnostics as D  # noqa: E402
from arm_planner.data_free import losses as L  # noqa: E402
from arm_planner.data_free.config import DataFreeConfig  # noqa: E402

STEP = 0.012          # spacing of skeleton samples in the shape's own units (extent about 1 to 2)


# ------------------------------------------------------------------ skeleton pieces
def seg(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(2, int(np.linalg.norm(b - a) / STEP) + 1)
    return a + np.linspace(0, 1, n)[:, None] * (b - a)


def arc(centre, radius, a0, a1, u=(1, 0, 0), v=(0, 1, 0)):
    """Circle arc from angle a0 to a1 (radians) in the plane spanned by u and v."""
    n = max(3, int(abs(a1 - a0) * radius / STEP) + 1)
    t = np.linspace(a0, a1, n)
    return np.asarray(centre, float) + radius * (np.cos(t)[:, None] * np.asarray(u, float) + np.sin(t)[:, None] * np.asarray(v, float))


def patch(fn, u0, u1, v0, v1, mask=None):
    """Points of the surface fn(u, v) -> (x, y, z); u and v are sampled at about STEP."""
    nu, nv = max(3, int((u1 - u0) / STEP) + 1), max(3, int((v1 - v0) / STEP) + 1)
    u, v = np.meshgrid(np.linspace(u0, u1, nu), np.linspace(v0, v1, nv), indexing="ij")
    pts = np.stack(fn(u, v), -1).reshape(-1, 3)
    return pts if mask is None else pts[mask(u, v).reshape(-1)]


def rect(a, b, origin=(0, 0, 0), e1=(1, 0, 0), e2=(0, 1, 0)):
    o, e1, e2 = np.asarray(origin, float), np.asarray(e1, float), np.asarray(e2, float)
    return patch(lambda u, v: tuple(o[i] + u * e1[i] + v * e2[i] for i in range(3)), -a / 2, a / 2, -b / 2, b / 2)


def disc(radius, origin=(0, 0, 0), e1=(1, 0, 0), e2=(0, 1, 0), squash=1.0):
    o, e1, e2 = np.asarray(origin, float), np.asarray(e1, float), np.asarray(e2, float)
    return patch(lambda u, v: tuple(o[i] + u * e1[i] + squash * v * e2[i] for i in range(3)), -radius, radius, -radius, radius,
                 mask=lambda u, v: u * u + v * v <= radius * radius)


def extrude(curve, length):
    """Sweep a planar curve (in xy) along z."""
    n = max(3, int(length / STEP) + 1)
    z = np.linspace(-length / 2, length / 2, n)
    return (curve[:, None, :] + np.stack([np.zeros(n), np.zeros(n), z], 1)[None]).reshape(-1, 3)


def lump(a, b, c):
    """Points filling a box (the skeleton of a rounded box)."""
    g = [np.linspace(-s / 2, s / 2, max(2, int(s / (2 * STEP)) + 1)) for s in (a, b, c)]
    return np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)


def cat(*parts):
    return np.concatenate(parts, 0)


def rad(x):
    return np.deg2rad(x)


# ------------------------------------------------------------------ families
# Each returns a list of (name suffix, groups, thin radius or None). groups = [(points, radius), ...] in the
# shape's own units; the first group's radius is set to "thin radius" in the trainer's units after posing
# (None: the proportions are kept as given, used for blobs).
def families(rng, thick=1.0):
    U = rng.uniform
    out = {}

    def add(kind, family, groups, thin, note=""):
        out.setdefault((kind, family), []).append({"groups": groups, "thin": thin, "note": note})

    for _ in range(3):
        r = thick * U(0.06, 0.10)
        add("rod", "straight rod", [(seg((0, 0, 0), (1, 0, 0)), 1)], r)
        t = rad(U(75, 110)); b = U(0.55, 1.0)
        add("rod", "L", [(cat(seg((1, 0, 0), (0, 0, 0)), seg((0, 0, 0), (b * np.cos(t), b * np.sin(t), 0))), 1)], r, f"{np.rad2deg(t):.0f} deg")
        t = rad(U(35, 55))
        add("rod", "V", [(cat(seg((1, 0, 0), (0, 0, 0)), seg((0, 0, 0), (np.cos(t), np.sin(t), 0))), 1)], r, f"{np.rad2deg(t):.0f} deg")
        phi = rad(U(150, 300))
        add("rod", "C arc", [(arc((0, 0, 0), 1, 0, phi), 1)], r, f"{np.rad2deg(phi):.0f} deg")
        leg = U(0.6, 1.4)
        add("rod", "U", [(cat(seg((-0.5, leg, 0), (-0.5, 0, 0)), arc((0, 0, 0), 0.5, np.pi, 2 * np.pi), seg((0.5, 0, 0), (0.5, leg, 0))), 1)], r)
        add("rod", "hook", [(cat(seg((-0.4, 1.6, 0), (-0.4, 0, 0)), arc((0, 0, 0), 0.4, np.pi, 2 * np.pi), seg((0.4, 0, 0), (0.4, U(0.1, 0.5), 0))), 1)], r)
        phi = rad(U(120, 190))
        add("rod", "S", [(cat(arc((0, 0.5, 0), 0.5, -np.pi / 2, -np.pi / 2 + phi), arc((0, -0.5, 0), 0.5, np.pi / 2, np.pi / 2 + phi)), 1)], r, f"{np.rad2deg(phi):.0f} deg arcs")
        d, ln = U(0.8, 1.0), U(0.9, 1.3)
        add("rod", "fold-back", [(cat(seg((0, d, 0), (ln, d, 0)), arc((ln, d / 2, 0), d / 2, np.pi / 2, -np.pi / 2), seg((ln, 0, 0), (0, 0, 0)),
                                       arc((0, -d / 2, 0), d / 2, np.pi / 2, 3 * np.pi / 2), seg((0, -d, 0), (ln, -d, 0))), 1)], r, "three passes")
        turns = U(1.15, 1.5); th = np.linspace(0, 2 * np.pi * turns, 500)
        rr = 0.45 + 0.75 * th / (2 * np.pi)
        add("rod", "spiral", [(np.stack([rr * np.cos(th), rr * np.sin(th), 0 * th], 1), 1)], r, f"{turns:.1f} turns")
        turns, pitch = U(0.9, 1.5), U(0.7, 1.1); th = np.linspace(0, 2 * np.pi * turns, 400)
        add("rod", "helix", [(np.stack([0.6 * np.cos(th), 0.6 * np.sin(th), pitch * th / (2 * np.pi)], 1), 1)], r, f"{turns:.1f} turns")
        a, b, c = U(0.7, 1.1), U(0.6, 1.0), U(0.6, 1.0)
        add("rod", "3-D crank", [(cat(seg((0, 0, 0), (a, 0, 0)), seg((a, 0, 0), (a, b, 0)), seg((a, b, 0), (a, b, c))), 1)], r)
        ang = [rad(90), rad(90 + U(105, 135)), rad(90 - U(105, 135))]
        add("rod", "Y", [(cat(*[seg((0, 0, 0), (U(0.6, 1.0) * np.cos(q), U(0.6, 1.0) * np.sin(q), 0)) for q in ang]), 1)], r)
        add("rod", "T", [(cat(seg((-U(0.6, 1.0), 0, 0), (U(0.6, 1.0), 0, 0)), seg((0, 0, 0), (0, -U(0.8, 1.4), 0))), 1)], r)
        add("rod", "plus", [(cat(seg((-U(0.5, 1), 0, 0), (U(0.5, 1), 0, 0)), seg((0, -U(0.5, 1), 0), (0, U(0.5, 1), 0))), 1)], r)
        add("rod", "corner frame", [(cat(seg((0, 0, 0), (U(0.7, 1), 0, 0)), seg((0, 0, 0), (0, U(0.7, 1), 0)), seg((0, 0, 0), (0, 0, U(0.7, 1)))), 1)], r)
        tet = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / 3 ** 0.5
        add("rod", "tetrapod", [(cat(*[seg((0, 0, 0), U(0.7, 1.0) * q) for q in tet]), 1)], r)
        add("rod", "F branch", [(cat(seg((0, 0, 0), (0, 1.6, 0)), seg((0, 1.6, 0), (U(0.6, 0.9), 1.6, 0)), seg((0, U(0.7, 1.0), 0), (U(0.4, 0.7), U(0.7, 1.0), 0))), 1)], r)
        big = U(2.0, 2.8)
        add("rod", "dumbbell", [(seg((0, 0, 0), (1.4, 0, 0)), 1), (np.array([[0.0, 0, 0], [1.4, 0, 0]]), big)], r, "thick ends")
        pts = seg((0, 0, 0), (1.0, 0, 0)); k = 10; big = U(1.8, 2.6)
        add("rod", "club", [(c_, 1 + (big - 1) * (i / (k - 1)) ** 1.5) for i, c_ in enumerate(np.array_split(pts, k))], r, "tapered")

    for _ in range(3):
        hth = thick * U(0.055, 0.08)
        a = U(1.0, 3.0)
        add("sheet", "plate", [(rect(a, 1), 1)], hth, f"{a:.1f} : 1")
        add("sheet", "disc plate", [(disc(1, squash=U(0.6, 1.0)), 1)], hth)
        t = rad(U(60, 120)); w = U(0.8, 1.6)
        add("sheet", "angle plate", [(cat(extrude(seg((1, 0, 0), (0, 0, 0)), w), extrude(seg((0, 0, 0), (np.cos(t), np.sin(t), 0)), w)), 1)], hth, f"{np.rad2deg(t):.0f} deg")
        leg, w = U(0.3, 0.9), U(0.8, 1.6)
        add("sheet", "U channel", [(extrude(cat(seg((-0.5, leg, 0), (-0.5, 0, 0)), arc((0, 0, 0), 0.5, np.pi, 2 * np.pi), seg((0.5, 0, 0), (0.5, leg, 0))), w), 1)], hth)
        phi = rad(U(150, 270)); w = U(0.8, 1.8)
        add("sheet", "C sheet", [(extrude(arc((0, 0, 0), 0.7, 0, phi), w), 1)], hth, f"{np.rad2deg(phi):.0f} deg")
        phi = rad(U(110, 170)); w = U(0.8, 1.5)
        add("sheet", "S sheet", [(extrude(cat(arc((0, 0.5, 0), 0.5, -np.pi / 2, -np.pi / 2 + phi), arc((0, -0.5, 0), 0.5, np.pi / 2, np.pi / 2 + phi)), w), 1)], hth)
        d, ln, w = U(0.8, 1.0), U(0.8, 1.1), U(0.7, 1.2)
        add("sheet", "Z fold sheet", [(extrude(cat(seg((0, d, 0), (ln, d, 0)), arc((ln, d / 2, 0), d / 2, np.pi / 2, -np.pi / 2), seg((ln, 0, 0), (0, 0, 0)),
                                                   arc((0, -d / 2, 0), d / 2, np.pi / 2, 3 * np.pi / 2), seg((0, -d, 0), (ln, -d, 0))), w), 1)], hth)
        al = rad(U(50, 110))
        add("sheet", "bowl", [(patch(lambda u, v: (np.sin(u) * np.cos(v), np.sin(u) * np.sin(v), -np.cos(u)), 0, al, 0, 2 * np.pi), 1)], hth, f"{np.rad2deg(al):.0f} deg half-angle")
        cs = U(0.5, 0.9)
        add("sheet", "saddle", [(patch(lambda u, v: (u, v, cs * (u * u - v * v)), -1, 1, -1, 1, mask=lambda u, v: u * u + v * v <= 1), 1)], hth)
        tw, ln = rad(U(120, 270)), U(1.8, 2.6)
        add("sheet", "twisted ribbon", [(patch(lambda u, v: (u * np.cos(tw * v / ln), u * np.sin(tw * v / ln), v), -0.5, 0.5, -ln / 2, ln / 2), 1)], hth, f"{np.rad2deg(tw):.0f} deg twist")
        ha = rad(U(22, 40))
        add("sheet", "cone shell", [(patch(lambda u, v: (u * np.sin(ha) * np.cos(v), u * np.sin(ha) * np.sin(v), u * np.cos(ha)), 0, 1.4, 0, 2 * np.pi), 1)], hth)
        w = U(0.9, 1.6); fin = U(0.5, 0.9)
        add("sheet", "T plates", [(cat(rect(1.4, w, e1=(1, 0, 0), e2=(0, 0, 1)), rect(fin, w, origin=(0, fin / 2, 0), e1=(0, 1, 0), e2=(0, 0, 1))), 1)], hth)
        add("sheet", "X plates", [(cat(rect(1.4, w, e1=(1, 0, 0), e2=(0, 0, 1)), rect(U(0.9, 1.4), w, e1=(0, 1, 0), e2=(0, 0, 1))), 1)], hth)
        s = U(0.8, 1.0)
        add("sheet", "corner plates", [(cat(rect(s, s, origin=(s / 2, s / 2, 0)), rect(s, s, origin=(s / 2, 0, s / 2), e2=(0, 0, 1)),
                                            rect(s, s, origin=(0, s / 2, s / 2), e1=(0, 1, 0), e2=(0, 0, 1))), 1)], hth)
        s, ht = 1.0, U(0.5, 0.9)
        add("sheet", "open box", [(cat(rect(s, s), rect(s, ht, origin=(0, -s / 2, ht / 2), e2=(0, 0, 1)), rect(s, ht, origin=(0, s / 2, ht / 2), e2=(0, 0, 1)),
                                       rect(s, ht, origin=(-s / 2, 0, ht / 2), e1=(0, 1, 0), e2=(0, 0, 1)), rect(s, ht, origin=(s / 2, 0, ht / 2), e1=(0, 1, 0), e2=(0, 0, 1))), 1)], hth)
        w = U(0.9, 1.5)
        add("sheet", "three-fin fan", [(cat(*[extrude(seg((0, 0, 0), (U(0.6, 0.9) * np.cos(q), U(0.6, 0.9) * np.sin(q), 0)), w) for q in (rad(90), rad(210), rad(330))]), 1)], hth)

    # Several rod lengths are offered; the outer-radius test keeps those that fit. (Under the mass-based
    # pose rule, pose_power 2, a rod on a plate or a ball can only stick out a little: the size is then
    # set by the heavy part alone. Under the extent-based default the heavy part comes out smaller.)
    for k in range(5):
        r = thick * U(0.06, 0.09)
        short = 1.0 - 0.15 * k
        add("mixed", "paddle", [(cat(disc(0.45, squash=U(0.7, 1.0)), seg((0.45, 0, 0), (0.45 + short * U(0.7, 1.0), 0, 0))), 1)], r, "plate + rod")
        add("mixed", "table", [(cat(disc(U(0.7, 0.9)), seg((0, 0, 0), (0, 0, -U(0.8, 1.3)))), 1)], r, "plate + stem")
        al = rad(U(60, 90))
        add("mixed", "spoon", [(cat(patch(lambda u, v: (0.5 * np.sin(u) * np.cos(v), 0.5 * np.sin(u) * np.sin(v), -0.5 * np.cos(u) + 0.5 * np.cos(al)), 0, al, 0, 2 * np.pi),
                                    seg((0.5 * np.sin(al), 0, 0), (0.5 * np.sin(al) + short * U(0.7, 1.0), 0, 0))), 1)], r, "bowl + rod")
        add("mixed", "ball on a stick", [(seg((0, 0, 0), (short * U(2.6, 3.6), 0, 0)), 0.24 * thick), (np.zeros((1, 3)), 1.0)], None)
        dirs = SD.fibonacci_sphere(7)[rng.choice(7, 3, replace=False)]
        add("mixed", "ball with limbs", [(cat(*[seg((0, 0, 0), short * U(2.2, 3.0) * q) for q in dirs]), 0.24 * thick), (np.zeros((1, 3)), 1.0)], None)
        add("mixed", "stool", [(cat(disc(0.7), *[seg((0.45 * np.cos(q), 0.45 * np.sin(q), 0), (0.6 * np.cos(q), 0.6 * np.sin(q), -U(0.7, 1.0))) for q in (rad(90), rad(210), rad(330))]), 1)], r, "plate + 3 legs")
        add("mixed", "plate with a post", [(cat(rect(U(1.2, 1.8), 1.0), seg((U(-0.4, 0.4), 0, 0), (U(-0.4, 0.4), 0, short * U(0.7, 1.1)))), 1)], r, "plate + rod")
        add("mixed", "plate with a hook", [(cat(rect(1.2, 0.9), seg((-0.4, 0, 0), (-0.4, 0, 0.8)), arc((0, 0, 0.8), 0.4, np.pi, 0, u=(1, 0, 0), v=(0, 0, 1)),
                                               seg((0.4, 0, 0.8), (0.4, 0, U(0.5, 0.65)))), 1)], r, "nearly a handle")

    for _ in range(3):
        add("blob", "capsule", [(seg((0, 0, 0), (1, 0, 0)), U(0.35, 0.8))], None)
        add("blob", "lentil", [(disc(1, squash=U(0.7, 1.0)), U(0.3, 0.6))], None)
        add("blob", "rounded box", [(lump(1, U(0.5, 1.0), U(0.3, 0.8)), U(0.15, 0.3))], None)
        pts = seg((0, 0, 0), (1, 0, 0)); k = 10
        add("blob", "pear", [(c_, 0.25 + 0.45 * (i / (k - 1))) for i, c_ in enumerate(np.array_split(pts, k))], None)
        lobes = int(rng.integers(3, 5))
        centres = (SD.fibonacci_sphere(4) if lobes == 4 else np.array([[np.cos(q), np.sin(q), 0] for q in rad(np.array([90, 210, 330]))])) * U(0.5, 0.75)
        add("blob", "clover", [(cat(centres, np.zeros((1, 3))), U(0.5, 0.65))], None, f"{lobes} lobes")
        add("blob", "wedge", [(cat(*[rect(1.0 * (1 - z), 0.9, origin=(0, 0, 0.5 * z)) for z in np.linspace(0, 0.9, 30)]), 0.16)], None)
    return out


# ------------------------------------------------------------------ fields and pose
def grid_points(resolution: int) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, resolution)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)


def field(groups, points) -> np.ndarray:
    best = np.full(len(points), -np.inf)
    for pts, radius in groups:
        d, _ = cKDTree(pts).query(points)
        best = np.maximum(best, radius - d)
    return best


def moments(t, points, tau):
    w = 1.0 / (1.0 + np.exp(-np.clip(t / tau, -60, 60)))
    total = w.sum() + 1e-6
    centroid = (w[:, None] * points).sum(0) / total
    off = points - centroid
    return centroid, (w[:, None] * off).T @ off / total


def extent(t, points, tau, power, start=None):
    """Centre and size under the trainer's rule ``pose_power``: the point minimising the weighted
    ``power``-th moment of the distance (Newton steps), and ``moment^(1/power)`` about it."""
    w = 1.0 / (1.0 + np.exp(-np.clip(t / tau, -60, 60)))
    centre = (w[:, None] * points).sum(0) / (w.sum() + 1e-6) if start is None else np.asarray(start, float).copy()
    for _ in range(8 if power != 2.0 else 0):
        y = points - centre
        rho = np.maximum(np.linalg.norm(y, axis=1), 1e-9)
        q = w * rho ** (power - 2.0)
        unit = y / rho[:, None]
        hessian = q.sum() * np.eye(3) + (power - 2.0) * np.einsum("n,ni,nj->ij", q, unit, unit)
        centre = centre + np.linalg.solve(hessian, (q[:, None] * y).sum(0))
    rho = np.linalg.norm(points - centre, axis=1)
    return centre, float(((w * rho ** power).sum() / (w.sum() + 1e-6)) ** (1.0 / power))


def near_identity_rotation(cov):
    _, vec = np.linalg.eigh(cov)
    rows, cols = linear_sum_assignment(-np.abs(vec))
    rot = np.zeros((3, 3))
    for axis, col in zip(rows, cols):
        v = vec[:, col]
        rot[axis] = v if v[axis] >= 0 else -v
    if np.linalg.det(rot) < 0:
        rot[np.argmin(np.abs(np.diag(rot)))] *= -1
    return rot


def pose(groups, thin, cfg, points, rng, fit_reach=0.0):
    """Groups in the trainer's pose convention, and whether the shape had to be scaled down to fit.

    The first group's radius equals ``thin`` when given."""
    every = np.concatenate([p for p, _ in groups])
    centre = every.mean(0)
    _, vec = np.linalg.eigh(np.cov((every - centre).T) + 1e-9 * np.eye(3))
    rot = vec[:, rng.permutation(3)].T * rng.choice([-1.0, 1.0], size=3)[:, None]     # random axis order and signs
    if np.linalg.det(rot) < 0:
        rot[0] *= -1
    scale = 0.5 / max(1e-6, np.linalg.norm(every - centre, axis=1).max())
    groups = [((p - centre) @ rot.T * scale, r * scale) for p, r in groups]
    if thin is not None:
        ratio = thin / groups[0][1]
        groups = [(p, r * ratio) for p, r in groups]
    tau, power, target = float(cfg.soft_tau), float(cfg.pose_power), L.pose_size_target(cfg)
    for round_ in range(8):
        t = field(groups, points)
        _, cov = moments(t, points, tau)
        if not np.isfinite(cov).all() or np.trace(cov) <= 0:
            return None, False
        centre, size = extent(t, points, tau, power)                  # position and size: the trainer's rule
        s = target / size
        rot = near_identity_rotation(cov)                             # orientation: principal axes of the mass
        groups = [((p - centre) @ rot.T * s, r * s) for p, r in groups]
        if thin is not None:                                         # thickness is absolute: put it back
            ratio = thin / groups[0][1]
            groups = [(p, r * ratio) for p, r in groups]
    # A shape that reaches further than fit_reach at the trainer's size is scaled down until it fits
    # (it is then smaller than the size target on purpose; train with anchor_exempt_size).
    limited = False
    for _ in range(5 if fit_reach else 0):
        reach = max(float(np.linalg.norm(p, axis=1).max() + r) for p, r in groups)
        if reach <= fit_reach:
            break
        limited = True
        k = 0.995 * fit_reach / reach
        groups = [(p * k, r * k) for p, r in groups]
        if thin is not None:
            ratio = thin / groups[0][1]
            groups = [(p, r * ratio) for p, r in groups]
        t = field(groups, points)                                     # centre and axes again, size left alone
        _, cov = moments(t, points, tau)
        centre, _ = extent(t, points, tau, power)
        rot = near_identity_rotation(cov)
        groups = [((p - centre) @ rot.T, r) for p, r in groups]
    return groups, limited


def check(grid, groups, cfg, outer, thin_limit):
    res = grid.shape[0]
    h = 2.0 / (res - 1)
    reach = max(float(np.linalg.norm(p, axis=1).max() + r) for p, r in groups)
    if reach > outer:
        return None, "outside the outer radius"
    solid = grid > 0
    topo = D._topology_of_mask(solid)
    if topo["pieces"] != 1 or topo["cavities"] != 0 or topo["handles"] != 0:
        return None, "not one simple piece on the grid"
    half = 0.5 * float(cfg.min_feature)
    core = grid > half
    if not core.any():
        return None, "thinner than the minimum feature everywhere"
    opened = ndi.distance_transform_edt(~core, sampling=h) <= half + 0.5 * h
    thin = float((solid & ~opened).sum()) / float(solid.sum())
    axis = np.linspace(-1, 1, res)
    radius = np.sqrt(axis[:, None, None] ** 2 + axis[None, :, None] ** 2 + axis[None, None, :] ** 2)
    far = grid < -half
    closed = ndi.distance_transform_edt(~far, sampling=h) <= half + 0.5 * h
    gap = float((~solid & (radius < float(cfg.bound_radius)) & ~closed).sum()) / float(solid.sum())
    if thin > thin_limit:
        return None, "thin features"
    if gap > thin_limit:
        return None, "thin gaps"
    return {"max_radius": reach, "thin_fraction": thin, "gap_fraction": gap, "volume_fraction": float(solid.mean())}, ""


def work(job):
    key, index, spec, seed, resolution, outer, thin_limit, min_gap, power, fit_reach = job
    cfg = DataFreeConfig(pose_power=power)
    rng = np.random.default_rng([seed, index])
    groups, limited = pose(spec["groups"], spec["thin"], cfg, grid_points(64), rng, fit_reach)
    if groups is None:
        return key, index, None, "pose failed"
    grid = field(groups, grid_points(resolution)).reshape(resolution, resolution, resolution)
    stats, reason = check(grid, groups, cfg, outer, thin_limit)
    if stats is None:
        return key, index, None, reason
    centroid, cov = moments(grid.reshape(-1), grid_points(resolution), float(cfg.soft_tau))
    centre, size = extent(grid.reshape(-1), grid_points(resolution), float(cfg.soft_tau), float(cfg.pose_power))
    stats.update({"scaled_down_to_fit": bool(limited),
                  "centre_offset": float(np.linalg.norm(centre)), "size": size, "size_target": L.pose_size_target(cfg),
                  "centroid_offset": float(np.linalg.norm(centroid)), "trace": float(np.trace(cov)),
                  "offdiag_max": float(np.abs(cov - np.diag(np.diag(cov))).max()),
                  "axes": [float(x) for x in np.sqrt(np.sort(np.diag(cov))[::-1])]})
    desc = SD.describe(grid, soft_tau=float(cfg.soft_tau), seed=index)
    if 0.0 < desc["gap"] < min_gap:
        return key, index, None, "nearest gap narrower than the margin"
    return key, index, {"grid": grid.astype(np.float16), "stats": stats, "descriptor": desc, "note": spec["note"]}, ""


# ------------------------------------------------------------------ picture
def render(grid, size=150):
    """Shaded picture of the solid from the usual viewing angle (elevation 22, azimuth 35)."""
    res = grid.shape[0]
    elev, azim, extent = np.deg2rad(22.0), np.deg2rad(35.0), 0.92
    w = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
    u = np.cross([0.0, 0.0, 1.0], w); u /= np.linalg.norm(u)
    v = np.cross(w, u)
    axis = np.linspace(-extent, extent, size)
    a, b, c = np.meshgrid(axis, axis, axis, indexing="ij")
    pts = a[..., None] * u + b[..., None] * v + c[..., None] * w
    F = ndi.map_coordinates(np.asarray(grid, np.float32), ((pts + 1) / 2 * (res - 1)).reshape(-1, 3).T, order=1,
                            mode="constant", cval=-1.0).reshape(size, size, size)
    cell = 2 * extent / (size - 1)
    gu, gv, gw = np.gradient(F, cell)
    first = (F[..., ::-1] > 0).argmax(-1)
    idx = (size - 1 - first)[..., None]
    normal = -np.stack([np.take_along_axis(g, idx, -1)[..., 0] for g in (gu, gv, gw)], -1)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6)
    light = np.array([-0.45, 0.55, 0.70]); light /= np.linalg.norm(light)
    shade = 0.38 + 0.62 * np.clip(normal @ light, 0, None)
    cover = np.clip(F.max(-1) / cell + 0.5, 0, 1)
    blue, paper = np.array([0x7f, 0xa8, 0xd6]) / 255, np.array([252, 252, 251]) / 255
    img = cover[..., None] * shade[..., None] * blue + (1 - cover[..., None]) * paper
    return img.transpose(1, 0, 2)[::-1]


def contact_sheet(records, grids, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted = "#1f1f1e", "#8a8985"
    cols = 8
    rows = int(np.ceil(len(records) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.35 * cols, 2.95 * rows), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes.ravel():
        ax.set_axis_off()
    for k, (rec, grid) in enumerate(zip(records, grids)):
        ax = axes[k // cols, k % cols]
        ax.imshow(render(grid), interpolation="bilinear")
        d = rec["descriptor"]
        star = " *" if rec.get("scaled_down_to_fit") else ""
        ax.set_title(f'{rec["rank"]}  {rec["family"]}{star}', fontsize=9.5, color=ink, pad=3)
        gap = "none" if d["gap"] <= 0 else f'{d["gap"]:.2f}'
        ax.text(0.5, -0.02, f'rod {d["linear"]:.2f}  sheet {d["planar"]:.2f}  blob {d["round"]:.2f}\n'
                            f'rays {d["ray_crossings"]:.1f}  fold {d["fold"]:.2f}  solid {d["solidity"]:.2f}\n'
                            f'thick {d["thickness"]:.2f}  branch {d["branches"]:.0f}  gap {gap}',
                transform=ax.transAxes, ha="center", va="top", fontsize=7.2, color=muted, linespacing=1.25)
    fig.suptitle(title, fontsize=13, color=ink, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975], h_pad=2.6)
    fig.savefig(path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--count", type=int, default=64)
    ap.add_argument("--resolution", type=int, default=96)
    ap.add_argument("--outer", type=float, default=0.83, help="largest radius allowed (decoder limit 0.85)")
    ap.add_argument("--thin-limit", type=float, default=0.06)
    ap.add_argument("--min-gap", type=float, default=0.14,
                    help="narrowest gap allowed between two parts of a shape (the trainer's minimum feature is 0.08)")
    ap.add_argument("--thickness", type=float, default=1.0,
                    help="multiplier on the rod radii (0.06 to 0.10) and sheet half-thicknesses (0.055 to 0.08). "
                         "At 1 three quarters of the anchors are under 0.16 across and the decoder reproduces the "
                         "rods as strings of beads (run P3); the Thingi anchors that worked have a median of 0.37")
    ap.add_argument("--fit-reach", type=float, default=0.0,
                    help="scale a shape that reaches further than this down until it fits (0: reject it instead); "
                         "such anchors are smaller than the trainer's size target, so train with anchor_exempt_size")
    ap.add_argument("--pose-power", type=float, default=None,
                    help="the trainer's pose_power the anchors are posed for (default: the trainer's default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--sheet", default="")
    ap.add_argument("--kinds", default="rod,sheet,mixed,blob", help="which kinds of shape to offer as candidates")
    ap.add_argument("--extend-from", default="",
                    help="folder of an existing anchor set: its anchors are kept, in their order, and --count NEW "
                         "ones are chosen to be as different as possible from them and from each other")
    ap.add_argument("--tag", default="", help="a word put in front of the family name of the anchors chosen here")
    ap.add_argument("--id-base", type=int, default=9_000_000, help="file ids are this plus the candidate number")
    args = ap.parse_args()
    if os.path.exists(os.path.join(args.output_dir, "anchors.pt")):
        raise SystemExit(f"{args.output_dir} already holds anchors.pt; refusing to overwrite")
    os.makedirs(args.output_dir, exist_ok=True)
    power = float(DataFreeConfig().pose_power if args.pose_power is None else args.pose_power)
    cfg = DataFreeConfig(pose_power=power)
    kept_records, kept_grids = [], None
    if args.extend_from:
        old = torch.load(os.path.join(args.extend_from, "anchors.pt"), map_location="cpu", weights_only=False)
        if int(old["resolution"]) != args.resolution or abs(float(old.get("pose_power", 2.0)) - power) > 1e-9:
            raise SystemExit("the set to extend has another resolution or pose rule")
        kept_grids = old["grid"].numpy()
        kept_records = json.load(open(os.path.join(args.extend_from, "anchors.json")))["anchors"]
        if len(kept_records) != kept_grids.shape[0]:
            raise SystemExit("anchors.json and anchors.pt of the set to extend disagree")
        if any(args.id_base <= int(r["file_id"]) < args.id_base + 100_000 for r in kept_records):
            raise SystemExit("--id-base overlaps the ids of the set to extend; give another one")
    wanted = set(k.strip() for k in args.kinds.split(",") if k.strip())
    specs = {key: v for key, v in families(np.random.default_rng(args.seed), args.thickness).items() if key[0] in wanted}
    jobs, index = [], 0
    for key, variants in specs.items():
        for spec in variants:
            jobs.append((key, index, spec, args.seed, args.resolution, args.outer, args.thin_limit, args.min_gap, power,
                         args.fit_reach))
            index += 1
    print(f"{len(jobs)} candidates in {len(specs)} families; {args.workers} workers", flush=True)
    with mp.get_context("fork").Pool(args.workers) as pool:
        results = pool.map(work, jobs, chunksize=1)
    reasons, good = {}, []
    for key, index, result, reason in results:
        if result is None:
            reasons[reason] = reasons.get(reason, 0) + 1
            reasons.setdefault("by family", {}).setdefault(key[1], []).append(reason)
        else:
            good.append({"kind": key[0], "family": key[1], "index": index, **result})
    print(f"{len(good)} eligible; rejected: {json.dumps(reasons)}", flush=True)
    if len(good) < args.count:
        print(f"only {len(good)} eligible for {args.count} anchors: keeping all of them", flush=True)

    # selection in descriptor space: one per family first, then the rest; always the farthest from the chosen
    vec = np.stack([SD.vector(g["descriptor"]) for g in good])
    kept_vec = np.stack([SD.vector(r["descriptor"]) for r in kept_records]) if kept_records else np.zeros((0, vec.shape[1]))
    every = np.concatenate([vec, kept_vec])
    scale = every.std(0); scale[scale < 1e-6] = 1.0
    z = (vec - every.mean(0)) / scale
    ball = (np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]) - every.mean(0)) / scale   # a ball's descriptor
    nearest = np.linalg.norm(z - ball, axis=1)
    for old_z in (kept_vec - every.mean(0)) / scale:                   # the anchors already in the set count as chosen
        nearest = np.minimum(nearest, np.linalg.norm(z - old_z, axis=1))
    chosen, used = [], set()
    order = {"rod": 0, "sheet": 1, "mixed": 2, "blob": 3}
    while len(chosen) < min(args.count, len(good)):
        pool_ = [i for i in range(len(good)) if i not in chosen and good[i]["family"] not in used]
        if not pool_:
            used = set()                                              # every family has one: start another round
            continue
        pick = max(pool_, key=lambda i: nearest[i])
        chosen.append(pick); used.add(good[pick]["family"])
        nearest = np.minimum(nearest, np.linalg.norm(z - z[pick], axis=1))
    chosen.sort(key=lambda i: (order[good[i]["kind"]], good[i]["family"], good[i]["index"]))

    records, grids = list(kept_records), ([] if kept_grids is None else list(kept_grids))
    for pick in chosen:
        g = good[pick]
        grids.append(g["grid"])
        records.append({"rank": len(records), "file_id": args.id_base + g["index"], "kind": g["kind"],
                        "family": (args.tag + " " if args.tag else "") + g["family"],
                        "note": g["note"], "descriptor": g["descriptor"], **g["stats"]})
    grids = np.stack(grids)
    torch.save({"grid": torch.from_numpy(grids), "resolution": args.resolution, "positive_inside": True,
                "pose_power": power, "file_ids": [r["file_id"] for r in records]},
               os.path.join(args.output_dir, "anchors.pt"))
    json.dump({"source": "procedural (scripts/build_data_free_procedural_anchors.py)", "arguments": vars(args),
               "candidates": len(jobs), "eligible": len(good), "rejected": reasons,
               "pose_power": power, "size_target": L.pose_size_target(cfg),
               "cov_trace_target": float(cfg.cov_trace_target), "soft_tau": float(cfg.soft_tau),
               "descriptor_keys": list(SD.KEYS), "anchors": records,
               "unused": [{"kind": g["kind"], "family": g["family"], "descriptor": g["descriptor"]}
                          for i, g in enumerate(good) if i not in chosen]},
              open(os.path.join(args.output_dir, "anchors.json"), "w"), indent=1)
    print("saved", os.path.join(args.output_dir, "anchors.pt"), flush=True)
    for kind in order:
        rows = [r for r in records if r["kind"] == kind]
        print(f"{kind}: {len(rows)} anchors in {len(set(r['family'] for r in rows))} families", flush=True)
    new = records[len(kept_records):]
    if kept_records and new:
        thick = [r["descriptor"]["thickness"] for r in new]
        before = [r["descriptor"]["thickness"] for r in kept_records]
        print(f"added {len(new)}: thickness {min(thick):.2f} to {max(thick):.2f} (median {float(np.median(thick)):.2f}); "
              f"the {len(kept_records)} kept ones: {min(before):.2f} to {max(before):.2f} (median {float(np.median(before)):.2f}); "
              f"scaled down to fit: {sum(bool(r.get('scaled_down_to_fit')) for r in new)}", flush=True)
    if args.sheet:
        lead = (f"{len(records)} procedural anchor objects: the {len(kept_records)} of the earlier set (rods, sheets, mixed, blobs), "
                f"then {len(new)} added ones" if kept_records else
                f"{len(records)} procedural anchor objects, in the order rods, sheets, mixed, blobs")
        contact_sheet(records, grids, args.sheet,
                      lead
                      + ("   (* = scaled down to fit inside the outer radius)\n" if any(r.get("scaled_down_to_fit") for r in records) else "\n") +
                      "Under each: local character (rod / sheet / blob shares); mean surface crossings of rays from the "
                      "centre (1 = star-shaped); fold; solidity; thickness; branches; nearest gap")
        print("saved", args.sheet, flush=True)


if __name__ == "__main__":
    main()
