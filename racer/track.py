"""The circuit.

A track is authored as a handful of clicked waypoints in *image pixel*
coordinates (see track_editor.py). This module turns that into a smooth,
uniformly sampled centreline in *metres*, plus left/right walls, curvature,
and the queries the simulator needs every frame:

    project(x, y)                 -> where am I on the track?
    raycast(x, y, heading, ...)   -> how far to the edges in these directions?
    pose_at(s, lateral)           -> put a car on the track at arc-length s
"""

import json
import os

import numpy as np

from .geometry import catmull_rom, ray_hit_segments, resample_by_arclength


class Projection:
    """Result of snapping a world point onto the centreline."""

    __slots__ = ("s", "lateral", "tangent", "curvature", "width", "index", "point")

    def __init__(self, s, lateral, tangent, curvature, width, index, point):
        self.s = s                  # arc-length along the centreline, metres
        self.lateral = lateral      # signed offset; + is LEFT of the racing direction
        self.tangent = tangent      # centreline heading here, radians
        self.curvature = curvature  # 1/radius, + is a left-hand bend
        self.width = width          # full track width here, metres
        self.index = index          # nearest centreline sample index
        self.point = point          # (x, y) of the closest centreline point


class Track:
    def __init__(self, data, base_dir="."):
        self.name = data.get("name", "unnamed")
        self.closed = bool(data.get("closed", True))
        self.mpp = float(data.get("meters_per_pixel", 0.65))
        self.sample_spacing = float(data.get("sample_spacing", 2.0))

        self.image_path = None
        if data.get("image"):
            candidate = os.path.join(base_dir, data["image"])
            if os.path.exists(candidate):
                self.image_path = candidate

        raw = np.asarray(data["centerline"], dtype=float)
        if raw.shape[1] == 2:                       # no per-point width given
            w = np.full((len(raw), 1), float(data.get("default_width", 24.0)))
            raw = np.hstack([raw, w])

        # Pixels -> metres. Image y grows downward but maths (and every
        # atan2 in this project) assumes y grows upward, so we flip it here,
        # once. After this line, "left" in the code is left on screen too.
        raw = raw * self.mpp
        raw[:, 1] *= -1.0

        smooth = catmull_rom(raw, closed=self.closed, samples_per_segment=24)
        samples = resample_by_arclength(smooth, self.sample_spacing, closed=self.closed)

        self.center = samples[:, :2]
        self.width = samples[:, 2]
        self.n = len(self.center)

        self._build_frames()
        self._build_walls()

        # Checkpoints stop an AI from cutting the whole circuit.
        self.n_checkpoints = max(int(self.length // 60), 12)
        self.checkpoint_spacing = self.length / self.n_checkpoints

    # ------------------------------------------------------------------ setup

    def _build_frames(self):
        c = self.center
        nxt = np.roll(c, -1, axis=0)
        prv = np.roll(c, 1, axis=0)

        seg = np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
        self.length = float(seg.sum())

        d = nxt - prv
        self.tangent = np.arctan2(d[:, 1], d[:, 0])
        self.normal = np.stack([-np.sin(self.tangent), np.cos(self.tangent)], axis=1)

        # curvature = d(heading)/ds, via a wrapped finite difference
        dtheta = np.arctan2(np.sin(np.roll(self.tangent, -1) - np.roll(self.tangent, 1)),
                            np.cos(np.roll(self.tangent, -1) - np.roll(self.tangent, 1)))
        ds = np.roll(seg, -1) + seg
        self.curvature = dtheta / np.maximum(ds, 1e-6)

        # light smoothing so the AI sees a clean curvature profile
        k = np.array([0.15, 0.2, 0.3, 0.2, 0.15])
        pad = np.concatenate([self.curvature[-2:], self.curvature, self.curvature[:2]])
        self.curvature = np.convolve(pad, k, mode="valid")

    def _build_walls(self):
        half = (self.width * 0.5)[:, None]
        self.left = self.center + self.normal * half
        self.right = self.center - self.normal * half

        def to_segments(poly):
            a = poly
            b = np.roll(poly, -1, axis=0)
            if not self.closed:
                a, b = a[:-1], b[:-1]
            return a, b

        la, lb = to_segments(self.left)
        ra, rb = to_segments(self.right)
        self.wall_a = np.vstack([la, ra])
        self.wall_b = np.vstack([lb, rb])

        # For fast raycasting: a bounding sphere per segment. A ray of length
        # R from point P can only hit a segment whose midpoint lies within
        # R + half its length of P. Testing that is one cheap vector op and
        # typically throws away ~90% of the walls before the real maths.
        self.wall_mid = (self.wall_a + self.wall_b) * 0.5
        self.wall_halflen = 0.5 * np.linalg.norm(self.wall_b - self.wall_a, axis=1)

    # ---------------------------------------------------------------- queries

    def project(self, x, y, hint=None, window=12):
        """Snap a world point onto the centreline.

        `hint` is the index returned last time for this car. A car moves at
        most ~1 m per tick, so searching a small window around the previous
        answer is enough; we fall back to a full search if the best match
        sits on the window edge (which means the car jumped or we guessed
        badly).
        """
        p = np.array([x, y], dtype=float)

        if hint is None:
            d2 = ((self.center - p) ** 2).sum(axis=1)
            i = int(np.argmin(d2))
        else:
            idx = (np.arange(hint - window, hint + window + 1)) % self.n
            d2w = ((self.center[idx] - p) ** 2).sum(axis=1)
            k = int(np.argmin(d2w))
            if k <= 1 or k >= len(idx) - 2:      # hugging the edge: re-search
                d2 = ((self.center - p) ** 2).sum(axis=1)
                i = int(np.argmin(d2))
            else:
                i = int(idx[k])

        best = None
        for j in (i - 1, i):
            a = self.center[j % self.n]
            b = self.center[(j + 1) % self.n]
            ab = b - a
            denom = float(ab @ ab)
            if denom < 1e-9:
                continue
            u = float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
            closest = a + ab * u
            dist2 = float(((p - closest) ** 2).sum())
            if best is None or dist2 < best[0]:
                best = (dist2, j % self.n, u, closest, ab)

        _, j, u, closest, ab = best
        s = (self.s[j] + u * float(np.linalg.norm(ab))) % self.length

        tangent = float(self.tangent[j])
        tv = np.array([np.cos(tangent), np.sin(tangent)])
        rel = p - closest
        lateral = float(tv[0] * rel[1] - tv[1] * rel[0])   # + = left

        return Projection(
            s=s,
            lateral=lateral,
            tangent=tangent,
            curvature=float(self.curvature[j]),
            width=float(self.width[j]),
            index=j,
            point=closest,
        )

    def raycast(self, x, y, heading, rel_angles, max_dist):
        """Distances from (x, y) to the track walls, in car-relative directions."""
        origin = np.array([x, y], dtype=float)
        angles = np.asarray(rel_angles, dtype=float) + heading

        # Discard segments that are simply too far away to be hit. Purely a
        # speed optimisation - the result is identical to testing them all.
        reach = max_dist + self.wall_halflen
        near = ((self.wall_mid - origin) ** 2).sum(axis=1) <= reach ** 2
        if not near.any():
            return np.full(len(angles), max_dist)

        return ray_hit_segments(origin, angles,
                                self.wall_a[near], self.wall_b[near], max_dist)

    def pose_at(self, s, lateral=0.0):
        """Inverse of project(): world (x, y, heading) at arc-length s."""
        s = s % self.length
        i = int(np.searchsorted(self.s, s, side="right") - 1) % self.n
        j = (i + 1) % self.n
        span = (self.s[j] - self.s[i]) % self.length
        u = 0.0 if span < 1e-9 else (s - self.s[i]) % self.length / span

        point = self.center[i] * (1 - u) + self.center[j] * u
        heading = float(self.tangent[i])
        normal = np.array([-np.sin(heading), np.cos(heading)])
        point = point + normal * lateral
        return float(point[0]), float(point[1]), heading

    def sample_ahead(self, s, distances):
        """Centreline preview: world points, widths and curvatures ahead of s."""
        targets = (s + np.asarray(distances, dtype=float)) % self.length
        idx = (np.searchsorted(self.s, targets, side="right") - 1) % self.n
        pts = self.center[idx]
        w = self.width[idx]
        k = self.curvature[idx]
        return [(float(pts[i, 0]), float(pts[i, 1]), float(w[i]), float(k[i]))
                for i in range(len(idx))]

    def info(self):
        """Read-only description handed to every Driver once, before the race.

        Everything an AI needs to precompute a racing line offline.
        """
        return {
            "name": self.name,
            "length": self.length,
            "closed": self.closed,
            "sample_spacing": self.sample_spacing,
            "centerline": self.center.copy(),      # (n, 2) metres
            "width": self.width.copy(),            # (n,)   metres
            "curvature": self.curvature.copy(),    # (n,)   1/m, + = left bend
            "tangent": self.tangent.copy(),        # (n,)   radians
            "s": self.s.copy(),                    # (n,)   arc-length
        }

    # ----------------------------------------------------------------- loading

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(data, base_dir=os.path.dirname(os.path.abspath(path)))
