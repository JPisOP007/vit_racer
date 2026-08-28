"""Small vectorised geometry helpers.

Nothing here is game-specific; it is just maths that the track and the
sensors need. Kept separate so it is easy to unit-test.
"""

import numpy as np


def cross2(a, b):
    """2D scalar cross product. Works on (..., 2) arrays."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def ray_hit_segments(origin, angles, seg_a, seg_b, max_dist):
    """Cast R rays from one point against S line segments, all at once.

    origin  : (2,)     world position of the sensor
    angles  : (R,)     absolute world angles of each ray, radians
    seg_a   : (S, 2)   segment start points
    seg_b   : (S, 2)   segment end points
    max_dist: float    rays longer than this report max_dist ("nothing seen")

    Returns (R,) array of distances to the nearest hit.

    Maths: ray is P + t*d (t >= 0), segment is A + u*(B-A) (0 <= u <= 1).
    Setting them equal and taking cross products gives closed forms for t, u.
    """
    origin = np.asarray(origin, dtype=float)
    d = np.stack([np.cos(angles), np.sin(angles)], axis=1)   # (R, 2)
    s = seg_b - seg_a                                        # (S, 2)
    ap = seg_a - origin                                      # (S, 2)

    denom = d[:, None, 0] * s[None, :, 1] - d[:, None, 1] * s[None, :, 0]  # (R, S)

    with np.errstate(divide="ignore", invalid="ignore"):
        # t numerator does not depend on the ray direction -> shape (S,)
        t = (ap[:, 0] * s[:, 1] - ap[:, 1] * s[:, 0])[None, :] / denom
        u = (ap[None, :, 0] * d[:, None, 1] - ap[None, :, 1] * d[:, None, 0]) / denom

    ok = (np.abs(denom) > 1e-12) & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)
    t = np.where(ok, t, np.inf)
    return np.minimum(t.min(axis=1), max_dist)


def catmull_rom(points, closed=True, samples_per_segment=24):
    """Smooth a polyline of hand-clicked points into a flowing curve.

    points: (N, K) array. Columns beyond the first two (e.g. track width)
    are interpolated with the same weights, which is exactly what we want.
    """
    p = np.asarray(points, dtype=float)
    n = len(p)
    if n < 3:
        return p.copy()

    out = []
    last = n if closed else n - 1
    for i in range(last):
        p0 = p[(i - 1) % n] if closed else p[max(i - 1, 0)]
        p1 = p[i]
        p2 = p[(i + 1) % n] if closed else p[min(i + 1, n - 1)]
        p3 = p[(i + 2) % n] if closed else p[min(i + 2, n - 1)]
        t = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)[:, None]
        out.append(
            0.5 * ((2 * p1)
                   + (-p0 + p2) * t
                   + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                   + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)
        )
    return np.concatenate(out, axis=0)


def resample_by_arclength(poly, spacing, closed=True):
    """Re-space a polyline so consecutive points are `spacing` apart.

    poly: (N, K), first two columns are x, y. Extra columns ride along.
    """
    p = np.asarray(poly, dtype=float)
    xy = p[:, :2]
    if closed:
        xy = np.vstack([xy, xy[:1]])
        p = np.vstack([p, p[:1]])

    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]

    n = max(int(round(total / spacing)), 8)
    targets = np.linspace(0.0, total, n, endpoint=not closed)

    out = np.empty((len(targets), p.shape[1]))
    for k in range(p.shape[1]):
        out[:, k] = np.interp(targets, cum, p[:, k])
    return out


def wrap_angle(a):
    """Fold an angle into (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi
