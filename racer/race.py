"""The race engine.

Owns the simulation loop: builds each driver's Observation, calls their
`drive`, applies the returned Control, resolves collisions, counts laps
and decides the finishing order. Deterministic given the same drivers.
"""

import math
import time
import traceback
from dataclasses import dataclass, field
from typing import List

import numpy as np

from .car import CarSpec, CarState
from .driver_api import Control, Observation, Opponent


DEFAULT_RAY_ANGLES = tuple(math.radians(a) for a in
                           (-90, -60, -40, -25, -12, 0, 12, 25, 40, 60, 90))
DEFAULT_PREVIEW = (5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 55.0, 70.0, 90.0, 120.0)


@dataclass
class Entrant:
    """One car plus the brain steering it, plus its bookkeeping."""
    index: int
    name: str
    driver: object
    car: CarState

    lap: int = 0
    progress: float = 0.0          # lap * length + s, used for ranking
    s: float = 0.0
    prev_s: float = 0.0
    checkpoints: set = field(default_factory=set)

    lap_start: float = 0.0
    race_start: float = 0.0        # when this car actually crossed the line
    lap_times: List[float] = field(default_factory=list)
    best_lap: float = 0.0
    last_lap: float = 0.0

    finished: bool = False
    finish_time: float = 0.0
    dnf: bool = False
    dnf_reason: str = ""

    off_track_time: float = 0.0
    wall_hits: int = 0
    collisions: int = 0

    stuck_timer: float = 0.0
    best_progress: float = 0.0

    calls: int = 0
    total_call_time: float = 0.0
    max_call_time: float = 0.0
    errors: int = 0
    first_error: str = ""

    position: int = 1
    last_control: Control = field(default_factory=Control)
    last_index: int = 0            # projection hint, keeps the search local
    post_proj: object = None       # this tick's post-move projection, if valid


class Race:
    def __init__(self, track, drivers, laps=3, dt=1.0 / 60.0,
                 time_limit=900.0, stuck_limit=15.0, ray_max=120.0,
                 collisions=True, seed=0, start="grid"):
        self.track = track
        self.laps = laps
        self.dt = dt
        self.time_limit = time_limit
        self.stuck_limit = stuck_limit
        self.ray_max = ray_max
        self.collisions_on = collisions
        self.start_mode = start          # "grid" (2-by-2) or "spread"
        self.rng = np.random.default_rng(seed)

        self.ray_angles = DEFAULT_RAY_ANGLES
        self.preview_distances = DEFAULT_PREVIEW

        self.time = 0.0
        self.finished = False
        self._contacts = set()
        self.entrants: List[Entrant] = []

        spec = CarSpec()
        for i, (name, drv) in enumerate(drivers):
            self.entrants.append(Entrant(index=i, name=name, driver=drv,
                                         car=CarState(spec=spec)))

        self._place_on_grid()
        self._call_setup()

    # ------------------------------------------------------------------ setup

    def _place_on_grid(self):
        """Put the cars on track.

        "grid"   - a two-by-two racing grid behind the start line.
        "spread" - evenly spaced around the whole lap, so a quick car is not
                   simply stuck in a train from lap one. Better for judging
                   raw pace when the drivers have no overtaking logic yet.
        """
        n = len(self.entrants)
        L = self.track.length
        for i, e in enumerate(self.entrants):
            if self.start_mode == "spread":
                start_s = (L - 8.0 - i * (L / max(n, 1))) % L
                offset = 0.0
            else:
                row, side = divmod(i, 2)
                start_s = L - (8.0 + row * 9.0)
                offset = self.track.width[0] * (0.22 if side == 0 else -0.22)
            x, y, h = self.track.pose_at(start_s, offset)
            e.car.reset(x, y, h)

            proj = self.track.project(x, y)
            e.s = e.prev_s = proj.s
            # lap = -1 means "on the grid, race not started". The first time
            # a car crosses the line its lap becomes 0 and its clock starts.
            # This keeps `progress` continuous across that crossing.
            e.lap = -1
            e.progress = proj.s - self.track.length
            e.best_progress = e.progress
            e.lap_start = 0.0
            e.checkpoints = set(range(self.track.n_checkpoints))

    def _call_setup(self):
        info = self.track.info()
        for e in self.entrants:
            race_info = {"laps": self.laps, "n_cars": len(self.entrants),
                         "your_index": e.index, "dt": self.dt}
            try:
                if hasattr(e.driver, "reset"):
                    e.driver.reset()
                e.driver.setup(info, e.car.spec, race_info)
            except Exception:
                e.errors += 1
                e.first_error = e.first_error or traceback.format_exc(limit=4)

    # ------------------------------------------------------------- simulation

    def step(self):
        if self.finished:
            return

        for e in self.entrants:
            e.post_proj = None
            if e.finished or e.dnf:
                e.car.speed *= 0.94
                continue

            proj = self.track.project(e.car.x, e.car.y, hint=e.last_index)
            e.last_index = proj.index
            obs = self.build_observation(e, proj)

            t0 = time.perf_counter()
            try:
                result = e.driver.drive(obs)
            except Exception:
                e.errors += 1
                if not e.first_error:
                    e.first_error = traceback.format_exc(limit=6)
                result = Control(0.0, 0.0, 1.0)
            elapsed = time.perf_counter() - t0

            e.calls += 1
            e.total_call_time += elapsed
            e.max_call_time = max(e.max_call_time, elapsed)

            if isinstance(result, tuple):
                result = Control(*result[:3])
            elif result is None:
                result = Control()
            ctrl = result.sanitised()
            e.last_control = ctrl

            on_track = abs(proj.lateral) <= proj.width * 0.5
            e.car.step(ctrl.steer, ctrl.throttle, ctrl.brake, self.dt, on_track)
            if not on_track:
                e.off_track_time += self.dt

            self._keep_in_world(e)

        if self.collisions_on:
            self._resolve_collisions()

        self.time += self.dt
        for e in self.entrants:
            self._update_progress(e)

        self._rank()

        alive = [e for e in self.entrants if not (e.finished or e.dnf)]
        if not alive or self.time >= self.time_limit:
            for e in alive:
                e.dnf = True
                e.dnf_reason = e.dnf_reason or "time limit"
            self.finished = True

    def _keep_in_world(self, e):
        """Soft wall at the edge of the run-off.

        We push the car back inside and scrub speed, but we do NOT pin it:
        a car facing the wall must still be able to drive itself out, or a
        single mistake would end the race.
        """
        proj = self.track.project(e.car.x, e.car.y, hint=e.last_index)
        e.last_index = proj.index
        limit = proj.width * 0.5 + 14.0
        over = abs(proj.lateral) - limit
        if over <= 0:
            # The car did not move, so this projection is still the truth
            # after the tick. _update_progress reuses it instead of
            # repeating the search, which is a third of the tick cost.
            e.post_proj = proj
            return

        e.wall_hits += 1
        nrm = np.array([-math.sin(proj.tangent), math.cos(proj.tangent)])
        outward = nrm * math.copysign(1.0, proj.lateral)
        e.car.x -= float(outward[0]) * over
        e.car.y -= float(outward[1]) * over

        heading_vec = np.array([math.cos(e.car.heading), math.sin(e.car.heading)])
        if float(heading_vec @ outward) * e.car.speed > 0:
            e.car.speed *= 0.35        # only punish driving *into* the wall
        else:
            e.car.speed *= 0.92        # already turning away: let them recover

    def _resolve_collisions(self):
        """Circle-vs-circle contact between cars.

        Cars are pushed apart and lose some closing speed. Contacts are
        tracked with hysteresis so a car riding a bumper counts as one
        collision, not one per tick.
        """
        act = [e for e in self.entrants if not (e.finished or e.dnf)]
        touching = set()

        for i in range(len(act)):
            for j in range(i + 1, len(act)):
                a, b = act[i].car, act[j].car
                dx, dy = b.x - a.x, b.y - a.y
                d = math.hypot(dx, dy)
                rsum = a.spec.radius + b.spec.radius
                if d > rsum * 1.35 or d < 1e-6:
                    continue

                pair = (act[i].index, act[j].index)
                overlapping = d < rsum
                was_touching = pair in self._contacts

                # The wide band only *sustains* an existing contact. It must
                # never create one, or the first real hit is never counted.
                if overlapping or was_touching:
                    touching.add(pair)

                if not overlapping:
                    continue
                if not was_touching:
                    act[i].collisions += 1
                    act[j].collisions += 1

                nx, ny = dx / d, dy / d
                overlap = (rsum - d) * 0.5 + 0.01
                a.x -= nx * overlap
                a.y -= ny * overlap
                b.x += nx * overlap
                b.y += ny * overlap
                act[i].post_proj = act[j].post_proj = None   # they moved

                for c, sgn in ((a, -1.0), (b, 1.0)):
                    vx = c.speed * math.cos(c.heading)
                    vy = c.speed * math.sin(c.heading)
                    if (vx * nx + vy * ny) * sgn < 0:   # driving into the other car
                        c.speed *= 0.86

        self._contacts = touching

    def _update_progress(self, e):
        if e.finished or e.dnf:
            return

        L = self.track.length
        proj = e.post_proj
        if proj is None:
            proj = self.track.project(e.car.x, e.car.y, hint=e.last_index)
        e.post_proj = None
        e.last_index = proj.index
        e.s = proj.s

        e.checkpoints.add(int(proj.s // self.track.checkpoint_spacing))

        delta = e.s - e.prev_s
        if delta < -L * 0.5:                       # crossed the line forwards
            need = self.track.n_checkpoints * 0.75
            if len(e.checkpoints) >= need:
                if e.lap < 0:
                    e.lap = 0                      # race starts now
                    e.lap_start = self.time
                    e.race_start = self.time
                else:
                    lap_time = self.time - e.lap_start
                    e.lap += 1
                    e.lap_times.append(lap_time)
                    e.last_lap = lap_time
                    e.best_lap = (lap_time if not e.best_lap
                                  else min(e.best_lap, lap_time))
                    e.lap_start = self.time
                    if e.lap >= self.laps:
                        e.finished = True
                        e.finish_time = self.time
                e.checkpoints = {0}
        elif delta > L * 0.5:                      # went backwards over it
            e.lap = max(e.lap - 1, -1)
            e.checkpoints = set()

        e.prev_s = e.s
        e.progress = e.lap * L + e.s

        if e.progress > e.best_progress + 0.5:
            e.best_progress = e.progress
            e.stuck_timer = 0.0
        else:
            e.stuck_timer += self.dt
            if e.stuck_timer > self.stuck_limit:
                e.dnf = True
                e.dnf_reason = "stuck / no progress"

    def _rank(self):
        order = sorted(self.entrants,
                       key=lambda e: (not e.finished, e.finish_time if e.finished else 0,
                                      -e.progress))
        for pos, e in enumerate(order, start=1):
            e.position = pos

    # ----------------------------------------------------------- observations

    def build_observation(self, e, proj=None):
        car = e.car
        if proj is None:
            proj = self.track.project(car.x, car.y, hint=e.last_index)

        rays = self.track.raycast(car.x, car.y, car.heading,
                                  self.ray_angles, self.ray_max)

        cos_h, sin_h = math.cos(-car.heading), math.sin(-car.heading)

        preview = []
        for (px, py, w, k) in self.track.sample_ahead(proj.s, self.preview_distances):
            rx, ry = px - car.x, py - car.y
            fwd = rx * cos_h - ry * sin_h
            lft = rx * sin_h + ry * cos_h
            preview.append((fwd, lft, w, k))

        opponents = []
        for o in self.entrants:
            if o.index == e.index:
                continue
            rx, ry = o.car.x - car.x, o.car.y - car.y
            fwd = rx * cos_h - ry * sin_h
            lft = rx * sin_h + ry * cos_h
            opponents.append(Opponent(
                index=o.index,
                forward=fwd,
                left=lft,
                distance=math.hypot(rx, ry),
                rel_speed=o.car.speed - car.speed,
                heading_diff=_wrap(o.car.heading - car.heading),
                gap_on_track=o.progress - e.progress,
            ))

        half = proj.width * 0.5
        return Observation(
            time=self.time, dt=self.dt,
            speed=car.speed, heading=car.heading, yaw_rate=car.yaw_rate,
            steer=car.steer, slip=min(car.slip, 5.0), on_track=car.on_track,
            x=car.x, y=car.y,
            progress=proj.s, lap=max(e.lap, 0),
            lateral=proj.lateral, track_width=proj.width,
            dist_left_edge=max(half - proj.lateral, 0.0),
            dist_right_edge=max(half + proj.lateral, 0.0),
            heading_error=_wrap(car.heading - proj.tangent),
            curvature=proj.curvature,
            ray_angles=self.ray_angles,
            rays=tuple(float(r) for r in rays),
            ray_max=self.ray_max,
            preview_distances=self.preview_distances,
            preview=tuple(preview),
            opponents=opponents,
            position=e.position,
            total_laps=self.laps,
            last_lap_time=e.last_lap,
            best_lap_time=e.best_lap,
        )

    # --------------------------------------------------------------- results

    def run(self, max_steps=None):
        steps = 0
        while not self.finished:
            self.step()
            steps += 1
            if max_steps and steps >= max_steps:
                break
        return self.results()

    def results(self):
        rows = []
        order = sorted(self.entrants,
                       key=lambda e: (not e.finished,
                                      e.finish_time if e.finished else 0.0,
                                      -e.progress))
        for pos, e in enumerate(order, start=1):
            rows.append({
                "position": pos,
                "name": e.name,
                "index": e.index,
                "finished": e.finished,
                "dnf": e.dnf,
                "dnf_reason": e.dnf_reason,
                "time": e.finish_time if e.finished else None,
                "race_time": (e.finish_time - e.race_start) if e.finished else None,
                "laps": max(e.lap, 0),
                "progress": e.progress,
                "best_lap": e.best_lap or None,
                "lap_times": list(e.lap_times),
                "off_track_time": e.off_track_time,
                "collisions": e.collisions,
                "wall_hits": e.wall_hits,
                "errors": e.errors,
                "first_error": e.first_error,
                "mean_call_ms": 1000 * e.total_call_time / max(e.calls, 1),
                "max_call_ms": 1000 * e.max_call_time,
            })
        return rows


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
