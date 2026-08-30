"""The learning environment: one car, one episode, reset, repeat.

This is the bridge between the race engine and the training loop, and its
single most important property is that it *does not simulate anything of
its own*. It drives the same `Race` object that play.py and evaluate.py
drive, and it hands the agent the same `Observation` those produce. What
an agent learns here is exactly what it will do on Sunday.

The interface is the usual one:

    obs = env.reset()
    obs, reward, done, info = env.step(action)

Two decisions worth knowing about:

*Episodes start anywhere.* A car that always begins on the start line
spends its first hundred thousand steps relearning turn one and never sees
the hairpin. So most episodes drop the car at a random point on the
circuit, at a random speed the corner there can actually hold, and run for
a few seconds. One episode in six starts properly from the grid, at rest,
so the standing start is learned too.

*Episodes end early when they have stopped being informative.* Off in the
scenery, parked, or reversing - cut it, reset, spend the compute somewhere
useful.
"""

import math

import numpy as np

from .agent_api import StepInfo
from .driver_api import Control, Driver
from .loader import load_drivers
from .race import Race

BIG = 1e9


class _Puppet(Driver):
    """A driver that returns whatever the environment last told it to."""

    name = "puppet"

    def __init__(self):
        self.control = Control()
        self.obs = None

    def setup(self, track_info, car_spec, race_info):
        pass

    def reset(self):
        self.control = Control()

    def drive(self, obs):
        self.obs = obs
        return self.control


class CarMonitor:
    """One car's episode bookkeeping: progress made, trouble found, when to stop.

    Split out of RacerEnv because the arena runs thirty-two of these at once
    in a single Race, and "when has this car had enough" must mean exactly
    the same thing there as it does in training.
    """

    def __init__(self, track, entrant, offtrack_limit=3.0, stall_limit=2.0):
        self.track = track
        self.e = entrant
        self.offtrack_limit = offtrack_limit
        self.stall_limit = stall_limit
        self.reset()

    def reset(self):
        e = self.e
        self.prev_s = e.s
        self.counters = (e.off_track_time, e.wall_hits, e.collisions, e.lap)
        self.offtrack_run = 0.0
        self.stall_run = 0.0
        self.prev_control = Control()
        self.distance = 0.0
        self.reward = 0.0
        self.steps = 0
        self.dead = False

    def update(self, obs, next_obs, control, dt):
        """Call once per decision, after the physics ticks have run."""
        e = self.e
        now = (e.off_track_time, e.wall_hits, e.collisions, e.lap)
        p_off, p_walls, p_hits, p_lap = self.counters
        off, walls, hits, lap = now
        self.counters = now

        ds = _wrap_s(e.s - self.prev_s, self.track.length)
        self.prev_s = e.s

        off_time = off - p_off
        wall_hit = walls > p_walls
        collision = hits > p_hits

        # Persistent trouble: off the road, or not going anywhere.
        self.offtrack_run = self.offtrack_run + dt if off_time > 0 else 0.0
        moving = abs(e.car.speed) > 1.5 and ds > -0.5
        self.stall_run = 0.0 if moving else self.stall_run + dt

        lost = self.offtrack_run >= self.offtrack_limit
        stalled = self.stall_run >= self.stall_limit
        terminal = bool(wall_hit or lost or stalled or e.dnf)

        info = StepInfo(
            obs=obs, next_obs=next_obs, control=control,
            prev_control=self.prev_control,
            ds=ds, dt=dt, off_track_time=off_time,
            wall_hit=bool(wall_hit or lost), collision=collision,
            stalled=stalled, lap_done=lap > p_lap, terminal=terminal,
        )
        self.prev_control = control
        self.distance += ds
        self.steps += 1
        return info


class RacerEnv:
    def __init__(self, track, agent, horizon=400, opponents=(), seed=0,
                 random_start=True, grid_prob=0.15, dt=1.0 / 60.0,
                 offtrack_limit=3.0, stall_limit=2.0):
        self.track = track
        self.agent = agent
        self.horizon = int(horizon)
        self.dt = dt
        self.random_start = random_start
        self.grid_prob = grid_prob
        self.offtrack_limit = offtrack_limit
        self.stall_limit = stall_limit
        self.repeat = max(int(getattr(agent, "action_repeat", 1)), 1)
        self.rng = np.random.default_rng(seed)

        self.opponent_names = list(opponents)
        self.opponents = load_drivers(self.opponent_names) if opponents else []

        from .car import CarSpec
        spec = CarSpec()
        self.mu_g = spec.grip_on * 9.81
        info = track.info()
        agent.setup(info, spec, {"laps": 1, "n_cars": 1 + len(self.opponents),
                                 "your_index": 0, "dt": dt})

        self.race = None
        self.entrant = None
        self.puppet = _Puppet()
        self.obs = None
        self.steps = 0
        self.episode_distance = 0.0
        self.episode_reward = 0.0

    # ------------------------------------------------------------------ life
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        entries = [(self.agent.name, self.puppet)] + self.opponents
        self.race = Race(self.track, entries, laps=10 ** 6, dt=self.dt,
                         time_limit=BIG, stuck_limit=BIG,
                         collisions=bool(self.opponents), start="spread")
        self.entrant = self.race.entrants[0]
        self.agent.reset()

        self._place()

        self.steps = 0
        self.monitor = CarMonitor(self.track, self.entrant,
                                  self.offtrack_limit, self.stall_limit)
        self.episode_distance = 0.0
        self.episode_reward = 0.0

        self.obs = self.race.build_observation(self.entrant)
        return self.obs

    def _place(self):
        """Drop the cars somewhere useful for learning."""
        L = self.track.length
        grid = (not self.random_start) or self.rng.random() < self.grid_prob
        s0, lateral, speed, wobble = sample_start(self.track, self.rng,
                                                  self.mu_g, grid)
        seat_car(self.track, self.entrant, s0, lateral, speed, wobble)

        # Traffic, if any, sprinkled up and down the road from us.
        gaps = (30.0, -30.0, 60.0, -60.0, 90.0, -90.0)
        for n, e in enumerate(self.race.entrants[1:]):
            gap = gaps[n % len(gaps)]
            side = 0.22 * (1 if n % 2 else -1)
            i = int(np.searchsorted(self.track.s, (s0 + gap) % L,
                                    side="right") - 1) % self.track.n
            seat_car(self.track, e, s0 + gap,
                     side * float(self.track.width[i]),
                     max(speed * 0.9, 6.0), 0.0)

    # ------------------------------------------------------------------ step
    def step(self, action):
        e = self.entrant
        control = self.agent.to_control(action)
        self.puppet.control = control

        ticks = 0
        for _ in range(self.repeat):
            self.race.step()
            ticks += 1
            if e.dnf or e.finished:
                break

        next_obs = self.race.build_observation(e)
        info = self.monitor.update(self.obs, next_obs, control, ticks * self.dt)
        reward = float(self.agent.reward(info))

        self.obs = next_obs
        self.steps += 1
        self.episode_distance = self.monitor.distance
        self.episode_reward += reward

        timeout = self.steps >= self.horizon and not info.terminal
        return next_obs, reward, info.terminal, {
            "step": info, "timeout": timeout, "ds": info.ds,
            "distance": self.episode_distance, "return": self.episode_reward,
            "laps": self.entrant.lap, "crash": info.wall_hit,
            "stalled": info.stalled,
        }

    # ------------------------------------------------------------- utilities
    def obs_dim(self):
        obs = self.obs if self.obs is not None else self.reset()
        return int(len(self.agent.features(obs)))

    def rollout_episode(self, policy, rng=None, deterministic=True, seed=None):
        """One episode, no bookkeeping. Used by evolution strategies."""
        obs = self.reset(seed=seed)
        total = 0.0
        for _ in range(self.horizon):
            feat = self.agent.features(obs)
            action, _, _ = policy.act(feat, rng, deterministic)
            obs, r, done, _info = self.step(action)
            total += r
            if done:
                break
        return total, self.episode_distance


def sample_start(track, rng, mu_g, grid=False):
    """Where an episode begins: (arc-length, lateral, speed, heading wobble).

    A grid start is the real thing - on the line, stationary. Otherwise
    anywhere on the circuit, at a speed the corner underneath can actually
    hold, because an agent punished for a crash it was born into learns
    nothing except that the world is unfair.
    """
    L = track.length
    if grid:
        return L - 8.0, 0.0, 0.0, 0.0

    s0 = float(rng.uniform(0.0, L))
    i = int(np.searchsorted(track.s, s0 % L, side="right") - 1) % track.n
    width = float(track.width[i])
    k = abs(float(track.curvature[i]))
    v_corner = math.sqrt(mu_g / max(k, 1e-4))
    top = float(min(45.0, 0.9 * v_corner))
    return (s0,
            float(rng.uniform(-0.3, 0.3)) * width,
            float(rng.uniform(2.0, max(top, 3.0))),
            float(rng.uniform(-0.15, 0.15)))


def seat_car(track, e, s, lateral, speed, wobble):
    """Put one entrant on the track and reset its lap bookkeeping."""
    x, y, h = track.pose_at(s % track.length, lateral)
    e.car.reset(x, y, h + wobble)
    e.car.speed = speed
    proj = track.project(x, y)
    e.last_index = proj.index
    e.s = e.prev_s = proj.s
    e.lap = 0
    e.progress = proj.s
    e.best_progress = e.progress
    e.stuck_timer = 0.0
    e.checkpoints = set()
    e.lap_start = 0.0
    e.race_start = 0.0
    e.dnf = False
    e.dnf_reason = ""
    e.finished = False


def _wrap_s(ds, length):
    """Arc-length difference, folded so crossing the line is not a huge jump."""
    half = length * 0.5
    if ds > half:
        ds -= length
    elif ds < -half:
        ds += length
    return ds
