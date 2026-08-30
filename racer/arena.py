"""The whole population, on the map, at the same time.

Evolution already works by running a crowd of slightly different drivers and
keeping what the good ones did. Normally that crowd is spread across worker
processes and you never see it. The arena puts all of them in one Race
instead, starting from the same spot, driving simultaneously, passing
through each other like ghosts.

What you watch is the actual algorithm, not an animation of it. Every car
on screen is a real candidate being scored, and the ones still moving at the
end are the ones that decide where the next generation is sampled from.

    python train.py --driver my_driver --algo cem --arena

Two honest caveats.

*It is slower than training properly.* Thirty-two cars in one process is
about a hundred times less throughput than `--workers 12` headless. The
arena is for seeing what your reward function is actually rewarding; when
you want a fast agent, run it headless.

*"Dying" is a picture, not the mechanism* - under `--algo es`. There, a
crashed car still votes, it just votes for going somewhere else. Under
`--algo cem` the picture is literal: the best few are averaged into the new
centre and everyone else is thrown away.
"""

import time

import numpy as np

from .env import BIG, CarMonitor, sample_start, seat_car
from .learn import make_optimizer
from .policy import Policy
from .race import Race


class Arena:
    def __init__(self, track, agent_cls, policy, cfg, seed=0):
        self.track = track
        self.cfg = cfg
        self.policy = policy
        self.horizon = int(cfg["horizon"])
        self.episodes = max(int(cfg.get("es_episodes", 1)), 1)
        self.dt = cfg.get("dt", 1.0 / 60.0)
        self.rng = np.random.default_rng(seed)

        self.opt = make_optimizer(policy, cfg, self.rng, seed)
        self.pop = self.opt.pop
        self.repeat = max(int(getattr(agent_cls(), "action_repeat", 1)), 1)

        # Same as headless evolution: fit the feature statistics once from
        # the newborn policy, then freeze them, so every candidate in every
        # generation is judged on identically scaled inputs.
        if policy.norm.count < 10.0:
            from .learn import make_env, warmup_normalizer
            warmup_normalizer(make_env(track, agent_cls, policy, cfg, seed + 99),
                              policy, 2000, self.rng)

        # One agent and one set of weights per candidate. They all share the
        # feature statistics, which evolution freezes anyway.
        self.agents, self.policies = [], []
        for i in range(self.pop):
            p = Policy(policy.obs_dim, policy.act_dim, policy.hidden)
            p.norm.load(*policy.norm.state())
            self.policies.append(p)
            self.agents.append(agent_cls().attach(p))

        from .car import CarSpec
        self.mu_g = CarSpec().grip_on * 9.81

        self.generation = 0
        self.race = None
        self.fitness = np.zeros(self.pop, dtype=np.float32)
        self.distance = np.zeros(self.pop, dtype=np.float32)
        self.episode = 0
        self.history = []
        self.start_generation()

    # ------------------------------------------------------------ lifecycle
    def start_generation(self):
        self.generation += 1
        members = self.opt.ask()
        for p, th in zip(self.policies, members):
            p.actor.set_flat(th)
        self.fitness[:] = 0.0
        self.distance[:] = 0.0
        self.episode = 0
        self._start_episode()

    def _start_episode(self):
        # Name the cars for what they are, so the leaderboard shows the
        # survivors holding station among their own children.
        survivors = set(self.opt.survivors)
        entries = [((f"elite{i:02d}" if i in survivors else f"car{i:02d}"), a)
                   for i, a in enumerate(self.agents)]
        self.race = Race(self.track, entries, laps=10 ** 6, dt=self.dt,
                         time_limit=BIG, stuck_limit=BIG, collisions=False,
                         start="grid")

        grid = self.rng.random() < 0.15
        s0, lateral, speed, wobble = sample_start(self.track, self.rng,
                                                  self.mu_g, grid)
        for e in self.race.entrants:
            seat_car(self.track, e, s0, lateral, speed, wobble)

        self.monitors = [CarMonitor(self.track, e) for e in self.race.entrants]
        self.steps = 0
        self.alive = self.pop

    # ----------------------------------------------------------------- step
    def step(self):
        """Advance every car by one decision. False when the generation ends."""
        for _ in range(self.repeat):
            self.race.step()

        dt = self.repeat * self.dt
        for i, (e, mon) in enumerate(zip(self.race.entrants, self.monitors)):
            if mon.dead:
                continue
            agent = self.agents[i]
            next_obs = self.race.build_observation(e)
            info = mon.update(agent.last_obs, next_obs, e.last_control, dt)
            mon.reward += float(agent.reward(info))
            if info.terminal:
                mon.dead = True
                self.alive -= 1
                # Park it where it died so the screen shows the wreck, and
                # so Race stops spending observations on a dead candidate.
                e.dnf = True
                e.dnf_reason = "stalled" if info.stalled else "off"

        self.steps += 1
        if self.alive > 0 and self.steps < self.horizon:
            return True

        for i, mon in enumerate(self.monitors):
            self.fitness[i] += mon.reward / self.episodes
            self.distance[i] += mon.distance / self.episodes

        self.episode += 1
        if self.episode < self.episodes:
            self._start_episode()
            return True
        return False

    def finish_generation(self):
        """Fold this generation's scores into the centre. Returns stats."""
        theta = self.opt.tell(self.fitness)
        self.policy.actor.set_flat(theta)
        order = np.argsort(self.fitness)
        stats = {
            "generation": self.generation,
            "return": float(self.fitness.mean()),
            "best_return": float(self.fitness.max()),
            "distance": float(self.distance.mean()),
            "best_distance": float(self.distance.max()),
            "survivors": len(self.opt.survivors),
            "winner": int(order[-1]),
            "steps": self.pop * self.horizon * self.episodes,
        }
        self.history.append(stats)
        return stats

    # ------------------------------------------------------------- for a HUD
    def status(self):
        return (f"generation {self.generation}   "
                f"alive {self.alive}/{self.pop}   {self.opt.describe()}   "
                f"furthest {self._leader_distance():.0f} m")

    def _leader_distance(self):
        return max((m.distance for m in self.monitors), default=0.0)

    def leader(self):
        """Index of the car that has got furthest this episode."""
        best, who = -1e18, 0
        for i, m in enumerate(self.monitors):
            if m.distance > best:
                best, who = m.distance, i
        return who


def arena_iterations(track, agent_cls, policy, cfg, seed=0, show_map=True):
    """Generator of per-generation stats, drawing the population as it goes.

    Deliberately the same shape as learn.ppo() and learn.es(), so train.py
    logs, evaluates and checkpoints an arena run without knowing it is one.
    Closing the window ends the generator, and training shuts down cleanly.
    """
    from .render import Renderer

    arena = Arena(track, agent_cls, policy, cfg, seed)
    view = Renderer(arena.race, show_map=show_map)
    view.follow = True

    for _gen in range(int(cfg["iterations"])):
        t0 = time.perf_counter()
        finished = False
        while not finished:
            event = view.handle_events()
            if not event:
                return                      # window closed: stop training
            if event == "restart":
                arena.start_generation()

            if not view.paused:
                for _ in range(view.speed_mult):
                    if not arena.step():
                        finished = True
                        break

            view.race = arena.race          # a new episode makes a new Race
            view.focus = arena.leader()
            view.banner = arena.status()
            view.draw()

        stats = arena.finish_generation()
        yield {
            "iteration": stats["generation"],
            "steps": stats["steps"] * stats["generation"],
            "return": stats["return"],
            "distance": stats["distance"],
            "crash_rate": 0.0,
            "std": arena.opt.sigma,
            "iter_s": time.perf_counter() - t0,
            "collect_s": time.perf_counter() - t0,
        }
        arena.start_generation()
        view.race = arena.race
