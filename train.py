#!/usr/bin/env python3
"""Train a driver. This is the loop that replaces you writing the controller.

    python train.py --driver my_driver
    python train.py --driver my_driver --iterations 600 --workers 8
    python train.py --driver my_driver --algo es --workers 12
    python train.py --driver my_driver --traffic baseline baseline
    python train.py --driver my_driver --resume

Every iteration the agent drives a few thousand steps, is scored, and its
weights are nudged. Every tenth iteration it is put out for a real time
trial from the grid, and if that lap is the fastest yet the weights are
written to runs/<driver>/best.npz - which is the file play.py and
evaluate.py will race.

Stop it whenever you like with Ctrl+C; the run is saved on the way out.
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np

from racer.arena import arena_iterations
from racer.env import RacerEnv
from racer.learn import evolve, lap_eval, load_agent_class, make_env, ppo
from racer.policy import Policy
from racer.tournament import format_time
from racer.track import Track


def build_config(agent_cls, args):
    cfg = dict(agent_cls.train)
    if args.algo:
        cfg["algo"] = args.algo
    if args.iterations:
        cfg["iterations"] = args.iterations
    if args.horizon:
        cfg["horizon"] = args.horizon
    if args.steps:
        cfg["steps_per_iter"] = args.steps
    if args.lr:
        cfg["lr"] = args.lr
    if args.episodes:
        cfg["es_episodes"] = args.episodes
    cfg["traffic"] = list(args.traffic or ())
    cfg["hidden"] = list(agent_cls.hidden)
    cfg["track_path"] = os.path.abspath(args.track)
    cfg["driver"] = args.driver
    return cfg


def watch_window(args, out):
    """Open play.py alongside, following this run's checkpoint.

    A 600-iteration run is about 23 hours of simulated driving done in 13
    minutes of wall clock, so watching every step is not a thing that can
    exist. What you can watch is the newest weights driving at normal speed:
    the viewer reloads them whenever an iteration lands, so the car in the
    window keeps lapping and quietly gets better while the table scrolls.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, os.path.join(here, "play.py"),
           "--drivers", args.driver, "--track", args.track,
           "--live", "last", "--weights", os.path.join(out, "last.npz")]
    try:
        proc = subprocess.Popen(cmd, cwd=here)
    except OSError as exc:
        print(f"could not open the viewer ({exc}); training anyway",
              file=sys.stderr)
        return None
    print("watching in a separate window (close it any time)")
    return proc


def main():
    ap = argparse.ArgumentParser(description="Train a racing agent.")
    ap.add_argument("--driver", default="my_driver",
                    help="module in drivers/ that defines an Agent subclass")
    ap.add_argument("--track", default="track_data/vit_bhopal.json")
    ap.add_argument("--algo", choices=["ppo", "es", "cem", "ga"],
                    help="ppo: gradients, one driver. es/cem/ga: evolution, "
                         "a population per generation (ga breeds them)")
    ap.add_argument("--iterations", type=int)
    ap.add_argument("--steps", type=int, help="PPO: env steps per iteration")
    ap.add_argument("--horizon", type=int, help="env steps per episode")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel rollout processes (try your core count)")
    ap.add_argument("--traffic", nargs="*", default=[],
                    help="drivers to put on track during training")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="run directory (default runs/<driver>)")
    ap.add_argument("--resume", action="store_true",
                    help="carry on from runs/<driver>/last.npz")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-laps", type=int, default=2)
    ap.add_argument("--watch", action="store_true",
                    help="open a window alongside, showing the weights as "
                         "they are being learned")
    ap.add_argument("--arena", action="store_true",
                    help="evolution only: run the whole population on the "
                         "map at once and watch the bad ones drop out")
    ap.add_argument("--episodes", type=int,
                    help="evolution only: episodes each candidate is scored "
                         "over (1 is the most watchable in the arena)")
    args = ap.parse_args()

    agent_cls = load_agent_class(args.driver)
    if not hasattr(agent_cls, "features"):
        print(f"drivers/{args.driver}.py is a hand-written driver, not an "
              f"Agent - there is nothing to train.", file=sys.stderr)
        return 2

    track = Track.load(args.track)
    cfg = build_config(agent_cls, args)

    if args.arena:
        # The arena is a view of evolution: a population, scored side by
        # side. PPO has no population to show - it improves one driver.
        if args.algo == "ppo":
            print("--arena shows a population evolving; PPO trains a single "
                  "driver. Use --algo ga, es or cem.", file=sys.stderr)
            return 2
        if cfg["algo"] == "ppo":
            cfg["algo"] = "ga"
            print("--arena: using --algo ga (the best cars survive and breed "
                  "the next generation, which is what you will be watching)")
        if not args.episodes:
            # Scoring every candidate over three episodes triples how long
            # you wait between generations, and you are here to watch. One
            # is noisier and still ranks the population fairly, because all
            # of them drive the same start.
            cfg["es_episodes"] = 1
        if args.workers > 1:
            print("--arena runs in one process; ignoring --workers "
                  f"{args.workers}. Train headless when you want speed.")
            args.workers = 1
    out = args.out or os.path.join("runs", args.driver)
    os.makedirs(out, exist_ok=True)

    # How wide is this agent's feature vector? Ask it.
    probe = RacerEnv(track, agent_cls().blank(), horizon=8, seed=args.seed)
    obs_dim = probe.obs_dim()
    cfg["obs_dim"] = obs_dim
    cfg["init_std"] = cfg.get("init_std", 0.4)

    policy = Policy(obs_dim, 2, agent_cls.hidden,
                    init_std=cfg["init_std"], seed=args.seed)
    best_lap = None
    best_distance = -1e9
    start_iter = 0
    start_steps = 0

    if args.resume:
        last = os.path.join(out, "last.npz")
        if os.path.exists(last):
            policy = Policy.load(last)
            start_iter = int(policy.meta.get("iterations", 0))
            start_steps = int(policy.meta.get("steps", 0))
            print(f"resumed from {last} at iteration {start_iter}")
        bestf = os.path.join(out, "best.npz")
        if os.path.exists(bestf):
            m = Policy.load(bestf).meta
            best_lap = float(m["best_lap"]) if "best_lap" in m else None
            best_distance = float(m.get("distance", -1e9))
    else:
        # A fresh run starts with no record to beat, so its first evaluation
        # would overwrite whatever is already there - possibly something far
        # quicker. Keep the old one instead of quietly binning it.
        bestf = os.path.join(out, "best.npz")
        if os.path.exists(bestf):
            keep = os.path.join(out, "previous_best.npz")
            os.replace(bestf, keep)
            print(f"kept the existing best.npz as {keep}"
                  f"  (--resume to carry on from it instead)")

    algo = cfg["algo"]
    print(f"Track:    {track.name}  ({track.length:.0f} m)")
    print(f"Agent:    {args.driver}  ({obs_dim} features -> "
          f"{'x'.join(str(h) for h in agent_cls.hidden)} -> 2 controls, "
          f"{policy.actor_size():,} weights)")
    print(f"Algorithm:{algo.upper()}  x {cfg['iterations']} iterations"
          f"{'  x %d workers' % args.workers if args.workers > 1 else ''}"
          f"{'  vs ' + ', '.join(cfg['traffic']) if cfg['traffic'] else ''}")
    if args.arena:
        print(f"Arena:    {cfg['population']} cars per generation, "
              f"{cfg['es_episodes']} episode(s) each, "
              f"top {cfg.get('elite', 8)} survive. Watching is slow - train "
              f"headless with --workers when you want the lap time.")
    print(f"Saving to {out}/")

    viewer = watch_window(args, out) if args.watch else None
    print()

    header = (f"{'iter':>5} {'steps':>9} {'reward':>9} {'metres':>8} "
              f"{'crash':>6} {'noise':>6} {'sec':>6}   {'time trial':>12}")
    print(header)
    print("-" * len(header))

    log_path = os.path.join(out, "log.csv")
    log = open(log_path, "a" if args.resume else "w", encoding="utf-8")
    if not args.resume or os.path.getsize(log_path) == 0:
        log.write("iteration,steps,return,distance,crash_rate,std,"
                  "iter_s,best_lap\n")

    if args.arena:
        runner = arena_iterations(track, agent_cls, policy, cfg, seed=args.seed)
    elif algo == "ppo":
        runner = ppo(track, agent_cls, policy, cfg, seed=args.seed,
                     workers=args.workers)
    else:
        runner = evolve(track, agent_cls, policy, cfg, seed=args.seed,
                        workers=args.workers)

    t_start = time.perf_counter()
    interrupted = False
    stats = None
    try:
        for stats in runner:
            it = start_iter + stats["iteration"]
            steps = start_steps + stats["steps"]
            lap_note = ""

            due = (stats["iteration"] % args.eval_every == 0
                   or stats["iteration"] == cfg["iterations"])
            if due:
                row = lap_eval(track, agent_cls, policy, laps=args.eval_laps)
                lap = row["best_lap"]
                dist = row["distance"]
                improved = False
                if lap:
                    lap_note = format_time(lap)
                    if best_lap is None or lap < best_lap:
                        best_lap, improved = lap, True
                else:
                    lap_note = f"{dist:7.0f} m"
                    if best_lap is None and dist > best_distance:
                        best_distance, improved = dist, True
                if improved:
                    lap_note += " *"
                    policy.save(os.path.join(out, "best.npz"), {
                        "best_lap": best_lap if best_lap else 0.0,
                        "distance": max(dist, 0.0),
                        "iterations": it,
                    })

            print(f"{it:5d} {steps:9,} {stats['return']:9.1f} "
                  f"{stats['distance']:8.1f} {stats['crash_rate']:6.0%} "
                  f"{stats['std']:6.3f} {stats['iter_s']:6.1f}   "
                  f"{lap_note:>12}")

            log.write(f"{it},{steps},{stats['return']:.3f},"
                      f"{stats['distance']:.2f},{stats['crash_rate']:.3f},"
                      f"{stats['std']:.4f},{stats['iter_s']:.2f},"
                      f"{best_lap if best_lap else ''}\n")
            log.flush()

            policy.save(os.path.join(out, "last.npz"),
                        {"iterations": it, "steps": steps})
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted - saving where we got to")
    finally:
        log.close()
        if viewer and viewer.poll() is None:
            viewer.terminate()

    it = start_iter + (stats["iteration"] if stats else 0)
    policy.save(os.path.join(out, "last.npz"),
                {"iterations": it, "steps": start_steps +
                 (stats["steps"] if stats else 0)})

    mins = (time.perf_counter() - t_start) / 60.0
    print(f"\n{'stopped' if interrupted else 'done'} after {it} iterations "
          f"({mins:.1f} min).")
    if best_lap:
        print(f"best lap: {format_time(best_lap)}   ->  {out}/best.npz")
        print(f"\n  python play.py --drivers {args.driver}")
        print(f"  python evaluate.py --drivers {args.driver} baseline")
    else:
        print("no complete lap yet. Train for longer, or look at the reward:")
        print("  a car that never finishes a lap is usually being punished")
        print("  harder for leaving the road than it is paid for going fast.")
    return 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    sys.exit(main())
