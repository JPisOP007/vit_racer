#!/usr/bin/env python3
"""Watch a race.

    python play.py --drivers my_driver
    python play.py --drivers my_driver baseline --laps 3
    python play.py --drivers my_driver friend_ai --start spread --sensors

    python play.py --drivers my_driver --live      # watch it while it trains

`--live` reloads the agent's weights from disk whenever training writes a
new checkpoint, without interrupting the car. Leave it open in one terminal
and run train.py in another: the same lap gets visibly faster as the
iterations land.

Keys: TAB switch car | C follow/overview | S sensor rays | M map
      SPACE pause | 1-4 speed | R restart | ESC quit
"""

import argparse
import os
import sys
import time

from racer.agent_api import Agent
from racer.loader import available_drivers, describe, load_drivers, warn_untrained
from racer.policy import Policy
from racer.race import Race
from racer.render import Renderer
from racer.track import Track
from racer.tournament import format_time


def build_race(track, entries, laps, start):
    return Race(track, entries, laps=laps, start=start)


class LiveWeights:
    """Hot-reloads trained weights into the cars while they are driving.

    Training rewrites its checkpoint about once a second; a lap takes a
    minute. Restarting the race on every write would mean never seeing a
    complete lap, so the policy is swapped underneath the running car
    instead. The car does not notice, and neither does the physics.
    """

    POLL = 0.5      # seconds between checks; stat() is cheap but not free

    def __init__(self, entries, which="last", override=None):
        self.agents = []
        for _label, inst in entries:
            if isinstance(inst, Agent):
                path = override or os.path.join(
                    "runs", type(inst).module_name(), which + ".npz")
                override = None            # only the first agent gets it
                inst.checkpoint = path
                self.agents.append([inst, path, None])   # agent, path, mtime
        self.next_poll = 0.0
        self.status = ""

    def active(self):
        return bool(self.agents)

    def poll(self, force=False):
        now = time.monotonic()
        if not force and now < self.next_poll:
            return
        self.next_poll = now + self.POLL

        notes = []
        for entry in self.agents:
            agent, path, seen = entry
            try:
                stamp = os.path.getmtime(path)
            except OSError:
                notes.append(f"{agent.name}: waiting for {path}")
                continue
            if stamp != seen:
                try:
                    agent.policy = Policy.load(path)
                    agent.trained = True
                    entry[2] = stamp
                except Exception:
                    continue          # half-written; try again next poll
            notes.append(self._describe(agent))
        self.status = "   ".join(n for n in notes if n)

    @staticmethod
    def _describe(agent):
        meta = getattr(agent.policy, "meta", {}) or {}
        bits = [agent.name]
        if "iterations" in meta:
            bits.append(f"iteration {int(meta['iterations'])}")
        if "best_lap" in meta and float(meta["best_lap"]):
            bits.append(f"best {format_time(float(meta['best_lap']))}")
        return "  ".join(bits)


def main():
    ap = argparse.ArgumentParser(description="Race some AIs, with a window.")
    ap.add_argument("--drivers", nargs="+", default=["baseline"],
                    help="module names from drivers/")
    ap.add_argument("--track", default="track_data/vit_bhopal.json")
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--start", choices=["grid", "spread"], default="grid")
    ap.add_argument("--sensors", action="store_true",
                    help="show the sensor rays from the start")
    ap.add_argument("--no-map", action="store_true",
                    help="hide the satellite background")
    ap.add_argument("--live", nargs="?", const="last", choices=["last", "best"],
                    help="follow a training run: reload weights as they are "
                         "written, and start the next race automatically")
    ap.add_argument("--weights",
                    help="with --live, an explicit checkpoint file to follow "
                         "instead of runs/<driver>/")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("available drivers:")
        for n in available_drivers():
            try:
                kind, path = describe(n)
            except Exception as exc:
                kind, path = f"broken: {exc}", ""
            print(f"  {n:<16} {kind:<18} {path}")
        return 0

    track = Track.load(args.track)
    entries = load_drivers(args.drivers)

    live = (LiveWeights(entries, args.live, args.weights)
            if args.live else None)
    if live and not live.active():
        print("--live needs at least one trained agent among the drivers.",
              file=sys.stderr)
        return 2
    if live:
        live.poll(force=True)      # pick up whatever exists before we start
    else:
        warn_untrained(args.drivers)

    race = build_race(track, entries, args.laps, args.start)

    view = Renderer(race, show_map=not args.no_map)
    view.show_rays = args.sensors

    running = True
    while running:
        event = view.handle_events()
        if event == "restart":
            race = build_race(track, entries, args.laps, args.start)
            view.race = race
            view.focus = 0
            continue
        running = bool(event)

        if live:
            live.poll()
            view.banner = live.status
            if race.finished:
                race = build_race(track, entries, args.laps, args.start)
                view.race = race

        if not view.paused:
            for _ in range(view.speed_mult):
                if not race.finished:
                    race.step()
        view.draw()

    if live:
        return 0

    print(f"\nTrack: {track.name}  ({track.length:.0f} m)")
    print(f"{'':3} {'driver':<16} {'race time':>10} {'best lap':>10} "
          f"{'off-trk':>8} {'hits':>5}")
    print("-" * 58)
    for r in race.results():
        note = f"  DNF: {r['dnf_reason']}" if r["dnf"] else ""
        print(f"{r['position']:<3} {r['name']:<16} "
              f"{format_time(r['race_time']):>10} {format_time(r['best_lap']):>10} "
              f"{r['off_track_time']:7.1f}s {r['collisions']:5d}{note}")
        if r["first_error"]:
            print("     exception in driver:")
            for line in r["first_error"].strip().splitlines()[-3:]:
                print("       " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
