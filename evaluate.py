#!/usr/bin/env python3
"""Headless scoring. No window, runs far faster than real time.

    python evaluate.py --drivers my_driver baseline
    python evaluate.py --drivers my_driver friend_ai llm_gpt --mode championship
    python evaluate.py --drivers my_driver --laps 5 --mode trial
"""

import argparse
import sys
import time

from racer.loader import available_drivers, describe, warn_untrained
from racer.tournament import championship, format_time, time_trial
from racer.track import Track


def main():
    ap = argparse.ArgumentParser(description="Score racing AIs, no graphics.")
    ap.add_argument("--drivers", nargs="+",
                    help="module names from drivers/, e.g. my_driver baseline")
    ap.add_argument("--track", default="track_data/vit_bhopal.json")
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--mode", choices=["trial", "championship", "both"],
                    default="both")
    ap.add_argument("--races", type=int, default=5,
                    help="races in the championship")
    ap.add_argument("--start", choices=["grid", "spread"], default="grid")
    ap.add_argument("--list", action="store_true", help="list drivers and exit")
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

    if not args.drivers:
        ap.error("give me at least one driver, or --list to see them")

    track = Track.load(args.track)
    warn_untrained(args.drivers)
    print(f"Track: {track.name}  ({track.length:.0f} m)")
    print(f"Drivers: {', '.join(args.drivers)}\n")

    t0 = time.perf_counter()

    if args.mode in ("trial", "both"):
        rows = time_trial(track, args.drivers, laps=args.laps)
        print("TIME TRIAL  (alone on track - pure pace)")
        print(f"{'':3} {'driver':<16} {'best lap':>10} {'total':>10} "
              f"{'off-trk':>8} {'walls':>6} {'ms/tick':>8}")
        print("-" * 68)
        for i, r in enumerate(rows, 1):
            note = ""
            if r["dnf"]:
                note = f"  DNF: {r['dnf_reason']}"
            if r["errors"]:
                note += f"  [{r['errors']} exceptions]"
            print(f"{i:<3} {r['driver']:<16} {format_time(r['best_lap']):>10} "
                  f"{format_time(r['total']):>10} {r['off_track']:7.1f}s "
                  f"{r['wall_hits']:6d} {r['mean_call_ms']:7.2f}{note}")
            if r["first_error"]:
                print("      first exception:")
                for line in r["first_error"].strip().splitlines()[-3:]:
                    print("        " + line)

        if len(rows) > 1 and rows[0]["best_lap"] and rows[-1]["best_lap"]:
            gap = rows[-1]["best_lap"] - rows[0]["best_lap"]
            print(f"\n  {rows[0]['driver']} is {gap:.3f}s a lap faster "
                  f"than {rows[-1]['driver']}.")
        print()

    if args.mode in ("championship", "both") and len(args.drivers) > 1:
        table, log = championship(track, args.drivers, races=args.races,
                                  laps=args.laps, start=args.start)
        print(f"CHAMPIONSHIP  ({args.races} races x {args.laps} laps, "
              f"{args.start} starts - racecraft)")
        print(f"{'':3} {'driver':<16} {'points':>7} {'wins':>5} {'DNF':>5} "
              f"{'hits':>6} {'best lap':>10}")
        print("-" * 60)
        for i, r in enumerate(table, 1):
            print(f"{i:<3} {r['driver']:<16} {r['points']:7d} {r['wins']:5d} "
                  f"{r['dnfs']:5d} {r['collisions']:6d} "
                  f"{format_time(r['best_lap']):>10}")

        print("\n  race by race:")
        for k, race_log in enumerate(log, 1):
            order = " > ".join(f"{d}{'(dnf)' if dnf else ''}"
                               for _, d, _, dnf in sorted(race_log))
            print(f"    R{k}: {order}")
        print()

    print(f"(evaluated in {time.perf_counter() - t0:.1f}s of wall clock)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
