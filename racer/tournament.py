"""Fair comparison between drivers.

Two modes, because they measure different things:

  TIME TRIAL - each driver alone on an empty track. Measures pure pace and
               nothing else. This is the honest "whose AI is faster" number.

  CHAMPIONSHIP - everyone on track together, several races, starting order
               rotated each time so nobody keeps the good grid slot. This
               measures racecraft: overtaking, defending, not crashing.

A driver that wins the time trial but loses the championship has pace but
no awareness of other cars. That is a useful thing to learn about your AI.
"""

from .loader import load_drivers
from .race import Race


POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]


def time_trial(track, driver_names, laps=3, dt=1.0 / 60.0):
    """Run every driver alone. Returns rows sorted by best lap."""
    rows = []
    for name in driver_names:
        entry = load_drivers([name])
        race = Race(track, entry, laps=laps, dt=dt, collisions=False)
        r = race.run()[0]
        rows.append({
            "driver": name,
            "label": r["name"],
            "best_lap": r["best_lap"],
            "total": r["race_time"],
            "laps": r["laps"],
            "dnf": r["dnf"],
            "dnf_reason": r["dnf_reason"],
            "off_track": r["off_track_time"],
            "wall_hits": r["wall_hits"],
            "errors": r["errors"],
            "first_error": r["first_error"],
            "mean_call_ms": r["mean_call_ms"],
            "max_call_ms": r["max_call_ms"],
            "lap_times": r["lap_times"],
        })
    rows.sort(key=lambda r: (r["best_lap"] is None, r["best_lap"] or 0.0))
    return rows


def championship(track, driver_names, races=5, laps=3, dt=1.0 / 60.0,
                 start="grid"):
    """Run several races, rotating the starting order. Returns (table, log)."""
    totals = {n: {"points": 0, "wins": 0, "finishes": 0, "dnfs": 0,
                  "collisions": 0, "best_lap": None, "label": n}
              for n in driver_names}
    log = []

    n = len(driver_names)
    for k in range(races):
        order = [driver_names[(i + k) % n] for i in range(n)]
        race = Race(track, load_drivers(order), laps=laps, dt=dt,
                    seed=k, start=start)
        results = race.run()

        # Map back from the race's display labels to the module names.
        by_index = {i: order[i] for i in range(len(order))}
        race_log = []
        for r in results:
            key = by_index[r["index"]]
            t = totals[key]
            t["label"] = r["name"]
            if r["position"] <= len(POINTS) and not r["dnf"]:
                t["points"] += POINTS[r["position"] - 1]
            if r["position"] == 1 and not r["dnf"]:
                t["wins"] += 1
            t["finishes"] += 1 if r["finished"] else 0
            t["dnfs"] += 1 if r["dnf"] else 0
            t["collisions"] += r["collisions"]
            if r["best_lap"]:
                t["best_lap"] = (r["best_lap"] if t["best_lap"] is None
                                 else min(t["best_lap"], r["best_lap"]))
            race_log.append((r["position"], key, r["race_time"], r["dnf"]))
        log.append(race_log)

    table = [dict(driver=k, **v) for k, v in totals.items()]
    table.sort(key=lambda r: (-r["points"], -r["wins"],
                              r["best_lap"] or 1e9))
    return table, log


def format_time(t):
    if t is None:
        return "   --   "
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:06.3f}" if m else f"  {s:6.3f}"
