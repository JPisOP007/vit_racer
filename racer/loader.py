"""Load driver modules by name.

`load_driver("my_driver")` imports drivers/my_driver.py and instantiates its
`Driver` class. This is what lets you drop a file into drivers/ and race it
without touching the engine.
"""

import importlib
import os
import sys


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_driver(name):
    """name is a module in drivers/, e.g. 'my_driver' or 'rival_ai'."""
    root = _project_root()
    if root not in sys.path:
        sys.path.insert(0, root)

    module = importlib.import_module(f"drivers.{name}")
    importlib.reload(module)

    if not hasattr(module, "Driver"):
        raise AttributeError(
            f"drivers/{name}.py must define a class named `Driver`."
        )
    instance = module.Driver()
    label = getattr(instance, "name", None) or name
    return label, instance


def load_drivers(names):
    """Returns [(display_name, instance), ...]; duplicate labels get suffixed."""
    out, seen = [], {}
    for n in names:
        label, inst = load_driver(n)
        seen[label] = seen.get(label, 0) + 1
        if seen[label] > 1:
            label = f"{label} #{seen[label]}"
        out.append((label, inst))
    return out


def available_drivers():
    d = os.path.join(_project_root(), "drivers")
    return sorted(f[:-3] for f in os.listdir(d)
                  if f.endswith(".py") and not f.startswith("_"))


def describe(name):
    """('scripted'|'agent'|'agent (untrained)', checkpoint path or '')."""
    from .agent_api import Agent

    module = importlib.import_module(f"drivers.{name}")
    cls = getattr(module, "Driver", None)
    if cls is None or not (isinstance(cls, type) and issubclass(cls, Agent)):
        return "scripted", ""
    path = cls().checkpoint_path()
    return ("agent" if os.path.exists(path) else "agent (untrained)"), path


def warn_untrained(names, stream=None):
    """Say something useful before a race full of empty neural networks."""
    import sys

    stream = stream or sys.stderr
    missing = []
    for n in names:
        try:
            kind, path = describe(n)
        except Exception:
            continue
        if kind.endswith("(untrained)"):
            missing.append((n, path))

    for n, path in missing:
        print(f"warning: {n} has no weights at {path} - it will not move.\n"
              f"         train it first:  python train.py --driver {n}",
              file=stream)
    return missing
