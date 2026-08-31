"""YOUR AGENT. This is the only file you need to write.

You are not writing a controller any more. You are writing the three
choices that decide what a controller *becomes*:

    features(obs)   what the network is allowed to see
    reward(step)    what you are paying it to do
    train = {...}   how the search for the weights is run

Then you run iterations and watch the lap time fall.

    python train.py --driver my_driver --iterations 300
    python play.py  --drivers my_driver
    python evaluate.py --drivers my_driver baseline

Training writes to runs/my_driver/. `best.npz` is the fastest agent seen so
far and is what the drivers above will race; `last.npz` is where training
resumes from; `log.csv` is one row per iteration if you want to plot it.

------------------------------------------------------------------------
WHERE THE LAP TIME IS, NOW THAT A NETWORK IS DRIVING
------------------------------------------------------------------------
  1. Train the defaults, unchanged, for 300 iterations. Watch it. It will
     be scruffy but it will lap. That is your control experiment - every
     idea below has to beat it.

  2. Read the reward. `w_offtrack` and `w_crash` are the two knobs with
     personality: too soft and it learns to cut every corner, too hard and
     it crawls round terrified. Move one at a time.

  3. Look at what `features` is NOT telling it. The default sees ten
     preview points and eleven rays. It does not see how far it is into
     the corner, or where the apex is, or what the track does after the
     next one. Adding one good feature beats a hundred more iterations.

  4. Precompute a racing line in `prepare()` and feed the network its
     offset from that line instead of from the centreline. This is the
     step that separates a fast agent from a tidy one, and it is where
     knowing the circuit yourself still wins.

  5. Train with traffic (`--traffic baseline baseline`) if you care about
     the championship. An agent trained alone has never seen another car
     and will sit behind one for a whole lap.
------------------------------------------------------------------------
"""

import numpy as np

from racer.agent_api import Agent


class Driver(Agent):
    # Shown on the HUD and in results tables. Make it yours.
    name = "my_driver"

    # ------------------------------------------------------------ the network
    hidden = (64, 64)
    action_repeat = 2          # decisions at 30 Hz; the car is steady at that

    # ------------------------------------------------------------- the reward
    w_progress = 1.0           # per metre gained. The only term that says GO.
    w_offtrack = 6.0           # per second in the dirt
    w_slip = 1.5               # per second sliding past the grip limit
    w_jerk = 0.05              # per unit of steering wobble
    w_crash = 20.0             # one-off, for ending up in the scenery
    w_contact = 1.0            # one-off, for leaning on another car

    # ---------------------------------------------------------- the iterations
    train = dict(
        Agent.train,
        algo="ppo",            # "ppo" gradients | "es" / "cem" evolution
        iterations=500,
        horizon=600,           # env steps per episode (600 x 2 ticks = 20 s)
        gamma=0.995,           # ~6 s of foresight; enough to brake in time
        steps_per_iter=4096,
        lr=3e-4,
        entropy=0.003,

        # Only used by the evolution algorithms. `population` is how many
        # cars you see at once under --arena; `elite` is how many of them
        # survive the generation. The mutation knobs are what a genetic
        # algorithm (--algo ga) uses to make children differ from parents:
        # too little and the whole population converges into one driver and
        # stops improving, too much and good parents produce broken children.
        population=32,
        elite=8,
        mutation_rate=0.12,
        mutation_sigma=0.06,
    )

    # ------------------------------------------------------------------ seeing
    # These two both hand straight back to the stock implementations in
    # racer/agent_api.py. They are here as the place to start cutting: add
    # a term, drop one, and see what it does to the lap time. Deleting the
    # methods entirely changes nothing.
    def features(self, obs):
        return super().features(obs)

    # ----------------------------------------------------------------- wanting
    def reward(self, step):
        return super().reward(step)

    # --------------------------------------------------------------- precompute
    def prepare(self, track_info, car_spec, race_info):
        """Runs once per race, before the first tick. Free thinking time.

        `track_info` is the whole circuit as numpy arrays: centerline (n,2),
        width (n,), curvature (n,), tangent (n,), s (n,). Step 4 above
        lives here.
        """
        self.limit_speed = np.sqrt(
            self.mu_g / np.maximum(np.abs(track_info["curvature"]), 1e-4))
