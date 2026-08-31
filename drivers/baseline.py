"""A deliberately DUMB driver.

It exists so you can (a) check the harness works and (b) have something
easy to beat on day one. It aims at a point 25 m ahead on the centreline
and mashes the throttle unless it is obviously going too fast.

It has no racing line, no braking points, and no idea other cars exist.
Beating it should take you about an hour. Beating it by 10 seconds a lap
is the actual exercise.
"""

import math

from racer.driver_api import Control, Driver


class Driver(Driver):
    name = "baseline"

    TARGET_SPEED = 13.0  # m/s everywhere - slow enough to survive the hairpin

    def setup(self, track_info, car_spec, race_info):
        self.max_steer = car_spec.max_steer

    def drive(self, obs):
        # 1. aim at a point ahead, further away the faster we are going
        lookahead = 6.0 + 0.9 * max(obs.speed, 0.0)
        idx = min(range(len(obs.preview_distances)),
                  key=lambda i: abs(obs.preview_distances[i] - lookahead))
        fwd, left, _w, _k = obs.preview[idx]

        # 2. steer at it. Positive = left, for the target and the wheel alike.
        angle = math.atan2(left, max(fwd, 1.0))
        steer = angle / self.max_steer

        # 3. crude speed control
        if obs.speed < self.TARGET_SPEED:
            throttle, brake = 1.0, 0.0
        else:
            throttle, brake = 0.0, 0.3

        return Control(steer=steer, throttle=throttle, brake=brake)
