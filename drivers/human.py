"""Drive it yourself, with the arrow keys.

    python play.py --drivers human
    python play.py --drivers human baseline

Not an AI - it just reads the keyboard. Worth ten minutes before you write
a line of controller code, because it makes the grip limit something you
have felt rather than something you read about. Watch the `slip` bar in the
telemetry panel: when it fills up you are asking for more grip than the
tyres have, and the car runs wide no matter how hard you turn the wheel.

Only works with play.py (it needs a window to read keys from).
"""

from racer.driver_api import Control, Driver

try:
    import pygame
except ImportError:                       # headless evaluate.py run
    pygame = None


class Driver(Driver):
    name = "human"

    STEER_RATE = 2.6      # how fast the input moves toward the key you hold
    RETURN_RATE = 4.0     # how fast it recentres when you let go

    def setup(self, track_info, car_spec, race_info):
        self.steer = 0.0

    def reset(self):
        self.steer = 0.0

    def drive(self, obs):
        if pygame is None or not pygame.get_init():
            return Control(0.0, 0.0, 1.0)

        keys = pygame.key.get_pressed()
        left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        up = keys[pygame.K_UP] or keys[pygame.K_w]
        down = keys[pygame.K_DOWN] or keys[pygame.K_s]

        # Ramp the steering instead of snapping to +/-1, or it is undrivable.
        target = (1.0 if left else 0.0) - (1.0 if right else 0.0)
        rate = self.STEER_RATE if target else self.RETURN_RATE
        self.steer += max(-rate * obs.dt, min(target - self.steer, rate * obs.dt))

        return Control(steer=self.steer,
                       throttle=1.0 if up else 0.0,
                       brake=1.0 if down else 0.0)
