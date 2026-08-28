"""Vehicle physics.

A kinematic bicycle model with a friction-circle grip limit. It is simple
enough to read in one sitting, but it punishes the two mistakes that make
racing AI interesting:

  * turning faster than the tyres can hold  -> the car washes wide (understeer)
  * using all the grip for braking          -> nothing left for cornering

Every quantity is SI: metres, seconds, radians, newtons, kilograms.
"""

import math
from dataclasses import dataclass, field

G = 9.81


@dataclass
class CarSpec:
    mass: float = 1100.0
    wheelbase: float = 2.6
    length: float = 4.4
    width: float = 1.9
    radius: float = 1.7          # collision circle

    engine_force: float = 9500.0   # N at full throttle
    brake_force: float = 15000.0   # N at full brake
    max_speed: float = 62.0        # m/s hard cap
    reverse_force: float = 3500.0

    max_steer: float = 0.55        # radians at |steer| = 1
    steer_rate: float = 3.4        # radians/s of steering-wheel travel

    drag: float = 0.62             # N per (m/s)^2
    rolling: float = 0.014         # rolling resistance coefficient, on track

    grip_on: float = 1.35          # tyre friction coefficient on tarmac
    grip_off: float = 0.55         # ...and off it
    rolling_off: float = 0.14      # dirt drags you down hard


@dataclass
class CarState:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0     # radians, 0 = +x
    speed: float = 0.0       # m/s along the heading
    steer: float = 0.0       # current wheel angle, normalised -1..1
    yaw_rate: float = 0.0
    slip: float = 0.0        # 0 = gripping, 1 = at the limit, >1 = sliding
    on_track: bool = True
    spec: CarSpec = field(default_factory=CarSpec)

    def step(self, steer_cmd, throttle, brake, dt, on_track):
        sp = self.spec
        self.on_track = on_track

        # --- steering actuator: the wheel cannot snap instantly -------------
        # Sign convention, used identically everywhere in this project:
        #   POSITIVE = LEFT.  Positive steer -> positive yaw rate -> the car
        #   rotates counter-clockwise, which is left on screen.
        steer_cmd = _clamp(steer_cmd, -1.0, 1.0)
        max_delta = sp.steer_rate * dt
        self.steer += _clamp(steer_cmd - self.steer, -max_delta, max_delta)

        grip = sp.grip_on if on_track else sp.grip_off
        roll = sp.rolling if on_track else sp.rolling_off
        mu_g = grip * G                       # max total acceleration, m/s^2

        # --- longitudinal forces -------------------------------------------
        throttle = _clamp(throttle, 0.0, 1.0)
        brake = _clamp(brake, 0.0, 1.0)

        v = self.speed
        drive = throttle * sp.engine_force
        if v < 0.0:                            # reversing
            drive = throttle * sp.reverse_force

        # Brake opposes motion; below walking pace let it reverse the car.
        if v > 0.5:
            braking = -brake * sp.brake_force
        elif v < -0.5:
            braking = brake * sp.brake_force
        else:
            braking = -brake * sp.reverse_force

        resist = -sp.drag * v * abs(v) - roll * sp.mass * G * _sign(v)
        a_long = (drive + braking + resist) / sp.mass

        # Traction limit: you cannot accelerate harder than the tyres allow.
        a_long = _clamp(a_long, -mu_g * 1.05, mu_g * 1.05)

        # --- lateral: how much turn does the requested steering ask for? ----
        delta = self.steer * sp.max_steer
        v_eff = abs(v)
        yaw_want = v * math.tan(delta) / sp.wheelbase
        a_lat_want = abs(v_eff * yaw_want)

        # Friction circle: braking/accelerating eats into cornering grip.
        a_lat_max = math.sqrt(max(mu_g ** 2 - min(a_long ** 2, mu_g ** 2), 0.0))
        a_lat_max = max(a_lat_max, 0.15 * mu_g)   # never fully zero

        if a_lat_want > a_lat_max:
            scale = a_lat_max / a_lat_want
            self.yaw_rate = yaw_want * scale       # the car runs wide
            self.slip = a_lat_want / max(a_lat_max, 1e-6)
            # Scrubbing tyres bleed speed, but cap the penalty: an absurd
            # slip number must not be able to freeze the car solid.
            scrub = min(0.4 * (a_lat_want - a_lat_max), mu_g)
            a_long -= scrub * _sign(v)
        else:
            self.yaw_rate = yaw_want
            self.slip = a_lat_want / max(a_lat_max, 1e-6)

        # --- integrate ------------------------------------------------------
        new_speed = v + a_long * dt
        if v > 0 and new_speed < 0 and brake > 0:
            new_speed = 0.0                        # brakes stop, not reverse
        self.speed = _clamp(new_speed, -12.0, sp.max_speed)

        self.heading = _wrap(self.heading + self.yaw_rate * dt)
        self.x += self.speed * math.cos(self.heading) * dt
        self.y += self.speed * math.sin(self.heading) * dt

    def reset(self, x, y, heading):
        self.x, self.y, self.heading = x, y, heading
        self.speed = 0.0
        self.steer = 0.0
        self.yaw_rate = 0.0
        self.slip = 0.0
        self.on_track = True


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _sign(v):
    return 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
