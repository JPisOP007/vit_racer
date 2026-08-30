"""Pygame view of a race.

Purely a viewer: it never touches the simulation state, so anything you see
here is exactly what the headless evaluator computes.

World coordinates are metres with y pointing UP. Screen pixels have y
pointing DOWN, so every conversion flips the sign of y exactly once, in
`to_screen`.
"""

import math

import numpy as np
import pygame

CAR_COLOURS = [
    (239, 83, 80), (66, 165, 245), (255, 202, 40), (102, 187, 106),
    (171, 71, 188), (255, 138, 60), (38, 198, 218), (236, 100, 170),
]

ASPHALT = (58, 58, 62)
EDGE = (238, 238, 238)
BG = (18, 20, 24)
TEXT = (235, 235, 235)
DIM = (150, 150, 155)


class Renderer:
    def __init__(self, race, width=1400, height=800, show_map=True):
        pygame.init()
        pygame.display.set_caption(f"VIT Racer - {race.track.name}")
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.w, self.h = width, height
        self.race = race
        self.track = race.track

        self.font = pygame.font.SysFont("consolas,dejavusansmono,monospace", 15)
        self.big = pygame.font.SysFont("consolas,dejavusansmono,monospace", 20, bold=True)
        self.small = pygame.font.SysFont("consolas,dejavusansmono,monospace", 12)

        self.focus = 0
        self.follow = True
        self.banner = ""          # live training status, drawn bottom-left
        self.show_rays = False
        self.show_map = show_map
        self.paused = False
        self.speed_mult = 1

        self.zoom = 1.0
        self.cam = np.array([0.0, 0.0])
        self._fit_track()

        self.map_img = None
        self._map_cache = (None, None)
        if self.track.image_path:
            try:
                self.map_img = pygame.image.load(self.track.image_path).convert()
            except pygame.error:
                self.map_img = None

    # ------------------------------------------------------------- camera

    def _fit_track(self):
        lo = self.track.center.min(axis=0) - 40
        hi = self.track.center.max(axis=0) + 40
        span = hi - lo
        self.overview_zoom = min(self.w / span[0], self.h / span[1])
        self.overview_cam = (lo + hi) * 0.5
        self.zoom = self.overview_zoom
        self.cam = self.overview_cam.copy()

    def to_screen(self, x, y):
        sx = (x - self.cam[0]) * self.zoom + self.w * 0.5
        sy = self.h * 0.5 - (y - self.cam[1]) * self.zoom   # the one y flip
        return int(sx), int(sy)

    def _update_camera(self):
        if self.follow and self.race.entrants:
            car = self.race.entrants[self.focus].car
            target = np.array([car.x, car.y])
            self.cam += (target - self.cam) * 0.15
            desired = 3.2
            self.zoom += (desired - self.zoom) * 0.08
        else:
            self.cam += (self.overview_cam - self.cam) * 0.15
            self.zoom += (self.overview_zoom - self.zoom) * 0.08

    # ------------------------------------------------------------ drawing

    def _draw_map(self):
        if not (self.show_map and self.map_img):
            return
        mpp = self.track.mpp
        scale = mpp * self.zoom
        key = round(scale, 3)
        if self._map_cache[0] != key:
            w = max(int(self.map_img.get_width() * scale), 1)
            h = max(int(self.map_img.get_height() * scale), 1)
            if w * h > 40_000_000:
                return
            self._map_cache = (key, pygame.transform.smoothscale(self.map_img, (w, h)))
        # Image pixel (0,0) sits at world (0,0); rows run toward -y.
        self.screen.blit(self._map_cache[1], self.to_screen(0.0, 0.0))

    def _draw_track(self):
        left = [self.to_screen(p[0], p[1]) for p in self.track.left]
        right = [self.to_screen(p[0], p[1]) for p in self.track.right]

        ribbon = left + right[::-1]
        if self.show_map and self.map_img:
            surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
            pygame.draw.polygon(surf, (*ASPHALT, 190), ribbon)
            self.screen.blit(surf, (0, 0))
        else:
            pygame.draw.polygon(self.screen, ASPHALT, ribbon)

        pygame.draw.lines(self.screen, EDGE, True, left, 2)
        pygame.draw.lines(self.screen, EDGE, True, right, 2)

        # dashed centreline
        step = max(int(6 / self.track.sample_spacing), 1)
        c = self.track.center
        for i in range(0, len(c) - step * 2, step * 2):
            pygame.draw.line(self.screen, (120, 120, 125),
                             self.to_screen(*c[i]), self.to_screen(*c[i + step]), 1)

        # start / finish
        a = self.track.left[0]
        b = self.track.right[0]
        pygame.draw.line(self.screen, (255, 255, 255),
                         self.to_screen(a[0], a[1]), self.to_screen(b[0], b[1]), 4)

    def _draw_rays(self, e):
        obs = self.race.build_observation(e)
        car = e.car
        origin = self.to_screen(car.x, car.y)
        for ang, dist in zip(obs.ray_angles, obs.rays):
            a = car.heading + ang
            end = self.to_screen(car.x + math.cos(a) * dist,
                                 car.y + math.sin(a) * dist)
            hit = dist < obs.ray_max - 0.5
            pygame.draw.line(self.screen, (255, 90, 90) if hit else (70, 110, 70),
                             origin, end, 1)

    def _draw_car(self, e, colour):
        car = e.car
        if e.dnf:
            # A wreck stays on screen - in the arena the pattern of where
            # cars died is most of what there is to read - but it fades back
            # so the survivors are the ones your eye follows.
            colour = tuple(int(c * 0.32 + 12) for c in colour)
        L, W = car.spec.length * 0.5, car.spec.width * 0.5
        ch, sh = math.cos(car.heading), math.sin(car.heading)
        corners = []
        for dx, dy in ((L, W), (L, -W), (-L, -W), (-L, W)):
            corners.append(self.to_screen(car.x + dx * ch - dy * sh,
                                          car.y + dx * sh + dy * ch))
        pygame.draw.polygon(self.screen, colour, corners)
        pygame.draw.polygon(self.screen, (15, 15, 15), corners, 1)

        # a nose marker so heading is obvious
        nose = self.to_screen(car.x + (L + 0.9) * ch, car.y + (L + 0.9) * sh)
        pygame.draw.circle(self.screen, (255, 255, 255), nose, 2)

        if self.zoom > 1.6:
            label = self.small.render(e.name, True, colour)
            px, py = self.to_screen(car.x, car.y)
            self.screen.blit(label, (px + 12, py - 22))

    # ---------------------------------------------------------------- HUD

    MAX_HUD_ROWS = 8

    def _draw_hud(self):
        r = self.race
        pad = 10
        shown = min(len(r.entrants), self.MAX_HUD_ROWS)
        extra = 22 if len(r.entrants) > shown else 0
        panel = pygame.Surface((330, 34 + 22 * shown + extra), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 165))
        self.screen.blit(panel, (pad, pad))

        self.screen.blit(self.font.render(
            f"lap {min(max(r.entrants[0].lap,0)+1, r.laps)}/{r.laps}"
            f"    t {r.time:6.1f}s    x{self.speed_mult}"
            f"{'  PAUSED' if self.paused else ''}",
            True, TEXT), (pad + 8, pad + 6))

        order = sorted(r.entrants, key=lambda e: e.position)
        for i, e in enumerate(order[:shown]):
            colour = CAR_COLOURS[e.index % len(CAR_COLOURS)]
            best = f"{e.best_lap:6.2f}" if e.best_lap else "   -- "
            status = "FIN" if e.finished else ("DNF" if e.dnf else f"L{max(e.lap,0)+1}")
            marker = ">" if e.index == self.focus else " "
            line = f"{marker}{e.position}. {e.name[:12]:<12} {status:>4} {best}"
            self.screen.blit(self.font.render(line, True, colour),
                             (pad + 8, pad + 30 + 22 * i))
        if extra:
            out = sum(1 for e in r.entrants if e.dnf)
            self.screen.blit(self.font.render(
                f"  ... {len(r.entrants) - shown} more, {out} out", True, DIM),
                (pad + 8, pad + 30 + 22 * shown))

        self._draw_telemetry()
        hint = ("TAB car   C camera   S sensors   M map   SPACE pause   "
                "1-4 speed   R restart   ESC quit")
        self.screen.blit(self.small.render(hint, True, DIM), (pad, self.h - 20))
        if self.banner:
            self.screen.blit(self.big.render(self.banner, True, TEXT),
                             (pad, self.h - 46))

    def _draw_telemetry(self):
        e = self.race.entrants[self.focus]
        car = e.car
        ctl = e.last_control
        x0, y0 = self.w - 250, 10

        panel = pygame.Surface((240, 150), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 165))
        self.screen.blit(panel, (x0, y0))

        colour = CAR_COLOURS[e.index % len(CAR_COLOURS)]
        self.screen.blit(self.big.render(e.name[:16], True, colour), (x0 + 10, y0 + 6))
        self.screen.blit(self.big.render(f"{car.speed * 3.6:5.0f} km/h", True, TEXT),
                         (x0 + 10, y0 + 32))

        rows = [
            ("throttle", ctl.throttle, (100, 220, 100)),
            ("brake", ctl.brake, (230, 90, 90)),
            ("steer", abs(ctl.steer), (110, 170, 240)),
            ("slip", min(car.slip, 1.5) / 1.5, (240, 190, 80)),
        ]
        for i, (label, val, col) in enumerate(rows):
            y = y0 + 62 + i * 21
            self.screen.blit(self.small.render(label, True, DIM), (x0 + 10, y))
            pygame.draw.rect(self.screen, (55, 55, 60), (x0 + 78, y + 2, 150, 11))
            pygame.draw.rect(self.screen, col,
                             (x0 + 78, y + 2, int(150 * max(0.0, min(val, 1.0))), 11))

        if not car.on_track:
            self.screen.blit(self.font.render("OFF TRACK", True, (255, 170, 60)),
                             (x0 + 10, y0 + 150))

    # --------------------------------------------------------------- loop

    def handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                elif ev.key == pygame.K_TAB:
                    self.focus = (self.focus + 1) % len(self.race.entrants)
                elif ev.key == pygame.K_c:
                    self.follow = not self.follow
                elif ev.key == pygame.K_s:
                    self.show_rays = not self.show_rays
                elif ev.key == pygame.K_m:
                    self.show_map = not self.show_map
                elif ev.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif ev.key == pygame.K_r:
                    return "restart"
                elif pygame.K_1 <= ev.key <= pygame.K_4:
                    self.speed_mult = [1, 2, 4, 8][ev.key - pygame.K_1]
        return True

    def draw(self):
        self.screen.fill(BG)
        self._update_camera()
        self._draw_map()
        self._draw_track()

        if self.show_rays and self.race.entrants:
            self._draw_rays(self.race.entrants[self.focus])

        for e in self.race.entrants:
            self._draw_car(e, CAR_COLOURS[e.index % len(CAR_COLOURS)])

        self._draw_hud()
        pygame.display.flip()
        self.clock.tick(60)
