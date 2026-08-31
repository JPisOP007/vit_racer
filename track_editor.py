#!/usr/bin/env python3
"""Trace a circuit on top of the campus map.

    python track_editor.py                          # edit the VIT circuit
    python track_editor.py --track track_data/x.json --image track_data/map.png

Waypoints are stored in MAP PIXEL coordinates, so what you click is what
gets saved. The smoothing and the metre conversion happen later, at load
time, in racer/track.py.

Mouse
    left click empty space ... add a waypoint at the end of the chain
    left drag a waypoint  ... move it
    right click a waypoint ... delete it
    scroll wheel ........... zoom
    middle drag / arrows ... pan

Keys
    [ / ]  narrow / widen the track at the nearest waypoint
    { / }  narrow / widen the WHOLE track
    I      insert a waypoint into the nearest segment
    P      preview the smoothed racing surface (what the game actually uses)
    C      open / closed circuit
    Ctrl+Z undo
    S      save        R  reload from disk        ESC quit
"""

import argparse
import json
import os
import sys

import numpy as np
import pygame

from racer.geometry import catmull_rom

BG = (16, 18, 22)
POINT = (255, 214, 64)
POINT_SEL = (255, 90, 90)
LINE = (120, 200, 255)
SURFACE = (255, 255, 255)
TEXT = (235, 235, 235)
DIM = (150, 150, 155)

HIT_RADIUS = 9          # screen pixels


class Editor:
    def __init__(self, track_path, image_path=None, width=1400, height=820):
        pygame.init()
        pygame.display.set_caption("VIT Racer - track editor")
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.w, self.h = width, height
        self.font = pygame.font.SysFont("consolas,dejavusansmono,monospace", 14)
        self.small = pygame.font.SysFont("consolas,dejavusansmono,monospace", 12)

        self.track_path = track_path
        self.data = self._load_data(track_path)
        self.points = [list(p) for p in self.data["centerline"]]
        self.closed = bool(self.data.get("closed", True))

        img = image_path or os.path.join(os.path.dirname(track_path),
                                         self.data.get("image", "map.png"))
        self.image = None
        if img and os.path.exists(img):
            try:
                self.image = pygame.image.load(img).convert()
            except pygame.error:
                self.image = None

        self.zoom = 1.0
        self.pan = [0.0, 0.0]
        if self.image:
            self.zoom = min(self.w / self.image.get_width(),
                            self.h / self.image.get_height())
        self._img_cache = (None, None)

        self.selected = None
        self.dragging = False
        self.panning = False
        self.show_preview = True
        self.status = "loaded %d waypoints" % len(self.points)
        self.undo_stack = []

    # ------------------------------------------------------------- storage

    @staticmethod
    def _load_data(path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {"name": "new track", "image": "map.png", "closed": True,
                "meters_per_pixel": 0.62, "default_width": 26,
                "sample_spacing": 2.0, "centerline": []}

    def save(self):
        self.data["centerline"] = [[round(p[0], 1), round(p[1], 1), round(p[2], 1)]
                                   for p in self.points]
        self.data["closed"] = self.closed
        with open(self.track_path, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2)
        self.status = f"saved {len(self.points)} waypoints -> {self.track_path}"

    def reload(self):
        self.data = self._load_data(self.track_path)
        self.points = [list(p) for p in self.data["centerline"]]
        self.closed = bool(self.data.get("closed", True))
        self.status = "reloaded from disk"

    def push_undo(self):
        self.undo_stack.append([list(p) for p in self.points])
        del self.undo_stack[:-40]

    def undo(self):
        if self.undo_stack:
            self.points = self.undo_stack.pop()
            self.selected = None
            self.status = "undo"

    # -------------------------------------------------------------- camera

    def to_screen(self, px, py):
        return (int(px * self.zoom + self.pan[0]),
                int(py * self.zoom + self.pan[1]))

    def to_image(self, sx, sy):
        return ((sx - self.pan[0]) / self.zoom,
                (sy - self.pan[1]) / self.zoom)

    def zoom_at(self, sx, sy, factor):
        ix, iy = self.to_image(sx, sy)
        self.zoom = max(0.15, min(self.zoom * factor, 8.0))
        self.pan[0] = sx - ix * self.zoom
        self.pan[1] = sy - iy * self.zoom

    def nearest_point(self, sx, sy):
        best, best_d = None, HIT_RADIUS ** 2
        for i, p in enumerate(self.points):
            px, py = self.to_screen(p[0], p[1])
            d = (px - sx) ** 2 + (py - sy) ** 2
            if d < best_d:
                best, best_d = i, d
        return best

    def nearest_segment(self, sx, sy):
        """Index i such that the click is closest to segment i -> i+1."""
        if len(self.points) < 2:
            return None
        ix, iy = self.to_image(sx, sy)
        p = np.array([ix, iy])
        n = len(self.points)
        last = n if self.closed else n - 1
        best, best_d = None, None
        for i in range(last):
            a = np.array(self.points[i][:2])
            b = np.array(self.points[(i + 1) % n][:2])
            ab = b - a
            denom = float(ab @ ab)
            if denom < 1e-9:
                continue
            u = float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
            d = float(((p - (a + ab * u)) ** 2).sum())
            if best_d is None or d < best_d:
                best, best_d = i, d
        return best

    # ------------------------------------------------------------- drawing

    def _draw_image(self):
        if not self.image:
            return
        key = round(self.zoom, 3)
        if self._img_cache[0] != key:
            w = max(int(self.image.get_width() * self.zoom), 1)
            h = max(int(self.image.get_height() * self.zoom), 1)
            if w * h > 40_000_000:
                return
            self._img_cache = (key, pygame.transform.smoothscale(self.image, (w, h)))
        self.screen.blit(self._img_cache[1], (self.pan[0], self.pan[1]))

    def _draw_preview(self):
        """Show the actual driving surface: smoothed centreline +/- width/2."""
        if not self.show_preview or len(self.points) < 3:
            return
        arr = np.array(self.points, dtype=float)
        smooth = catmull_rom(arr, closed=self.closed, samples_per_segment=12)
        xy = smooth[:, :2]

        nxt = np.roll(xy, -1, axis=0)
        prv = np.roll(xy, 1, axis=0)
        d = nxt - prv
        norm = np.linalg.norm(d, axis=1, keepdims=True)
        d = d / np.maximum(norm, 1e-9)
        nrm = np.stack([-d[:, 1], d[:, 0]], axis=1)
        half = (smooth[:, 2] * 0.5)[:, None]

        for side in (xy + nrm * half, xy - nrm * half):
            pts = [self.to_screen(p[0], p[1]) for p in side]
            if len(pts) > 2:
                pygame.draw.lines(self.screen, SURFACE, self.closed, pts, 2)

        pts = [self.to_screen(p[0], p[1]) for p in xy]
        if len(pts) > 2:
            pygame.draw.lines(self.screen, LINE, self.closed, pts, 1)

    def _draw_points(self):
        pts = [self.to_screen(p[0], p[1]) for p in self.points]
        if len(pts) > 1 and not self.show_preview:
            pygame.draw.lines(self.screen, LINE, self.closed, pts, 1)

        for i, (sx, sy) in enumerate(pts):
            colour = POINT_SEL if i == self.selected else POINT
            pygame.draw.circle(self.screen, colour, (sx, sy), 5)
            pygame.draw.circle(self.screen, (20, 20, 20), (sx, sy), 5, 1)
            if i == 0:
                self.screen.blit(self.small.render("START", True, POINT),
                                 (sx + 8, sy - 16))

    def _draw_hud(self):
        mpp = float(self.data.get("meters_per_pixel", 0.62))
        widths = [p[2] for p in self.points] or [0]
        length_px = 0.0
        n = len(self.points)
        last = n if self.closed else n - 1
        for i in range(max(last, 0)):
            a = self.points[i]
            b = self.points[(i + 1) % n]
            length_px += ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

        panel = pygame.Surface((430, 92), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 175))
        self.screen.blit(panel, (10, 10))

        lines = [
            f"{self.data.get('name', 'track')}   "
            f"{'closed loop' if self.closed else 'open'}",
            f"{n} waypoints   ~{length_px * mpp:.0f} m   "
            f"width {min(widths) * mpp:.1f}-{max(widths) * mpp:.1f} m",
            self.status,
        ]
        for i, text in enumerate(lines):
            self.screen.blit(self.font.render(text, True,
                                              TEXT if i < 2 else (140, 220, 140)),
                             (20, 18 + i * 22))

        hint = ("L-click add/drag   R-click delete   I insert   [ ] width   "
                "P preview   C closed   Ctrl+Z undo   S save   ESC quit")
        self.screen.blit(self.small.render(hint, True, DIM), (10, self.h - 20))

    def draw(self):
        self.screen.fill(BG)
        self._draw_image()
        self._draw_preview()
        self._draw_points()
        self._draw_hud()
        pygame.display.flip()

    # -------------------------------------------------------------- events

    def default_width(self):
        if self.points:
            return self.points[-1][2]
        return float(self.data.get("default_width", 26))

    def handle_event(self, ev):
        """Returns False to quit."""
        if ev.type == pygame.QUIT:
            return False

        if ev.type == pygame.MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            self.zoom_at(mx, my, 1.12 if ev.y > 0 else 1 / 1.12)

        elif ev.type == pygame.MOUSEBUTTONDOWN:
            mx, my = ev.pos
            if ev.button == 1:
                hit = self.nearest_point(mx, my)
                if hit is None:
                    self.push_undo()
                    ix, iy = self.to_image(mx, my)
                    self.points.append([ix, iy, self.default_width()])
                    self.selected = len(self.points) - 1
                    self.dragging = True     # so you can place-and-drag in one go
                    self.status = f"added waypoint {self.selected}"
                else:
                    self.push_undo()
                    self.selected = hit
                    self.dragging = True
            elif ev.button == 2:
                self.panning = True
            elif ev.button == 3:
                hit = self.nearest_point(mx, my)
                if hit is not None:
                    self.push_undo()
                    self.points.pop(hit)
                    self.selected = None
                    self.status = "deleted a waypoint"

        elif ev.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
            self.panning = False

        elif ev.type == pygame.MOUSEMOTION:
            if self.dragging and self.selected is not None:
                ix, iy = self.to_image(*ev.pos)
                self.points[self.selected][0] = ix
                self.points[self.selected][1] = iy
            elif self.panning:
                self.pan[0] += ev.rel[0]
                self.pan[1] += ev.rel[1]

        elif ev.type == pygame.KEYDOWN:
            return self.handle_key(ev)

        return True

    def handle_key(self, ev):
        # Read the modifier from the event, not the live keyboard: the event
        # is what actually accompanied this keypress.
        mods = getattr(ev, "mod", 0)
        if ev.key == pygame.K_ESCAPE:
            return False
        if ev.key == pygame.K_z and (mods & pygame.KMOD_CTRL):
            self.undo()
        elif ev.key == pygame.K_s:
            self.save()
        elif ev.key == pygame.K_r:
            self.reload()
        elif ev.key == pygame.K_p:
            self.show_preview = not self.show_preview
        elif ev.key == pygame.K_c:
            self.closed = not self.closed
            self.status = "closed loop" if self.closed else "open track"
        elif ev.key == pygame.K_i:
            seg = self.nearest_segment(*pygame.mouse.get_pos())
            if seg is not None:
                self.push_undo()
                n = len(self.points)
                a, b = self.points[seg], self.points[(seg + 1) % n]
                mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2]
                self.points.insert(seg + 1, mid)
                self.selected = seg + 1
                self.status = f"inserted waypoint {seg + 1}"
        elif ev.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
            delta = 2.0 if ev.key == pygame.K_RIGHTBRACKET else -2.0
            if mods & pygame.KMOD_SHIFT:
                self.push_undo()
                for p in self.points:
                    p[2] = max(6.0, p[2] + delta)
                self.status = "resized whole track"
            else:
                idx = self.selected
                if idx is None:
                    idx = self.nearest_point(*pygame.mouse.get_pos())
                if idx is not None:
                    self.push_undo()
                    self.points[idx][2] = max(6.0, self.points[idx][2] + delta)
                    self.status = f"waypoint {idx} width {self.points[idx][2]:.0f}px"
        elif ev.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN):
            step = 40
            self.pan[0] += step * ((ev.key == pygame.K_LEFT) - (ev.key == pygame.K_RIGHT))
            self.pan[1] += step * ((ev.key == pygame.K_UP) - (ev.key == pygame.K_DOWN))
        return True

    def run(self):
        running = True
        while running:
            for ev in pygame.event.get():
                if not self.handle_event(ev):
                    running = False
                    break
            self.draw()
            self.clock.tick(60)
        pygame.quit()


def main():
    ap = argparse.ArgumentParser(description="Trace a circuit on a map image.")
    ap.add_argument("--track", default="track_data/vit_bhopal.json")
    ap.add_argument("--image", default=None)
    args = ap.parse_args()
    Editor(args.track, args.image).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
