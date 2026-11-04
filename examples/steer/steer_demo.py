"""Steering demo: a field of moving stripes on a 4x4 chare array whose
wave number, speed and color can be changed from a CCS client while it
runs (charm4py.steer). The picture is served on the liveViz handlers
lvConfig / lvImage, so a liveViz client (appSpector's viewer) shows it;
they are answered here, by a gather over the array, rather than through
charm4py.liveviz.

    python -m charmrun.start +p4 ++local steer_demo.py ++server ++server-port 1234 [++server-auth KEYFILE]

Then, from a client (appSpector shell 'steer', or ccsio.steer):
    list                    the calls below
    set_waves 8             8 stripes across the image (broadcast to every chare, waits for all)
    set_speed 0.5           phase advance per step
    set_color 40 200 255    stripe color
    state                   steps done and every chare's parameters (ret=True gather)
    slow 5                  sleeps 5 s in a coroutine: PE 0 keeps answering other requests meanwhile
    eval steps              only with steer.init(allow_eval=True) and ++server-auth

Optional argument: run time in seconds (default 600), then the program exits.
"""

import math
import struct
import time

from charm4py import charm, Chare, Array, coro, steer

NX, NY = 4, 4           # chares
TW, TH = 32, 32         # pixels per chare
W, H = NX * TW, NY * TH

steps = 0               # on PE 0: steps the main loop has done


class Tile(Chare):

    def __init__(self):
        self.waves = 4.0
        self.speed = 0.2
        self.color = (255, 160, 40)
        self.phase = 0.0

    def step(self):
        self.phase += self.speed

    def set_waves(self, waves):
        self.waves = waves

    def set_speed(self, speed):
        self.speed = speed

    def set_color(self, color):
        self.color = tuple(color)

    def params(self):
        return {"tile": list(self.thisIndex), "waves": self.waves, "speed": self.speed,
                "color": list(self.color), "phase": round(self.phase, 3)}

    def pixels(self):
        """This tile's RGB rows: (i, j, bytes)."""
        i, j = self.thisIndex
        r, g, b = self.color
        k = 2 * math.pi * self.waves / W
        row = bytearray()
        for x in range(TW):
            v = 0.5 + 0.5 * math.sin(k * (i * TW + x) + self.phase)
            row += bytes((int(r * v), int(g * v), int(b * v)))
        return (i, j, bytes(row) * TH)


class Images(Chare):
    """lvConfig / lvImage on PE 0: each image request is answered from a
    gather of every tile's pixels, in a coroutine (delayed reply)."""

    def __init__(self):
        charm.CcsRegisterHandler("lvConfig", self.config)
        charm.CcsRegisterHandler("lvImage", self.image)

    def config(self, data):
        charm.CcsSendReply(struct.pack(">iiii", 1, 1, 1, 0))   # version 1, color, push, 2D

    def image(self, data):
        w, h = struct.unpack_from(">ii", data, 8)
        self.thisProxy.render(charm.CcsDelayReply(), w, h)

    @coro
    def render(self, token, w, h):
        img = bytearray(W * H * 3)
        for i, j, px in tiles.pixels(ret=True).get():
            for y in range(TH):
                o = ((j * TH + y) * W + i * TW) * 3
                img[o:o + TW * 3] = px[y * TW * 3:(y + 1) * TW * 3]
        out = bytearray(w * h * 3)          # the request's size: crop or pad
        cw = min(w, W) * 3
        for y in range(min(h, H)):
            out[y * w * 3:y * w * 3 + cw] = img[y * W * 3:y * W * 3 + cw]
        charm.CcsSendDelayedReply(token, bytes(out))


# -- the steering interface --------------------------------------------------

@steer.expose
def set_waves(waves: float):
    """Number of stripes across the image (every chare; returns once all have it)."""
    if waves <= 0:
        raise ValueError("waves must be positive")
    tiles.set_waves(float(waves), awaitable=True).get()
    return waves


@steer.expose
def set_speed(speed: float):
    """Phase advance per step, in radians."""
    tiles.set_speed(float(speed), awaitable=True).get()
    return speed


@steer.expose
def set_color(r: int, g: int, b: int):
    """Stripe color, 0..255 each."""
    color = [max(0, min(255, int(c))) for c in (r, g, b)]
    tiles.set_color(color, awaitable=True).get()
    return color


@steer.expose
def state():
    """Steps done, and every chare's parameters (a gather over the array)."""
    per_tile = tiles.params(ret=True).get()
    return {"steps": steps, "pes": charm.numPes(), "tiles": len(per_tile),
            "waves": sorted({t["waves"] for t in per_tile}),
            "speed": sorted({t["speed"] for t in per_tile}),
            "color": sorted({tuple(t["color"]) for t in per_tile}),
            "first": per_tile[0]}


@steer.expose
def slow(secs: float = 5.0):
    """Wait secs seconds without blocking PE 0 (charm.sleep), then answer."""
    t0 = time.time()
    charm.sleep(float(secs))
    return {"slept": round(time.time() - t0, 3), "steps": steps}


def main(args):
    global tiles, steps
    run_secs = float(args[1]) if len(args) > 1 else 600.0
    tiles = Array(Tile, dims=(NX, NY))
    Chare(Images, onPE=0)
    steer.init(allow_eval=True, verbose=True)     # eval only if the job also has ++server-auth
    print("steer_demo: %dx%d image, %d chares; CCS handlers lvConfig, lvImage, c4pSteer ready" % (W, H, NX * NY))
    t_end = time.time() + run_secs
    while time.time() < t_end:
        tiles.step(awaitable=True).get()
        steps += 1
        if steps % 200 == 0:
            print("steer_demo: step %d" % steps)
        charm.sleep(0.02)
    print("steer_demo: done after %d steps" % steps)
    charm.exit()


charm.start(main)
