"""Unit tests for charm4py/liveviz.py that need no Charm++ runtime.

Run from the repository root:  python -m unittest tests/liveviz/test_liveviz_unit.py
(charm4py itself is not imported: a stub charm4py.charm stands in for it.)
"""
import importlib.util
import io
import os
import struct
import sys
import types
import unittest
from contextlib import redirect_stderr

import numpy as np

LIVEVIZ_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'charm4py', 'liveviz.py')


class FakeCharm(object):
    """Records CCS calls the way charmlib would see them."""

    def __init__(self):
        self.reducers = types.SimpleNamespace(addReducer=self._add_reducer)
        self.reset()

    def _add_reducer(self, func, pre=None, post=None):
        setattr(self.reducers, func.__name__, func)

    def reset(self):
        self.handlers = {}
        self.replies = []           # bytes sent with CcsSendReply
        self.delayed = {}           # token -> bytes sent with CcsSendDelayedReply
        self.next_token = 1

    def CcsRegisterHandler(self, name, handler):
        self.handlers[name] = handler

    def CcsSendReply(self, message):
        assert type(message) is bytes, type(message)   # charmlib requires bytes
        self.replies.append(message)

    def CcsDelayReply(self):
        token = self.next_token
        self.next_token += 1
        return token

    def CcsSendDelayedReply(self, token, message):
        assert type(message) is bytes, type(message)
        assert token not in self.delayed, 'token %d answered twice' % token
        self.delayed[token] = message


def load_liveviz():
    fake = FakeCharm()
    pkg = types.ModuleType('charm4py')
    pkg.__path__ = []
    charm_mod = types.ModuleType('charm4py.charm')
    charm_mod.charm = fake
    charm_mod.Chare = type('Chare', (object,), {})
    charm_mod.register = lambda cls: cls
    saved = {k: sys.modules.get(k) for k in ('charm4py', 'charm4py.charm', 'charm4py.liveviz')}
    sys.modules['charm4py'] = pkg
    sys.modules['charm4py.charm'] = charm_mod
    try:
        spec = importlib.util.spec_from_file_location('charm4py.liveviz', LIVEVIZ_PATH)
        mod = importlib.util.module_from_spec(spec)
        sys.modules['charm4py.liveviz'] = mod
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod, fake


lv, fake_charm = load_liveviz()


def reference_composite(contribs, is_color):
    """Straightforward per-pixel definition of the composite."""
    _, _, _, _, _, th, tw = contribs[0]
    bpp = 3 if is_color else 1
    out = [0] * (tw * th * bpp)
    for data, sx, sy, lh, lw, _, _ in contribs:
        data = bytes(np.asarray(data, dtype=np.uint8).reshape(-1)) \
            if isinstance(data, np.ndarray) else bytes(data)
        for y in range(lh):
            for x in range(lw):
                gx, gy = sx + x, sy + y
                if not (0 <= gx < tw and 0 <= gy < th):
                    continue
                for c in range(bpp):
                    s = (y * lw + x) * bpp + c
                    v = data[s] if s < len(data) else 0
                    d = (gy * tw + gx) * bpp + c
                    out[d] = (out[d] + v) % 256
    return bytes(out)


def v1(w, h, code=1):
    return struct.pack('>iiii', 1, code, w, h)


def v2(w, h, ctype=0, quality=0, code=1):
    return struct.pack('>iiiiii', 2, code, w, h, ctype, quality)


VIEW = struct.pack('>14d', *[float(i) for i in range(1, 15)])


class TestImageRequest(unittest.TestCase):

    def test_v1(self):
        r = lv.ImageRequest.from_bytes(v1(1100, 400, code=7))
        self.assertEqual((r.version, r.request_type, r.width, r.height), (1, 7, 1100, 400))
        self.assertEqual((r.compression_type, r.compression_quality), (0, 0))
        self.assertIsNone(r.x)

    def test_v2_appspector(self):
        # what appSpector's ccsio.liveviz.pack_request sends for --jpeg 75
        r = lv.ImageRequest.from_bytes(v2(1100, 400, 1, 75))
        self.assertEqual((r.version, r.width, r.height), (2, 1100, 400))
        self.assertEqual((r.compression_type, r.compression_quality), (1, 75))
        self.assertIsNone(r.x)

    def test_v1_3d(self):
        r = lv.ImageRequest.from_bytes(v1(10, 20) + VIEW)
        self.assertEqual((r.width, r.height, r.compression_type), (10, 20, 0))
        self.assertEqual((r.x.x, r.x.y, r.x.z), (1.0, 2.0, 3.0))
        self.assertEqual((r.o.x, r.o.y, r.o.z), (10.0, 11.0, 12.0))
        self.assertEqual((r.minZ, r.maxZ), (13.0, 14.0))

    def test_v2_3d(self):
        r = lv.ImageRequest.from_bytes(v2(10, 20, 1, 50) + VIEW)
        self.assertEqual((r.compression_type, r.compression_quality), (1, 50))
        self.assertEqual((r.y.x, r.z.z, r.maxZ), (4.0, 9.0, 14.0))

    def test_short(self):
        with self.assertRaises(ValueError):
            lv.ImageRequest.from_bytes(b'\0' * 15)
        with self.assertRaises(ValueError):
            lv.ImageRequest.from_bytes(struct.pack('>iiiii', 2, 1, 2, 2, 0))

    def test_trailing_bytes_ignored(self):
        r = lv.ImageRequest.from_bytes(v1(3, 4) + b'xyz')
        self.assertEqual((r.width, r.height), (3, 4))
        self.assertIsNone(r.x)


class TestByteImage(unittest.TestCase):

    def check(self, contribs, is_color):
        img = lv.ByteImage.from_contributions(contribs, is_color)
        self.assertIs(type(img.data), bytes)
        self.assertEqual(img.data, reference_composite(contribs, is_color))
        return img

    def test_color_tiles(self):
        rng = np.random.default_rng(1)
        contribs = []
        for ty in range(2):
            for tx in range(3):
                d = bytearray(rng.integers(0, 256, 5 * 4 * 3, dtype=np.uint8).tobytes())
                contribs.append((d, tx * 5, ty * 4, 4, 5, 8, 15))
        img = self.check(contribs, True)
        self.assertEqual((img.width, img.height, img.bytes_per_pixel), (15, 8, 3))

    def test_grey_and_overlap_wraps(self):
        a = bytes([200] * 16)
        b = bytes([100] * 16)
        img = self.check([(a, 0, 0, 4, 4, 6, 6), (b, 2, 2, 4, 4, 6, 6)], False)
        px = np.frombuffer(img.data, np.uint8).reshape(6, 6)
        self.assertEqual(px[0, 0], 200)
        self.assertEqual(px[3, 3], (200 + 100) % 256)
        self.assertEqual(px[5, 5], 100)
        self.assertEqual(px[0, 5], 0)

    def test_input_types(self):
        raw = np.arange(2 * 3 * 3, dtype=np.uint8)
        for d in (raw.tobytes(), bytearray(raw.tobytes()), memoryview(raw.tobytes()),
                  raw, raw.reshape(2, 3, 3), raw.astype(np.int64)):
            img = lv.ByteImage.from_contributions([(d, 1, 1, 2, 3, 4, 5)], True)
            self.assertEqual(img.data, reference_composite([(raw.tobytes(), 1, 1, 2, 3, 4, 5)], True))

    def test_short_data_padded(self):
        self.check([(bytes([9] * 7), 0, 0, 3, 3, 3, 3)], False)

    def test_out_of_bounds_clipped(self):
        d = bytes(range(1, 4 * 4 * 3 + 1))
        img = self.check([(d, 3, 3, 4, 4, 5, 5), (d, -2, -1, 4, 4, 5, 5),
                          (d, 10, 10, 4, 4, 5, 5)], True)
        self.assertEqual(len(img.data), 5 * 5 * 3)

    def test_empty_contribution(self):
        self.check([(b'', 0, 0, 0, 0, 2, 2), (bytes([1, 2, 3, 4]), 0, 0, 2, 2, 2, 2)], False)

    def corner(self, src, w, h):
        out = lv.ByteImage.with_image_in_corner(src, w, h)
        self.assertIs(type(out.data), bytes)
        self.assertEqual((out.width, out.height, out.is_color), (w, h, src.is_color))
        bpp = src.bytes_per_pixel
        expected = bytearray(w * h * bpp)
        for y in range(min(h, src.height)):
            for x in range(min(w, src.width)):
                s = (y * src.width + x) * bpp
                d = (y * w + x) * bpp
                if s + bpp <= len(src.data):
                    expected[d:d + bpp] = src.data[s:s + bpp]
        self.assertEqual(out.data, bytes(expected))

    def test_corner_crop_pad_same(self):
        for color in (True, False):
            bpp = 3 if color else 1
            src = lv.ByteImage(bytes(np.arange(7 * 5 * bpp, dtype=np.uint8)), 7, 5, color)
            for w, h in ((7, 5), (3, 2), (10, 9), (3, 9), (10, 2), (0, 0)):
                self.corner(src, w, h)

    def test_corner_short_source(self):
        src = lv.ByteImage(bytes([5] * 10), 4, 4, False)
        self.corner(src, 4, 4)


class TestEncodeReply(unittest.TestCase):

    def image(self, color):
        bpp = 3 if color else 1
        rng = np.random.default_rng(2)
        return lv.ByteImage(rng.integers(0, 256, 32 * 16 * bpp, dtype=np.uint8).tobytes(),
                            32, 16, color)

    def test_raw(self):
        img = self.image(True)
        self.assertEqual(lv.encode_reply(img, lv.ImageRequest.from_bytes(v1(32, 16))), img.data)
        self.assertEqual(lv.encode_reply(img, lv.ImageRequest.from_bytes(v2(32, 16))), img.data)

    def test_jpeg(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest('Pillow not installed')
        for color, mode in ((True, 'RGB'), (False, 'L')):
            for q in (10, 75, 0, 150):
                reply = lv.encode_reply(self.image(color),
                                        lv.ImageRequest.from_bytes(v2(32, 16, 1, q)))
                self.assertIs(type(reply), bytes)
                self.assertEqual(reply[:2], b'\xff\xd8')
                pil = Image.open(io.BytesIO(reply))
                self.assertEqual((pil.size, pil.mode), ((32, 16), mode))
        small = lv.encode_reply(self.image(True), lv.ImageRequest.from_bytes(v2(32, 16, 1, 5)))
        big = lv.encode_reply(self.image(True), lv.ImageRequest.from_bytes(v2(32, 16, 1, 95)))
        self.assertLess(len(small), len(big))

    def test_jpeg_without_pillow(self):
        saved = sys.modules.get('PIL')
        sys.modules['PIL'] = None   # makes "from PIL import Image" raise ImportError
        lv._warned.clear()
        err = io.StringIO()
        try:
            with redirect_stderr(err):
                req = lv.ImageRequest.from_bytes(v2(32, 16, 1, 75))
                self.assertEqual(lv.encode_reply(self.image(True), req), b'')
                self.assertEqual(lv.encode_reply(self.image(True), req), b'')
        finally:
            if saved is None:
                sys.modules.pop('PIL', None)
            else:
                sys.modules['PIL'] = saved
        self.assertEqual(err.getvalue().count('WARNING'), 1)   # warned once

    def test_unknown_compression(self):
        lv._warned.clear()
        err = io.StringIO()
        with redirect_stderr(err):
            for ctype in (2, 7, -1):
                req = lv.ImageRequest.from_bytes(v2(32, 16, ctype, 0))
                self.assertEqual(lv.encode_reply(self.image(False), req), b'')
        self.assertIn('compressionType 7', err.getvalue())


def contribs_for(value, w=4, h=2):
    return [(bytes([value] * (w * h * 3)), 0, 0, h, w, h, w)]


class TestGroup(unittest.TestCase):

    def setUp(self):
        fake_charm.reset()
        lv.LiveViz.cfg = lv.Config()
        self.callbacks = []

    def make(self, poll):
        g = lv.LiveVizGroup(self.callbacks.append, poll)
        self.assertIn('lvImage', fake_charm.handlers)
        return g

    def test_push_single(self):
        g = self.make(False)
        fake_charm.handlers['lvImage'](v1(4, 2))
        self.assertEqual(len(self.callbacks), 1)
        self.assertEqual(self.callbacks[0].width, 4)
        g.send(contribs_for(7))
        self.assertEqual(fake_charm.delayed, {1: bytes([7] * 24)})

    def test_push_overlapping_requests(self):
        # Two clients ask before either image is ready: each gets its own
        # reply, at its own size and compression, in request order.
        g = self.make(False)
        fake_charm.handlers['lvImage'](v1(4, 2))
        fake_charm.handlers['lvImage'](v2(2, 1))
        self.assertEqual(len(self.callbacks), 2)
        self.assertEqual(fake_charm.delayed, {})
        g.send(contribs_for(1))
        self.assertEqual(fake_charm.delayed, {1: bytes([1] * 24)})
        g.send(contribs_for(2))
        self.assertEqual(fake_charm.delayed, {1: bytes([1] * 24), 2: bytes([2] * 6)})

    def test_push_jpeg(self):
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest('Pillow not installed')
        g = self.make(False)
        fake_charm.handlers['lvImage'](v2(4, 2, 1, 75))
        g.send(contribs_for(9))
        self.assertEqual(fake_charm.delayed[1][:2], b'\xff\xd8')

    def test_push_unrequested_image_dropped(self):
        g = self.make(False)
        with redirect_stderr(io.StringIO()):
            g.send(contribs_for(3))
        self.assertEqual((fake_charm.delayed, fake_charm.replies), ({}, []))

    def test_poll_image_first(self):
        g = self.make(True)
        g.send(contribs_for(5))
        g.send(contribs_for(6))
        fake_charm.handlers['lvImage'](v1(2, 2))     # crop+pad to 2x2
        fake_charm.handlers['lvImage'](v2(4, 2, 0, 0))
        self.assertEqual(fake_charm.replies,
                         [bytes([5] * 12), bytes([6] * 24)])
        self.assertEqual(self.callbacks, [])

    def test_poll_request_first(self):
        g = self.make(True)
        fake_charm.handlers['lvImage'](v1(4, 2))
        fake_charm.handlers['lvImage'](v2(4, 2, 5, 0))   # unknown compression
        self.assertEqual(fake_charm.replies, [])
        g.send(contribs_for(8))
        with redirect_stderr(io.StringIO()):
            g.send(contribs_for(9))
        self.assertEqual(fake_charm.delayed, {1: bytes([8] * 24), 2: b''})


class TestConfig(unittest.TestCase):

    def test_to_binary(self):
        self.assertEqual(lv.Config().to_binary(), struct.pack('>iiii', 1, 1, 1, 0))
        cfg = lv.Config(isColor=False, isPush=False, is3d=True, min=(1, 2, 3), max=(4, 5, 6))
        self.assertEqual(cfg.to_binary(),
                         struct.pack('>iiii6d', 1, 0, 0, 1, 1, 2, 3, 4, 5, 6))


if __name__ == '__main__':
    unittest.main()
