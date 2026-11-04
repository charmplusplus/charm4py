"""Unit tests for charm4py.steer's request handling, without Charm++.

The charm4py runtime is replaced by stubs: a fake ``charm`` that records CCS
replies, an identity ``register``/``coro`` and a plain ``Chare`` base whose
``thisProxy.run`` calls the method at once. Run with
``python -m unittest tests/steer/test_steer_unit.py`` from the repository
root; the live test is tests/steer/steer_live.py.
"""

import importlib.util
import json
import os
import sys
import types
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class FakeCharm:
    def __init__(self):
        self.handlers = {}
        self.replies = []           # (token or None, bytes)
        self.next_token = 100

    def CcsRegisterHandler(self, name, fn):
        self.handlers[name] = fn

    def CcsSendReply(self, msg):
        self.replies.append((None, msg))

    def CcsDelayReply(self):
        self.next_token += 1
        return self.next_token

    def CcsSendDelayedReply(self, token, msg):
        self.replies.append((token, msg))


class FakeFuture:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def load_steer():
    """Import charm4py/steer.py against stub charm4py modules."""
    fake = FakeCharm()
    pkg = types.ModuleType("charm4py")
    pkg.__path__ = [os.path.join(ROOT, "charm4py")]
    charm_mod = types.ModuleType("charm4py.charm")

    class Chare:
        def __new__(cls, C=None, args=(), onPE=-1):
            if C is None:
                return object.__new__(cls)
            obj = object.__new__(C)
            obj.thisProxy = obj
            obj.__init__(*args)
            return types.SimpleNamespace(obj=obj)   # a "proxy": no second __init__

    charm_mod.charm = fake
    charm_mod.Chare = Chare
    charm_mod.register = lambda C: C
    em_mod = types.ModuleType("charm4py.entry_method")
    em_mod.coro = lambda f: f
    th_mod = types.ModuleType("charm4py.threads")
    th_mod.Future = FakeFuture
    th_mod.CollectiveFuture = FakeFuture
    saved = {k: sys.modules.get(k) for k in ("charm4py", "charm4py.charm", "charm4py.entry_method",
                                              "charm4py.threads", "charm4py.steer")}
    sys.modules.update({"charm4py": pkg, "charm4py.charm": charm_mod,
                        "charm4py.entry_method": em_mod, "charm4py.threads": th_mod})
    spec = importlib.util.spec_from_file_location("charm4py.steer", os.path.join(ROOT, "charm4py", "steer.py"))
    steer = importlib.util.module_from_spec(spec)
    sys.modules["charm4py.steer"] = steer
    spec.loader.exec_module(steer)
    return steer, fake, saved


def unload(saved):
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


class SteerTests(unittest.TestCase):
    def setUp(self):
        self.steer, self.charm, self.saved = load_steer()
        self.argv = sys.argv
        self.ns = {"x": 41}

    def tearDown(self):
        sys.argv = self.argv
        unload(self.saved)

    def start(self, allow_eval=False, auth=False):
        sys.argv = ["prog.py", "++server"] + (["++server-auth", "key"] if auth else [])
        self.steer.init(allow_eval=allow_eval, namespace=self.ns)

    def ask(self, req):
        """Send one request (dict or raw bytes); return (delayed, reply dict)."""
        data = req if isinstance(req, bytes) else json.dumps(req).encode()
        n = len(self.charm.replies)
        self.charm.handlers["c4pSteer"](data)
        self.assertEqual(len(self.charm.replies), n + 1, "exactly one reply per request")
        token, msg = self.charm.replies[-1]
        return token is not None, json.loads(msg)

    def test_expose_forms(self):
        s = self.steer

        @s.expose
        def a(x):
            return x

        @s.expose("bee")
        def b():
            pass

        @s.expose(name="cee")
        def c():
            pass

        s.expose(len, "length")
        self.assertEqual(sorted(s.exposed()), ["a", "bee", "cee", "length"])
        with self.assertRaises(ValueError):
            s.expose(a, "list")
        with self.assertRaises(TypeError):
            s.expose(3, "three")
        s.unexpose("cee")
        self.assertNotIn("cee", s.exposed())

    def test_list(self):
        def set_rate(rate: float, clamp=True):
            """Set the rate.

            More text."""
        self.steer.expose(set_rate)
        self.start()
        delayed, r = self.ask({"call": "list"})
        self.assertFalse(delayed)
        self.assertEqual(r, {"ok": True, "result": {"eval": False, "calls": [
            {"name": "set_rate", "signature": "(rate: float, clamp=True)", "doc": "Set the rate.\n\nMore text."}]}})

    def test_call_runs_delayed_with_args_and_kwargs(self):
        seen = []
        self.steer.expose(lambda a, b=0: seen.append((a, b)) or a + b, "add")
        self.start()
        delayed, r = self.ask({"call": "add", "args": [2], "kwargs": {"b": 3}})
        self.assertTrue(delayed)
        self.assertEqual(r, {"ok": True, "result": 5})
        self.assertEqual(seen, [(2, 3)])
        self.assertEqual(self.ask(b'{"call": "add", "args": [1]}\x00\x00')[1], {"ok": True, "result": 1})

    def test_future_result_is_waited_for(self):
        self.steer.expose(lambda: FakeFuture([1, 2, 3]), "bcast")
        self.start()
        self.assertEqual(self.ask({"call": "bcast"})[1], {"ok": True, "result": [1, 2, 3]})

    def test_errors(self):
        def bad(x):
            raise ValueError("x must be positive")
        self.steer.expose(bad)
        self.start()
        delayed, r = self.ask({"call": "bad", "args": [-1]})
        self.assertTrue(delayed)
        self.assertFalse(r["ok"])
        self.assertEqual(r["error"], "ValueError: x must be positive")
        self.assertIn("Traceback", r["traceback"])
        r = self.ask({"call": "bad", "args": [1, 2]})[1]
        self.assertTrue(r["error"].startswith("TypeError"))
        for req, text in [({"call": "nope"}, "no steering call 'nope' (exposed: bad)"),
                          (b"not json", "JSONDecodeError"),
                          ({"args": []}, '"call"'),
                          ([1, 2], "JSON object"),
                          ({"call": "bad", "args": 5}, '"args" must be a list'),
                          ({"call": "bad", "kwargs": []}, '"kwargs" must be an object')]:
            delayed, r = self.ask(req)
            self.assertFalse(delayed, req)
            self.assertFalse(r["ok"])
            self.assertIn(text, r["error"])
            self.assertNotIn("traceback", r)

    def test_json_conversion(self):
        class Thing:
            def __repr__(self):
                return "<thing>"
        self.steer.expose(lambda: {"t": (1, 2), "s": {3}, "b": b"hi", "o": Thing(), "n": float("inf")}, "mix")
        self.start()
        self.assertEqual(self.ask({"call": "mix"})[1]["result"],
                         {"t": [1, 2], "s": [3], "b": "hi", "o": "<thing>", "n": float("inf")})
        loop = []
        loop.append(loop)
        self.steer.expose(lambda: loop, "loop")
        r = self.ask({"call": "loop"})[1]
        self.assertFalse(r["ok"])
        self.assertIn("not JSON-serializable", r["error"])

    def test_numpy(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("no numpy")
        self.steer.expose(lambda: {"a": np.arange(3), "f": np.float32(0.5), "i": np.int64(7)}, "np")
        self.start()
        self.assertEqual(self.ask({"call": "np"})[1]["result"], {"a": [0, 1, 2], "f": 0.5, "i": 7})

    def test_eval_off_by_default(self):
        self.start(auth=True)
        r = self.ask({"call": "eval", "args": ["1+1"]})[1]
        self.assertFalse(r["ok"])
        self.assertIn("PermissionError: eval is off: the program did not enable it", r["error"])
        self.assertFalse(self.ask({"call": "list"})[1]["result"]["eval"])

    def test_eval_needs_auth(self):
        self.start(allow_eval=True, auth=False)
        r = self.ask({"call": "eval", "args": ["1+1"]})[1]
        self.assertIn("without ++server-auth", r["error"])
        self.assertFalse(self.ask({"call": "list"})[1]["result"]["eval"])

    def test_eval_with_auth(self):
        self.start(allow_eval=True, auth=True)
        self.assertTrue(self.ask({"call": "list"})[1]["result"]["eval"])
        self.assertEqual(self.ask({"call": "eval", "args": ["x + 1"]})[1], {"ok": True, "result": 42})
        self.assertEqual(self.ask({"call": "eval", "args": ["y = x * 2\ny"]})[1], {"ok": True, "result": 82})
        self.assertEqual(self.ask({"call": "eval", "args": ["z = 1"]})[1], {"ok": True, "result": None})
        self.assertEqual(self.ns["z"], 1)
        r = self.ask({"call": "eval", "args": ["1/0"]})[1]
        self.assertEqual(r["error"], "ZeroDivisionError: division by zero")
        r = self.ask({"call": "eval", "args": ["1", "2"]})[1]
        self.assertIn("one argument", r["error"])

    def test_auth_detection(self):
        self.assertTrue(self.steer.auth_enabled(["p", "++server-auth", "f"]))
        self.assertTrue(self.steer.auth_enabled(["p", "++server-auth=f"]))
        self.assertFalse(self.steer.auth_enabled(["p", "++server", "++server-port", "1"]))


if __name__ == "__main__":
    unittest.main()
