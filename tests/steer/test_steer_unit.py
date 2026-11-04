"""Unit tests for charm4py.steer's request handling, without Charm++.

The charm4py runtime is replaced by stubs: a fake ``charm`` that records CCS
replies, an identity ``register``/``coro`` and a plain ``Chare`` base whose
``thisProxy.run`` calls the method at once. Run with
``python -m unittest tests/steer/test_steer_unit.py`` from the repository
root. A live program to try against is examples/steer/steer_demo.py.
"""

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class FakeCharm:
    def __init__(self):
        self.handlers = {}
        self.replies = []           # (token or None, bytes)
        self.next_token = 100
        self.timers = []            # (callable, secs, args)
        self.fail_sends = 0         # make the next N sends raise

    def CcsRegisterHandler(self, name, fn):
        self.handlers[name] = fn

    def _check_send(self, msg):
        if not isinstance(msg, bytes):
            raise TypeError("reply must be bytes")
        if self.fail_sends:
            self.fail_sends -= 1
            raise RuntimeError("send failed")

    def CcsSendReply(self, msg):
        self._check_send(msg)
        self.replies.append((None, msg))

    def CcsDelayReply(self):
        self.next_token += 1
        return self.next_token

    def CcsSendDelayedReply(self, token, msg):
        self._check_send(msg)
        self.replies.append((token, msg))

    def scheduleCallableAfter(self, cb, secs, args=[]):
        self.timers.append((cb, secs, args))


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


VALID_KEY = "0 0123456789abcdef0123456789abcdef\n"


class SteerTests(unittest.TestCase):
    def setUp(self):
        self.steer, self.charm, self.saved = load_steer()
        self.argv = sys.argv
        self.ns = {"x": 41}
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        sys.argv = self.argv
        unload(self.saved)
        self.tmp.cleanup()

    def keyfile(self, text, name="key"):
        path = os.path.join(self.tmp.name, name)
        with open(path, "wb") as f:
            f.write(text.encode() if isinstance(text, str) else text)
        return path

    def start(self, allow_eval=False, auth=False, keytext=VALID_KEY, **kw):
        """auth=True: ++server-auth with a key file holding ``keytext``."""
        sys.argv = ["prog.py", "++server"] + (["++server-auth", self.keyfile(keytext)] if auth else [])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.steer.init(allow_eval=allow_eval, namespace=self.ns, **kw)
        return out.getvalue()

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
        self.assertNotIn("traceback", r)        # no auth, not verbose: no traceback
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

        class LooksLikeNumpy:                   # must not be called: not a numpy type
            def tolist(self):
                raise AssertionError("tolist called on a non-numpy object")

            def item(self):
                raise AssertionError("item called on a non-numpy object")

            def __repr__(self):
                return "<proxy>"
        nan, inf = float("nan"), float("inf")
        self.steer.expose(lambda: {"t": (1, 2), "s": {3}, "b": b"hi", "o": Thing(), "n": inf,
                                   "l": [nan, -inf, 1.5], 7: "int key", None: 0, (1, 2): "tuple key",
                                   "p": LooksLikeNumpy()}, "mix")
        self.start()
        delayed, r = self.ask({"call": "mix"})
        msg = self.charm.replies[-1][1]
        self.assertNotIn(b"NaN", msg)
        self.assertNotIn(b"Infinity", msg)
        self.assertEqual(r["result"], {"t": [1, 2], "s": [3], "b": "hi", "o": "<thing>", "n": None,
                                       "l": [None, None, 1.5], "7": "int key", "None": 0,
                                       "(1, 2)": "tuple key", "p": "<proxy>"})
        loop = []
        loop.append(loop)
        self.steer.expose(lambda: loop, "loop")
        r = self.ask({"call": "loop"})[1]
        self.assertFalse(r["ok"])
        self.assertIn("not JSON-serializable", r["error"])
        shared = [1]
        self.steer.expose(lambda: [shared, shared], "shared")   # not a cycle
        self.assertEqual(self.ask({"call": "shared"})[1]["result"], [[1], [1]])

    def test_numpy(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("no numpy")
        self.steer.expose(lambda: {"a": np.arange(3), "f": np.float32(0.5), "i": np.int64(7),
                                   "b": np.bool_(True), "nan": np.array([np.nan, 1.0]),
                                   np.int64(3): "np key"}, "np")
        self.start()
        self.assertEqual(self.ask({"call": "np"})[1]["result"],
                         {"a": [0, 1, 2], "f": 0.5, "i": 7, "b": True, "nan": [None, 1.0], "3": "np key"})

    def test_eval_off_by_default(self):
        self.start(auth=True)
        r = self.ask({"call": "eval", "args": ["1+1"]})[1]
        self.assertFalse(r["ok"])
        self.assertIn("PermissionError: eval is off: the program did not enable it", r["error"])
        self.assertFalse(self.ask({"call": "list"})[1]["result"]["eval"])

    def test_eval_needs_auth(self):
        self.start(allow_eval=True, auth=False)
        r = self.ask({"call": "eval", "args": ["1+1"]})[1]
        self.assertIn("eval is off: the CCS server does not require authentication", r["error"])
        self.assertFalse(self.ask({"call": "list"})[1]["result"]["eval"])

    def test_eval_needs_level0_key(self):
        for text in ["", "1 0123456789abcdef\n", "# 0 0123456789abcdef\n"]:
            self.steer, self.charm, _ = load_steer()
            out = self.start(allow_eval=True, auth=True, keytext=text)
            self.assertIn("no level-0 key", out)
            self.assertIn("eval stays off", out)
            r = self.ask({"call": "eval", "args": ["1+1"]})[1]
            self.assertFalse(r["ok"])
            self.assertIn("PermissionError", r["error"])
            self.assertNotIn(self.tmp.name, r["error"])     # key file path not sent to clients
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
        self.assertIn("Traceback", r["traceback"])         # authenticated: tracebacks sent
        r = self.ask({"call": "eval", "args": ["1", "2"]})[1]
        self.assertIn("one argument", r["error"])

    def test_auth_detection(self):
        st = self.steer
        good = self.keyfile(VALID_KEY, "good")
        self.assertTrue(st.auth_enabled(["p", "++server-auth", good]))
        self.assertFalse(st.auth_enabled(["p", "++server-auth=" + good]))   # not a Charm++ form
        self.assertFalse(st.auth_enabled(["p", "++server", "++server-port", "1"]))
        self.assertFalse(st.auth_enabled(["p", "++server-auth"]))
        bad = self.keyfile("1 abcd\n", "bad")
        self.assertFalse(st.auth_enabled(["p", "++server-auth", good, "++server-auth", bad]))

    # key files, parsed as Charm++'s CcsSecMan_default does

    def auth(self, text):
        return self.steer.auth_status(["p", "++server-auth", self.keyfile(text)])

    def test_keyfile_empty(self):
        ok, why = self.auth("")
        self.assertFalse(ok)
        self.assertIn("no level-0 key", why)

    def test_keyfile_only_level1(self):
        self.assertFalse(self.auth("1 0123456789abcdef0123456789abcdef\n")[0])
        self.assertFalse(self.auth("1 abcd\n2 OTPsecret\n   7 ffff\n")[0])

    def test_keyfile_comments(self):
        self.assertFalse(self.auth("# a CCS key file\n# 0 0123456789abcdef\n\n")[0])
        self.assertFalse(self.auth("; 0 abcd\nkey 0 abcd\n")[0])

    def test_keyfile_valid_level0(self):
        self.assertTrue(self.auth(VALID_KEY)[0])
        self.assertTrue(self.auth("# comment\n1 abcd\n0 abcd\n")[0])
        self.assertTrue(self.auth("  \t+0   ABcd  trailing words\n")[0])
        self.assertTrue(self.auth("0abcd\n")[0])           # sscanf "%d%s" needs no space
        self.assertTrue(self.auth("-0 abcd")[0])            # no final newline
        self.assertTrue(self.auth("00 OTPseed\n")[0])       # one-time key

    def test_keyfile_missing(self):
        ok, why = self.steer.auth_status(["p", "++server-auth", os.path.join(self.tmp.name, "nope")])
        self.assertFalse(ok)
        self.assertIn("cannot read", why)
        sys.argv = ["prog.py", "++server-auth", os.path.join(self.tmp.name, "nope")]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.steer.init(allow_eval=True, namespace=self.ns)
        self.assertIn("cannot read", out.getvalue())
        self.assertIn("eval stays off", out.getvalue())
        self.assertFalse(self.ask({"call": "list"})[1]["result"]["eval"])

    def test_keyfile_parse_edges(self):
        k = self.steer.keyfile_has_level0
        self.assertFalse(k(b"0\n"))                 # level without a key
        self.assertFalse(k(b"0 a\n"))               # one hex digit: Charm++ rejects it
        self.assertFalse(k(b"0 xyz\n"))             # not hex, not OTP
        self.assertFalse(k(b"0x10 abcd\n"))         # %d reads 0, the key is "x10"
        self.assertFalse(k(b"1 abcd 0 abcd\n"))     # one key per line
        self.assertFalse(k(b"4294967296 abcd\n"))   # overflows a C int: not taken as 0
        self.assertFalse(k(b"\0 0 abcd\n"))         # sscanf stops at NUL
        # fgets(line, 200) splits long lines: "0" ends a 199-byte piece, its key is in the next
        self.assertFalse(k(b" " * 198 + b"0 abcd\n"))
        # ... and a long comment line can hide a key at its 200th byte
        self.assertTrue(k(b"#" * 199 + b"0 abcd\n"))

    # exactly one reply, bounds, timeouts

    def test_base_exceptions_still_reply(self):
        def bye():
            raise SystemExit(3)

        def gen_exit():
            raise GeneratorExit()
        self.steer.expose(bye)
        self.steer.expose(gen_exit)
        self.start()
        for name, exc in (("bye", SystemExit), ("gen_exit", GeneratorExit)):
            n = len(self.charm.replies)
            with self.assertRaises(exc):
                self.charm.handlers["c4pSteer"](json.dumps({"call": name}).encode())
            self.assertEqual(len(self.charm.replies), n + 1)
            token, msg = self.charm.replies[-1]
            self.assertIsNotNone(token)
            self.assertEqual(json.loads(msg)["error"].split(":")[0], exc.__name__)
        self.assertEqual(self.steer._agent.obj.calls, {})       # no leaked call state

    def test_send_failure_falls_back(self):
        self.steer.expose(lambda: 1, "one")
        self.start()
        self.charm.fail_sends = 1
        delayed, r = self.ask({"call": "one"})
        self.assertTrue(delayed)
        self.assertEqual(r, {"ok": False, "error": "charm4py.steer: the reply could not be encoded"})
        self.charm.fail_sends = 2                                # both attempts fail: no crash
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.charm.handlers["c4pSteer"](b'{"call": "one"}')
        self.assertIn("could not send a reply", out.getvalue())
        self.assertEqual(self.steer._agent.obj.calls, {})

    def test_unencodable_result(self):
        class BadRepr:
            def __repr__(self):
                raise RuntimeError("no repr")
        self.steer.expose(lambda: BadRepr(), "bad")
        self.start()
        r = self.ask({"call": "bad"})[1]
        self.assertFalse(r["ok"])
        self.assertIn("not JSON-serializable", r["error"])

    def test_busy(self):
        handler = lambda req: self.charm.handlers["c4pSteer"](json.dumps(req).encode())
        inner = []

        def outer():
            n = len(self.charm.replies)
            handler({"call": "noop"})           # the stub runs calls at once: this one is nested
            inner.append(json.loads(self.charm.replies[n][1]))
            return "outer done"
        self.steer.expose(outer)
        self.steer.expose(lambda: None, "noop")
        self.start(max_inflight=1)
        handler({"call": "outer"})
        self.assertEqual(inner, [{"ok": False, "error": "busy"}])
        self.assertEqual(json.loads(self.charm.replies[-1][1]), {"ok": True, "result": "outer done"})
        self.assertEqual(self.ask({"call": "noop"})[1], {"ok": True, "result": None})   # slot freed
        self.steer._agent = None
        for bad in ({"max_inflight": 0}, {"max_inflight": True}, {"timeout": 0}, {"timeout": -1}):
            with self.assertRaises(ValueError):
                self.steer.init(**bad)

    def test_timeout_replies_once_and_drops_late_result(self):
        def slow():
            # the timer fires while the call is still running
            cb, secs, args = self.charm.timers[-1]
            self.assertEqual(secs, 2.5)
            cb(*args)
            return "late"
        self.steer.expose(slow)
        self.steer.expose(lambda: "fast", "fast")
        self.start(timeout=2.5)
        n = len(self.charm.replies)
        self.charm.handlers["c4pSteer"](b'{"call": "slow"}')
        self.assertEqual(len(self.charm.replies), n + 1, "the late result must be dropped")
        r = json.loads(self.charm.replies[-1][1])
        self.assertFalse(r["ok"])
        self.assertTrue(r["error"].startswith("timeout"))
        self.assertEqual(self.steer._agent.obj.calls, {})
        # a timer firing after its call finished sends nothing
        self.assertEqual(self.ask({"call": "fast"})[1], {"ok": True, "result": "fast"})
        n = len(self.charm.replies)
        cb, secs, args = self.charm.timers[-1]
        cb(*args)
        self.assertEqual(len(self.charm.replies), n)

    def test_verbose_logging_and_tracebacks(self):
        def bad(secret):
            raise ValueError("no")
        self.steer.expose(bad)
        self.start()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            r = self.ask({"call": "bad", "args": ["hunter2"]})[1]
        self.assertNotIn("hunter2", out.getvalue())
        self.assertNotIn("traceback", r)
        self.steer, self.charm, _ = load_steer()
        self.steer.expose(bad)
        self.start(verbose=True)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            r = self.ask({"call": "bad", "args": ["hunter2"]})[1]
        self.assertIn("hunter2", out.getvalue())
        self.assertIn("Traceback", r["traceback"])


if __name__ == "__main__":
    unittest.main()
