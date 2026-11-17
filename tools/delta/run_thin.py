#!/usr/bin/env python3
"""Run the thin charm4py test suite (test_config.json plus tests/thin) one test at a
time, continuing past failures, inside a Slurm allocation on Delta.

usage: run_thin.py <outdir> [--mode srun|charmrun] [--counts 1,2] [--filter REGEX]
Run from the charm4py clone with the venv active.

Counts: a test with force_min_processes runs once at max(2, that value); every
other test runs at each count in --counts (default 1,2).
srun mode:     srun --mpi=pmix -n N -c 4 --gres=gpu:G --cpu-bind=none [prefix] python test args +pe N
charmrun mode: python -m charmrun.start [prefix] python test args +pN
A multi-process result counts only if the log shows
"Reconverse> Starting Reconverse with N processes"; otherwise it is BANNER?.
"""
import argparse, json, os, re, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument('outdir')
ap.add_argument('--mode', default='srun')
ap.add_argument('--counts', default='1,2')
ap.add_argument('--filter', default='.')
ap.add_argument('--gpus', default=os.environ.get('NGPUS', '2'))
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)
py = sys.executable
counts = [int(c) for c in a.counts.split(',')]
try:
    import numba  # noqa
    numba_ok = True
except Exception:
    numba_ok = False
try:
    import cupy  # noqa
    from charm4py import _ckdyn
    device_ok = bool(_ckdyn.has_device_support())
except Exception:
    device_ok = False

tests = json.load(open('test_config.json'))
# tests/thin, as the Mac gates ran them (checkpoint_restart is a pair of runs)
thin = [
    {'path': 'tests/thin/smoke_ckdyn.py', 'env': {'CHARM_NOLOAD': '1'}},
    {'path': 'tests/thin/sections_extra.py'},
    {'path': 'tests/thin/sections_migrate.py', 'args': '+balancer RotateLB', 'min': 2},
    {'path': 'tests/thin/atsync_in_coroutine.py', 'args': '+balancer RotateLB', 'min': 2},
    {'path': 'tests/thin/atsync_in_coroutine.py', 'args': 'norecurse'},
    {'path': 'tests/thin/views_retained.py', 'args': 'keep', 'only': [2]},
    {'path': 'tests/thin/views_retained.py', 'args': 'copy', 'only': [2]},
    {'path': 'tests/thin/views_retained.py', 'args': 'drop', 'only': [2]},
    {'path': 'tests/thin/checkpoint_restart.py', 'args': 'ckrs.ckpt', 'restart': 'ckrs.ckpt'},
    {'path': 'tests/migration/test_migrate.py', 'args': '+balancer RotateLB', 'min': 2},
    {'path': 'examples/particle/particle.py', 'args': '8 4 2500 +balancer RotateLB', 'min': 2},
]


def launch(n, t, extra):
    args = t['args'].split() if t.get('args') else []
    prog = [py] + ([t['path']] if not t.get('interactive') else ['-m', 'charm4py.interactive'])
    pre = [t['prefix']] if t.get('prefix') else []
    if a.mode == 'srun':
        return (['srun', '--mpi=pmix', '-n', str(n), '-c', '4', '--gres=gpu:%s' % a.gpus,
                 '--cpu-bind=none', '--kill-on-bad-exit=1'] + pre + prog + args + extra + ['+pe', str(n)])
    return [py, '-m', 'charmrun.start'] + pre + prog + args + extra + ['+p%d' % n]


def run_one(n, t, label, extra=(), suffix=''):
    cmd = launch(n, t, list(extra))
    name = (t['path'].replace('/', '_')[:-3] + ('_' + t['args'].split()[0].lstrip('+') if t.get('args') else '')
            + ('_exec' if t.get('prefix') else '') + suffix + '_n%d' % n)
    log = os.path.join(a.outdir, name + '.log')
    env = dict(os.environ); env.update(t.get('env', {}))
    limit = max(150, t.get('timeout_override', 0))
    t0 = time.time()
    with open(log, 'w') as f:
        f.write('CMD ' + ' '.join(cmd) + '\n'); f.flush()
        stdin = open(t['path']) if t.get('interactive') else subprocess.DEVNULL
        p = subprocess.Popen(cmd, stdin=stdin, stdout=f, stderr=subprocess.STDOUT, env=env,
                             start_new_session=True)
        try:
            rc = p.wait(limit)
        except subprocess.TimeoutExpired:
            p.terminate()
            try: p.wait(10)
            except subprocess.TimeoutExpired:
                p.terminate(); time.sleep(2); p.kill(); p.wait()
            rc = 'TIMEOUT'
    out = open(log, errors='replace').read()
    m = re.search(r'Starting Reconverse with (\d+) process', out)
    banner = int(m.group(1)) if m else None
    if rc == 0:
        res = 'PASS' if banner == n else 'BANNER?(%s)' % banner
    else:
        res = 'FAIL(%s)' % rc
    el = time.time() - t0
    line = '%-58s n=%d %-14s %6.1fs' % (label + suffix, n, res, el)
    print(line, flush=True)
    return res


summary = {'PASS': 0, 'other': 0, 'SKIP': 0}
for t in tests + thin:
    label = t['path'] + (' ' + t['args'] if t.get('args') else '') + (' [exec]' if t.get('prefix') else '')
    if not re.search(a.filter, label):
        continue
    c = t.get('condition')
    if c == 'numbaInstalled' and not numba_ok:
        print('%-58s SKIP(no numba)' % label, flush=True); summary['SKIP'] += 1; continue
    if c == 'not numbaInstalled' and numba_ok:
        continue
    if c == 'deviceSupport' and not device_ok:
        print('%-58s SKIP(no device support or no cupy)' % label, flush=True); summary['SKIP'] += 1; continue
    if 'only' in t:
        ns = t['only']
    elif 'force_min_processes' in t or 'min' in t:
        ns = [max(2, t.get('force_min_processes', 0), t.get('min', 0))]
    else:
        ns = counts
    for n in ns:
        r = run_one(n, t, label)
        if t.get('restart') and r == 'PASS':
            r2 = run_one(n, t, label, extra=['+restart', t['restart']], suffix=' +restart')
            summary['PASS' if r2 == 'PASS' else 'other'] += 1
        summary['PASS' if r == 'PASS' else 'other'] += 1
print('SUMMARY pass=%d not-pass=%d skipped=%d' % (summary['PASS'], summary['other'], summary['SKIP']), flush=True)
