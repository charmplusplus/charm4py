#!/usr/bin/env python3
"""Average the Jacobi3D times of scripts/charm4py.sh output files per mode.

Usage: process_times.py JOB_NAME START_NODES END_NODES
Reads JOB_NAME-n<nodes>*.out for nodes = START, 2*START, ..., END (the output
name pattern of charm4py.sh is jacobi3d-charm4py-n<nodelist>-<jobid>.out; rename
or link the files to JOB_NAME-n<nodes>-<anything>.out). Each run is attributed to
the mode on its "Mode:" line, so the order of the runs in a file does not matter
(PR #208 assumed three host runs followed by three device runs). Writes
JOB_NAME.csv with one row per node count: for each mode the mean and standard
deviation of the average time per iteration and of the communication time per
iteration, in ms.
"""
import csv
import glob
import re
import statistics
import sys

MODES = ('staged', 'device', 'persistent')


def parse(path):
    """ {mode: ([total ms], [comm ms])} of one output file """
    out = {}
    mode = None
    with open(path) as f:
        for line in f:
            m = re.search(r'Mode: (\w+)', line)
            if m:
                mode = m.group(1)
                out.setdefault(mode, ([], []))
            m = re.search(r'Average time per iteration: ([0-9.eE+-]+) ms', line)
            if m and mode:
                out[mode][0].append(float(m.group(1)))
            m = re.search(r'Communication time per iteration: ([0-9.eE+-]+) ms', line)
            if m and mode:
                out[mode][1].append(float(m.group(1)))
    return out


def stats(xs):
    if not xs:
        return '', ''
    return round(statistics.mean(xs), 3), round(statistics.stdev(xs), 3) if len(xs) > 1 else 0.0


def main():
    if len(sys.argv) != 4:
        print('usage:', sys.argv[0], 'JOB_NAME START_NODES END_NODES')
        sys.exit(1)
    job, start, end = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    with open(job + '.csv', 'w', newline='') as f:
        w = csv.writer(f)
        header = ['Number of nodes']
        for mode in MODES:
            header += [mode + '-total', 'error', mode + '-comm', 'error']
        w.writerow(header)
        nodes = start
        while nodes <= end:
            runs = {mode: ([], []) for mode in MODES}
            for path in sorted(glob.glob('%s-n%d*.out' % (job, nodes))):
                for mode, (tot, comm) in parse(path).items():
                    if mode in runs:
                        runs[mode][0].extend(tot)
                        runs[mode][1].extend(comm)
            row = [nodes]
            print('Nodes:', nodes)
            for mode in MODES:
                tot, comm = runs[mode]
                print('  %-10s total %s comm %s' % (mode, tot, comm))
                row += [*stats(tot), *stats(comm)]
            w.writerow(row)
            nodes *= 2
    print('wrote', job + '.csv')


if __name__ == '__main__':
    main()
