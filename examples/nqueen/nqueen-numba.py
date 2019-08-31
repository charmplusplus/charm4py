from charm4py import charm
from time import time
import numpy
import numba

# See README.rst


def queen(current_row, solution):
    if NUM_ROWS - current_row <= GRAINSIZE:
        # number of rows left <= grainsize, so now we do sequential computation
        global solution_count
        solution_count += queen_seq(current_row, solution)
    else:
        for column in range(NUM_ROWS):
            if valid_move(current_row, solution, column):
                solution[current_row] = column
                # charm will spawn the task on any available PE
                charm.pool.Task(queen, args=[current_row + 1, solution])


@numba.jit(nopython=True, cache=False)
def queen_seq(current_row, solution):
    num_solutions = 0
    for column in range(NUM_ROWS):
        if valid_move(current_row, solution, column):
            if current_row + 1 == NUM_ROWS:
                # valid solution with one queen per row found
                return 1
            solution[current_row] = column
            num_solutions += queen_seq(current_row + 1, solution)
    return num_solutions


@numba.jit(nopython=True, cache=False)
def valid_move(cur_row, solution, column):
    for row in range(cur_row):
        q_col = solution[row]
        # if in the same column or same diagonal, can't place queen here
        if q_col == column or (cur_row - row) == abs(column - q_col):
            return False
    return True


def numbaPrecompile():
    # trigger compilation by running the function with dummy data (but correct types)
    queen_seq(NUM_ROWS-1, numpy.full(NUM_ROWS, -1, dtype=numpy.int8))


def main(args):
    NUM_ROWS = 13  # size of board is NUM_ROWS x NUM_ROWS
    if len(args) > 1:
        NUM_ROWS = int(args[1])
    if len(args) > 2:
        GRAINSIZE = min(int(args[2]), NUM_ROWS)
    else:
        GRAINSIZE = max(1, NUM_ROWS - 2)

    print('\nUsage: nqueen [numqueens] [grainsize]')
    print('Number of queens is', NUM_ROWS, ', grainsize is', GRAINSIZE)

    # set NUM_ROWS and GRAINSIZE as global variables on every PE
    global_data = {}
    global_data['NUM_ROWS'] = NUM_ROWS
    global_data['GRAINSIZE'] = GRAINSIZE
    global_data['solution_count'] = 0  # to count number of solutions found on each PE
    charm.thisProxy.updateGlobals(global_data, ret=1).get()

    # compile numba functions on every PE before starting, to get
    # consistent benchmark results
    charm.thisProxy.rexec('numbaPrecompile()', ret=1).get()

    startTime = time()
    # initialize empty solution, solution holds the column number where a queen is placed, for each row
    solution = numpy.full(NUM_ROWS, -1, dtype=numpy.int8)
    queen(0, solution)
    # wait until there is no work being done on any PE (quiescence detection)
    charm.waitQD()
    elapsed = time() - startTime
    numSolutions = sum(charm.thisProxy.eval('solution_count', ret=2).get())
    print('There are', numSolutions, 'solutions to', NUM_ROWS, 'queens. Time taken:', round(elapsed, 3), 'secs')
    exit()


charm.start(main)