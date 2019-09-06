from charm4py import charm, Chare, Group
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


class Util(Chare):

    def compile(self):
        # trigger compilation by running the function with dummy data (but correct types)
        queen_seq(NUM_ROWS-1, numpy.full(NUM_ROWS, -1, dtype=numpy.int8))

    def getSolutionCount(self):
        return solution_count


def main(args):
    global NUM_ROWS, GRAINSIZE
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
    charm.thisProxy.updateGlobals(global_data, awaitable=True).get()

    # compile numba functions on every PE before starting, to get
    # consistent benchmark results
    util = Group(Util)
    util.compile(awaitable=True).get()

    startTime = time()
    # initialize empty solution, solution holds the column number where a queen is placed, for each row
    solution = numpy.full(NUM_ROWS, -1, dtype=numpy.int8)
    queen(0, solution)
    # wait until there is no work being done on any PE (quiescence detection)
    charm.waitQD()
    elapsed = time() - startTime
    numSolutions = sum(util.getSolutionCount(ret=True).get())
    print('There are', numSolutions, 'solutions to', NUM_ROWS, 'queens. Time taken:', round(elapsed, 3), 'secs')
    exit()


charm.start(main)
