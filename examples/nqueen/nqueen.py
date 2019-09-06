from charm4py import charm, Chare, Group
from time import time
import array

# See README.rst
# This runs much faster with PyPy, and the Numba version (nqueen-numba.py)
# runs much faster than this and PyPy


def queen(current_row, solution):
    global solution_count
    # if number of rows left <= grainsize, we stop spawning tasks, so we just
    # do a recursive algorithm on this PE
    spawn_tasks = (NUM_ROWS - current_row) > GRAINSIZE
    for column in range(NUM_ROWS):
        if valid_move(current_row, solution, column):
            if current_row + 1 == NUM_ROWS:
                # valid solution with one queen per row found
                solution_count += 1
                continue
            solution[current_row] = column  # mark the partial solution
            if spawn_tasks:
                # charm will spawn the task on any available PE
                charm.pool.Task(queen, args=[current_row + 1, solution])
            else:
                queen(current_row + 1, solution)


def valid_move(cur_row, solution, column):
    for row in range(cur_row):
        q_col = solution[row]
        # if in the same column or same diagonal, can't place queen here
        if q_col == column or (cur_row - row) == abs(column - q_col):
            return False
    return True


class Util(Chare):
    def getSolutionCount(self):
        return solution_count


def main(args):
    global NUM_ROWS, GRAINSIZE
    NUM_ROWS = 5  # size of board is NUM_ROWS x NUM_ROWS
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

    startTime = time()
    # initialize empty solution, solution holds the column number where a queen is placed, for each row
    solution = array.array('b', [-1] * NUM_ROWS)
    queen(0, solution)
    # wait until there is no work being done on any PE (quiescence detection)
    charm.waitQD()
    elapsed = time() - startTime
    numSolutions = sum(Group(Util).getSolutionCount(ret=True).get())
    print('There are', numSolutions, 'solutions to', NUM_ROWS, 'queens. Time taken:', round(elapsed, 3), 'secs')
    exit()


charm.start(main)
