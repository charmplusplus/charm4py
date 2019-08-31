from charm4py import charm, Chare, Group, Future

# This example implements a simple distributed job scheduler.
# Jobs consist of one or multiple tasks. The type of job supported in this
# example is "parallel map", that applies a function to the elements of an
# iterable, using all available processes, returning the list of results.
# The scheduler is distributed in the sense that it will send tasks to any
# available processes, including processes on remote hosts.
# But there is only one central scheduler on process 0. The scheduler keeps
# track of which processes are idle (not running any tasks) and sends tasks
# to processes as they become idle (thus it balances load in a natural manner).
# Multiple jobs can be submitted to the scheduler, at the same or different
# times.

# NOTE: This is not meant to be the most scalable or efficient implementation.
# For example, grouping tasks into chunks for better efficiency is not
# implemented in this example.
# Charm4py has a more advanced implementation of this in charm4py/pool.py


class Job(object):
    """ This class is mainly for book-keeping (store and manage job state) """

    def __init__(self, job_id, func, tasks, callback):
        self.id = job_id
        self.func = func  # the function to use for the parallel map job
        self.tasks = tasks  # in this example, each task is an element of the iterable
        self.callback = callback  # user-supplied callback to send results of the job
        self.current_task_idx = 0
        self.resultsReceived = 0
        self.results = [None] * len(self.tasks)

    def isDone(self):
        return self.resultsReceived == len(self.tasks)

    def addResult(self, task_id, result):
        self.results[task_id] = result
        self.resultsReceived += 1

    def nextTask(self):
        if self.current_task_idx >= len(self.tasks):
            return None, None
        idx = self.current_task_idx
        self.current_task_idx += 1
        return self.tasks[idx], idx


class Scheduler(Chare):
    """ The scheduler sends tasks to distributed workers """

    def __init__(self):
        # create a Worker on every process, pass them a reference (proxy) to myself
        self.workers = Group(Worker, args=[self.thisProxy])
        # scheduler will send tasks to processes from 1 to N-1 (keep 0 free)
        self.free_workers = set(range(1, charm.numPes()))
        self.next_job_id = 0
        self.jobs = {}

    def map_async(self, func, iterable, callback):
        """ Start a new parallel map job (apply func to elements in iterable).
            The result will be sent back via the provided callback """
        self.addJob(func, list(iterable), callback)
        self.schedule()

    def addJob(self, func, tasks, callback):
        job = Job(self.next_job_id, func, tasks, callback)
        self.jobs[self.next_job_id] = job
        self.next_job_id += 1

    def schedule(self):
        for job in self.jobs.values():
            while len(self.free_workers) > 0:
                task, task_id = job.nextTask()
                if task is None:
                    # this job has no more tasks left to submit
                    break
                free_worker = self.free_workers.pop()
                # send task to a free worker
                self.workers[free_worker].apply(job.func, task, task_id, job.id)

    def taskDone(self, worker_id, task_id, job_id, result):
        """ Called by workers to tell the scheduler that they are done with a task """
        self.free_workers.add(worker_id)
        job = self.jobs[job_id]
        job.addResult(task_id, result)
        if job.isDone():
            self.jobs.pop(job.id)
            # job is done, send the result back to whoever submitted the job
            job.callback(job.results)  # callback is a callable
        self.schedule()


class Worker(Chare):

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def apply(self, func, arg, task_id, job_id):
        """ Apply function to argument and send the result to the scheduler """
        result = func(arg)
        self.scheduler.taskDone(self.thisIndex, task_id, job_id, result)


def square(x):
    return x**2


def main(args):
    assert charm.numPes() >= 2
    # create the Scheduler on PE 0
    scheduler = Chare(Scheduler, onPE=0)
    # create Futures to receive the results of two jobs
    future1 = Future()
    future2 = Future()
    # send two map_async jobs at the same time to the scheduler
    scheduler.map_async(square, [1, 2, 3, 4, 5], callback=future1)
    scheduler.map_async(square, [1, 3, 5, 7, 9], callback=future2)
    # wait for the two jobs to complete and print the results
    print('Final results are:')
    print(future1.get())
    print(future2.get())
    exit()


charm.start(main)
