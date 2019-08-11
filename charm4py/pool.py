from . import charm, Chare, Group, coro_ext, threads, Future
from .charm import Charm4PyError
from .threads import NotThreadedError
from collections import defaultdict
import sys


INITIAL_MAX_JOBS = 2048


class Task(object):

    def __init__(self, data, result_dest, func=None):
        if func is not None:
            self.func = func
        self.data = data
        # destination of task's result. can be a future or int. if int, it
        # indicates the position in the result list
        self.result_dest = result_dest


class Chunk(object):

    def __init__(self, tasks, result_dest):
        self.data = tasks
        self.result_dest = result_dest
        self.func = None


class Job(object):

    def __init__(self, id, func, tasks, result, ncores, chunksize, threaded):
        self.id = id
        self.max_cores = ncores
        self.n_avail = ncores
        self.func = func  # if func is not None, function is the same for all tasks in the job
        self.workers = []  # ID of workers who have executed tasks from this job
        self.chunked = chunksize > 1
        self.threaded = threaded
        self.failed = False
        assert chunksize > 0
        if self.chunked:
            if isinstance(result, threads.Future):
                self.results = [None] * len(tasks)
                self.future = result
                self.tasks = [Chunk(tasks[i:i+chunksize], i) for i in range(0, len(tasks), chunksize)]
            else:
                self.tasks = [Chunk(tasks[i:i+chunksize], result[i:i+chunksize]) for i in range(0, len(tasks), chunksize)]
        else:
            if isinstance(result, threads.Future):
                self.results = [None] * len(tasks)
                self.future = result
                if func is not None:
                    self.tasks = [Task(args, i) for i, args in enumerate(tasks)]
                else:
                    self.tasks = [Task(args, i, func) for i, (func, args) in enumerate(tasks)]
            else:
                if func is not None:
                    self.tasks = [Task(args, result[i]) for i, args in enumerate(tasks)]
                else:
                    self.tasks = [Task(args, result[i], func) for i, (func, args) in enumerate(tasks)]
        # print('Created job with', len(self.tasks), 'tasks')
        self.tasks_pending = len(self.tasks)

    def getTask(self):
        if self.n_avail > 0:
            self.n_avail -= 1
            return self.tasks.pop()
        else:
            return None

    def taskDone(self):
        self.n_avail += 1
        self.tasks_pending -= 1


class PoolScheduler(Chare):

    def __init__(self):
        self.workers = None
        self.idle_workers = set(range(1, charm.numPes()))
        self.num_workers = len(self.idle_workers)
        self.jobs = [None] * INITIAL_MAX_JOBS
        self.job_id_pool = set(range(INITIAL_MAX_JOBS))
        self.job_next = None
        self.job_last = self
        self.worker_knows = defaultdict(set)
        self.setMigratable(False)

    def start(self, func, tasks, result, ncores, chunksize, allow_nested):
        assert ncores != 0
        if ncores < 0:
            ncores = self.num_workers
        elif ncores > self.num_workers:
            print('charm.pool Warning: requested more cores than are '
                  'available. Using max available cores')
            ncores = self.num_workers

        if self.workers is None:
            assert self.num_workers > 0, 'Run with more than 1 PE to use charm.pool'
            # first time running a job, create Group of workers
            print('Initializing charm.pool with', self.num_workers, 'worker PEs. '
                  'Warning: charm.pool is experimental (API and performance '
                  'is subject to change)')
            self.workers = Group(Worker, args=[self.thisProxy])

        if len(self.job_id_pool) == 0:
            oldSize = len(self.jobs)
            newSize = int(oldSize * 1.5)
            self.job_id_pool.update(range(oldSize, newSize))
            self.jobs.extend([None] * (newSize - oldSize))

        if charm.interactive:
            try:
                if func is not None:
                    self.workers.check(func.__module__, func.__name__, ret=1).get()
                else:
                    for func_, args in tasks:
                        self.workers.check(func_.__module__, func_.__name__, ret=1).get()
            except Exception as e:
                if isinstance(result, threads.Future):
                    result.send(e)
                else:
                    for f in result:
                        f.send(e)
                return

        job_id = self.job_id_pool.pop()
        job = Job(job_id, func, tasks, result, ncores, chunksize, allow_nested)
        self.jobs[job_id] = job
        self.job_last.job_next = job
        self.job_last = job
        job.job_next = None

        if job.chunked:
            if job.func is not None:
                if job.threaded:
                    job.remote = self.workers.runChunkSingleFunc_th
                else:
                    job.remote = self.workers.runChunkSingleFunc
            else:
                if job.threaded:
                    job.remote = self.workers.runChunk_th
                else:
                    job.remote = self.workers.runChunk
        else:
            if job.func is not None:
                if job.threaded:
                    job.remote = self.workers.runTaskSingleFunc_th
                else:
                    job.remote = self.workers.runTaskSingleFunc
            else:
                if job.threaded:
                    job.remote = self.workers.runTask_th
                else:
                    job.remote = self.workers.runTask

        self.schedule()

    def schedule(self):
        job = self.job_next
        prev = self
        while job is not None:
            if len(self.idle_workers) == 0:
                return
            while True:
                if not job.failed:
                    task = job.getTask()
                    if task is None:
                        break
                    worker_id = self.idle_workers.pop()
                    # print('Sending task to worker', worker_id)

                    if job.func is not None:
                        func = None
                        if job.id not in self.worker_knows[worker_id]:
                            func = job.func
                            job.workers.append(worker_id)
                            self.worker_knows[worker_id].add(job.id)
                    else:
                        func = task.func
                    # NOTE: this is a non-standard way of using proxies, but is
                    # faster and allows the scheduler to reuse the same proxy
                    self.workers.elemIdx = worker_id
                    job.remote(func, task.data, task.result_dest, job.id)

                if len(job.tasks) == 0:
                    prev.job_next = job.job_next
                    if job == self.job_last:
                        self.job_last = prev
                    # print('Deleted job set')
                    job = None
                    break
                if len(self.idle_workers) == 0:
                    return
            # go to next job
            if job is not None:
                prev = job
                job = job.job_next
            else:
                job = prev.job_next

    def taskFinished(self, worker_id, job_id, result=None):
        # print('Job finished')
        job = self.jobs[job_id]
        if job.failed:
            return self.taskError(worker_id, job_id, job.exception)
        if result is not None:
            if job.chunked:
                i, results = result
                n = len(results)
                job.results[i:i+n] = results
            else:
                i, _result = result
                job.results[i] = _result
        self.idle_workers.add(worker_id)
        job.taskDone()
        if job.tasks_pending == 0:
            self.jobs[job_id] = None
            self.job_id_pool.add(job_id)
            for worker_id in job.workers:
                self.worker_knows[worker_id].remove(job.id)
            if result is not None:
                job.future.send(job.results)
        self.schedule()

    def threadPaused(self, worker_id):
        self.idle_workers.add(worker_id)
        self.schedule()

    def threadResumed(self, worker_id):
        self.idle_workers.discard(worker_id)

    def migrated(self):
        charm.abort('Someone migrated PoolScheduler which is non-migratable')

    def taskError(self, worker_id, job_id, exception):
        job = self.jobs[job_id]
        job.exception = exception
        self.idle_workers.add(worker_id)
        # marking as failed will allow the scheduler to delete it from the linked list
        # NOTE that we will only delete from the 'jobs' list once all the pending tasks are done
        job.failed = True
        if not hasattr(job, 'future'):
            if job.chunked:
                for chunk in job.tasks:
                    for f in chunk.result_dest:
                        f.send(job.exception)
            else:
                for t in job.tasks:
                    t.result_dest.send(job.exception)
        job.tasks = []
        job.taskDone()
        if job.n_avail == job.max_cores:  # all the running tasks are done
            self.jobs[job_id] = None
            self.job_id_pool.add(job_id)
            for worker_id in job.workers:
                self.worker_knows[worker_id].remove(job.id)
            if hasattr(job, 'future'):
                job.future.send(job.exception)
        self.schedule()


class Worker(Chare):

    def __init__(self, scheduler):
        self.scheduler = scheduler
        assert len(self.scheduler.elemIdx) > 0  # make sure points to the element, not collection
        self.__addThreadEventSubscriber__(scheduler, self.thisIndex)
        # TODO: when to purge entries from this dict?
        self.funcs = {}  # job ID -> function used by this job ID

    @coro_ext(event_notify=True)
    def runTaskSingleFunc_th(self, func, args, result_destination, job_id):
        self.runTaskSingleFunc(func, args, result_destination, job_id)

    def runTaskSingleFunc(self, func, args, result_destination, job_id):
        if func is not None:
            self.funcs[job_id] = func
        else:
            func = self.funcs[job_id]
        self.runTask(func, args, result_destination, job_id)

    @coro_ext(event_notify=True)
    def runTask_th(self, func, args, result_destination, job_id):
        self.runTask(func, args, result_destination, job_id)

    def runTask(self, func, args, result_destination, job_id):
        try:
            result = func(args)
            if isinstance(result_destination, int):
                self.scheduler.taskFinished(self.thisIndex, job_id, (result_destination, result))
            else:
                # assume result_destination is a future
                result_destination.send(result)
                self.scheduler.taskFinished(self.thisIndex, job_id)
        except Exception as e:
            if isinstance(e, NotThreadedError):
                e = Charm4PyError('Use allow_nested=True to allow launching tasks'
                                  ' from tasks with charm.pool')
            charm.prepareExceptionForSend(e)
            self.scheduler.taskError(self.thisIndex, job_id, e)
            if not isinstance(result_destination, int):
                result_destination.send(e)

    @coro_ext(event_notify=True)
    def runChunkSingleFunc_th(self, func, chunk, result_destination, job_id):
        self.runChunkSingleFunc(func, chunk, result_destination, job_id)

    def runChunkSingleFunc(self, func, chunk, result_destination, job_id):
        try:
            if func is not None:
                self.funcs[job_id] = func
            else:
                func = self.funcs[job_id]
            results = [func(args) for args in chunk]
            self.send_chunk_results(results, result_destination, job_id)
        except Exception as e:
            self.send_chunk_exc(e, result_destination, job_id)

    @coro_ext(event_notify=True)
    def runChunk_th(self, _, chunk, result_destination, job_id):
        try:
            results = [func(args) for func, args in chunk]
            self.send_chunk_results(results, result_destination, job_id)
        except Exception as e:
            self.send_chunk_exc(e, result_destination, job_id)

    def runChunk(self, _, chunk, result_destination, job_id):
        try:
            results = [func(args) for func, args in chunk]
            self.send_chunk_results(results, result_destination, job_id)
        except Exception as e:
            self.send_chunk_exc(e, result_destination, job_id)

    def send_chunk_results(self, results, result_destination, job_id):
        if isinstance(result_destination, int):
            self.scheduler.taskFinished(self.thisIndex, job_id, (result_destination, results))
        else:
            # assume result_destination is a list of futures
            # TODO: should send all results together to PE where future was created,
            # and then send from there to destination future
            for i, result in enumerate(results):
                result_destination[i].send(result)
            self.scheduler.taskFinished(self.thisIndex, job_id)

    def send_chunk_exc(self, e, result_destination, job_id):
        if isinstance(e, NotThreadedError):
            e = Charm4PyError('Use allow_nested=True to allow launching tasks'
                              ' from tasks with charm.pool')
        charm.prepareExceptionForSend(e)
        self.scheduler.taskError(self.thisIndex, job_id, e)
        if not isinstance(result_destination, int):
            for f in result_destination:
                f.send(e)

    def check(self, func_module, func_name):
        eval(func_name, sys.modules[func_module].__dict__)


# This acts as an interface to charm.pool. It is not a chare
# An instance of this object exists on every process
class Pool(object):

    def __init__(self, pool_scheduler):
        # proxy to PoolScheduler singleton chare
        self.pool_scheduler = pool_scheduler

    def map_async(self, func, iterable, ncores=-1, multi_future=False, chunksize=1, allow_nested=False):
        if isinstance(iterable, list):
            tasks = iterable
        else:
            tasks = list(iterable)
        if multi_future:
            result = [Future() for _ in range(len(tasks))]
        else:
            result = Future()
        # TODO shouldn't send task objects to a central place. what if they are large?
        self.pool_scheduler.start(func, tasks, result, ncores, chunksize, allow_nested)
        return result

    def map(self, func, iterable, ncores=-1, chunksize=1, allow_nested=False):
        return self.map_async(func, iterable, ncores, multi_future=False,
                              chunksize=chunksize,
                              allow_nested=allow_nested).get()

    # iterable is sequence of (function, args) tuples
    # NOTE: this API may change in the future
    def submit_async(self, iterable, ncores=-1, multi_future=False, chunksize=1, allow_nested=False):
        if isinstance(iterable, list):
            tasks = iterable
        else:
            tasks = list(iterable)
        if multi_future:
            result = [Future() for _ in range(len(tasks))]
        else:
            result = Future()
        # TODO shouldn't send task objects to a central place. what if they are large?
        self.pool_scheduler.start(None, tasks, result, ncores, chunksize, allow_nested)
        return result

    def submit(self, iterable, ncores=-1, multi_future=False, chunksize=1, allow_nested=False):
        return self.submit_async(iterable, ncores, multi_future=False,
                                 chunksize=chunksize,
                                 allow_nested=allow_nested).get()
