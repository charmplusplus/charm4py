from charm4py import charm, Chare, Group

# this class is mainly for bookkeeping to store and manage job state
class Job:
    def __init__(self, job_id, tasks, procs, future):
        self.id = job_id
        self.tasks = tasks
        self.curTask = 0
        self.procs = procs
        self.resultsReceived = 0
        self.future = future
        self.results = [None] * len(self.tasks)

    def isDone(self):
        return self.resultsReceived == len(self.tasks)

    def addResult(self, task_id, result):
        self.results[task_id] = result
        self.resultsReceived += 1

    def nextTask(self):
        t = self.curTask
        if t >= len(self.tasks): return None
        self.curTask += 1
        return t


class Master(Chare):

    def __init__(self):
        # create a Worker in every processor
        self.workers = Group(Worker)
        self.free_procs = set(range(1, charm.numPes()))
        self.next_job_id = 0
        self.jobs = {}

    def addJob(self, tasks, procs, future):
        job = Job(self.next_job_id, tasks, procs, future)
        self.jobs[self.next_job_id] = job
        self.next_job_id += 1
        return job

    def map_async(self, func, numProcs, tasks, future):
        """ start a new map job (to apply func to tasks), using numProcs processors
            result is sent to given future """
        free = [self.free_procs.pop() for i in range(numProcs)] # select free processors
        job = self.addJob(tasks, free, future)
        # tell workers in selected processors to start the job
        for p in free:
            self.workers[p].start(job.id, func, tasks, self.thisProxy)

    def getTask(self, src, job_id, prev_task=None, prev_result=None):
        """ called by worker to get a new task """
        job = self.jobs[job_id]
        if prev_task is not None:
            job.addResult(prev_task, prev_result)
        if not job.isDone():
            next_task = job.nextTask()
            if next_task is not None: self.workers[src].apply(next_task)
        else:
            for p in job.procs: self.free_procs.add(p)
            self.jobs.pop(job.id)
            job.future.send(job.results)


class Worker(Chare):

    def start(self, job_id, f, tasks, master):
        self.job_id = job_id
        self.func   = f
        self.tasks  = tasks
        self.master = master
        # request a new task
        master.getTask(self.thisIndex, job_id)

    def apply(self, task_id):
        """ apply function to task and send result to master """
        result = self.func(self.tasks[task_id])
        self.master.getTask(self.thisIndex, self.job_id, task_id, result)


def f(x):
    return x*x

def main(args):
    if charm.numPes() < 5:
        print("\nRun this example with at least 5 PEs\n")
        exit()
    pool = Chare(Master, onPE=0) # create one Master called 'pool' on PE 0
    f1 = charm.createFuture()
    f2 = charm.createFuture()
    tasks1 = [1, 2, 3, 4, 5]
    tasks2 = [1, 3, 5, 7, 9]
    pool.map_async(f, 2, tasks1, f1)
    pool.map_async(f, 2, tasks2, f2)
    print("Final results are", f1.get(), f2.get()) # wait on futures
    exit()

charm.start(main)
