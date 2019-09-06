from charm4py import charm


charm.options.local_msg_buf_size = 1000
charm.options.remote_exec = True


class MyException(Exception):
    def __init__(self):
        super(MyException, self).__init__()


myfunc = None

myfunc_bad_source = '''
def myfunc(x):
    raise MyException
    return x**2
'''

myfunc_good_source = """
def myfunc(x):
    return x**2
"""


def main(args):
    num_tasks = (charm.numPes() - 1) * 20

    for _ in range(5):
        for trial in range(2):
            if trial == 0:
                # use bad func
                charm.thisProxy.rexec(myfunc_bad_source, awaitable=True).get()
            else:
                # use good func
                charm.thisProxy.rexec(myfunc_good_source, awaitable=True).get()

            for func in (myfunc, None):
                for multi_future in (False, True):
                    for chunk_size in (1, 4):
                        try:
                            if func is None:
                                tasks = [(myfunc, i) for i in range(num_tasks)]
                                result = charm.pool.submit_async(tasks, multi_future=multi_future, chunksize=chunk_size)
                            else:
                                tasks = range(num_tasks)
                                result = charm.pool.map_async(func, tasks, multi_future=multi_future, chunksize=chunk_size)
                            if multi_future:
                                result = [f.get() for f in result]
                            else:
                                result = result.get()
                            assert trial == 1 and result == [x**2 for x in range(num_tasks)]
                        except MyException:
                            assert trial == 0
    exit()


charm.start(main)
