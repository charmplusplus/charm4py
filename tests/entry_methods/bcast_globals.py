from charm4py import charm, Chare, Group


group1_proxy = group2_proxy = done_future = None


class Test(Chare):

    def start(self):
        group2_proxy[(self.thisIndex + 1) % charm.numPes()].ping()

    def ping(self):
        group1_proxy[(self.thisIndex - 1) % charm.numPes()].pong()

    def pong(self):
        self.contribute(None, None, done_future)


def main(args):
    g1 = Group(Test)
    g2 = Group(Test)
    done = charm.Future()

    main_globals = {}
    main_globals['group1_proxy'] = g1
    main_globals['group2_proxy'] = g2
    main_globals['done_future'] = done
    charm.thisProxy.updateGlobals(main_globals, module_name='__main__', awaitable=True).get()

    group1_proxy.start()
    done.get()
    exit()


charm.start(main)
