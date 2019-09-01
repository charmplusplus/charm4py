from charm4py import charm, Chare, Group, coro
import sys
import socket
import subprocess

# Very simple hardware monitor that reports high CPU temperatures of hosts on a
# cluster. There is one Monitor chare running on each host. Monitors send high
# temperature reports to a Controller running on process 0


EXIT_AFTER_SECS = 30


class Controller(Chare):

    @coro
    def start(self, monitors, logfilename=None):
        print('\nStarting hardware monitor...')
        if logfilename is not None:
            self.log = open(logfilename, 'a')
        else:
            self.log = sys.stdout
        self.hosts = monitors.getHostName(ret=True).get()
        for i, host in enumerate(self.hosts):
            print('Monitor', i, 'running on host', host)
        print('Going to run for', EXIT_AFTER_SECS, 'secs')
        monitors.start(self.thisProxy)
        charm.scheduleCallableAfter(self.thisProxy.close, EXIT_AFTER_SECS)

    def close(self):
        self.log.close()
        exit()

    def reportAboveThreshold(self, values, from_id):
        self.log.write('Host ' + str(self.hosts[from_id]) + ' is running hot: ' + str(values) + '\n')
        self.log.flush()


class Monitor(Chare):

    @coro
    def start(self, controller, threshold=50.0):
        while True:
            temperatures = self.read_sensor()
            t_above_threshold = [t for t in temperatures if t >= threshold]
            if len(t_above_threshold) > 0:
                # this is asynchronous and monitor doesn't wait for completion
                controller.reportAboveThreshold(t_above_threshold, charm.myHost())
            # put coroutine to sleep, returning control to the scheduler
            charm.sleep(1.0)

    def getHostName(self):
        return socket.gethostname()

    def read_sensor(self):
        # note that this depends on specific output format of the sensors
        # command, which could change in the future. Adapt as needed
        lines = subprocess.check_output('sensors').decode().split('\n')
        temps = []
        for l in lines:
            fields = l.split()
            if len(fields) > 0 and fields[0] == 'Core':
                temps.append(float(fields[2][1:-2]))
        return temps


def main(args):
    # create a single controller on PE 0
    controller = Chare(Controller, onPE=0)
    # We only need one monitor per host. For this example, we could just launch
    # the program with one process per host and create a Group of monitors.
    # But more complex applications might need multiple processes per host
    # and still need to restrict certain groups of chares to certain PEs. Here
    # we illustrate how to do that:
    rank0pes = [charm.getHostFirstPe(host) for host in range(charm.numHosts())]
    # create monitors only on the first process of each host
    monitors = Group(Monitor, onPEs=rank0pes)
    controller.start(monitors)


charm.start(main)
