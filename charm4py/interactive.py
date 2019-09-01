from . import charm, Chare, coro, Future
from . import chare
import sys
import time
import re
from code import InteractiveInterpreter
import os
import inspect


HANG_CHECK_FREQ = 0.5  # in secs


def future_():
    f = Future()
    charm.dynamic_register['_f'] = f
    return f


class InteractiveConsole(Chare, InteractiveInterpreter):

    def __init__(self, args):
        global Charm4PyError
        from .charm import Charm4PyError
        # restore original tty stdin and stdout (else readline won't work correctly)
        os.dup2(charm.origStdinFd, 0)
        os.dup2(charm.origStoutFd, 1)
        charm.dynamic_register['future'] = future_
        charm.dynamic_register['self'] = self
        InteractiveInterpreter.__init__(self, locals=charm.dynamic_register)
        self.filename = '<console>'
        self.resetbuffer()
        # regexp to detect when user defines a new chare type
        self.regexpChareDefine = re.compile('class\s*(\S+)\s*\(.*Chare.*\)\s*:')
        # regexps to detect import statements
        self.regexpImport1 = re.compile('\s*from\s*(\S+) import')
        self.regexpImport2 = re.compile('import\s*(\S+)')
        self.options = charm.options.interactive

        try:
            import readline
            import rlcompleter
            readline.parse_and_bind('tab: complete')
        except:
            pass

        try:
            sys.ps1
        except AttributeError:
            sys.ps1 = '>>> '
        try:
            sys.ps2
        except AttributeError:
            sys.ps2 = '... '
        self.thisProxy.start()

    def resetbuffer(self):
        self.buffer = []

    def null(self):
        return

    def write(self, data, sched=True):
        sys.stdout.write(data)
        sys.stdout.flush()
        if sched:
            # go through charm scheduler to keep things moving
            self.thisProxy.null(awaitable=True).get()

    @coro
    def start(self):
        self.write('\nCharm4py interactive shell (beta)\n')
        self.write('charm.options.interactive.verbose = ' + str(self.options.verbose) + '\n')

        charm.scheduleCallableAfter(self.thisProxy.hang_check_phase1, HANG_CHECK_FREQ)
        self.monitorFutures = []
        self.interactive_running = False

        more = 0
        tick = time.time()
        while 1:
            if time.time() - tick > 0.01:
                try:
                    if more:
                        prompt = sys.ps2
                    else:
                        prompt = sys.ps1
                    try:
                        line = self.raw_input(prompt)
                        tick = time.time()
                    except EOFError:
                        self.write('\n')
                        break
                    else:
                        more = self.push(line)
                except KeyboardInterrupt:
                    self.write('\nKeyboardInterrupt\n')
                    self.resetbuffer()
                    more = 0

    def push(self, line):
        self.buffer.append(line)
        source = '\n'.join(self.buffer)
        more = self.runsource(source, self.filename)
        if not more:
            self.resetbuffer()
            self.thisProxy.null(awaitable=True).get()
        return more

    def runcode(self, code):
        try:
            for line in self.buffer:
                m = self.regexpChareDefine.search(line)
                if m is not None:
                    newChareTypeName = m.group(1)
                    source = '\n'.join(self.buffer)
                    charm.thisProxy.registerNewChareType(newChareTypeName, source, awaitable=True).get()
                    if self.options.verbose > 0:
                        self.write('Charm4py> Broadcasted Chare definition\n')
                    return

            line = self.buffer[0]
            module_name = None
            if 'import' in line:
                m = self.regexpImport1.search(line)
                if m is not None:
                    module_name = m.group(1)
                else:
                    m = self.regexpImport2.match(line)
                    if m is not None:
                        module_name = m.group(1)
            if module_name is not None:
                prev_modules = set(sys.modules.keys())
                InteractiveInterpreter.runcode(self, code)
                if module_name not in sys.modules:  # error importing the module
                    return
                if self.options.broadcast_imports:
                    charm.thisProxy.rexec('\n'.join(self.buffer), awaitable=True).get()
                    if self.options.verbose > 0:
                        self.write('Charm4py> Broadcasted import statement\n')

                new_modules = set(sys.modules.keys()) - prev_modules
                chare_types = []
                for module_name in new_modules:
                    try:
                        members = inspect.getmembers(sys.modules[module_name], inspect.isclass)
                    except:
                        # some modules can throw exceptions with inspect.getmembers, ignoring them for now
                        continue
                    for C_name, C in members:
                        if C.__module__ != chare.__name__ and hasattr(C, 'mro'):
                            if chare.ArrayMap in C.mro():
                                chare_types.append(C)
                            elif Chare in C.mro():
                                chare_types.append(C)
                            elif chare.Group in C.mro() or chare.Array in C.mro() or chare.Mainchare in C.mro():
                                raise Charm4PyError('Chares must not inherit from Group, Array or'
                                                    ' Mainchare. Refer to new API')
                if len(chare_types) > 0:
                    if self.options.broadcast_imports:
                        charm.thisProxy.registerNewChareTypes(chare_types, awaitable=True).get()
                        if self.options.verbose > 0:
                            self.write('Broadcasted the following chare definitions: ' + str([str(C) for C in chare_types]) + '\n')
                    else:
                        self.write('Charm4py> ERROR: import module(s) contain Chare definitions but the import was not broadcasted\n')
                return
        except:
            self.showtraceback()

        self.interactive_running = True
        InteractiveInterpreter.runcode(self, code)
        self.interactive_running = False

    def raw_input(self, prompt=''):
        return input(prompt)

    def hang_check_phase1(self):
        self.monitorFutures = [f for f in self.monitorFutures if f.blocked]
        if self.interactive_running:
            for f in charm.threadMgr.futures.values():
                if f.blocked and not hasattr(f, 'ignorehang') and not hasattr(f, 'timestamp'):
                    f.timestamp = time.time()
                    self.monitorFutures.append(f)
            for f in self.monitorFutures:
                if time.time() - f.timestamp >= 2.0:
                    charm.startQD(self.thisProxy.hang_check_phase2)
                    return
        charm.scheduleCallableAfter(self.thisProxy.hang_check_phase1, HANG_CHECK_FREQ)

    def hang_check_phase2(self):
        monitor_futures = self.monitorFutures
        self.monitorFutures = []
        charm.scheduleCallableAfter(self.thisProxy.hang_check_phase1, HANG_CHECK_FREQ)
        for f in monitor_futures:
            if f.blocked:
                self.write('\nError: system is idle, canceling block on future\n', sched=False)
                charm.threadMgr.cancelFuture(f)

    def showtraceback(self):
        error_type, error, tb = sys.exc_info()
        if hasattr(error, 'remote_stacktrace'):
            origin, stacktrace = error.remote_stacktrace
            self.write('----------------- Python Stack Traceback from PE ' + str(origin) + ' -----------------\n')
            self.write(stacktrace + '\n')
            self.write(error_type.__name__ + ': ' + str(error) + ' (PE ' + str(origin) + ')\n')
        else:
            super(InteractiveConsole, self).showtraceback()


if __name__ == '__main__':
    charm.start(interactive=True)
