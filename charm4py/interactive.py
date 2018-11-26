from . import charm, Chare, threaded
import sys
import time
import re
from code import InteractiveInterpreter
import os


class InteractiveConsole(Chare, InteractiveInterpreter):

    def __init__(self, args):
        # restore original tty stdin and stdout (else readline won't work correctly)
        os.dup2(charm.origStdinFd, 0)
        os.dup2(charm.origStoutFd, 1)
        InteractiveInterpreter.__init__(self, locals=charm.dynamic_register)
        self.filename = '<console>'
        self.resetbuffer()
        # regexp to detect when user defines a new chare type
        self.regexpChareDefine = re.compile('class\s*(\S+)\s*\(.*Chare.*\)\s*:')

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

    def write(self, data):
        # print(data)
        sys.stdout.write(data)
        sys.stdout.flush()
        # go through charm scheduler to keep things moving
        self.thisProxy.null(ret=True).get()

    @threaded
    def start(self):
        self.write('Charm4py interactive shell\n')
        more = 0
        tick = time.time()
        while 1:
            # self.thisProxy.null(ret=True).get()
            # time.sleep(0.001)
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
            self.thisProxy.null(ret=True).get()
        return more

    def runcode(self, code):
        chareTypeDefined = False
        for line in self.buffer:
            m = self.regexpChareDefine.search(line)
            if m is not None:
                newChareTypeName = m.group(1)
                source = '\n'.join(self.buffer)
                # print("Chare type '" + newChareTypeName + "' source:\n", source)
                charm.thisProxy.registerNewChareType(newChareTypeName, source, ret=True).get()
                chareTypeDefined = True
        if not chareTypeDefined:
            InteractiveInterpreter.runcode(self, code)

    def raw_input(self, prompt=''):
        return input(prompt)
