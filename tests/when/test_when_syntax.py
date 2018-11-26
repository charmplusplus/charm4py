from charm4py import charm
from charm4py import wait
import re


# This program tests that different types of @when conditional statements (given
# by strings) are transformed into the correct objects (from wait module) that
# handle that type of condition

# NOTE: this is not a parallel program

def parseMethodArgs(s):
    arg_names = re.split(', *', s[1:-1])
    method_args = {}
    for i in range(1, len(arg_names)):
      method_args[arg_names[i]] = i-1
    return method_args


def main(args):

    when_cond = 'self.iterations == iter'
    method    = '(self, iter, x, y)'
    cond = wait.parse_cond_str(when_cond, parseMethodArgs(method))
    assert isinstance(cond, wait.MsgTagCond)
    assert cond.attrib_name == 'iterations'
    assert cond.arg_idx == 0

    when_cond = 'self.x == x'
    method    = '(self, iter, x, y)'
    cond = wait.parse_cond_str(when_cond, parseMethodArgs(method))
    assert isinstance(cond, wait.MsgTagCond)
    assert cond.attrib_name == 'x'
    assert cond.arg_idx == 1

    when_cond = 'y    ==    self.x  '
    method    = '(self, iter, x, y)'
    cond = wait.parse_cond_str(when_cond, parseMethodArgs(method))
    assert isinstance(cond, wait.MsgTagCond)
    assert cond.attrib_name == 'x'
    assert cond.arg_idx == 2

    when_cond = 'self.x == x + y'
    method    = '(self, iter, x, y)'
    cond = wait.parse_cond_str(when_cond, parseMethodArgs(method))
    assert isinstance(cond, wait.ChareStateMsgCond)

    when_cond = 'x < y'
    method    = '(self, iter, x, y)'
    cond = wait.parse_cond_str(when_cond, parseMethodArgs(method))
    assert isinstance(cond, wait.ChareStateMsgCond)

    when_cond = 'y == y'
    method    = '(self, iter, x, y)'
    cond = wait.parse_cond_str(when_cond, parseMethodArgs(method))
    assert isinstance(cond, wait.ChareStateMsgCond)

    when_cond = 'iter'
    method    = '(self, iter, x, y)'
    cond = wait.parse_cond_str(when_cond, parseMethodArgs(method))
    assert isinstance(cond, wait.ChareStateMsgCond)

    when_cond = 'self.x'
    method    = '(self, iter, x, y)'
    cond = wait.parse_cond_str(when_cond, parseMethodArgs(method))
    assert isinstance(cond, wait.ChareStateCond)

    when_cond = 'self.x + self.y == 3'
    method    = '(self, iter, x, y)'
    cond = wait.parse_cond_str(when_cond, parseMethodArgs(method))
    assert isinstance(cond, wait.ChareStateCond)

    when_cond = 'self.x > (self.y + 2/3 + self.z + error)'
    method    = '(self, iter, x, y)'
    cond = wait.parse_cond_str(when_cond, parseMethodArgs(method))
    assert isinstance(cond, wait.ChareStateCond)

    exit()


charm.start(main)
