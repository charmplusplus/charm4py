from collections import defaultdict
import ast


# This condition object is used to match one argument of a @when entry method to
# an attribute of a chare
# Example:
#           @when("self.cur_iteration == args[0]")
#           def method(self, iter, x, y, z)
#               # invoke method only if self.cur_iteration == iter
class MsgTagCond(object):

    group = True

    def __init__(self, cond_str, attrib_name, arg_idx):
        self.cond_str    = cond_str
        self.attrib_name = attrib_name
        self.arg_idx     = arg_idx

    def evaluateWhen(self, obj, args):
        return args[self.arg_idx] == getattr(obj, self.attrib_name)

    def createWaitCondition(self):
        c = object.__new__(MsgTagCond)
        c.cond_str    = self.cond_str
        c.attrib_name = self.attrib_name
        c.arg_idx     = self.arg_idx
        c.wait_queue  = defaultdict(list)
        return c

    def enqueue(self, elem):
        elem_type, em, header, args = elem
        self.wait_queue[args[self.arg_idx]].append((em, header, args))

    def check(self, obj):
        dequeued = False
        while True:
            attrib = getattr(obj, self.attrib_name)
            if attrib not in self.wait_queue:
                # no msg waiting for this attribute value
                break
            else:
                msgs = self.wait_queue[attrib]
                em, header, args = msgs.pop()
                em.run(obj, header, args)
                if len(msgs) == 0:
                    del self.wait_queue[attrib]

                dequeued = True
        return dequeued, len(self.wait_queue) == 0


# Manage a conditional statement involving a chare's state and the contents of a message
# Example:
#           @when("self.check == args[0] + args[1]")
#           def method(self, x, y, z)
#               # invoke method only if self.check == x + y
class ChareStateMsgCond(object):

    # these condition objects don't group elements because each can have different msg arguments
    group = False

    def __init__(self, cond_str):
        self.cond_str  = cond_str
        self.cond_func = eval('lambda self, args: ' + cond_str)

    def createWaitCondition(self):
        c = object.__new__(ChareStateMsgCond)
        c.cond_str  = self.cond_str
        c.cond_func = self.cond_func
        return c

    def evaluateWhen(self, obj, args):
        #return eval(cond_str)    # eval is very slow
        return self.cond_func(obj, args)

    def enqueue(self, elem):
        self.elem = elem

    def check(self, obj):
        t, em, header, args = self.elem
        #if eval(me.cond_str):    # eval is very slow
        if self.cond_func(obj, args):
            em.run(obj, header, args)
            return True, True
        return False, False

    def __getstate__(self):
        return self.cond_str, self.elem, self._cond_next

    def __setstate__(self, state):
        self.cond_str, self.elem, self._cond_next = state
        self.cond_func = eval('lambda self, args: ' + self.cond_str)


# Conditional statements involving only a chare's state
# Examples:
#           @when("self.ready")
#           def method(self, x, y, z)
#               # invoke method only if self.ready is True
#
#           @threaded
#           def method(self, ...):
#               for nb in nbs: nb.work(...)
#               self.msgsRecvd = 0
#               self.wait("self.msgsRecvd == len(self.nbs)")
#               ...
class ChareStateCond(object):

    group = True

    def __init__(self, cond_str, charm):
        self.charm     = charm
        self.cond_str  = cond_str
        self.cond_func = eval('lambda self: ' + cond_str)

    def createWaitCondition(self):
        c = object.__new__(ChareStateCond)
        c.charm      = self.charm
        c.cond_str   = self.cond_str
        c.cond_func  = self.cond_func
        c.wait_queue = []
        return c

    def evaluateWhen(self, obj, args):
        #return eval(me.cond_str)   # eval is very slow
        return self.cond_func(obj)

    def enqueue(self, elem):
        self.wait_queue.append(elem)

    def check(self, obj):
        dequeued = False
        while self.cond_func(obj):
        #while eval(me.cond_str):   # eval is very slow
            elem = self.wait_queue.pop()
            if elem[0] == 0:
                # is msg
                t, em, header, args = elem
                em.run(obj, header, args)
            elif elem[0] == 1:
                # is thread
                tid = elem[1]
                self.charm.threadMgr.resumeThread(tid, None)
            dequeued = True
            if len(self.wait_queue) == 0: break
        return dequeued, len(self.wait_queue) == 0

    def __getstate__(self):
        return self.cond_str, self.wait_queue, self._cond_next

    def __setstate__(self, state):
        self.cond_str, self.wait_queue, self._cond_next = state
        self.cond_func = eval('lambda self: ' + self.cond_str)
        import charmpy
        self.charm = charmpy.charm


def is_tag_cond(root_ast):
    """ Determine if the AST corresponds to a 'when' condition of the form
        `self.xyz == args[x]` where xyz is the name of an attribute, x is an
        integer. if True, returns the condition string, the name of the attribute
        (e.g. xyz) and the integer index (e.g. x). Otherwise returns None """
    try:
        if len(root_ast.body) != 1:
            return None
        expr = root_ast.body[0]
        if not isinstance(expr, ast.Expr):
            return None

        compare = expr.value
        if (len(compare.ops) != 1) or (not isinstance(compare.ops[0], ast.Eq)):
            return None

        left, right = compare.left, compare.comparators
        if len(right) != 1:
            return None
        right = right[0]

        attrib, args = None, None
        if isinstance(left, ast.Attribute) and (isinstance(right, ast.Subscript)):
            attrib, args = left, right
        elif isinstance(right, ast.Attribute) and (isinstance(left, ast.Subscript)):
            attrib, args = right, left

        if (attrib is None) or (attrib.value.id != 'self'):
            return None

        if args.value.id != 'args':
            return None

        if isinstance(args.slice.value, ast.UnaryOp):
            if not isinstance(args.slice.value.op, ast.USub):
                return None
            else:
                idx = -args.slice.value.operand.n
        else:
            idx = args.slice.value.n
        if not isinstance(idx, int):
            return None

        return ('self.' + attrib.attr + ' == args[' + str(idx) + ']', attrib.attr, idx)
    except:
      return None
