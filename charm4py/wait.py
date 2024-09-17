from collections import defaultdict
import ast
from importlib import import_module


# This condition object is used to match one argument of a @when entry method to
# an attribute of a chare
# Example:
#           @when("self.cur_iteration == iter")
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
#           @when("self.check == x + y")
#           def method(self, x, y, z)
#               # invoke method only if self.check == x + y
class ChareStateMsgCond(object):

    # these condition objects don't group elements because each can have different msg arguments
    group = False

    def __init__(self, cond_str, cond_func):
        self.cond_str  = cond_str
        self.cond_func = cond_func

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
        em = self.elem[1]
        self.cond_func = em.when_cond_func


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

    def __init__(self, cond_str, module_name):
        self.cond_str  = cond_str
        self.globals_module_name = module_name
        self.cond_func = eval('lambda self: ' + cond_str,
                              import_module(module_name).__dict__)

    def createWaitCondition(self):
        c = object.__new__(ChareStateCond)
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
        #while eval(me.cond_str):   # eval is very slow
        while self.cond_func(obj):
            elem = self.wait_queue.pop()
            if elem[0] == 0:
                # is msg
                t, em, header, args = elem
                em.run(obj, header, args)
            elif elem[0] == 1:
                # is thread
                tid = elem[1]
                charm.threadMgr.resumeThread(tid, None)
            dequeued = True
            if len(self.wait_queue) == 0:
                break
        return dequeued, len(self.wait_queue) == 0

    def __getstate__(self):
        return self.cond_str, self.wait_queue, self._cond_next, self.globals_module_name

    def __setstate__(self, state):
        self.cond_str, self.wait_queue, self._cond_next, self.globals_module_name = state
        self.cond_func = eval('lambda self: ' + self.cond_str,
                              import_module(self.globals_module_name).__dict__)


def is_tag_cond(root_ast):
    """ Determine if the AST corresponds to a 'when' condition of the form
        `self.xyz == args[x]` where xyz is the name of an attribute, x is an
        integer. if True, returns the condition string, the name of the attribute
        (e.g. xyz) and the integer index (e.g. x). Otherwise returns None """
    try:
        if not isinstance(root_ast.body, ast.Compare):
            return None
        compare = root_ast.body
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

        idx = args.slice.value
        if isinstance(idx, ast.Num):
            idx = idx.n
        elif isinstance(idx, ast.Constant):
            idx = idx.value

        if not isinstance(idx, int):
            return None

        return ('self.' + attrib.attr + ' == args[' + str(idx) + ']', attrib.attr, idx)
    except:
        return None


class MsgArgsTransformer(ast.NodeTransformer):

    def __init__(self, method_arguments):
        self.method_arguments = method_arguments
        self.num_msg_args = 0

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id in self.method_arguments and node.value.id != 'self':
            idx = self.method_arguments[node.value.id]
            self.num_msg_args += 1
            return ast.copy_location(ast.Attribute(
                value=ast.Subscript(
                    value=ast.Name(id='args', ctx=ast.Load()),
                    slice=ast.Index(value=ast.Num(n=idx)),
                    ctx=node.ctx
                ),
                attr=node.attr,
                ctx=node.ctx
            ), node)
        else:
            return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id in self.method_arguments:
            idx = self.method_arguments[node.id]
            self.num_msg_args += 1
            return ast.copy_location(ast.Subscript(
                value=ast.Name(id='args', ctx=ast.Load()),
                slice=ast.Index(value=ast.Num(n=idx)),
                ctx=node.ctx
            ), node)
        else:
            return node


#import astunparse

def parse_cond_str(cond_str, module_name, method_arguments={}):

    #print("Original condition string is", cond_str)
    t = ast.parse(cond_str, filename='<string>', mode='eval')
    if len(method_arguments) > 0:
        # in the AST, convert names of method arguments to `args[x]`, where x is the
        # position of the argument in the function definition
        transformer = MsgArgsTransformer(method_arguments)
        transformer.visit(t)
        #print("Transformed to", astunparse.unparse(t), "num args detected=", transformer.num_msg_args)
        if transformer.num_msg_args == 0:
            return ChareStateCond(cond_str, module_name)
    else:
        return ChareStateCond(cond_str, module_name)

    tag_cond = is_tag_cond(t)
    if tag_cond is not None:
        return MsgTagCond(*tag_cond)

    # compile AST to code, then eval to a lambda function
    new_tree = ast.parse("lambda self, args: x", filename='<string>', mode='eval')
    new_tree.body.body = t.body
    new_tree = ast.fix_missing_locations(new_tree)
    lambda_func = eval(compile(new_tree, '<string>', 'eval'),
                       import_module(module_name).__dict__)
    return ChareStateMsgCond(cond_str, lambda_func)


def charmStarting():
    global charm
    from .charm import charm
