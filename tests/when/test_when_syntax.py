from charmpy import charm

# NOTE: this is not really a charm program

def main(args):

    import ast
    import wait

    good = []
    good.append(("self.xyz == args[0]", 'xyz', 0))
    good.append(("self.xyz == args[3]", 'xyz', 3))
    good.append(("self.xyz  ==  args[3 ]", 'xyz', 3))
    good.append(("args[0] == self.xyz", 'xyz', 0))
    good.append(("args[3] == self.abc", 'abc', 3))
    good.append(("args[ 3 ] == self.abc", 'abc', 3))
    good.append(("args[ 3]  ==   self.abc  ", 'abc', 3))
    good.append(("args[-1]  ==  self.my_iterations", 'my_iterations', -1))

    for cond_str, attrib_name, idx in good:
        r = wait.is_tag_cond(ast.parse(cond_str))
        assert (r is not None) and (r[1] == attrib_name) and (r[2] == idx), cond_str

    bad = []
    bad.append("xyz")
    bad.append("xyz == args[0]")
    bad.append("args[3] == xyz")
    bad.append("self.xyz <= args[0]")
    bad.append("self.xyz + args[0]")
    bad.append("self.xyz == abc")
    bad.append("self.xyz == args[0] + args[1]")
    bad.append("self.xyz == args[0")
    bad.append("self.xyz == arg[0]")
    bad.append("self.xyz == x[0]")
    bad.append("self.xyz + 2 == args[0]")

    for cond_str in bad:
        try:
            syntax_tree = ast.parse(cond_str)
            r = wait.is_tag_cond(syntax_tree)
        except:
            r = None
        assert r is None

    charm.exit()


charm.start(main)
