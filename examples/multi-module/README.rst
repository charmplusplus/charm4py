

This example shows how to run a Charm4py program that has chare classes defined
in different modules.

Start main.py to run this program. For example:
$ python3 -m charmrun.start +p4 main.py

In this example, there are chare types defined in main.py, hello.py and
goodbye.py. The modules which contain chare definitions are specified when
charm.start() is called. Note that the script used to launch the program
(in this case main.py) does not need to be specified.

The program creates 2 Groups of chares of type hello.Hello and goodbye.Goodbye
respectively. The remote method 'updateGlobals' of the charm runtime is used
in this example to set global variables of the hello and goodbye modules on
every process, so that chares from these modules can access them. This is
simply meant to illustrate how updateGlobals can be used to update the global
variables of different modules.

At the start of the program, the mainchare broadcasts a message to the "hello"
chares. On receiving the broadcast message, each of these sends a point-to-point
message to the "goodbye" chare on its PE. The goodbye chares then perform a
global reduction, after which the 'done' method of the mainchare is called and
the program exits.
