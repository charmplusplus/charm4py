from charm4py import charm, Chare, Array, Future, Reducer, Group

def handler(msg):
    print("CCS Ping handler called on " + str(charm.myPe()))
    msg = msg.decode('utf-8')
    msg = msg.rstrip('\x00')
    answer = "Hello to sender " + str(msg) + " from PE " + str(charm.myPe()) + ".\n"
    answer_bytes = answer.encode('utf-8')
    charm.CcsSendReply(answer_bytes)

class RegisterPerChare(Chare):

    def register(self, return_future, handler):
        charm.CcsRegisterHandler("ping2", handler)
        charm.CcsRegisterHandler("ping", handler)
        self.reduce(return_future, Reducer.nop)

def main(args):
    # No need to initialize converse, because charm.start does this
    # just register the handler
    reg_wait = Future()
    registers = Group(RegisterPerChare)
    registers.register(reg_wait, handler)
    reg_wait.get()
    print("CCS Handlers registered . Waiting for net requests...")


charm.start(main)