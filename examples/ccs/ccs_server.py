from charm4py import charm, Chare, Array, Future, Reducer, Group

class RegisterPerChare(Chare):

    def register(self, return_future):
        charm.CcsRegisterHandler("ping2", handler)
        charm.CcsRegisterHandler("ping", handler)
        self.reduce(return_future, 0, Reducer.sum)

def handler(msg):
    if charm.CcsIsRemoteRequest():
        answer = "Hello to sender " + msg.payload + "from PE " + str(charm.myPe())
        # send the answer back to the client
        print("CCS Ping handler called on " + str(charm.myPe()))
        charm.CcsSendReply(answer)

def main(args):
    # No need to initialize converse, because charm.start does this
    # just register the handler
    reg_wait = Future()
    registers = Group(RegisterPerChare)
    registers.register(reg_wait)
    reg_wait.get()
    print("CCS Handlers registered . Waiting for net requests...")


charm.start(main)