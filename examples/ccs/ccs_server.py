from charm4py import charm, Chare, Array

def handler(msg):
    if charm.CcsIsRemoteRequest():
        answer = "Hello to sender " + msg.payload + "from PE " + str(charm.myPe())
        # send the answer back to the client
        print("CCS Ping handler called on " + str(charm.myPe()))
        charm.CcsSendReply(answer)

def main(args):
    # No need to initialize converse, because charm.start does this
    # just register the handler
    charm.CcsRegisterHandler("ping2", handler)
    charm.CcsRegisterHandler("ping", handler)
    print("CCS Handlers registered . Waiting for net requests...")


charm.start(main)