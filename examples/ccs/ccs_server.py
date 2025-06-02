from charm4py import charm, Chare, Array, Future, Reducer, Group, liveviz, coro

def handler(msg):
    print("CCS Ping handler called on " + str(charm.myPe()))

    msg = msg.rstrip('\x00') #removes null characters from the end
    answer = "Hello to sender " + str(msg) + " from PE " + str(charm.myPe()) + ".\n"
    # send the answer back to the client
    charm.CcsSendReply(answer)

class RegisterPerChare(Chare):

  def register(self, request):
    data = bytearray(25 * 25 * 3)
    for i in range(25*25):
      data[i*3 + self.thisIndex%3] = 200  # Red
    liveviz.LiveViz.deposit(data, self, self.thisIndex*25, 0, 25, 25, 25, 100)
        

def main(args):
    # No need to initialize converse, because charm.start does this
    # just register the handler
    reg_wait = Future()
    registers = Group(RegisterPerChare)
    config = liveviz.Config()
    liveviz.LiveViz.init(config, registers, registers.register)
    print("CCS Handlers registered . Waiting for net requests...")


charm.start(main, modules=['charm4py.liveviz'])