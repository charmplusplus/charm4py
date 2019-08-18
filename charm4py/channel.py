from .threads import threadMgr, LocalFuture


class Channel(object):

    def __new__(cls, chare, remote, local=None):
        if not hasattr(chare, '__channels__'):
            chare.__initchannelattrs__()
        ch = chare.__findPendingChannel__(remote, False)
        if ch is None:
            local_port = len(chare.__channels__)
            ch = _Channel(local_port, remote, True)
            chare.__channels__.append(ch)
            chare.__pendingChannels__.append(ch)
        else:
            ch.setEstablished()
        if local is None:
            # if local is None, we assume local endpoint is the individual chare
            if hasattr(chare, 'thisIndex'):
                local = chare.thisProxy[chare.thisIndex]
            else:
                local = chare.thisProxy
        remote._channelConnect__(local, ch.port)
        return ch


CHAN_BUF_SIZE = 40000

class _Channel(object):

    def __init__(self, port, remote, locally_initiated):
        self.port = port
        self.remote = remote
        self.remote_port = -1
        self.send_seqno = 0
        self.recv_seqno = 0
        self.data = {}
        self.recv_fut = None
        self.established = False
        self.established_fut = None
        self.locally_initiated = locally_initiated

    def setEstablished(self):
        self.established = True
        del self.established_fut
        del self.locally_initiated

    def send(self, *msg):
        if not self.established:
            self.established_fut = LocalFuture()
            self.established_fut.get()
            self.setEstablished()
        self.remote._channelRecv__(self.remote_port, self.send_seqno, *msg)
        self.send_seqno = (self.send_seqno + 1) % CHAN_BUF_SIZE

    def recv(self):
        if self.recv_seqno in self.data:
            ret = self.data.pop(self.recv_seqno)
        else:
            self.recv_fut = LocalFuture()
            ch, ret = self.recv_fut.get()
            self.recv_fut = None
        self.recv_seqno = (self.recv_seqno + 1) % CHAN_BUF_SIZE
        return ret


def waitgen(channels):
    n = len(channels)
    f = LocalFuture()
    for ch in channels:
        if ch.recv_seqno in ch.data:
            n -= 1
            yield ch
        else:
            ch.recv_fut = f
    while n > 0:
        ch, msg = threadMgr.pauseThread()
        ch.data[ch.recv_seqno] = msg
        ch.recv_fut = None
        n -= 1
        yield ch
