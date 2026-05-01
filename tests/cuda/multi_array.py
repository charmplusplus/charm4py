from charm4py import charm, Chare, Array, coro, Future, Channel, Group, ArrayMap
import numpy as np
from numba import cuda
import array


class A(Chare):
    def __init__(self, msg_size):
        self.msg_size = msg_size
        if type(self.thisIndex) is tuple:
            self.idx = int(self.thisIndex[0])
        else:
            self.idx = self.thisIndex

    @coro
    def run(self, done_future, addr_optimization=False):
        partner = self.thisProxy[int(not self.idx)]
        partner_channel = Channel(self, partner)

        device_data = cuda.device_array(self.msg_size, dtype='int8')
        device_data2 = cuda.device_array(self.msg_size, dtype='int8')
        # if addr_optimization:
        d_addr = array.array('L', [0, 0])
        d_size = array.array('i', [0, 0])

        d_addr[0] = device_data.__cuda_array_interface__['data'][0]
        d_addr[1] = device_data2.__cuda_array_interface__['data'][0]

        d_size[0] = device_data.nbytes
        d_size[1] = device_data2.nbytes

        host_array = np.array(self.msg_size, dtype='int32')
        host_array.fill(42)

        if self.idx:
            h1 = np.ones(self.msg_size, dtype='int8')
            h2 = np.zeros(self.msg_size, dtype='int8')
            device_data.copy_to_device(h1)
            device_data2.copy_to_device(h2)
            if addr_optimization:
                partner_channel.send(20, host_array, src_ptrs=d_addr,
                                     src_sizes=d_size
                                     )
                partner_channel.recv()
            else:
                partner_channel.send(20, host_array, device_data, device_data2)
        else:
            if addr_optimization:
                f, g = partner_channel.recv(post_addresses=d_addr,
                                            post_sizes=d_size
                                            )
            else:
                f, g = partner_channel.recv(device_data, device_data2)
            partner_channel.send(1)
            h1 = device_data.copy_to_host()
            h2 = device_data2.copy_to_host()

            assert f == 20
            assert np.array_equal(host_array, g)
            assert np.array_equal(h1, np.ones(self.msg_size, dtype='int8'))
            assert np.array_equal(h2, np.zeros(self.msg_size, dtype='int8'))
        self.reduce(done_future)


class ArrMap(ArrayMap):
    def procNum(self, index):
        return index[0] % 2


def main(args):
    # if this is not a cuda-aware build,
    # vacuously pass the test
    if not charm.CkCudaEnabled():
        print("WARNING: Charm4Py was not build with CUDA-enabled Charm++. "
              "GPU-Direct functionality will not be tested"
              )
        charm.exit(0)

    peMap = Group(ArrMap)
    chares = Array(A, 2, args=[(1<<30)], map=peMap)
    done_fut = Future()
    chares.run(done_fut, addr_optimization=False)
    done_fut.get()

    done_fut = Future()
    chares.run(done_fut, addr_optimization=True)
    done_fut.get()

    chares = Group(A, args=[(1<<30)])
    done_fut = Future()
    chares.run(done_fut, addr_optimization=False)
    done_fut.get()

    done_fut = Future()
    chares.run(done_fut, addr_optimization=True)
    done_fut.get()
    charm.exit(0)


charm.start(main)
