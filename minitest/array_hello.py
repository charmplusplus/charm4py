from charm4py import charm, Chare, Array

# This example creates a multi-dimensional distributed chare array, and has
# elements of the array pass a message consecutively from one to the next in
# row-major order


class Hello(Chare):

    def __init__(self, array_dims):
        self.array_dims = array_dims

    def sayHi(self, hello_num):
        print('Hi[' + str(hello_num) + '] from element', self.thisIndex, 'on PE', charm.myPe())
        lastIdx = tuple([size-1 for size in self.array_dims])
        if self.thisIndex == lastIdx:
            # this is the last index, we are done
            print('All done')
            exit()
        else:
            # send a hello message to the next element (in row-major order)
            nextIndex = list(self.thisIndex)
            num_dims = len(self.array_dims)
            for i in range(num_dims-1, -1, -1):
                nextIndex[i] = (nextIndex[i] + 1) % self.array_dims[i]
                if nextIndex[i] != 0:
                    break
            return self.thisProxy[nextIndex].sayHi(hello_num + 1)


def main(args):
    print('\nUsage: array_hello.py [dim1_size dim2_size ...]')
    array_dims = (2, 2, 2)  # default: create a 2 x 2 x 2 chare array
    if len(args) > 1:
        array_dims = tuple([int(x) for x in args[1:]])

    num_elems = 1
    for size in array_dims:
        num_elems *= size
    print('Running Hello on', charm.numPes(), 'processors for', num_elems,
          'elements, array dimensions are', array_dims)

    # create a chare array of Hello chares, passing the array dimensions to
    # each element's constructor
    array_proxy = Array(Hello, array_dims, args=[array_dims])
    firstIdx = (0,) * len(array_dims)
    # send hello message to the first element
    array_proxy[firstIdx].sayHi(17)


charm.start(main)
