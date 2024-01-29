from charm4py import charm, ray


def square(x):
    return x**2

def main(args):
    ray.init()
    result = charm.pool.map_async(square, range(10), chunksize=2, multi_future=True)
    for element in result:
        print(element.get())
    #print(result)  # prints [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
    exit()

charm.start(main)
