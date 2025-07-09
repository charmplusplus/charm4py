from charm4py import charm, ray


def square(x):
    return x**2

def twice(x):
    return 2 * x

def main(args):
    ray.init()
    results = charm.pool.map_async(square, [4], chunksize=1, multi_future=True)
    results_twice = charm.pool.map_async(twice, results, chunksize=1, multi_future=True)

    for x in results_twice:
        print(x.get())

    #print(result)  # prints [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
    exit()

charm.start(main)
