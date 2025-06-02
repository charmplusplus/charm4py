from charm4py import ray, charm
import numpy as np
import matplotlib.pyplot as plt
import os

# Compute whether a point is in the Mandelbrot set
def mandelbrot_fast(re, im, max_iter):
    zr = zi = 0.0
    for i in range(max_iter):
        zr2 = zr * zr
        zi2 = zi * zi
        if zr2 + zi2 > 4.0:
            return i
        zi = 2 * zr * zi + im
        zr = zr2 - zi2 + re
    return max_iter

# Remote task to compute a tile
@ray.remote
def compute_tile(x_start, x_end, y_start, y_end, width, height, max_iter):
    tile = np.zeros((y_end - y_start, x_end - x_start), dtype=np.uint16)
    for y in range(y_start, y_end):
        for x in range(x_start, x_end):
            re = 3.5 * (x / width) - 2.5
            im = 2.0 * (y / height) - 1.0
            tile[y - y_start, x - x_start] = mandelbrot_fast(re, im, max_iter)
    return tile

def generate_mandelbrot_image_optimized(width=12000, height=8000, max_iter=200, tile_size=1000, max_pending=1000):
    # Pre-create the empty file with the correct size
    total_bytes = 2 * width * height  # 2 bytes per pixel (uint16)
    with open("output/mandelbrot_large.dat", "wb") as f:
        f.seek(total_bytes - 1)
        f.write(b'\0')
    result_image = np.memmap("output/mandelbrot_large.dat", dtype=np.uint16, mode='w+', shape=(height, width))
    pending = []

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)
            tile_ref = compute_tile.remote(x, x_end, y, y_end, width, height, max_iter)
            pending.append(((x, y), tile_ref))

            if len(pending) >= max_pending:
                (x0, y0), tile = pending.pop(0)
                tile = ray.get(tile)
                result_image[y0:y0+tile.shape[0], x0:x0+tile.shape[1]] = tile

    for (x0, y0), tile_ref in pending:
        tile = ray.get(tile_ref)
        result_image[y0:y0+tile.shape[0], x0:x0+tile.shape[1]] = tile

    return result_image


def main(args):
    output_path = "output/mandelbrot_large.dat"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ray.init()
    # Run the benchmark
    image = generate_mandelbrot_image_optimized(width=int(args[1]), height=int(args[2]), max_iter=int(args[3]), tile_size=int(args[4]))
    # Optional: show the result
    plt.imshow(image, cmap='hot')
    plt.title("Mandelbrot Set (Ray)")
    plt.axis('off')
    plt.savefig("mandelbrot_ray.png", dpi=300, bbox_inches='tight')
    os.remove('output/mandelbrot_large.dat')
    charm.exit()

charm.start(main)
