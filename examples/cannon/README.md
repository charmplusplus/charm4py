# Cannon's Matrix Multiplication Example

This example demonstrates an implementation of Cannon's Matrix Multiplication algorithm using Charm4py. It showcases several features of Charm4py, including 2-dimensional chare arrays, Channels, Reductions, and Futures. The example also utilizes Numba to accelerate computation.

## What it does

Cannon's algorithm is a distributed algorithm for matrix multiplication. It divides the input matrices into sub-matrices and distributes them among a 2D grid of processors (or chares, in this case). The algorithm then performs a series of local multiplications and data shifts to compute the final result.

Key features demonstrated:

1. Use of Numba for accelerated matrix multiplication
2. 2-dimensional chare arrays for distributed computation
3. Channels for efficient communication between chares
4. Reductions and Futures for synchronization

## How to run

1. Ensure Scipy is installed (required) and Numba is installed for improved performance (optional):
   ```
   pip install scipy numba
   ```

2. Run the example with the following command:
   ```
   python3 -m charmrun.start +pN cannon.py <matrix_dim> <chare_dim>
   ```
   Where:
   - `N` is the number of PEs
   - `<matrix_dim>` is the dimension of the input matrices (must be a perfect square)
   - `<chare_dim>` is the dimension of the chare grid (must be a perfect square)
   - `<matrix_dim>` must be divisible by `<chare_dim>`

   For example:
   ```
   python3 -m charmrun.start +p4 1000 10
   ```
   This will multiply two 1000x1000 matrices using a 10x10 grid of chares.

3. The program will output the size of each chare's sub-array and the total execution time.

## Requirements

- Charm4py (assumed to be installed)
- Numpy
- Numba (optional, for improved performance)
- Scipy

## Note

This example is designed to showcase Charm4py features and distributed computing concepts. For production use, consider using optimized libraries like ScaLAPACK for large-scale matrix operations.
