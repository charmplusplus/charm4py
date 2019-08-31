
**Jacobi iteration with a 2D array**

This code uses a 2D blocked decomposition of a 2D array with more than one
data element per chare. The code runs for a specified number of iterations,
or until convergence is reached. It uses a reduction after every iteration to
check for convergence, then proceeds to the next iteration.

A 5-point stencil pattern is used to update the value for each data element.
For example, the new value for element X is the current value of X plus the
current values of its left, right, top, and bottom neighbors::

     T
   L X R
     B

   X' = (X + L + R + T + B) / 5.0

Boundary conditions are obeyed in this example.
