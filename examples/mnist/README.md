## Distributed MNIST with Pytorch

This example implements a data-parallel distributed training algorithm on the MNIST dataset. The implementation uses pytorch to do computation on the GPU, if available. After compute at every time step, data is collected on the CPU for global reduction.

### Running the example

First, install the necessary dependencies:

`pip install torch torchvision`

The Charm4py implementation can be run with srun or charmrun:

`srun python mnist.py`
`python -m charmrun.start mnist.py +p2`

To run the mpi4py example with mpiexec on 2 processors:

`mpiexec -n 2 python -m mpi4py mnist-mpi4py.py`
