#!/bin/bash
#BSUB -W 30
#BSUB -P csc357
#BSUB -nnodes 256
#BSUB -J jacobi3d-charm4py-strong-n256

# These need to be changed between submissions
file=jacobi3d.py
n_nodes=256
n_procs=$((n_nodes * 6))
grid_width=3072
grid_height=3072
grid_depth=3072

# Function to display commands
exe() { echo "\$ $@" ; "$@" ; }

cd $HOME/work/charm4py/examples/cuda/gpudirect/jacobi3d

#exe conda init bash
#exe conda activate charm4py
exe source activate charm4py

exe export LD_LIBRARY_PATH=$HOME/work/ucx/install/lib:$HOME/work/pmix-3.1.5/install/lib:/sw/summit/gdrcopy/2.0/lib64:$LD_LIBRARY_PATH
exe export UCX_MEMTYPE_CACHE=n

ppn=1
pemap="L0,4,8,84,88,92"
n_iters=100
warmup_iters=10

echo "# Charm4py Jacobi3D Performance Benchmarking (GPUDirect off)"

for iter in 1 2 3
do
  date
  echo "# Run $iter"
  exe jsrun -n$n_procs -a1 -c$ppn -g1 -K3 -r6 --smpiargs="-disable_gpu_hooks" python3 ./$file -x $grid_width -y $grid_height -z $grid_depth -w $warmup_iters -i $n_iters +ppn $ppn +pemap $pemap
done

echo "# Charm4py Jacobi3D Performance Benchmarking (GPUDirect on)"

for iter in 1 2 3
do
  date
  echo "# Run $iter"
  exe jsrun -n$n_procs -a1 -c$ppn -g1 -K3 -r6 --smpiargs="-disable_gpu_hooks" python3 ./$file -x $grid_width -y $grid_height -z $grid_depth -w $warmup_iters -i $n_iters +ppn $ppn +pemap $pemap -d
done
