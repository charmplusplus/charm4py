# Distributed deep learning example with MNIST dataset, using mpi4py and PyTorch.
# Adapted from https://pytorch.org/tutorials/intermediate/dist_tuto.html
# and https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random
import time
import sys
from torch.autograd import Variable
from torchvision import datasets, transforms
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

# Dataset partitioning helper
class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = random.Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

# Neural network architecture
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Worker object (1 per MPI rank)
class Worker(object):

    def __init__(self, num_workers, epochs):
        self.num_workers = num_workers
        self.epochs = epochs
        self.agg_time = 0.0
        self.time_cnt = 0
        self.agg_time_all = 0.0

    # Partitioning MNIST dataset
    def partition_dataset(self):
        dataset = datasets.MNIST('./data', train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                 ]))
        size = self.num_workers
        bsz = int(128 / float(size))  # my batch size
        partition_sizes = [1.0 / size for _ in range(size)]
        partition = DataPartitioner(dataset, partition_sizes)
        partition = partition.use(rank)
        train_set = torch.utils.data.DataLoader(partition,
                                                batch_size=bsz,
                                                shuffle=True)
        return train_set, bsz

    # Distributed SGD
    def run(self):
        # Starting a new run
        self.train_set, bsz = self.partition_dataset()
        self.model = Net()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)
        self.num_batches = math.ceil(len(self.train_set.dataset) / float(bsz))
        self.epoch = 0

        while self.epoch < self.epochs:
            t0 = time.time()
            epoch_loss = 0.0
            for data, target in self.train_set:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                epoch_loss += loss.item()
                loss.backward()
                self.average_gradients(self.model)
                self.optimizer.step()
            print(f'Rank {rank:4d} | Epoch {self.epoch:4d} | Loss {(epoch_loss / self.num_batches):9.3f} | Time {(time.time() - t0):9.3f}')
            self.epoch += 1

        print(f'Rank {rank:4d} training complete, average allreduce time (us): {((self.agg_time / self.time_cnt) * 1000000):9.3f}')
        agg_time_arr = np.array([self.agg_time])
        agg_time_all_arr = np.array([0.0])
        comm.Allreduce(agg_time_arr, agg_time_all_arr, op=MPI.SUM)
        self.agg_time_all = agg_time_all_arr[0]
        if rank == 0:
            print(f'Rank {rank:4d} all average allreduce time (us): {((self.agg_time_all / self.num_workers / self.time_cnt) * 1000000):9.3f}')


    # Gradient averaging
    def average_gradients(self, model):
        for param in model.parameters():
            # Obtain numpy arrays from gradient data
            data_shape = param.grad.data.shape
            send_data = param.grad.data.numpy()
            recv_data = np.copy(send_data)

            # Blocking allreduce
            start_time = time.time()
            comm.Allreduce(send_data, recv_data, op=MPI.SUM)
            self.agg_time += time.time() - start_time
            self.time_cnt += 1

            # Restore original shape of gradient data
            param.grad.data = torch.from_numpy(recv_data)
            param.grad.data /= float(self.num_workers)

def main():
    # Initialize PyTorch on all PEs
    num_threads = 1
    torch.set_num_threads(num_threads)
    torch.manual_seed(1234)
    print(f'MPI rank {rank} initialized PyTorch with {num_threads} threads')

    # Create workers and start training
    epochs = 6
    workers = Worker(nprocs, epochs)
    t0 = time.time()
    print(f'Starting MNIST dataset training with {nprocs} MPI processes for {epochs} epochs')
    workers.run()

    comm.Barrier()

    # Training complete
    if rank == 0:
        print(f'Done. Elapsed time: {(time.time() - t0):9.3f} s')

main()
