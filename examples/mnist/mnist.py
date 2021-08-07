# Distributed deep learning example with MNIST dataset, using Charm4py and PyTorch.
# Adapted from https://pytorch.org/tutorials/intermediate/dist_tuto.html
# and https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py.
# Supports overdecomposition and load balancing.

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
from charm4py import charm, Chare, Group, Array, threaded, Reducer
import numpy as np

# Add LB command line arguments
sys.argv += ['+LBOff', '+LBCommOff', '+LBObjOnly']

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

# Initialize PyTorch on each PE
class TorchInit(Chare):

    def init(self, num_threads):
        torch.set_num_threads(num_threads)
        torch.manual_seed(1234)
        print(f'PE {charm.myPe()} initialized PyTorch with {num_threads} threads')

# Chare array
class Worker(Chare):

    def __init__(self, num_workers, epochs, lb_epochs):
        self.num_workers = num_workers
        self.epochs = epochs
        self.lb_epochs = lb_epochs
        self.agg_time = 0
        self.time_cnt = 0
        self.agg_time_all = 0
        if isinstance(self.thisIndex, tuple):
            # is chare array element (assume 1D chare array)
            assert len(self.thisIndex) == 1
            self.myrank = self.thisIndex[0]
        else:
            # is group element
            self.myrank = self.thisIndex

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
        partition = partition.use(self.myrank)
        train_set = torch.utils.data.DataLoader(partition,
                                                batch_size=bsz,
                                                shuffle=True)
        return train_set, bsz

    # Distributed SGD
    @threaded
    def run(self, done_future=None):
        if done_future is not None:
            # Starting a new run
            self.done_future = done_future
            self.train_set, bsz = self.partition_dataset()
            self.model = Net()
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)
            self.num_batches = math.ceil(len(self.train_set.dataset) / float(bsz))
            self.epoch = 0

        while self.epoch < self.epochs:
            if self.epoch == 0:
                charm.LBTurnInstrumentOn()
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
            print(f'Chare {self.thisIndex[0]:4d} | PE {charm.myPe():4d} | Epoch {self.epoch:4d} | Loss {(epoch_loss / self.num_batches):9.3f} | Time {(time.time() - t0):9.3f}')
            self.epoch += 1
            if (self.lb_epochs > 0) && (self.epoch % self.lb_epochs == 0):
                # Start load balancing
                self.AtSync()
                return

        print(f'Chare {self.thisIndex[0]:4d} training complete, average allreduce time (us): {((self.agg_time / self.time_cnt) * 1000000):9.3f}')
        self.agg_time_all = self.allreduce(self.agg_time, Reducer.sum).get()
        if self.myrank == 0:
            print(f'Chare {self.thisIndex[0]:4d} all average allreduce time (us): {((self.agg_time_all / self.num_workers / self.time_cnt) * 1000000):9.3f}')
        self.contribute(None, None, self.done_future)

    # Gradient averaging
    def average_gradients(self, model):
        for param in model.parameters():
            # Flatten gradient data
            data_shape = param.grad.data.shape
            reshaped_data = param.grad.data.reshape(-1)

            # Blocking allreduce
            start_time = time.time()
            agg_data = self.allreduce(reshaped_data, Reducer.sum).get()
            self.agg_time += time.time() - start_time
            self.time_cnt += 1

            # Restore original shape of gradient data
            param.grad.data = agg_data.reshape(data_shape) / float(self.num_workers)

    # Return method from load balancing
    def resumeFromSync(self):
        self.thisProxy[self.thisIndex].run()

def main(args):
    # Initialize PyTorch on all PEs
    Group(TorchInit).init(1, ret=True).get()

    # Create chare array and start training
    num_workers = charm.numPes()
    epochs = 6
    lb_epochs = 0
    workers = Array(Worker, num_workers, args=[num_workers, epochs, lb_epochs], useAtSync=True)
    t0 = time.time()
    done = charm.createFuture()
    print(f'Starting MNIST dataset training with {num_workers} chares on {charm.numPes()} PEs for {epochs} epochs (LB every {lb_epochs} epochs)')
    workers.run(done)
    done.get()

    # Training complete
    print(f'Done. Elapsed time: {(time.time() - t0):9.3f} s')
    charm.exit()

charm.start(main)
