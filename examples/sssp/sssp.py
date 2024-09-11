from charm4py import charm, Chare, Array, coro, Channel, Future, Reducer
import random
import time

class SsspChares(Chare):

    def __init__(self):
        self.local_graph = [] #[vertex_index, edge_list, distance]
        self.partition_indices = []
        self.start_vertex = 0
        self.num_local_vertices = 0
        self.my_index = charm.myPe()
    
    def get_partition(self, edge_list, partition_indices, callback):
        self.partition_indices = partition_indices
        self.start_vertex = partition_indices[self.my_index]
        self.num_local_vertices = partition_indices[self.my_index + 1] - partition_indices[self.my_index]
        self.local_graph = [[self.start_vertex + i, [], float('inf')] for i in range(self.num_local_vertices)]
        for i in range(len(edge_list)):
            self.local_graph[edge_list[i][0] - self.start_vertex][1].append((edge_list[i][1], edge_list[i][2]))
        self.reduce(callback, None, Reducer.nop)
    
    def calculate_destination(self, vertex_index):
        for i in range(len(self.partition_indices)-1):
            if vertex_index >= self.partition_indices[i] and vertex_index < self.partition_indices[i+1]:
                return i
        return len(self.partition_indices)-1
    
    def update_distance(self, update):
        local_index = update[0]-self.start_vertex
        if update[1] < self.local_graph[local_index][2]:
            self.local_graph[update[0]-self.start_vertex][2] = update[1]
            for i in range(len(self.local_graph[local_index][1])):
                dest_vertex = self.local_graph[local_index][1][i][0]
                dest_partition = self.calculate_destination(dest_vertex)
                cost = self.local_graph[local_index][2] + self.local_graph[local_index][1][i][1]
                new_update = (dest_vertex, cost)
                self.thisProxy[dest_partition].update_distance(new_update)
    
    def print_results(self, callback):
        max_local_cost = 0.0
        for i in range(len(self.local_graph)):
            #print("Final cost of vertex", self.local_graph[i][0], ":", self.local_graph[i][2])
            if self.local_graph[i][2] > max_local_cost:
                max_local_cost = self.local_graph[i][2]
        self.reduce(callback, max_local_cost, Reducer.max)


class Main(Chare):

    def __init__(self, args):
        if len(args) != 5:
            print("Wrong number of arguments. Usage: sssp.py <num_vertices> <num_edges> <random_seed> <source_vertex>")
            exit()
        #define parameters
        self.num_vertices = int(args[1])
        self.num_edges = int(args[2])
        self.random_seed = int(args[3])
        self.source_vertex = int(args[4])
        if self.source_vertex < 0 or self.source_vertex > self.num_vertices-1:
            print("Source vertex out of range")
            exit()
        #generate edges randomly and sort them by edge source
        begin_generation = time.time()
        random.seed(self.random_seed)
        self.edge_list = []
        for i in range(self.num_edges):
            edge_source = random.randint(0, self.num_vertices-1)
            edge_dest = random.randint(0, self.num_vertices-1)
            while edge_source==edge_dest:
                edge_dest = random.randint(0, self.num_vertices-1)
            edge_weight = random.random()
            self.edge_list.append((edge_source, edge_dest, edge_weight))
        self.edge_list.sort(key=lambda a: a[0])
        #initiate worker array
        num_partitions = charm.numPes()
        self.workers = Array(SsspChares, num_partitions)
        charm.awaitCreation(self.workers)
        #split edges by pe
        send_lists = [[] for _ in range(num_partitions)]
        avg_partition_size = self.num_edges // num_partitions
        for i in range(len(self.edge_list)):
            partition_num = i // avg_partition_size
            if partition_num >= num_partitions:
                partition_num = num_partitions - 1
            send_lists[partition_num].append(self.edge_list[i])
        #move edges to keep vertices intact
        for i in range(1, len(send_lists)):
            if len(send_lists[i-1])!=0 and send_lists[i-1][-1][0]==send_lists[i][0][0]:
                last_previous_vertex = send_lists[i-1][-1][0]
                while len(send_lists[i]) > 0 and send_lists[i][0][0] == last_previous_vertex:
                    edge_to_move = send_lists[i].pop(0)
                    send_lists[i-1].append(edge_to_move)
        #define partition indices
        partition_indices = []
        for i in range(len(send_lists)):
            if len(send_lists[i]) > 0:
                partition_indices.append(send_lists[i][0][0])
            else:
                partition_indices.append(partition_indices[-1])
        partition_indices.append(self.num_vertices)
        generation_length = time.time() - begin_generation
        #send information to pes
        f = Future()
        for i in range(num_partitions):
            self.workers[i].get_partition(send_lists[i], partition_indices, f)
        f.get()
        #find partition of start vertex
        source_partition = 0
        for i in range(len(partition_indices)-1):
            if self.source_vertex >= partition_indices[i] and self.source_vertex < partition_indices[i+1]:
                source_partition = i
                break
        begin_algo = time.time()
        self.workers[source_partition].update_distance((self.source_vertex, 0.0))
        charm.waitQD()
        algo_length = time.time()-begin_algo
        final_stats = Future()
        self.workers.print_results(final_stats)
        global_max = final_stats.get()
        print("Generation time:", generation_length)
        print("Global max cost:", global_max)
        print("Algorithm runtime:", algo_length)
        exit()

    



charm.start(Main)