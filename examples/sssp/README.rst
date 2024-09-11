
**Single source shortest paths (SSSP)**

This program solves the single source shortest paths (SSSP) problem in parallel.
This implementation uses a parallel Bellman-Ford approach, which repeatedly
relaxes edges until the distance values of each vertex from the source can no
longer be improved any further. Termination is detected using the quiescence
detection feature in Charm4Py. To run this program, run the following command::

    $ python3 -m charmrun.start +p<N> <num_vertices> <num_edges> <random_seed> <source_vertex>

where:
* num_vertices = the number of vertices in the graph
* num_edges = the number of edges in the graph
* random_seed = random seed used to generate graph information
* source_vertex = the vertex from which distances will be calculated

All graph information will be automatically generated before the algorithm executes.
The source and destination of each edge is uniformly generated over the interval [0, num_vertices).
The weights of each edge are uniformly generated over [0.0, 1.0).
The graph is a directed graph, and between any pair of vertices, multiple edges may exist.