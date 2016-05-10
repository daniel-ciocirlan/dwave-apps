__author__ = 'Daniel'

from collections import namedtuple
from numpy import dot, array
from dwave_sapi import local_connection, BlackBoxSolver

Node = namedtuple('Node', ['index', 'weight'])

class Graph(object):
    def __init__(self, filename):
        f = open(filename, 'r')
        lines = []
        for line in f:
            lines.append(line)
        f.close()

        data = lines[0].split()
        self.nnodes = int(data[0])
        self.nedges = int(data[1])
        self.nodes = []
        self.edges = []

        for i in range(self.nnodes):
            self.nodes.append(Node(i, float(lines[i+1])))
        for i in range(self.nedges):
            edge = lines[1 + self.nnodes + i].split()
            self.edges.append((int(edge[0]), int(edge[1])))

        self.norm = 1

    def has_edge(self, i,j):
        for edge in self.edges:
            if edge[0] == i and edge[1] == j:
                return True
            if edge[0] == j and edge[1] == i:
                return True
        return False

    def n_neighbors(self, i):
        neigh = 0
        for j in range(self.nnodes):
            if self.has_edge(i,j):
                neigh += 1
        return neigh

    def get_data(self):
        max_weight = 0
        for node in self.nodes:
            if max_weight < node.weight:
                max_weight = node.weight

        h = [self.norm * self.n_neighbors(node.index) - 2 * node.weight / max_weight for node in self.nodes]
        J = [[0] * self.nnodes for i in range(self.nnodes)]

        for i in range(self.nnodes):
            for k in range(i):
                if self.has_edge(k,i):
                    J[k][i] = self.norm

        return h,J

    def print_data(self):
        h, J = self.get_data()
        print 'h =', h
        print 'J = '
        for line in J:
            print '\t', line

class WMISIsingSolver(object):
    def __init__(self, graph):
        self.graph = graph
        self.h, self.J = self.graph.get_data()
        solver = local_connection.get_solver('c4-sw_optimize')
        self.blackbox_solver = BlackBoxSolver(solver)
        self.blackbox_parameter = 1
        self.cluster_parameter = 2

    def __call__(self, states, num_states):
        ret = []
        state_len = len(states)/num_states
        for i in range(num_states):
            state = states[i * state_len : (i+1) * state_len]
            # this is the objective function value
            obj_value = dot(self.h, state) + dot (state, dot(self.J,state))
            ret.append(obj_value)
        return tuple(ret)

    def compute(self):
        blackbox_answer = self.blackbox_solver.solve(self,
                                                    self.graph.nnodes,
                                                    cluster_num = self.cluster_parameter,
                                                    min_iter_inner = self.blackbox_parameter,
                                                    max_iter_outer = self.blackbox_parameter,
                                                    unchanged_threshold = self.blackbox_parameter,
                                                    max_unchanged_objective_outer = self.blackbox_parameter,
                                                    max_unchanged_objective_inner = self.blackbox_parameter,
                                                    unchanged_best_threshold = self.blackbox_parameter,
                                                    verbose=0)
        selected_nodes = filter(lambda node:blackbox_answer[node.index] == 1, self.graph.nodes)
        total_weight = reduce(lambda x, node: x + node.weight, selected_nodes, 0)

        return selected_nodes, total_weight

g = Graph('wmis.in')
print 'weights:', [node.weight for node in g.nodes]
print 'edges', g.edges
g.print_data()

solver = WMISIsingSolver(g)
selected_nodes, total_weight = solver.compute()
print
print 'selected nodes indices:', [node.index for node in selected_nodes], 'for a total weight =', total_weight