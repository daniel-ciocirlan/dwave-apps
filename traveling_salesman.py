import math
from dwave_sapi import local_connection, BlackBoxSolver
from numpy import dot, array, prod
from collections import namedtuple

Point = namedtuple('Point', ['index', 'x', 'y'])
Edge = namedtuple('Edge', ['a', 'b', 'length'])

class TSPEvaluator(object):
    def __init__(self, filename):
        self.points = self.get_points(filename)
        self.edges = self.get_edges(self.points)
        self.npoints = len(self.points)
        self.index_matrix = self.build_matrix()
        self.gamma = 999

    def get_points(self, filename):
        f = open(filename, 'r')

        points = []
        for line in f:
            data = [int(x) for x in line.split(' ')]
            points.append(Point(data[0], data[1], data[2]))
        f.close()
        return points

    def get_edges(self,points):
        edges = []
        npoints = len(points)

        for i in range(npoints):
            for j in range(i+1, npoints):
                xi = points[i].x
                yi = points[i].y
                xj = points[j].x
                yj = points[j].y
                dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)

                edges.append(Edge(points[i].index, points[j].index, dist))

        return edges

    def get_path_length(self, edges, edge_selection):
        selected_edges = [edges[i] for i in range(len(edges)) if edge_selection[i] == 1]
        current_node = 0
        prev_node = 0
        length = 0
        end = False

        while end == False:
            prev_node = current_node
            del_index = -1
            for i in range(len(selected_edges)):
                if selected_edges[i].a == current_node:
                    del_index = i
                    current_node = selected_edges[i].b
                    break
                elif selected_edges[i].b == current_node:
                    del_index = i
                    current_node = selected_edges[i].a
                    break

            if current_node == 0:
                length += 1
                end = True
            else:
                if prev_node == current_node:
                    return 0
                else:
                    length += 1
                    del selected_edges[del_index]
        return length

    def build_matrix(self):
        matrix = []
        for i in range(self.npoints):
            line = []
            for edge in self.edges:
                if edge.a == i or edge.b == i:
                    line.append(1)
                else:
                    line.append(0)

            matrix.append(line)
        return matrix

    def __call__(self, states, numStates):
        state_len = len(states)/numStates
        states_bin  = [(item+1)/2 for item in states]

        ret = []
        for state_number in range(numStates):
            # select one "state" from the array = list of +/-1 values meaning
            # whether an edge was considered in the tour or not
            w = array(states_bin[state_number*state_len:(state_number+1)*state_len])
            edge_lengths = [edge.length for edge in self.edges]

            # compute the tour length
            path_length = self.get_path_length(self.edges, w)
            if path_length == 0:
                # deal a massive penalty for non-valid paths
                evaluation = self.gamma**5
            else:
                # the total length of included edges
                evaluation = dot(edge_lengths, w)
                # vertices must be present in the selected edge list exactly twice
                # penalize any non-conforming paths
                for line in self.index_matrix:
                    evaluation += (self.gamma*(dot(line, w) - 2))**2
                # penalize any path longer or shorter than exactly one complete tour
                evaluation += (self.gamma * (path_length - self.npoints))**2
            ret.append(evaluation)

        return tuple(ret)

solver = local_connection.get_solver('c4-sw_optimize')
blackbox_solver = BlackBoxSolver(solver)

evaluator = TSPEvaluator('simple_tsp.in')
print 'The points on the map:'
for point in evaluator.points:
    print point

print 'Calling Blackbox...'
blackbox_parameter = 10
blackbox_answer = blackbox_solver.solve(evaluator,\
                                        len(evaluator.edges),\
                                        cluster_num = 10,\
                                        min_iter_inner = blackbox_parameter,\
                                        max_iter_outer = blackbox_parameter,\
                                        unchanged_threshold = blackbox_parameter,\
                                        max_unchanged_objective_outer = blackbox_parameter,\
                                        max_unchanged_objective_inner = blackbox_parameter,\
                                        unchanged_best_threshold = blackbox_parameter,\
                                        verbose=0)

blackbox_answer_bin = array([(item+1)/2 for item in blackbox_answer])

print 'The best bit string we found was:', blackbox_answer_bin

print 'This corresponds to the edges:'
total_journey = 0.0
for num in range(len(blackbox_answer_bin)):
    if blackbox_answer_bin[num] == 1:
        total_journey += evaluator.edges[num][2]
        print evaluator.edges[num]

energy = evaluator.__call__(blackbox_answer,1)
print 'For a total journey distance of:', total_journey, 'and an energy of', energy[0]