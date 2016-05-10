__author__ = 'Daniel'

from dwave_sapi import local_connection, BlackBoxSolver

class SATSolver(object):
    def __init__(self, filename):
        self.clauses = []
        self.process_file(filename)
        solver = local_connection.get_solver('c4-sw_optimize')
        self.blackbox_solver = BlackBoxSolver(solver)
        self.blackbox_parameter = 10
        self.gamma = 99999

    def process_file(self, filename):
        file = open(filename, 'r')
        # retain the clauses as lists of pairs (var_index, value=+/-1)
        for line in file:
            tokens = line.split()
            self.clauses.append([(int(tokens[2*i]), int(tokens[2*i+1])) for i in range(len(tokens) / 2)])
        file.close()
        # determine the number of variables in our list as the largest var index
        self.nvars = 0
        for clause in self.clauses:
            for pair in clause:
                if pair[0] > self.nvars:
                    self.nvars = pair[0]
        self.nvars += 1

    def __call__(self, states, num_states):
        ret = []
        for i in range(num_states):
            # retrieve a candidate solution
            state = states[i * self.nvars : (i+1) * self.nvars]
            # the product of (x_i + x_i+)
            sat_value = 1
            # each clause is a list of (index, x_index+) pairs
            for clause in self.clauses:
                clause_sum = 0
                # each pair contributes to the satisfiability measure
                for pair in clause:
                    # pair[1] = x_index+, state[pair[0]] = state[index] = x_index
                    clause_sum += (state[pair[0]] + pair[1]) ** 2
                sat_value *= clause_sum
            # final objective value
            ret.append(self.gamma - sat_value)
        return tuple(ret)

    def solve(self):
        blackbox_answer = self.blackbox_solver.solve(self,
                                                    self.nvars,
                                                    cluster_num = 10,
                                                    min_iter_inner = self.blackbox_parameter,
                                                    max_iter_outer = self.blackbox_parameter,
                                                    unchanged_threshold = self.blackbox_parameter,
                                                    max_unchanged_objective_outer = self.blackbox_parameter,
                                                    max_unchanged_objective_inner = self.blackbox_parameter,
                                                    unchanged_best_threshold = self.blackbox_parameter,
                                                    verbose=0)
        energy = self.__call__(blackbox_answer,1)[0]

        return (energy != self.gamma), blackbox_answer

sat_solver = SATSolver('simple_sat.in')
sat, answer = sat_solver.solve()

if sat:
    print 'The expression IS satisfiable and a possible assignment is', answer
else:
    print 'The expression is NOT satisfiable'