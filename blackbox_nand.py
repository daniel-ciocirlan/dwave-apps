__author__ = 'Daniel'

from dwave_sapi import local_connection, BlackBoxSolver

class NANDEvaluator(object):
    def __init__(self):
        self.gamma = 999
        # initialized, will be set from outside
        self.x = 1
        self.y = 1

    def __call__(self, states, num_states):
        len_state = len(states) / num_states

        ret = []
        for i in range(num_states):
            state = states[i * len_state : (i+1) * len_state]
            # the variables
            x = state[0]
            y = state[1]
            # the result
            g = state[2]

            # verify the NAND function and the required input values
            obj_value = (x * y + x + y + 2 * g - 1) ** 2 + self.gamma * ((x - self.x)**2 + (y - self.y)**2)
            ret.append(obj_value)
        return tuple(ret)

class QNand(object):
    def __init__(self):
        solver = local_connection.get_solver('c4-sw_optimize')
        self.blackbox_solver = BlackBoxSolver(solver)
        self.evaluator = NANDEvaluator()
        self.blackbox_parameter = 10

    def compute(self,x,y):
        # map 0/1 to qubit values +/-1
        self.evaluator.x = x * 2 - 1
        self.evaluator.y = y * 2 - 1
        blackbox_answer = self.blackbox_solver.solve(self.evaluator,
                                                    3,
                                                    cluster_num = 10,
                                                    min_iter_inner = self.blackbox_parameter,
                                                    max_iter_outer = self.blackbox_parameter,
                                                    unchanged_threshold = self.blackbox_parameter,
                                                    max_unchanged_objective_outer = self.blackbox_parameter,
                                                    max_unchanged_objective_inner = self.blackbox_parameter,
                                                    unchanged_best_threshold = self.blackbox_parameter,
                                                    verbose=0)

        # rebuild 0/1 booleans [a,b,h] from [x,y,g]
        blackbox_answer_bin = [(item+1)/2 for item in blackbox_answer]
        # the answer we desire is the third value in the list [a,b,h]
        return blackbox_answer_bin[2]

nand = QNand()

for x in range(2):
    for y in range(2):
        print x, 'NAND', y, '=', nand.compute(x,y)