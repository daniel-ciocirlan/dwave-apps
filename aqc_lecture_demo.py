from dwave_sapi import local_connection, RemoteConnection, BlackBoxSolver
from numpy import dot, array, prod

class MyClass(object):
    def __init__(self, subset_sum_array):
        self.subset_sum_array = subset_sum_array

    def __call__(self, states, numStates):
        // TODO write evaluator
        ret = []
        return tuple(ret)

solver = local_connection.get_solver('c4-sw_optimize')
blackbox_parameter = 10

// TODO prepare data
subset_sum_array = array ([-7, -4, -2, 5, 2])
s = 1
num_vars = len(subset_sum_array)

obj = MyClass(subset_sum_array)
blackbox_solver = BlackBoxSolver(solver)
print 'contacting super black box...'

blackbox_answer = blackbox_solver.solve(obj, num_vars, cluster_num = 10, \
    min_iter_inner = blackbox_parameter, max_iter_outer= blackbox_parameter, \
    unchanged_threshold=blackbox_parameter, max_unchanged_objective_outer=blackbox_parameter, \
    max_unchanged_objective_inner = blackbox_parameter, \
    unchanged_best_threshold = blackbox_parameter, verbose=0)

// TODO decode data
blackbox_answer_bin = array([(item+1)/2 for item in blackbox_answer])

print 'The best bit string we found was:', blackbox_answer_bin
subset_list = []
for k in range(num_vars):
    if blackbox_answer_bin[k] == 1:
        subset_list.append(subset_sum_array[k])
print 'The subset this corresponds to is:', subset_list

energy_of_best_solution_found = dot(subset_sum_array, blackbox_answer_bin)**2+prod(1-blackbox_answer_bin)
print 'Its energy is:', energy_of_best_solution_found

if energy_of_best_solution_found==0:
    print 'We found a solution... this set has a subset that sums to', s, '!'