from dwave_sapi import local_connection , RemoteConnection, BlackBoxSolver
# additional imports if necessary
# ...
class MyEvaluatorObject ( object ):
    def __init__ (self, working_set, target):
    # initialize the object just as any other
        pass

    def __call__ (self , states , numStates ):
    # initialize the return values as a list
        ret = []
        stateLen = len(states) / numStates

        for i in range(numStates):
            # array of +/-1
            state = [states[k] for k in range(i * stateLen, (i+1) * stateLen, 1)]
            state_bin = [(bit + 1)/ 2 for bit in state]

            result = 0
            # compute result for this state

            ret.append(result)

        return tuple ( ret )

# additional methods if necessary
# ...
# in the main part of the program

superset = [-7, 4, 2, 5, -8, 0, 1]
num_vars = len(superset)

solver = local_connection.get_solver('c4-sw_optimize')
blackbox_solver = BlackBoxSolver ( solver )
# initialize variables and all other necessary values
# ...

obj = MyEvaluatorObject ( superset , -1)

blackbox_parameter = 10
blackbox_answer = blackbox_solver.solve (obj , num_vars , \
    cluster_num = 10, \
    min_iter_inner = blackbox_parameter , \
    max_iter_outer = blackbox_parameter , \
    unchanged_threshold = blackbox_parameter ,\
    max_unchanged_objective_outer = blackbox_parameter , \
    max_unchanged_objective_inner = blackbox_parameter , \
    unchanged_best_threshold = 5,\
    verbose =0)

# process the answer if a list of +/ -1 is not suitable
print blackbox_answer

# decode
selected_values = []
for i in range(num_vars):
    if (blackbox_answer[i] == 1):
        selected_values.append(superset[i])

print selected_values
