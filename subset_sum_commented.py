from dwave_sapi import local_connection, RemoteConnection, BlackBoxSolver
from numpy import dot, array, prod

class MyClass(object):
#
# We send in any parameters we want to. In this case this is the values of our particular subset sum
# instance. (The set of numbers that we are trying to find a subset of)
#
    def __init__(self, subset_sum_array):
        self.subset_sum_array = subset_sum_array
#
# Now we define and compute the value of the generating function.
#
    def __call__(self, states, numStates):
#
# First we get the length of each state; this is just the number of variables in our problem. In our
# example, this will be five.
#
        stateLen = len(states)/numStates
#
# An important point is that BlackBox natively solves problems assuming that the variables are +1/-1
# variables. In cases where you'd prefer to use 0/1 variables (such as in this example), you need to
# explicitly convert. Here we convert the states list into a new list called states_bin which stores the
# suggested answers in 0/1 variables.
#
        states_bin  = [(item+1)/2 for item in states] # converting to 0/1 from -1/+1
#
# We now create a list called ret where we'll store all the results of computing the value of the
# generating function on the states sent in.
#
        ret = []
#
# Now cycle over all the states sent in.
#
        for state_number in range(numStates):
#
# The w array stores the current state of interest.
#
            w = array(states_bin[state_number*stateLen:(state_number+1)*stateLen])
#
# Now we compute the value of the generating function, given the bit string w.
#
            result = dot(self.subset_sum_array, w)**2+prod(1-w)
#
# We then append that result to the ret list.
#
            ret.append(result)
#
# Once we've done this for all the values sent in, the result is returned as a tuple.
#
        return tuple(ret)
#
##########################################################################################################]

##########################################################################################################
#
# This is the main program for using BlackBox to solve a subset sum instance.
#
# First we establish a connection to a local solver and choose a solver type. Here we'll use the 
# optimization solver defined over the full 128 variable
# graph representing a fully functional Rainier processor.
#

# create a remote connection using url and token

solver = local_connection.get_solver('c4-sw_optimize')
#
# The blackbox_parameter is an integer that sets a bunch of solver parameters within the BlackBox
# algorithm. As a developer you actually have a lot more flexibility than just setting a single parameter.
# But for the time being you can just use the rule of thumb that the bigger this is, the more work
# BlackBox will put into finding the best possible solution. Here we'll set it to 10. You can experiment
# with decreasing or increasing it.
#
blackbox_parameter = 10
#
# This is an array giving the values of our initial set.
#
subset_sum_array = array ([-7, -4, -2, 5, 2])
#
# num_vars is just the length of the set.
#
num_vars = len(subset_sum_array)
#
# obj is an object of the class ObjClass. Pass variables in here (in this case subset_sum_array).
#
obj = MyClass(subset_sum_array)
#
# Set up an instance of BlackBoxSolver
#
blackbox_solver = BlackBoxSolver(solver)
#
# Now run BlackBox. Here you have to pass in obj, num_vars, and cluster_num, but everything else is
# optional. For most users you can just do what I'm doing here and tie all of these parameters to a
# single common value.
#
print 'contacting super black box...'

blackbox_answer = blackbox_solver.solve(obj, num_vars, cluster_num = 10, \
    min_iter_inner = blackbox_parameter, max_iter_outer= blackbox_parameter, \
    unchanged_threshold=blackbox_parameter, max_unchanged_objective_outer=blackbox_parameter, \
    max_unchanged_objective_inner = blackbox_parameter, \
    unchanged_best_threshold = blackbox_parameter, verbose=0)
#
# blackbox_answer returns a list of 1/-1 variables denoting the best solution it found. Since we want
# 0/1 variables we convert it to that, and cast it as an array.
#
blackbox_answer_bin = array([(item+1)/2 for item in blackbox_answer])
#
# Now we output the answer!
# 
print 'The best bit string we found was:',blackbox_answer_bin
#
# This stores the subset found in a list.
#
subset_list = []
for k in range(num_vars):
    if blackbox_answer_bin[k] ==1:
        subset_list.append(subset_sum_array[k])
#
print 'The subset this corresponds to is:', subset_list
#
# This computes the value of the generating function at the best answer found.
#
energy_of_best_solution_found = dot(subset_sum_array, blackbox_answer_bin)**2+prod(1-blackbox_answer_bin)
print 'Its energy is:', energy_of_best_solution_found
#
if energy_of_best_solution_found==0:
    print 'We found a solution... this set has a subset that sums to zero!'
#
# And that's it!
#
##########################################################################################################
