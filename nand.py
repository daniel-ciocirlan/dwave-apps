__author__ = 'Daniel'

# Import D-Wave's Python API

from dwave_sapi import local_connection

solver = local_connection.get_solver('c4-sw_sample')

#define the problem

h = [0]*128
J = dict()

h[48] = -0.1
h[49] = -0.1
h[53] = -0.2

J[(48,53)] = 0.2
J[(48,52)] = 0.1
J[(49,52)] = -1
J[(49,53)] = 0.2

#send the problem to hardware

answer = solver.solve_ising(h,J,num_reads = 100)['solutions'][0]
print '48 = ', answer[48]
print '49 = ', answer[49]
print '52 = ', answer[52]
print '53 = ', answer[53]


answer = solver.solve_ising(h,J,num_reads = 100)['energies'][0]
print answer
