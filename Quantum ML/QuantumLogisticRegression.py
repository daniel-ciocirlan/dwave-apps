from dwave_sapi import local_connection, RemoteConnection, BlackBoxSolver
from numpy import dot, array, prod, arange, meshgrid, linspace
from struct import pack,unpack
from math import *
import matplotlib.pyplot as plt

class QuantumLR(object):
    def __init__(self, filename):
        f = open(filename)
        self.x = []
        self.y = []

        # state length = 3 * sizeof(float)
        self.stateLen = 128

        for line in f:
            tokens = line.split(",")
            e1 = float(tokens[0])
            e2 = float(tokens[1])
            adm = int(tokens[2])
            self.x.append([1, e1, e2, e1*e2])
            self.y.append(adm)

    # converter from qubit to binary values
    def qtobin(self, qubits):
        return [(x + 1) / 2 for x in qubits]

    # converter from binary to qubit values
    def bintoq(self, bin):
        return [2 * x - 1 for x in bin]

    # the sigmoid function
    def h(self, x, theta):
        result = 1 / (1 + exp(-dot(theta, x)))
        return result

    # cost for a single training example
    def singleCost(self, x, y, theta):
        result = - y * log(self.h(x, theta)) - (1 - y) * log(1 - self.h(x, theta))
        return result

    # total cost for a given theta parameter, on all the training set
    def totalCost(self, theta):
        result = 0
        for i in range(len(self.y)):
            # print "computing for "
            result += self.singleCost(self.x[i], self.y[i], theta)

        return result/len(self.y)

    # the callback of this objective function evaluator
    def __call__(self, states, numStates):
        ret = []
        states_bin = self.qtobin(states)

        for i in range(numStates):
            try:
                state = states_bin[i * self.stateLen : (i+1) * self.stateLen]
                theta0 = self.as_float32(state[:32])
                theta1 = self.as_float32(state[32:64])
                theta2 = self.as_float32(state[64:96])
                theta3 = self.as_float32(state[96:128])
                # theta4 = self.as_float32(state[128:])
                theta = [theta0, theta1, theta2, theta3]

                cost = self.totalCost(theta)
                ret.append(cost)
            except:
                ret.append(float("inf"))
                pass


        return tuple(ret)

    # Where the bits2int function converts bits to an integer.
    def bits2int(self, bits):
        # You may want to change ::-1 if depending on which bit is assumed
        # to be most significant.
        bits = [int(x) for x in bits[::-1]]

        x = 0
        for i in range(len(bits)):
            x += bits[i]*2**i
        return x

    def as_float32(self, bitlist):
        """
        See: http://en.wikipedia.org/wiki/IEEE_754-2008
        """
        return unpack("f",pack("I", self.bits2int(bitlist)))[0]

    def binary(self, num):
        import struct
        return ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', num))

    def convert_answer(self, bb_answer):
        bin = [(x + 1) / 2 for x in bb_answer]
        theta0 = self.as_float32(bin[:32])
        theta1 = self.as_float32(bin[32:64])
        theta2 = self.as_float32(bin[64:96])
        theta3 = self.as_float32(bin[96:128])
        # theta4 = self.as_float32(bin[128:])
        return [theta0, theta1, theta2, theta3]

solver = local_connection.get_solver('c4-sw_optimize')
blackbox_parameter = 10

obj = QuantumLR("data.txt")

blackbox_solver = BlackBoxSolver(solver)
print 'contacting super black box...'

blackbox_answer = blackbox_solver.solve(obj, obj.stateLen)

print blackbox_answer
theta = obj.convert_answer(blackbox_answer)
print theta
print obj.totalCost(theta)

r = arange(0, 100, 1)

x = linspace(30,100, 100)
y = linspace(30,100, 100)
X, Y = meshgrid(x,y)
F = -1.121 + 3.0517576306010596e-05 * X + 0.00390625 * Y + 0.00024414061044808477 * X * Y
plt.contour(X,Y,F,[0])

plt.plot([obj.x[i][1] for i in range(len(obj.y)) if obj.y[i] == 1], [obj.x[i][2] for i in range(len(obj.y)) if obj.y[i] == 1], "go")
plt.plot([obj.x[i][1] for i in range(len(obj.y)) if obj.y[i] == 0], [obj.x[i][2] for i in range(len(obj.y)) if obj.y[i] == 0], "ro")

plt.show()