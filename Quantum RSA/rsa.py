from dwave_sapi import local_connection, RemoteConnection, BlackBoxSolver
from numpy import dot, array, prod

SIZE = 32
debug = True

def binaryToInt(qubits):
    result = 0
    for qbit in qubits:
        bit = (qbit + 1) / 2
        result = result * 2 + bit
    return result

def extendedGcd(a, b):
    s = 0
    t = 1
    r = b

    old_s = 1
    old_t = 0
    old_r = a

    while r != 0:
        q = old_r / r
        old_r, r = (r, old_r - q * r)
        old_s, s = (s, old_s - q * s)
        old_t, t = (t, old_t - q * t)

    # Bezout coefficients old_s, old_t
    # gcd = old_r
    return (old_s, old_t, old_r)

def modInverse(a, m):
    t = 0
    r = m
    newt = 1
    newr = a

    while newr != 0:
        q = r / newr
        r, newr = (newr, r - q * newr)
        t, newt = (newt, t - q * newt)

    if r > 1:
        return 0
    while t < 0:
        t = t + m
    return t

def expmod(a,b,c):
    x = 1
    for i in xrange(0,b):
        x = (x*a)%c
    return x

class QFactoring(object):
    def __init__(self, n):
        self.n = n

    def __call__(self, states, numStates):
        ret = []
        for i in range(numStates):
            beginIndex = i * 2 * SIZE
            halfIndex = i * 2 * SIZE + SIZE
            endIndex = (i + 1) * 2 * SIZE

            num1 = binaryToInt(states[beginIndex:halfIndex])
            num2 = binaryToInt(states[halfIndex:endIndex])

            ret.append((num1 * num2 - self.n) ** 2)

        return tuple(ret)

    def solve(self):
        solver = local_connection.get_solver('c4-sw_optimize')
        blackbox_parameter = 10

        blackbox_solver = BlackBoxSolver(solver)
        if debug:
            print 'contacting super black box...'

        blackbox_answer = blackbox_solver.solve(self, 2 * SIZE, cluster_num = 10, \
            min_iter_inner = blackbox_parameter, max_iter_outer= blackbox_parameter, \
            unchanged_threshold=blackbox_parameter, max_unchanged_objective_outer=blackbox_parameter, \
            max_unchanged_objective_inner = blackbox_parameter, \
            unchanged_best_threshold = blackbox_parameter, verbose=0)

        blackbox_answer_bin = array([(item+1)/2 for item in blackbox_answer])
        a, b = (binaryToInt(blackbox_answer[0:SIZE]), binaryToInt(blackbox_answer[SIZE:]))

        if debug:
            print 'The best bit string we found was:',blackbox_answer_bin
            print 'the factors are ', a, ' and ', b

        return (a, b)

class RSACracker(object):
    def __init__(self, publicKey):
        self.m, self.e = publicKey
        # qf = QFactoring(self.m)
        # self.p, self.q = qf.solve()
        self.p, self.q = (61, 53)
        self.phi = (self.p - 1) * (self.q - 1)
        self.privateKey = modInverse(self.e, self.phi)

    def crack(self, encValue):
        return expmod(encValue, self.privateKey, self.m)

cracker = RSACracker((3233, 17))
print cracker.crack(2790)