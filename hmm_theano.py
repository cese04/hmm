import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt


def random_normalized(d1, d2):
    x = np.random.randn((d1, d2))
    return x / x.sum(axis=1, keepdims=True)


class HMM:

    def __init__(self, M):
        self.arg = M

    def fit(self, X, learning_rate=0.001, max_iter=10,
            V=None, p_cost=1.0, print_period=10):
        if V is None:
            V = max(max(x) for x in X) + 1
        N = len(X)

        pi0 = np.ones(self.M) / self.M
        A0 = random_normalized(self.M, self.M)
        B0 = random_normalized(self.M, V)

        thx, cost = self.set(pi0, A0, B0)

    def set(self, pi, A, B):
        self.pi = theano.shared(pi)
        self.A = theano.shared(A)
        self.B = theano.shared(B)

        thx = 

def fit_coin():
    X = []
    for line in open('coin_data.txt'):
        # 1 for H, 0 for T
        x = [1 if e == "H" else 0 for e in line.rstrip()]
        X.append(x)

    hmm = HMM(2)
    hmm.fit(X)
    L = hmm.get_cost_multi(X).sum()
    print("LL with fitted params: ", L)

    pi = np.array([0.5, 0.5])
    A = np.array([0.1, 0.9], [0.8, 0.2])
    B = np.array([0.6, 0.4], [0.3, 0.7])
    hmm.set(pi, A, B)
    L = hmm.get_cost_multi(X).sum()
    print("LL with real params: " L)

if __name__ == '__main__':
    fit_coin()
