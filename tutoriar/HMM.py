import numpy as np

class HMM:
    def __init__(self,Ann,Bnm,pi1n):
        '''
        HMM model
        :param Ann: the state transition matrix; n shape of hidden states
        :param Bnm: the confusion matrix; m: shape of
        :param pi1n: pi matrix ,the vector of the initial state probabilities;
        '''
        self.A = Ann
        self.B = Bnn
        self.pi = pi1n
        self.N = self.A.shape[0]
        self.M = self.B.shape[1]


    def printhmm(self):
        print("================describe HMM=========================")
        print("HMM content: N =", self.N, ",M =", self.M)
        for i in range(self.N):
            if i == 0:
                print("hmm.A ", self.A[i, :], " hmm.B ", self.B[i, :])
            else:
                print("      ", self.A[i, :], "       ", self.B[i, :])
        print("hmm.pi", self.pi)
        print("==================================================")

    def forward(self,O,T):
        O = T.shape[0]

