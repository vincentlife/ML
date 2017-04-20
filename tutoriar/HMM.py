import numpy as np
import pandas as pd
class HMM:
    def __init__(self,Ann,Bnm,pi1n,hstatelist,statelist):
        '''
        HMM model
        :param Ann: the state transition matrix; n shape of hidden states
                                next hidden state
                                [[ 0.5, 0.375, 0.125],
        current hidden  state   [0.25, 0.125, 0.625],
                                [0.25, 0.375, 0.375]]

        :param Bnm: the confusion matrix; m: shape of
                            Observable state
                         [[ 0.6, 0.2, 0.15, 0.05],
        hidden state    [0.25, 0.25, 0.25, 0.25],
                        [0.05, 0.10, 0.35, 0.5]]

        :param pi1n: pi matrix ,the vector of the initial state probabilities;
        '''
        self.A = pd.DataFrame(Ann,index=hstatelist,columns=hstatelist)
        self.B = pd.DataFrame(Bnm,index=hstatelist,columns=statelist)
        self.pi = pi1n
        self.N = len(hstatelist)
        self.M = len(statelist)


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

    def forward(self, T):
        '''
        前向算法，解决已知模型参数，计算某一给定可观察状态序列的概率
        :param T: 观察序列
        :return: 概率
        '''
        resultp = 1.0
        # Store intermediate vector
        temp_vector = self.pi

        for i in range(len(T)):
            resultp = resultp * np.sum(self.B.ix[:,T[i]].data*temp_vector.T,axis=1)[0]
            temp_vector = self.A.values.T.dot(temp_vector)

        return resultp

    def viterbi(self, O):
        '''
        viterbi算法, 给定了一个可观察序列和HMM，寻找最可能的隐藏序列。
        :param O: 观察序列
        :return: 隐藏序列
        '''
        olen = len(O)
        backpointer = np.zeros((self.N, olen))
        backpointer.fill(-1)
        tvector = self.B.ix[:,O[0]].values*self.pi.ravel()
        ttvector = tvector
        for i in range(1,olen):
            for x in range(self.N):
                # i = hiddenstates
                pri = tvector
                # pr of X when i happens
                prXi = self.A.iloc[:, x].values
                # pr of observation when X happens
                proX = self.B.ix[x,O[i]]
                px = pri*prXi*proX
                ttvector[x] = np.max(px)
                print(px)
                print(np.argmax(px))
                print("--------")
                backpointer[x,i] = np.argmax(px)
            tvector = ttvector

        print(backpointer)
        print(tvector)

    # def Baum_Welch(self,O):




if __name__ == '__main__':
    Ann = [[ 0.5, 0.375, 0.125],
          [0.25, 0.125, 0.625],
          [0.25, 0.375, 0.375]]
    Bnm = [[ 0.6, 0.2, 0.15, 0.05],
          [0.25, 0.25, 0.25, 0.25],
          [0.05, 0.10, 0.35, 0.5]]
    pi = np.array([[0.25, 0.5, 0.25]]).T
    hstate = ["sunny","cloudy","rainy"]
    state = ["dry","dryish","damp","soggy"]
    hmm = HMM(Ann,Bnm,pi,hstate,state)
    # hmm.forward([0,2,1,2,3])
    hmm.viterbi([0,2,1,2])



