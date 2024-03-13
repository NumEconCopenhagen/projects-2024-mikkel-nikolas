alpha = 1/3
beta = 2/3


w1A = 0.8
w2A = 0.3
p2 = 1
w1B = 1-w1A
w2B = 1-w2A
from types import SimpleNamespace

class inauguralprojectClass:

    def __init__(self):

        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3
        par.p2 = 1

    def utility_A(self,x1A,x2A):
        uA = x1A**alpha*x2A**(1-alpha)

    def utility_B(self,x1B,x2B):
        uB = x1B**beta*x2B**(1-beta)

    def demand_A(self,p1):
        IA = p1*w1A+p2*w2A
        return alpha*IA/p1

    def demand_B(self,p1):
        IB = p1*w1B+p2*w2B
        return beta*IB/p2


    def check_market_clearing(self,p1):

        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2