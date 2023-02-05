# -*- coding: utf-8 -*-

## Simulating a brain signal with RC circuits

# creating a class to simulate an axon

class Axon:

    def __init__(
        self, 
        V_s = 0.1,
        C = 1e-10,
        R1 = 1e8,
        R2 = 1e6,
        N = 1,
    ):

        self.R1 = R1
        self.R2 = R2
        self.C = C
        self.initial_conditions = np.zeros(N)
    
        # We make this two conditions below in order to ensure that V_s is (int,float) and still can be callable
        if isinstance(V_s, (int, float)): 
           self.V_s = lambda t: V_s
           self.initial_conditions[0] = V_s
        elif callable(V_s):
            self.V_s = V_s
            self.initial_conditions[0] = V_s(0)

    def dV(self, V_m, V_n):

        # Consider V_m == V_(N-1) and V_n == V_N
        I2 = (V_m - V_n) / self.R2
        I1 = V_m / self.R1

        dV =(I2 - I1) / self.C

        return dV

    def __call__(self, u, t):

        # u is an array with voltage in a given time t
        new_u = np.zeros_like(u)

        new_u[0] = self.dV(self.V_s(t), u[0])

        for i in range(len(u) - 1):
            new_u[i + 1] += self.dV(u[i], u[i + 1]) 

        return new_u
