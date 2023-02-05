# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt 
from matplotlib import cm, colors
from scipy.integrate import odeint
from scipy.signal import square

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
    
# here we aim to simulate how a neural signal with electric potential V_s
# propagates in function of time, through every rc circuit

axon = Axon(N=100)

# timestep
timestep = 1e-6 
# where we stop at each timestep
T = 1e-2
# paces
num_time_steps = int(T/timestep)
# defining linspace for chart
time_steps = np.linspace(0, T, num_time_steps)

# solving a linear system of differential equations with scipy
solver = odeint(axon,axon.initial_conditions,time_steps)

# now plotting the solutions of the linear system we just solved
new_cmap = colors.LinearSegmentedColormap.from_list("", cm.get_cmap("Greens")(np.linspace(0.4, 1))) 

for i in range(0, num_time_steps, 1000):
    plt.plot(
        solver[i, :],
        label = f"time step = {i}",
        color = new_cmap(i/num_time_steps))

plt.plot(solver[-1, :], label="last time step", color="blue")
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (V)')
plt.title('Neural signal in function of time')
plt.show()
