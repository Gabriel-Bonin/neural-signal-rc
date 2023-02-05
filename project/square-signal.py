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
      
# now we check how a square signal behaves
# the second term in the multiplication defines how many periods we want to see on the chart
# changing the value if t modifies the period of the function
# the third term in the multiplication limits the signal for visualization
V_s = lambda t: 0.1*np.heaviside(square(t*2*np.pi*200), 1)*np.heaviside(0.01 - t, 1) 

# timestep
timestep = 1e-6 
# time of simulation duration
T = 0.05 
num_time_steps = int(T/timestep)
time_steps = np.linspace(0, T, num_time_steps)

axon = Axon(V_s=V_s, N=100)
solver = odeint(axon,axon.initial_conditions,time_steps)

plt.subplot(211)
plt.plot(time_steps, V_s(time_steps))
plt.title("Input signal")
plt.ylabel('Voltage (V)')
plt.subplot(212)
plt.plot(time_steps, solver[:, -1])
plt.title("Output signal")
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (V)')
plt.show()
