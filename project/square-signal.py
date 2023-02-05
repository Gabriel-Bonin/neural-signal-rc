# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt 
from matplotlib import cm, colors
from scipy.integrate import odeint
from scipy.signal import square

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
