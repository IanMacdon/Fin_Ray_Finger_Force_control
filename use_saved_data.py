import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from System import System
from dmc import DMC
from scipy.signal import lti
from time import time as ti
from pid import PID


with open('saved_dictionary.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

print(loaded_dict.keys())
time = loaded_dict["time"]
pos = loaded_dict["pos"]
pos = [110 - i for i in pos]
force = loaded_dict["force"]
images = loaded_dict["images"]

ind = next(x[0] for x in enumerate(force) if x[1] > 0)

dt = 0.01
g = lti([33.04, 836.8], [1, 41.08, 809, 0])
Gpid = System(g, dt)
Gdmc = System(g, dt)
Fpid = [0]
t = [0]
pos = [i - pos[ind] for i in pos]
fit = np.polyfit(pos[ind+5:], force[ind+5:], 2)
pos2force = np.poly1d(fit)
est_force = pos2force(pos[ind:])
step = 100
# cdmc = DMC(g, dt, 3500, 500, 0.00001, 0.975, step)
cdmc = DMC(g, dt, 500, 30, 0.00001, 0.95, step)
Fdmc = [0]
ydmc = [0]
ypid = [0]
s = ti()
ulim = 50
cpid = PID(10, 5, 1, dt)
cpid.input_max = ulim
cpid.input_min = -ulim
cdmc.max_u = ulim
cdmc.min_u = -ulim
for i in range(1, int(3/dt)):
    udmc = cdmc.get_control_action(ydmc[i-1])
    ydmc.append(Gdmc.increment_system(udmc))
    Fdmc.append(pos2force(ydmc[i]))
    print(udmc)

    upid = cpid.update_controller(step, ypid[i-1])
    ypid.append(Gpid.increment_system(upid))
    Fpid.append(pos2force(ypid[i]))

    t.append(i*dt)

print(Fpid[-1])
plt.plot(t, Fdmc, t, Fpid)
plt.show()
