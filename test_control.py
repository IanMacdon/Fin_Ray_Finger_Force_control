from dmc import DMC
from WSG_Gripper import Gripper
from pid import PID
from scipy.signal import lti
import time
import matplotlib.pyplot as plt


wsg = Gripper()
dt = 0.01
G = lti([33.04, 836.8], [1, 41.08, 809, 0])
set = 50
c = DMC(G, dt, 800, 70, 0.0001, 0.985, 50)
ulim = 400
c.max_u = ulim
c.min_u = -ulim

start_time = time.time()
last_time = start_time
wsg.homing_gripper()
y = [110 - wsg.read_pos()]
t_vec = [0]
i = 0
cur_time = time.time()
u_vec = [0]
while i*dt < 5:
    cur_time = time.time()
    if cur_time - last_time > dt:
        pos = 110 - wsg.read_pos()
        if cur_time - start_time > 4:
            u = c.get_control_action(pos)
        else:
            u = 0

        u_vec.append(u)
        print('pos', pos, 'u', u, 'error', set-pos, 'dt', i*dt)
        i = i + 1
        wsg.send_speed(u)
        y.append(pos)
        last_time = cur_time
        t_vec.append(cur_time - start_time)

wsg.send_speed(0)
wsg.homing_gripper()
plt.plot(t_vec, y, t_vec, u_vec)
plt.show()
