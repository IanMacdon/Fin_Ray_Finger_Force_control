from scipy.signal import lti
import matplotlib.pyplot as plt
from pid import PID
from System import System
from dmc import DMC


dt = 0.01
g = lti([33.04, 836.8], [1, 41.08, 809, 0])
G = System(g, dt)

c = DMC(g, dt, 100, 25, 0.00001, 0.975, 50)
c.max_u = 100
c.min_u = -100
F = [0]
t = [0]
for i in range(1, int(5/dt)):
    u = c.get_control_action(F[i-1])
    F.append(G.increment_system(u))
    t.append(dt*i)

plt.plot(t, F)
plt.show()
