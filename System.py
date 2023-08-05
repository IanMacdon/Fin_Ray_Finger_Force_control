import numpy as np
from scipy.signal import lti, cont2discrete, TransferFunction


class System:
    def __init__(self, g, dt):
        self.dt = dt
        self.g = g
        self.gz = g.to_discrete(dt=dt, method='tustin')
        self.u_state = np.zeros((1, len(self.gz.num)))[0]
        self.y_state = np.zeros((1, len(self.gz.den)-1))[0]
        self.y = 0

    def increment_system(self, u):
        self.u_state = np.append(u, self.u_state[0: -1])
        u_terms = np.sum(np.multiply(self.u_state, self.gz.num))
        y_terms = np.sum(np.multiply(self.y_state, -1 * self.gz.den[1:]))
        y = u_terms + y_terms
        self.y_state = np.append(y, self.y_state[0: -1])
        self.y = y
        return y
