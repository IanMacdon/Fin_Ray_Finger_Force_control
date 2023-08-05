import numpy as np
from System import System
import matplotlib.pyplot as plt
from scipy.signal import lti


class DMC:
    def __init__(self, g, dt, p, m, la, alpha, sp):
        self.P, self.m, self.la, self.alpha, self.g = p, m, la, alpha, g
        self.dt = dt                                                                                  
        self.max_u = None
        self.min_u = None
        self.A = None
        self.u_last = np.zeros([1, m])[0]
        self.yhat = None
        self.setpoint = None
        self.sp = sp
        self.gz = g.to_discrete(dt=dt, method='tustin')
        print(self.gz)
        self.dyn_mat = self.get_dynamic_matrix()
        self.update_setpoint(0)

    def get_dynamic_matrix(self):
        yr = self.get_step_response()
        # plt.plot(yr)
        #plt.show()
        self.yhat = np.asarray(yr)
        f = yr
        A = np.zeros([self.P, self.m])
        A[:, 0] = f
        for i in range(1, self.m):
            f = np.insert(f[0:-1], 0, 0)
            A[:, i] = f
        self.A = A
        R = self.la * np.eye(self.m)
        d = np.matmul(np.linalg.inv(np.matmul(np.transpose(A), A) + R), np.transpose(A))
        return d

    def get_step_response(self):
        sys = System(self.g, self.dt)
        y = [0]
        for i in range(0, self.P - 1):
            y.append(sys.increment_system(1))
        y = np.asarray(y)
        return y

    def update_setpoint(self, y, order=0):
        if order == 0:
            y = [y]
            for j in range(1, self.P):
                y.append(self.alpha * y[j-1] + (1 - self.alpha)*self.sp)
            self.setpoint = y

    def get_control_action(self, y):
        phi = y - self.yhat[0]
        self.yhat = self.yhat + phi
        self.update_setpoint(y)
        e = self.setpoint - self.yhat
        du = np.matmul(self.dyn_mat, e)
        new_u = du + self.u_last

        if self.max_u is not None:
            new_u = [self.max_u if uk > self.max_u else self.min_u if uk < self.min_u else uk for uk in new_u]

        du = new_u - self.u_last
        dy = np.matmul(self.A, du)
        self.yhat = self.yhat + dy
        val = self.yhat[1:]
        self.yhat = np.append(val, val[-1])
        u = du + self.u_last
        self.u_last = u
        return u[0]
