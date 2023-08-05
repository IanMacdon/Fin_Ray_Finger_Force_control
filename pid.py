import numpy as np
import scipy


class PID:
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.output_max = None
        self.output_min = None
        self.input_min = None
        self.input_max = None
        self.last_time = None
        self.last_output = 0
        self.last_error = 0
        self.last_y = 0
        self.integral = 0

    def update_controller(self, setpoint, y):
        error = setpoint - y
        d_y = y - self.last_y
        d_error = error - self.last_error
        pro_term = self.kp * error
        self.integral = self.integral + self.ki * d_error
        der_term = -self.kd * d_y / self.dt
        u = pro_term + self.integral + der_term
        if self.input_max is not None:
            if u > self.input_max:
                u = self.input_max
            elif u < self.input_min:
                u = self.input_min
        self.last_output = u
        self.last_y = y
        self.last_error = error
        return u
